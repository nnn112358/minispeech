"""Self-aligning components for non-autoregressive TTS (FastPitch / RAD-TTS /
"One TTS Alignment To Rule Them All"). The FastSpeech learns the phoneme<->mel
alignment DURING training (no external aligner): a small AlignmentEncoder gives a
soft alignment, a forward-sum (CTC) loss drives it to be monotonic, and MAS
extracts hard durations to supervise the duration predictor. All training-only —
inference uses just the duration predictor."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sqzw.monotonic_align import maximum_path

_PRIOR_CACHE = {}


def _beta_binomial(P, M, scaling=1.0):
    """(M, P) beta-binomial prior peaking on the diagonal (frame i -> phoneme i*P/M)."""
    key = (P, M)
    if key not in _PRIOR_CACHE:
        from scipy.stats import betabinom
        x = np.arange(P)
        rows = [betabinom(P - 1, scaling * i, scaling * (M + 1 - i)).pmf(x) for i in range(1, M + 1)]
        _PRIOR_CACHE[key] = np.asarray(rows, dtype=np.float32)
    return _PRIOR_CACHE[key]


def prior_batch(phon_lens, mel_lens, T_phon, T_mel):
    """Padded beta-binomial prior (B, T_mel, T_phon); biases the soft attention toward
    a monotonic-diagonal alignment early in training (RAD-TTS / One-TTS-Alignment)."""
    B = len(phon_lens)
    out = np.zeros((B, T_mel, T_phon), dtype=np.float32)
    for b in range(B):
        P, M = phon_lens[b], mel_lens[b]
        out[b, :M, :P] = _beta_binomial(P, M)
    return torch.from_numpy(out)


class AlignmentEncoder(nn.Module):
    """Soft alignment between text (keys) and mel (queries) via cosine similarity
    with a learnable scale (so the attention can sharpen as features converge)."""
    def __init__(self, n_mel=80, n_text=256, n_att=256):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.key_proj = nn.Sequential(
            nn.Conv1d(n_text, n_text * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_text * 2, n_att, 1))
        self.query_proj = nn.Sequential(
            nn.Conv1d(n_mel, n_mel * 2, 3, padding=1), nn.ReLU(),
            nn.Conv1d(n_mel * 2, n_att, 1))

    def forward(self, mel, text, key_mask=None, attn_prior=None, prior_weight=1.0):
        # mel: (B, n_mel, T_mel)   text: (B, n_text, T_phon)   key_mask: (B, T_phon) True=pad
        # attn_prior: (B, T_mel, T_phon) beta-binomial diagonal prior (early-training guide)
        q = F.normalize(self.query_proj(mel), dim=1)            # (B, n_att, T_mel) unit per frame
        k = F.normalize(self.key_proj(text), dim=1)            # (B, n_att, T_phon) unit per phoneme
        qk = torch.bmm(q.transpose(1, 2), k)                    # cosine sim (B, T_mel, T_phon) in [-1,1]
        score = (self.scale * qk).unsqueeze(1)                  # (B, 1, T_mel, T_phon)
        if attn_prior is not None and prior_weight > 0:
            score = score + prior_weight * torch.log(attn_prior[:, None] + 1e-8)
        attn_logprob = score.clone()
        if key_mask is not None:
            score = score.masked_fill(key_mask[:, None, None, :], -float("inf"))
        attn_soft = F.softmax(score, dim=3)                     # over phonemes
        return attn_soft, attn_logprob


class ForwardSumLoss(nn.Module):
    """CTC forward-sum: maximize prob of the monotonic phoneme order over mel frames."""
    def __init__(self, blank_logprob=-1.0):
        super().__init__()
        self.blank = blank_logprob
        self.ctc = nn.CTCLoss(zero_infinity=True)

    def forward(self, attn_logprob, phon_lens, mel_lens):
        # attn_logprob: (B, 1, T_mel, T_phon)
        a = F.pad(attn_logprob, (1, 0), value=self.blank)       # blank class at index 0
        total = 0.0
        for b in range(a.shape[0]):
            tgt = torch.arange(1, phon_lens[b] + 1, device=a.device).unsqueeze(0)   # (1, T_phon)
            cur = a[b, :, : mel_lens[b], : phon_lens[b] + 1]                          # (1, T_mel, T_phon+1)
            cur = F.log_softmax(cur, dim=-1)
            total = total + self.ctc(cur.permute(1, 0, 2), tgt,
                                     mel_lens[b:b + 1], phon_lens[b:b + 1])
        return total / a.shape[0]


def hard_durations(attn_soft, phon_lens, mel_lens):
    """MAS on the soft alignment -> hard path -> per-phoneme durations (B, T_phon)."""
    B, _, T_mel, T_phon = attn_soft.shape
    dev = attn_soft.device
    mask = torch.zeros(B, T_mel, T_phon, device=dev)
    for b in range(B):
        mask[b, : mel_lens[b], : phon_lens[b]] = 1.0
    neg_cent = torch.log(attn_soft.squeeze(1).clamp_min(1e-8))   # (B, T_mel, T_phon)
    path = maximum_path(neg_cent, mask)                          # (B, T_mel, T_phon)
    return path.sum(1), path                                    # durations (B, T_phon)


def bin_loss(attn_soft, attn_hard):
    """Push the soft alignment toward the hard MAS path (KL-ish)."""
    return -(attn_hard * attn_soft.clamp_min(1e-8).log()).sum() / attn_hard.sum().clamp_min(1.0)
