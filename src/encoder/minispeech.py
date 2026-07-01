"""MiniSpeechEncoder: compact non-AR acoustic model. OpenJTalk phoneme ids -> 80-mel
(22050/hop256). Conv-based blocks (depthwise-separable), no self-attention.
A duration predictor is trained so inference needs no alignment."""
import json, math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset


class FSSet(Dataset):
    def __init__(self, manifest, cache_mels=False):
        self.items = json.load(open(manifest))
        # opt-in lazy RAM cache: with num_workers=0 the mel .npy reads are
        # synchronous and dominate epoch time for small models; caching the mels
        # (~70KB each) removes the disk-I/O stall. Off by default so very large
        # datasets don't blow up RAM.
        self.cache = {} if cache_mels else None
    def __len__(self): return len(self.items)
    def _mel(self, i, path):
        if self.cache is None:
            return torch.from_numpy(np.load(path)).float()
        m = self.cache.get(i)
        if m is None:
            m = torch.from_numpy(np.load(path)).float(); self.cache[i] = m
        return m
    def __getitem__(self, i):
        it = self.items[i]
        dur = torch.LongTensor(it["durations"]) if "durations" in it else torch.zeros(len(it["phoneme_ids"]), dtype=torch.long)
        return (torch.LongTensor(it["phoneme_ids"]), dur, self._mel(i, it["mel"]))


def collate(b):
    phs, durs, mels = zip(*b)
    B = len(b); Tp = max(p.shape[0] for p in phs); Tf = max(m.shape[1] for m in mels)
    ph = torch.zeros(B, Tp, dtype=torch.long); du = torch.zeros(B, Tp, dtype=torch.long)
    me = torch.zeros(B, 80, Tf); pl = torch.LongTensor([p.shape[0] for p in phs]); ml = torch.LongTensor([m.shape[1] for m in mels])
    for i, (p, d, m) in enumerate(zip(phs, durs, mels)):
        ph[i, :p.shape[0]] = p; du[i, :d.shape[0]] = d; me[i, :, :m.shape[1]] = m
    return ph, du, me, pl, ml


class ConvBlock(nn.Module):
    def __init__(self, d, k=5, drop=0.1):
        super().__init__()
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.pw = nn.Conv1d(d, d, 1)
        self.norm = nn.LayerNorm(d); self.ff = nn.Sequential(nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, d))
        self.norm2 = nn.LayerNorm(d); self.drop = nn.Dropout(drop)
    def forward(self, x):
        y = x.transpose(1, 2); y = self.pw(F.relu(self.dw(y))).transpose(1, 2)
        x = self.norm(x + self.drop(y))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DurationPredictor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(d, d, 3, padding=1), nn.ReLU(), nn.Conv1d(d, d, 3, padding=1), nn.ReLU())
        self.lin = nn.Linear(d, 1)
    def forward(self, x):
        return self.lin(self.net(x.transpose(1, 2)).transpose(1, 2)).squeeze(-1)


def length_regulate(x, durations, max_len):
    outs = []
    for b in range(x.shape[0]):
        ex = torch.repeat_interleave(x[b], durations[b].clamp(min=0), dim=0)
        if ex.shape[0] < max_len: ex = F.pad(ex, (0, 0, 0, max_len - ex.shape[0]))
        else: ex = ex[:max_len]
        outs.append(ex)
    return torch.stack(outs, 0)


def length_regulate_gather(hidden, dur):
    """ONNX-friendly length regulate for batch=1 using cumsum+gather.
    No repeat_interleave. All ops are ONNX opset-11+.

    Args:
        hidden: (1, L, d) encoder hidden states
        dur: (1, L) integer durations per phoneme

    Returns:
        expanded: (1, T, d) where T = sum(dur)
    """
    cumsum = torch.cumsum(dur[0].clamp(min=0), dim=0)  # (L,)
    mel_len = cumsum[-1]
    positions = torch.arange(mel_len, device=hidden.device)  # (T,)
    # For position t, phoneme index = count(cumsum <= t)
    indices = (cumsum.unsqueeze(-1) <= positions.unsqueeze(0)).long().sum(dim=0)
    indices = indices.clamp(max=hidden.shape[1] - 1)
    return hidden[:, indices, :]  # (1, T, d)


class MiniSpeechEncoder(nn.Module):
    def __init__(self, n_sym, d=256, n_enc=4, n_dec=4, n_mel=80, learn_alignment=False, max_len=2048):
        super().__init__()
        self.emb = nn.Embedding(n_sym, d, padding_idx=0)
        self.enc = nn.ModuleList([ConvBlock(d) for _ in range(n_enc)])
        self.dp = DurationPredictor(d)
        self.dec = nn.ModuleList([ConvBlock(d) for _ in range(n_dec)])
        self.out = nn.Linear(d, n_mel); self.d = d
        # Precomputed sinusoidal positional encoding (constant table, sliced to T).
        # Slicing a buffer avoids a runtime `arange().float()` -> Cast(to=FLOAT):
        # that cast keeps the exported ONNX graph heavier (ScatterND/Range) AND makes
        # float16 conversion produce a self-contradictory Cast node that ORT rejects.
        # Non-persistent so existing checkpoints load unchanged (and values are
        # bit-identical to the old runtime computation).
        self.register_buffer("_pe", self._build_pe(max_len, d), persistent=False)
        if learn_alignment:
            from encoder.alignment import AlignmentEncoder
            self.aligner = AlignmentEncoder(n_mel=n_mel, n_text=d)

    @staticmethod
    def _build_pe(L, d):
        pe = torch.zeros(L, d)
        pos = torch.arange(L).float().unsqueeze(1)
        dv = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * dv); pe[:, 1::2] = torch.cos(pos * dv)
        return pe

    def posenc(self, x):
        T = x.shape[1]
        if T <= self._pe.shape[0]:                       # ONNX-traced path (constant slice)
            return x + self._pe[:T].unsqueeze(0)
        # length beyond the precomputed table (rare; PyTorch-only fallback)
        d = x.shape[2]
        pos = torch.arange(T, device=x.device).float().unsqueeze(1)
        dv = torch.exp(torch.arange(0, d, 2, device=x.device).float() * (-math.log(10000.0) / d))
        pe = torch.zeros(T, d, device=x.device)
        pe[:, 0::2] = torch.sin(pos * dv); pe[:, 1::2] = torch.cos(pos * dv)
        return x + pe.unsqueeze(0)

    def encode(self, ph):
        x = self.posenc(self.emb(ph))
        for b in self.enc: x = b(x)
        return x

    def decode(self, y_lr):
        y = self.posenc(y_lr)
        for b in self.dec: y = b(y)
        return self.out(y).transpose(1, 2)

    def forward_infer(self, ph):
        """ONNX-friendly inference: phoneme_ids (1,L) -> mel (1,80,T).
        Uses cumsum+gather instead of repeat_interleave. All ops ONNX opset-11+."""
        x = self.encode(ph)
        log_dur = self.dp(x)
        dur = torch.clamp(torch.round(torch.exp(log_dur) - 1), min=1).long()
        expanded = length_regulate_gather(x, dur)
        return self.decode(expanded)

    def forward(self, ph, durations=None, max_frames=None):
        x = self.encode(ph)
        log_dur = self.dp(x.detach())
        if durations is None:
            durations = torch.clamp(torch.round(torch.exp(log_dur) - 1), min=1).long()
            max_frames = max(int(durations.sum(1).max().item()), 1)
        return self.decode(length_regulate(x, durations, max_frames)), log_dur

    def forward_align(self, ph, mel, pl, ml, attn_prior=None, prior_weight=1.0):
        from encoder.alignment import hard_durations
        x = self.encode(ph)
        log_dur = self.dp(x.detach())
        Tp = ph.shape[1]
        key_mask = torch.arange(Tp, device=ph.device)[None, :] >= pl.to(ph.device)[:, None]
        soft, logp = self.aligner(mel, x.transpose(1, 2), key_mask, attn_prior, prior_weight)
        dur, hard = hard_durations(soft, pl.tolist(), ml.tolist())
        mel_pred = self.decode(length_regulate(x, dur.long(), mel.shape[2]))
        return mel_pred, log_dur, dur, soft, hard, logp
