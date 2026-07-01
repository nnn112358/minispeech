"""EfficientEncoder: an EfficientSpeech-style lightweight acoustic model
(phoneme ids -> 80-mel), a drop-in alternative to MiniSpeechEncoder with the
same method contract (encode / decode / forward / forward_infer / forward_align).

Memory levers borrowed from EfficientSpeech (Atienza 2023, Apache-2.0):
  - a small *working dim* `w = embed_dim // reduction` (the whole net runs at w,
    only the embedding is full embed_dim),
  - a downsampling *pyramid* encoder + multi-scale Fuse,
  - `expansion=1` MixFFN, a depthwise-separable MelDecoder,
  - NO sinusoidal positional encoding (the depthwise convs carry position).

Two faithfulness levels (config flag `faithful`):
  - faithful=False (default, original port): standard MHA, duration predictor is
    MiniSpeech's (detached), MelDecoder fed only the fused features (w).
  - faithful=True: closer to the paper -- full-width multi-head attention
    (qkv = dim*3*heads), an AcousticDecoder-style duration decoder with LayerNorms
    that also emits features, and the MelDecoder is fed [fused, dur_features] (2w).
    (pitch/energy are still dropped: this pipeline has no such stats.)

Self-attention is masked during batched training (pad = phoneme id 0); at bs=1
inference the mask is None so the exported ONNX graph stays branch-free.
"""
import torch, torch.nn as nn, torch.nn.functional as F

from encoder.minispeech import length_regulate, length_regulate_gather


class MixFFN(nn.Module):
    """EfficientSpeech Mix-FFN: Linear -> conv -> GELU -> Linear."""
    def __init__(self, dim, expansion=1):
        super().__init__()
        h = dim * expansion
        self.mlp1 = nn.Linear(dim, h)
        self.conv = nn.Conv1d(h, h, 3, padding=1)
        self.mlp2 = nn.Linear(h, dim)
        self.act = nn.GELU()

    def forward(self, x):                      # (B, N, dim)
        x = self.mlp1(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.mlp2(self.act(x))


class SelfAttention(nn.Module):
    """Multi-head self-attention. `key_mask` (B, N) True=pad.
    full_width=True replicates EfficientSpeech: each head is full `dim` wide
    (qkv = Linear(dim, dim*3*heads)), giving `heads`x the attention capacity.
    full_width=False is standard MHA (dim split across heads)."""
    def __init__(self, dim, num_heads=1, full_width=False):
        super().__init__()
        self.h = num_heads
        self.full = full_width
        self.scale = (dim // num_heads) ** -0.5
        if full_width:
            self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=False)
            self.proj = nn.Linear(dim * num_heads, dim)
            self.hd = dim
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)
            self.proj = nn.Linear(dim, dim)
            self.hd = dim // num_heads

    def forward(self, x, key_mask=None):       # x (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                 # each (B, h, N, hd)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_mask is not None:
            attn = attn.masked_fill(key_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.h * self.hd)
        return self.proj(x)


class EncBlock(nn.Module):
    """Depthwise-sep merge (optionally stride>1) -> self-attn -> Mix-FFN, residual."""
    def __init__(self, dim_in, dim_out, kernel, stride, head, expansion, full_width):
        super().__init__()
        self.dw = nn.Conv1d(dim_in, dim_in, kernel, stride=stride, padding=kernel // 2, bias=False)
        self.pw = nn.Conv1d(dim_in, dim_out, 1, bias=False)
        self.attn = SelfAttention(dim_out, head, full_width)
        self.mixffn = MixFFN(dim_out, expansion)
        self.norm1 = nn.LayerNorm(dim_out)
        self.norm2 = nn.LayerNorm(dim_out)

    def forward(self, x, full_mask=None):      # x (B, N, Cin), full_mask (B, N0)
        y = self.pw(self.dw(x.transpose(1, 2))).transpose(1, 2)   # (B, N', Cout)
        km = None
        if full_mask is not None:
            mf = full_mask.float()[:, None]
            km = F.interpolate(mf, size=y.shape[1], mode="nearest")[:, 0] > 0.5
        y = self.norm1(self.attn(y, km) + y)
        y = self.norm2(self.mixffn(y) + y)
        return y


class PyramidEncoder(nn.Module):
    def __init__(self, n_sym, embed_dim, depth, reduction, kernel_size, head, expansion, full_width):
        super().__init__()
        w = embed_dim // reduction
        dim_ins = [embed_dim] + [w * (2 ** i) for i in range(depth - 1)]
        self.dim_outs = [w * (2 ** i) for i in range(depth)]
        kernels = [kernel_size - (2 if i > 0 else 0) for i in range(depth)]   # 3,1,1,...
        strides = [1] + [2] * (depth - 1)
        heads = []
        for i, do in enumerate(self.dim_outs):
            h = head * (i + 1)
            if not full_width:
                while do % h:                   # standard MHA needs dim divisible by heads
                    h -= 1
            heads.append(h)
        self.emb = nn.Embedding(n_sym, embed_dim, padding_idx=0)
        self.blocks = nn.ModuleList([
            EncBlock(di, do, k, s, h, expansion, full_width)
            for di, do, k, s, h in zip(dim_ins, self.dim_outs, kernels, strides, heads)])

    def forward(self, ph, mask=None):
        x = self.emb(ph)
        feats = []
        for blk in self.blocks:
            x = blk(x, mask)
            feats.append(x)
        return feats


class Fuse(nn.Module):
    """Project each pyramid scale to w, upsample back to phoneme length, concat, fuse."""
    def __init__(self, dims, kernel_size):
        super().__init__()
        w = dims[0]
        self.mlps = nn.ModuleList(nn.Linear(d, w) for d in dims)
        self.ups = nn.ModuleList(
            nn.ConvTranspose1d(w, w, kernel_size, stride=d // w) if d // w > 1 else nn.Identity()
            for d in dims)
        self.fuse = nn.Linear(w * len(dims), w)

    def forward(self, feats):
        out_len = feats[0].shape[1]
        outs = []
        for f, mlp, up in zip(feats, self.mlps, self.ups):
            x = up(mlp(f).transpose(1, 2))      # (B, w, ~N0)
            x = x[:, :, :out_len]
            if x.shape[2] < out_len:
                x = F.pad(x, (0, out_len - x.shape[2]))
            outs.append(x)
        x = torch.cat(outs, dim=1).transpose(1, 2)   # (B, N0, w*depth)
        return self.fuse(x)                          # (B, N0, w)


class DurationPredictor(nn.Module):
    """MiniSpeech's predictor: 2x Conv1d + Linear -> log-duration (B, N)."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU(),
                                 nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU())
        self.lin = nn.Linear(dim, 1)

    def forward(self, x):
        return self.lin(self.net(x.transpose(1, 2)).transpose(1, 2)).squeeze(-1)


class DurationDecoder(nn.Module):
    """EfficientSpeech AcousticDecoder (duration): conv+LN x2 -> log-dur, and the
    pre-linear `features` are also returned to enrich the MelDecoder input.
    (final ReLU of the paper's duration head is dropped: we predict log-dur to
    match the shared trainer's MSE-in-log-space contract.)"""
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU())
        self.norm1 = nn.LayerNorm(dim)
        self.conv2 = nn.Sequential(nn.Conv1d(dim, dim, 3, padding=1), nn.ReLU())
        self.norm2 = nn.LayerNorm(dim)
        self.lin = nn.Linear(dim, 1)

    def forward(self, x):                       # (B, N, dim)
        y = F.relu(self.norm1(self.conv1(x.transpose(1, 2)).transpose(1, 2)))
        feat = self.norm2(self.conv2(y.transpose(1, 2)).transpose(1, 2))
        return self.lin(feat).squeeze(-1), feat


class MelDecoder(nn.Module):
    """EfficientSpeech depthwise-separable mel decoder. Internal width dim2=min(4w,256)."""
    def __init__(self, in_dim, w, kernel_size, n_mel, n_blocks, block_depth):
        super().__init__()
        dim2 = min(4 * w, 256)
        pad = kernel_size // 2
        self.proj = nn.Sequential(nn.Linear(in_dim, dim2), nn.Tanh(), nn.LayerNorm(dim2))
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            convs = nn.ModuleList()
            for _ in range(block_depth):
                convs.append(nn.ModuleList([
                    nn.Sequential(
                        nn.Conv1d(dim2, dim2, kernel_size, padding=pad, groups=dim2),
                        nn.Conv1d(dim2, dim2, 1), nn.Tanh()),
                    nn.LayerNorm(dim2)]))
            self.blocks.append(nn.ModuleList([convs, nn.LayerNorm(dim2)]))
        self.mel_linear = nn.Linear(dim2, n_mel)

    def forward(self, x):                       # (B, T, in_dim) -> (B, n_mel, T)
        skip = self.proj(x)
        for convs, skip_norm in self.blocks:
            y = skip
            for conv, norm in convs:
                y = norm(conv(y.transpose(1, 2)).transpose(1, 2))
            skip = skip_norm(y + skip)
        return self.mel_linear(skip).transpose(1, 2)


class EfficientEncoder(nn.Module):
    """EfficientSpeech-style encoder with the MiniSpeechEncoder method contract."""
    def __init__(self, n_sym, embed_dim=128, depth=2, reduction=2, expansion=1,
                 kernel_size=3, head=1, n_mel=80, mel_blocks=2, mel_block_depth=2,
                 mel_kernel=3, learn_alignment=False, faithful=False, **_):
        super().__init__()
        self.faithful = faithful
        self.encoder = PyramidEncoder(n_sym, embed_dim, depth, reduction,
                                      kernel_size, head, expansion, full_width=faithful)
        w = embed_dim // reduction
        self.fuse = Fuse(self.encoder.dim_outs, kernel_size)
        if faithful:
            self.dur = DurationDecoder(w)
            self.decoder = MelDecoder(2 * w, w, mel_kernel, n_mel, mel_blocks, mel_block_depth)
        else:
            self.dp = DurationPredictor(w)
            self.decoder = MelDecoder(w, w, mel_kernel, n_mel, mel_blocks, mel_block_depth)
        self.d = w
        if learn_alignment:
            from encoder.alignment import AlignmentEncoder
            self.aligner = AlignmentEncoder(n_mel=n_mel, n_text=w)

    def encode(self, ph, mask=None):
        if mask is None and ph.shape[0] > 1:    # batched train: pad = phoneme id 0
            mask = (ph == 0)
        return self.fuse(self.encoder(ph, mask))     # (B, L, w)

    def _dec_input(self, fused):
        """-> (log_dur (B,L), decoder_input (B,L,in_dim)). faithful concatenates the
        duration-decoder features (2w); the original port feeds only fused (w) and
        keeps the duration predictor detached (MiniSpeech behaviour)."""
        if self.faithful:
            log_dur, feat = self.dur(fused)
            return log_dur, torch.cat([fused, feat], dim=-1)
        return self.dp(fused.detach()), fused

    def decode(self, expanded):
        return self.decoder(expanded)           # (B, n_mel, T)

    def forward_infer(self, ph):
        """ONNX-friendly inference: phoneme_ids (1,L) -> mel (1,80,T)."""
        fused = self.encode(ph)
        log_dur, dec_in = self._dec_input(fused)
        dur = torch.clamp(torch.round(torch.exp(log_dur) - 1), min=1).long()
        return self.decode(length_regulate_gather(dec_in, dur))

    def forward(self, ph, durations=None, max_frames=None):
        fused = self.encode(ph)
        log_dur, dec_in = self._dec_input(fused)
        if durations is None:
            durations = torch.clamp(torch.round(torch.exp(log_dur) - 1), min=1).long()
            max_frames = max(int(durations.sum(1).max().item()), 1)
        return self.decode(length_regulate(dec_in, durations, max_frames)), log_dur

    def forward_align(self, ph, mel, pl, ml, attn_prior=None, prior_weight=1.0):
        from encoder.alignment import hard_durations
        Tp = ph.shape[1]
        key_mask = torch.arange(Tp, device=ph.device)[None, :] >= pl.to(ph.device)[:, None]
        fused = self.encode(ph, key_mask if ph.shape[0] > 1 else None)
        log_dur, dec_in = self._dec_input(fused)
        soft, logp = self.aligner(mel, fused.transpose(1, 2), key_mask, attn_prior, prior_weight)
        dur, hard = hard_durations(soft, pl.tolist(), ml.tolist())
        mel_pred = self.decode(length_regulate(dec_in, dur.long(), mel.shape[2]))
        return mel_pred, log_dur, dur, soft, hard, logp


# ── config keys persisted in checkpoints (arch="efficient") ──────────────────
EFF_KEYS = ("embed_dim", "depth", "reduction", "expansion", "kernel_size", "head",
            "mel_blocks", "mel_block_depth", "mel_kernel", "faithful")


def encoder_from_config(n_sym, cfg, learn_alignment=False):
    """Build the right encoder from a checkpoint `config` dict (arch auto-detect)."""
    if cfg.get("arch") == "efficient":
        kw = {k: cfg[k] for k in EFF_KEYS if k in cfg}
        return EfficientEncoder(n_sym=n_sym, learn_alignment=learn_alignment, **kw)
    from encoder.minispeech import MiniSpeechEncoder
    return MiniSpeechEncoder(n_sym=n_sym, d=cfg.get("dim", 256),
                             n_enc=cfg.get("n_enc", 4), n_dec=cfg.get("n_dec", 4),
                             learn_alignment=learn_alignment)
