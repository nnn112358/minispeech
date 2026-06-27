"""Shared copy-synthesis evaluation loop for all vocoders.
Used by cli/eval_vocos.py, cli/eval_hifigan.py, cli/eval_mbistft.py."""
import os, time
import numpy as np
import torch
import soundfile as sf
from common.features import PiperMelFeatures

SR = 22050


def eval_copysynthesis(vocode_fn, filelist, n, out_dir, tag, dev, save_gt=False):
    """Copy-synthesis eval: wav -> mel -> vocode_fn(mel) -> wav.
    vocode_fn: callable(mel_tensor) -> audio_tensor (1D).
    Returns list of per-file dicts with aud_s, syn_ms, rtf, mel_l1."""
    feat = PiperMelFeatures().to(dev)
    os.makedirs(out_dir, exist_ok=True)
    files = [l for l in open(filelist).read().splitlines() if l.strip()][:n]
    results = []
    for i, f in enumerate(files):
        y, sr = sf.read(f, dtype="float32")
        if y.ndim > 1:
            y = y[:, 0]
        x = torch.from_numpy(y).unsqueeze(0).to(dev)
        with torch.no_grad():
            mel = feat(x)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            audio = vocode_fn(mel)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            a_np = audio.squeeze(0).cpu().numpy()
            mel2 = feat(torch.from_numpy(a_np).unsqueeze(0).to(dev))
            L = min(mel.size(2), mel2.size(2))
            l1 = (mel[:, :, :L] - mel2[:, :, :L]).abs().mean().item()
        adur = len(a_np) / SR
        sf.write(f"{out_dir}/{tag}_{i:02d}.wav", np.clip(a_np, -1, 1), SR)
        if save_gt:
            sf.write(f"{out_dir}/{tag}_{i:02d}_gt.wav", y, SR)
        print(f"  [{i}] {os.path.basename(f)} aud {adur:.2f}s syn {dt*1000:.1f}ms "
              f"RTF {dt/adur:.4f} melL1 {l1:.3f}", flush=True)
        results.append(dict(aud_s=adur, syn_ms=dt * 1000, rtf=dt / adur, mel_l1=l1))
    tot_syn = sum(r["syn_ms"] for r in results) / 1000
    tot_aud = sum(r["aud_s"] for r in results)
    mean_l1 = np.mean([r["mel_l1"] for r in results])
    print(f"\nsummary: {len(files)} files  total syn {tot_syn*1000:.0f}ms / aud {tot_aud:.2f}s "
          f"=> RTF {tot_syn/tot_aud:.4f} ({tot_aud/tot_syn:.0f}x RT)  mean melL1 {mean_l1:.3f}", flush=True)
    return results
