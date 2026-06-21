#!/usr/bin/env python3
"""Full 2-stage TTS: phoneme_ids -> FastSpeech mel -> SqueezeWave audio.
The speaker/voice comes from the FastSpeech checkpoint (the vocoder is mostly
speaker-agnostic). Phonemes come from a manifest entry or --phonemes; text->phoneme
(OpenJTalk g2p) is a separate piper-side step, not bundled here."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import soundfile as sf
from cli.infer import load_model
from cli.train_fastspeech import FastSpeech

SR = 22050


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-ckpt", default="fs/tyc_fs/fs_300.pth", help="FastSpeech (defines the voice)")
    ap.add_argument("--voc-ckpt", default="checkpoints/tyc_c64_gan/sqzwgan_bnrecal.pth")
    ap.add_argument("--manifest", default="fs_data/tyc/fs_manifest.json", help="phoneme_ids source")
    ap.add_argument("--idx", type=int, default=0, help="utterance index in the manifest")
    ap.add_argument("--phonemes", default="", help="comma-separated phoneme_ids (overrides manifest)")
    ap.add_argument("--sigma", type=float, default=0.7)
    ap.add_argument("--out", default="outputs/tts.wav")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fck = torch.load(a.fs_ckpt, map_location=dev)
    n_sym = fck["model"]["emb.weight"].shape[0]
    fs = FastSpeech(n_sym=n_sym).to(dev); fs.load_state_dict(fck["model"]); fs.eval()

    # vocoder: auto-detect SqueezeWave ("model"+"config") / Vocos ("G") / HiFi-GAN ("G"+"init_channels")
    vck = torch.load(a.voc_ckpt, map_location=dev)
    if "config" in vck:                       # SqueezeWave
        voc, _ = load_model(a.voc_ckpt, dev)
        vocode = lambda mel: voc.infer(mel, sigma=a.sigma).squeeze(0)
    elif "init_channels" in vck:              # HiFi-GAN
        from sqzw.hifigan_gen import Generator as HGen
        voc = HGen(init_channels=vck["init_channels"]).to(dev); voc.load_state_dict(vck["G"]); voc.eval()
        vocode = lambda mel: voc.hifigan(mel).squeeze(1).squeeze(0)
    else:                                     # Vocos
        from sqzw.vocos_gen import Generator as VGen
        voc = VGen().to(dev); voc.load_state_dict(vck["G"]); voc.eval()
        vocode = lambda mel: voc.head(voc.backbone(mel)).squeeze(0)

    if a.phonemes:
        ph_ids = [int(x) for x in a.phonemes.replace(" ", "").split(",") if x != ""]
    else:
        ph_ids = json.load(open(a.manifest))[a.idx]["phoneme_ids"]
    ph = torch.LongTensor(ph_ids).unsqueeze(0).to(dev)

    with torch.no_grad():
        mel, _ = fs(ph)                                          # (1,80,T) predicted-duration mel
        audio = vocode(mel).cpu().numpy()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out, np.clip(audio, -1, 1), SR)
    print(f"phonemes={len(ph_ids)} -> FS mel {tuple(mel.shape)} -> audio {len(audio)/SR:.2f}s "
          f"(fs={os.path.basename(a.fs_ckpt)} voc={os.path.basename(a.voc_ckpt)}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
