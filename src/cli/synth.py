#!/usr/bin/env python3
"""Full 2-stage TTS: text or phoneme_ids -> MiniSpeech mel -> vocoder audio.
The speaker/voice comes from the MiniSpeech checkpoint (the vocoder is mostly
speaker-agnostic). Input can be Japanese text (--text), manifest entry, or
comma-separated phoneme_ids (--phonemes)."""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import soundfile as sf
from encoder.minispeech import MiniSpeech

SR = 22050


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-ckpt", default="fs/tyc_fs/fs_300.pth", help="MiniSpeech checkpoint (defines the voice)")
    ap.add_argument("--voc-ckpt", default="checkpoints/vocos_lite_a/vocos_last.pth")
    ap.add_argument("--manifest", default="fs_data/tyc/fs_manifest.json", help="phoneme_ids source")
    ap.add_argument("--idx", type=int, default=0, help="utterance index in the manifest")
    ap.add_argument("--phonemes", default="", help="comma-separated phoneme_ids (overrides manifest)")
    ap.add_argument("--text", default="", help="Japanese text (g2p via OpenJTalk, overrides manifest)")
    ap.add_argument("--out", default="outputs/tts.wav")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fck = torch.load(a.fs_ckpt, map_location=dev)
    sd = {k.replace("dp.net.3.", "dp.net.2."): v for k, v in fck["model"].items()}
    n_sym = sd["emb.weight"].shape[0]
    cfg = fck.get("config", {})
    fs = MiniSpeech(n_sym=n_sym, d=cfg.get("dim", 256), n_enc=cfg.get("n_enc", 4), n_dec=cfg.get("n_dec", 4)).to(dev)
    fs.load_state_dict(sd); fs.eval()

    # vocoder: auto-detect Vocos / MB-iSTFT / HiFi-GAN
    vck = torch.load(a.voc_ckpt, map_location=dev)
    if "G" in vck and "config" in vck and "dim" in vck.get("config", {}):  # Vocos
        from decoders.vocos.generator import Generator as VGen
        cfg = vck["config"]
        voc = VGen(**cfg).to(dev); voc.load_state_dict(vck["G"]); voc.eval()
        vocode = lambda mel: voc.head(voc.backbone(mel)).squeeze(0)
    elif vck.get("vocoder_type") == "mbistft":  # MB-iSTFT
        from decoders.mb_istft.generator import Generator as MBGen
        voc = MBGen(init_channels=vck.get("init_channels", 256), n_fft=vck.get("n_fft", 16)).to(dev); voc.load_state_dict(vck["G"]); voc.eval()
        vocode = lambda mel: voc.mbistft(mel)[0].squeeze(1).squeeze(0)
    elif "init_channels" in vck:              # HiFi-GAN
        from decoders.hifigan.generator import Generator as HGen
        voc = HGen(init_channels=vck["init_channels"]).to(dev); voc.load_state_dict(vck["G"]); voc.eval()
        vocode = lambda mel: voc.hifigan(mel).squeeze(1).squeeze(0)
    else:                                     # Vocos (legacy, no config)
        from decoders.vocos.generator import Generator as VGen
        voc = VGen().to(dev); voc.load_state_dict(vck["G"]); voc.eval()
        vocode = lambda mel: voc.head(voc.backbone(mel)).squeeze(0)

    if a.text:
        from piper_plus_g2p.encode.id_maps import get_phoneme_id_map
        from piper_train.infer_onnx import text_to_phoneme_ids_and_prosody
        id_map = get_phoneme_id_map("ja")
        ids, _ = text_to_phoneme_ids_and_prosody(a.text, id_map, language="ja", language_id_map=None)
        ph_ids = [1] + ids
    elif a.phonemes:
        ph_ids = [int(x) for x in a.phonemes.replace(" ", "").split(",") if x != ""]
    else:
        ph_ids = json.load(open(a.manifest))[a.idx]["phoneme_ids"]
    ph = torch.LongTensor(ph_ids).unsqueeze(0).to(dev)

    with torch.inference_mode():
        mel, _ = fs(ph)                                          # (1,80,T) predicted-duration mel
        audio = vocode(mel).cpu().numpy()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sf.write(a.out, np.clip(audio, -1, 1), SR)
    print(f"phonemes={len(ph_ids)} -> FS mel {tuple(mel.shape)} -> audio {len(audio)/SR:.2f}s "
          f"(fs={os.path.basename(a.fs_ckpt)} voc={os.path.basename(a.voc_ckpt)}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
