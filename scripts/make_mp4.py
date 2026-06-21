#!/usr/bin/env python3
"""Build an MP4 demo from the 2-stage (MiniSpeech+Vocos) synthesized audio:
concat 4 utterances (gaps), render a waveform + mel-spectrogram background with
Japanese text labels, write combined.wav + bg.png. ffmpeg adds a moving playhead."""
import numpy as np, librosa
from scipy.io import wavfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import librosa.display

SR = 22050; GAP = 0.4
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"
jp = fm.FontProperties(fname=FONT)
OUT = "outputs"
items = [
    (0,    "水をマレーシアから買わなくてはならないのです。"),
    (100,  "次に、俳句を詠むときで避けるべき八ヶ条は、以下のようなものである。"),
    (1000, "兄の膝は、怪我で手術が必要かもしれない。"),
    (3000, "その素性は知れないが、言葉遣いや美しい声色は、女性的なものである。"),
]

# ---- concat with silence gaps, track segment spans ----
gap = np.zeros(int(GAP*SR), np.float32)
chunks, spans, t = [], [], 0.0
for idx, txt in items:
    sr, a = wavfile.read(f"{OUT}/u{idx}_2stage.wav")
    a = a.astype(np.float32)/32768.0
    dur = len(a)/SR
    spans.append((t, t+dur, idx, txt))
    chunks.append(a); chunks.append(gap); t += dur + GAP
audio = np.concatenate(chunks); DUR = len(audio)/SR
wavfile.write(f"{OUT}/demo_2stage.wav", SR, (np.clip(audio,-1,1)*32767).astype(np.int16))
print(f"combined {DUR:.2f}s -> demo_2stage.wav")

# ---- background figure: 1280x720 ----
W, H = 1280, 720
fig = plt.figure(figsize=(W/100, H/100), dpi=100, facecolor="#0e1117")
gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.2], hspace=0.18,
                      left=0.06, right=0.985, top=0.86, bottom=0.07)

# waveform
axw = fig.add_subplot(gs[0]); axw.set_facecolor("#0e1117")
tt = np.linspace(0, DUR, len(audio))
axw.plot(tt, audio, color="#3fb0ff", lw=0.4)
axw.set_xlim(0, DUR); axw.set_ylim(-1, 1); axw.set_yticks([])
for s in axw.spines.values(): s.set_visible(False)
axw.tick_params(colors="#8aa")
axw.set_ylabel("waveform", color="#9fb", fontsize=9)

# mel spectrogram
axs = fig.add_subplot(gs[1]); axs.set_facecolor("#0e1117")
S = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=1024,
        hop_length=256, n_mels=80), ref=np.max)
librosa.display.specshow(S, sr=SR, hop_length=256, x_axis="time", y_axis="mel",
                         ax=axs, cmap="magma")
axs.set_xlim(0, DUR)
axs.set_ylabel("mel", color="#9fb", fontsize=9)
axs.set_xlabel("time [s]", color="#9fb", fontsize=9)
axs.tick_params(colors="#8aa")

# segment boundaries + JP text labels
for (s0, s1, idx, txt) in spans:
    for ax in (axw, axs):
        ax.axvspan(s0, s1, color="#ffffff", alpha=0.03)
        ax.axvline(s0, color="#444", lw=0.6, ls=":")
    short = txt if len(txt) <= 22 else txt[:21]+"…"
    axw.text((s0+s1)/2, 1.18, f"u{idx}", color="#ffd166", fontsize=10,
             ha="center", va="bottom", fontproperties=jp, fontweight="bold")
    axs.text((s0+s1)/2, 88, short, color="#e8e8e8", fontsize=8.5,
             ha="center", va="bottom", fontproperties=jp)

fig.text(0.06, 0.94, "2-stage TTS  —  MiniSpeech (non-AR)  +  Vocos vocoder",
         color="#ffffff", fontsize=15, fontproperties=jp, fontweight="bold")
fig.text(0.06, 0.905, "JSUT 22.05 kHz / 80-mel  ·  4 utterances  ·  AX650N NPU3 U16: ~10.7 ms/utt (277× realtime)",
         color="#9fb4c8", fontsize=9.5, fontproperties=jp)
fig.savefig(f"{OUT}/demo_bg.png", dpi=100, facecolor=fig.get_facecolor())
print(f"bg -> demo_bg.png  (DUR={DUR:.3f})")
# emit DUR for ffmpeg
open(f"{OUT}/.demo_dur","w").write(f"{DUR:.4f}")
