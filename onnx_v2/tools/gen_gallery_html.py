#!/usr/bin/env python3
"""wav/ の全 encoder×vocoder スペクトログラムを一覧する自己完結 HTML を生成。

SPECTROGRAMS.md の HTML 版。サムネイルをクリックするとライトボックスで開き、
mp4(音声+再生ヘッド)があれば動画を、無ければ原寸 PNG を表示する。
PNG/mp4 は事前に用意しておく(gen_demos.py / 全組み合わせ生成)。

  uv run python tools/gen_gallery_html.py
"""
import html
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # onnx_v2/
OUT = os.path.join(HERE, "spectrograms.html")
ENCODERS = ["enc_d64", "enc_d96", "enc_d128", "enc_d192", "enc_d256", "enc_d384"]
VOCODERS = ["vocos_d64_n512", "vocos_d128_n512", "vocos_d256_n512",
            "vocos_d512_n1024", "hifigan_ch256", "mbistft_n16", "mbistft_n64"]
SHORT = {"vocos_d64_n512": "vocos_d64", "vocos_d128_n512": "vocos_d128",
         "vocos_d256_n512": "vocos_d256", "vocos_d512_n1024": "vocos_d512",
         "hifigan_ch256": "hifigan", "mbistft_n16": "mbistft_n16",
         "mbistft_n64": "mbistft_n64"}
TEXT = "おはようございます、こんにちは、こんばんは、おやすみなさい。"

CSS = """
:root { color-scheme: light dark; }
body { font-family: system-ui, sans-serif; margin: 0; padding: 24px; line-height: 1.6; }
h1 { margin: 0 0 8px; font-size: 1.4rem; }
p.desc { margin: 0 0 16px; color: #666; max-width: 70rem; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ccc; padding: 4px; text-align: center; vertical-align: top; }
thead th { background: #1b1b1b; color: #fff; position: sticky; top: 0; z-index: 2; }
tbody th { background: #1b1b1b; color: #fff; position: sticky; left: 0; z-index: 1; white-space: nowrap; }
.cell { position: relative; cursor: pointer; display: inline-block; }
.cell img { width: 230px; height: auto; display: block; border-radius: 4px; }
.cell:hover img { outline: 3px solid #00bcd4; }
.badge { position: absolute; top: 4px; right: 4px; background: #00bcd4; color: #002;
         font-size: 11px; font-weight: 700; padding: 1px 6px; border-radius: 10px; }
.noaud { opacity: 0.72; }
/* lightbox */
#lb { position: fixed; inset: 0; background: rgba(0,0,0,.9); display: none;
      align-items: center; justify-content: center; flex-direction: column; z-index: 9999; }
#lb.show { display: flex; }
#lb img, #lb video { max-width: 95vw; max-height: 86vh; border-radius: 6px; }
#lb .cap { color: #eee; margin-top: 10px; font-size: 14px; }
#lb .hint { color: #999; margin-top: 4px; font-size: 12px; }
"""

JS = """
const lb = document.getElementById('lb');
const box = document.getElementById('lb-box');
const cap = document.getElementById('lb-cap');
function openLB(el){
  const img = el.dataset.img, vid = el.dataset.video, label = el.dataset.label;
  box.innerHTML = vid
    ? `<video src="${vid}" controls autoplay></video>`
    : `<img src="${img}">`;
  cap.textContent = label + (vid ? '' : '  (音声なし — 画像のみ)');
  lb.classList.add('show');
}
function closeLB(){ lb.classList.remove('show'); box.innerHTML=''; }
document.querySelectorAll('.cell').forEach(c => c.addEventListener('click', () => openLB(c)));
lb.addEventListener('click', e => { if (e.target === lb || e.target.id === 'lb-close') closeLB(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLB(); });
"""


def cell_html(e, v):
    png = f"wav/{e}__{v}.png"
    mp4 = f"wav/{e}__{v}.mp4"
    has_png = os.path.exists(os.path.join(HERE, png))
    has_mp4 = os.path.exists(os.path.join(HERE, mp4))
    label = html.escape(f"{e} × {v}")
    if not has_png:
        return '<td>⚠️欠落</td>'
    data_vid = f' data-video="{mp4}"' if has_mp4 else ''
    badge = '<span class="badge">▶ 音声</span>' if has_mp4 else ''
    cls = 'cell' if has_mp4 else 'cell noaud'
    return (f'<td><div class="{cls}" data-img="{png}"{data_vid} data-label="{label}">'
            f'<img src="{png}" loading="lazy" alt="{label}">{badge}</div></td>')


def main():
    n = len(ENCODERS) * len(VOCODERS)
    n_aud = sum(os.path.exists(os.path.join(HERE, f"wav/{e}__{v}.mp4"))
                for e in ENCODERS for v in VOCODERS)
    head = "".join(f"<th>{SHORT[v]}</th>" for v in VOCODERS)
    rows = ""
    for e in ENCODERS:
        rows += "<tr><th>" + e + "</th>" + "".join(cell_html(e, v) for v in VOCODERS) + "</tr>\n"
    doc = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>スペクトログラム一覧 (encoder × vocoder = {n})</title>
<style>{CSS}</style>
</head>
<body>
<h1>スペクトログラム一覧 (encoder × vocoder = {n})</h1>
<p class="desc">
同一テキスト「{html.escape(TEXT)}」を、全 {len(ENCODERS)} encoder × {len(VOCODERS)} vocoder
= {n} 通りで合成した mel→audio の STFT スペクトログラム(CPU, fp16, magma, dB, 0–11kHz)。
セルをクリックすると拡大表示。<b>▶ 音声</b>付き({n_aud} 件)は動画(再生ヘッド付き)で再生、
それ以外は原寸 PNG を表示します。Markdown 版は <a href="SPECTROGRAMS.md">SPECTROGRAMS.md</a>。
</p>
<div class="wrap">
<table>
<thead><tr><th>encoder ＼ vocoder</th>{head}</tr></thead>
<tbody>
{rows}</tbody>
</table>
</div>
<div id="lb"><div id="lb-box"></div><div class="cap" id="lb-cap"></div>
<div class="hint">クリックまたは Esc で閉じる</div></div>
<script>{JS}</script>
</body>
</html>
"""
    with open(OUT, "w") as f:
        f.write(doc)
    print(f"wrote {OUT}  ({n} cells, {n_aud} with audio/video)")


if __name__ == "__main__":
    main()
