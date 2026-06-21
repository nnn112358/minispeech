#!/usr/bin/env python3
"""Watch the aux fine-tune log and exit when the mel-loss EMA has plateaued
(improvement over a window < threshold), or when training ends/crashes.
Prints a progress line each poll (to the output file); the single process-exit
is what notifies the caller."""
import sys
import os, sys; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time, re, sys, subprocess

LOG = "checkpoints/train_aux_r2.log"
WIN = 4000        # step window to measure improvement over
THRESH = 0.004    # mel EMA drop over WIN below this => saturated
MIN_STEP = 8000   # don't declare saturation before this
PAT = re.compile(r"step (\d+)/\d+ nll [-\d.]+ mel ([\d.]+) stft ([\d.]+)")


def rows():
    out = []
    try:
        for line in open(LOG):
            m = PAT.match(line)
            if m:
                out.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    except FileNotFoundError:
        pass
    return out


while True:
    alive = subprocess.run(["pgrep", "-f", "m3_sqzwave_finetune_aux"],
                           capture_output=True).returncode == 0
    r = rows()
    if r:
        s, mel, stft = r[-1]
        past = [x for x in r if x[0] <= s - WIN]
        dmel = (past[-1][1] - mel) if past else None
        msg = f"step {s} mel {mel:.4f} stft {stft:.4f}"
        if dmel is not None:
            msg += f"  dmel/{WIN}={dmel:.4f}"
        print(msg, flush=True)
        if dmel is not None and dmel < THRESH and s >= MIN_STEP:
            print(f"SATURATED at step {s}: dmel/{WIN}={dmel:.4f} < {THRESH} (mel {mel:.4f} stft {stft:.4f})", flush=True)
            break
    if not alive:
        done = False
        try:
            done = any(l.startswith("DONE") for l in open(LOG))
        except FileNotFoundError:
            pass
        print(f"TRAINING ENDED ({'DONE/max-steps' if done else 'process gone'})", flush=True)
        break
    time.sleep(120)
