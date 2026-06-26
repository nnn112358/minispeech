"""WavSet: random fixed-length audio segments from a filelist."""
import os, random
import numpy as np
import torch
import soundfile as sf
from torch.utils.data import Dataset


class WavSet(Dataset):
    """Random fixed-length audio segments from a filelist."""
    def __init__(self, filelist, num_samples=16384):
        self.files = [l for l in open(filelist).read().splitlines() if l.strip()]
        self.n = num_samples

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        y, sr = sf.read(self.files[i], dtype="float32")
        if y.ndim > 1:
            y = y[:, 0]
        if len(y) >= self.n:
            s = random.randint(0, len(y) - self.n)
            y = y[s:s + self.n]
        else:
            y = np.pad(y, (0, self.n - len(y)))
        return torch.from_numpy(y)


class MelWavSet(Dataset):
    """Pre-computed mel + random audio crop. Returns (mel_crop, audio_crop)."""
    def __init__(self, filelist, mel_dir, num_samples=16384, hop_length=256):
        self.files = [l for l in open(filelist).read().splitlines() if l.strip()]
        self.mel_dir = mel_dir
        self.n = num_samples
        self.hop = hop_length
        self.mel_frames = num_samples // hop_length  # 64 for 16384/256

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        y, sr = sf.read(self.files[i], dtype="float32")
        if y.ndim > 1:
            y = y[:, 0]

        basename = os.path.splitext(os.path.basename(self.files[i]))[0]
        mel = np.load(os.path.join(self.mel_dir, basename + ".npy"))  # (80, T)

        if len(y) >= self.n:
            max_start = len(y) - self.n
            s = random.randint(0, max_start)
            y = y[s:s + self.n]
            mel_start = s // self.hop
            mel = mel[:, mel_start:mel_start + self.mel_frames]
        else:
            y = np.pad(y, (0, self.n - len(y)))
            mel = np.pad(mel, ((0, 0), (0, max(0, self.mel_frames - mel.shape[1]))))

        mel = mel[:, :self.mel_frames]
        return torch.from_numpy(y), torch.from_numpy(mel)
