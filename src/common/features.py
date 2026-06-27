"""Custom Vocos feature extractor for the JSUT 2-stage TTS.
Produces EXACTLY the same 80-dim log-mel that the MiniSpeechEncoder acoustic model is
trained to output: spectrogram_torch (magnitude STFT, n_fft=1024, hop=256,
win=1024, center=False, reflect-pad) -> librosa mel basis (22050,1024,80) ->
log(clamp 1e-5). Self-contained: vocos + spectrogram_torch (common.mel) are vendored locally."""
import numpy as np
import torch
import librosa
from vocos.feature_extractors import FeatureExtractor
from common.mel import spectrogram_torch


class PiperMelFeatures(FeatureExtractor):
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80):
        super().__init__()
        self.n_fft = n_fft; self.hop = hop_length; self.win = win_length; self.sr = sample_rate
        mb = librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels).astype("float32")
        self.register_buffer("mel_basis", torch.from_numpy(mb))  # (n_mels, n_fft//2+1)

    def forward(self, audio, **kwargs):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        spec = spectrogram_torch(audio, self.n_fft, self.sr, self.hop, self.win, center=False)  # (B, F, L)
        mel = torch.matmul(self.mel_basis.to(spec.device), spec)  # (B, n_mels, L)
        return torch.log(torch.clamp(mel, min=1e-5))
