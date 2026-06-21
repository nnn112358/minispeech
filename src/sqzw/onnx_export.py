"""ONNX export wrapper for the SqueezeWave flow. The reverse pass needs Gaussian
noise; for an ONNX/NPU-friendly static graph we take it as an INPUT `z`
(1, n_audio_channel, l) instead of generating it inside (RandomNormal is
unsupported by Pulsar2). Fixed cap of 256 mel frames -> 65536 samples (2.972 s),
matching the Vocos vocoder_fix.onnx so the two are directly comparable."""
import torch
import torch.nn as nn

FRAMES = 256  # mel cap (== Vocos vocoder_fix.onnx)


class SqueezeWaveONNX(nn.Module):
    """Reverse SqueezeWave flow with noise supplied as input z."""
    def __init__(self, model, sigma=1.0):
        super().__init__()
        self.m = model
        self.sigma = sigma
        self.n_remaining = model.n_remaining_channels
        self.n_early = model.n_early_size
        self.n_every = model.n_early_every
        self.n_flows = model.n_flows

    def forward(self, mel, z):
        # mel: (1, 80, FRAMES);  z: (1, n_audio_channel, l)
        audio = self.sigma * z[:, :self.n_remaining, :]
        off = self.n_remaining
        for k in reversed(range(self.n_flows)):
            n_half = audio.size(1) // 2
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]
            output = self.m.WN[k]((audio_0, mel))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)
            audio = self.m.convinv[k](audio, reverse=True)
            if k % self.n_every == 0 and k > 0:
                zk = self.sigma * z[:, off:off + self.n_early, :]
                off += self.n_early
                audio = torch.cat((zk, audio), 1)
        audio = audio.permute(0, 2, 1).contiguous().view(1, 1, -1)
        return audio
