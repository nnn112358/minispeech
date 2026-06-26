"""Sub-band Multi-resolution STFT Loss for MB-iSTFT training.

Training-only loss module. Not included in inference.
Vendored from piper-plus (MIT License). See THIRD_PARTY_NOTICES.md.
"""

import torch
import torch.nn.functional as F
from torch import nn


class STFTLoss(nn.Module):
    """Single-resolution STFT loss (spectral convergence + log magnitude)."""

    def __init__(self, fft_size: int, hop_size: int, win_size: int) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer("window", torch.hann_window(win_size))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_mag = self._stft(x)
        y_mag = self._stft(y)
        sc = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro").clamp(min=1e-7)
        mag = F.l1_loss(torch.log(x_mag + 1e-7), torch.log(y_mag + 1e-7))
        return sc + mag

    def _stft(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.squeeze(1)
        stft = torch.stft(
            x, self.fft_size, self.hop_size, self.win_size,
            self.window, return_complex=True,
        )
        return torch.abs(stft)


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss for sub-band signals.

    Default parameters follow the MB-iSTFT-VITS2 paper.
    """

    def __init__(
        self,
        fft_sizes=(171, 384, 683),
        hop_sizes=(10, 30, 60),
        win_sizes=(60, 150, 300),
    ) -> None:
        super().__init__()
        self.stft_losses = nn.ModuleList()
        for fs, hs, ws in zip(fft_sizes, hop_sizes, win_sizes):
            self.stft_losses.append(STFTLoss(fs, hs, ws))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            B, S, T = x.shape
            x = x.reshape(B * S, T)
            y = y.reshape(B * S, T)

        loss = 0.0
        for stft_loss in self.stft_losses:
            loss += stft_loss(x, y)
        return loss / len(self.stft_losses)
