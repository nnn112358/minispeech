"""ONNX-compatible iSTFT using conv_transpose1d.

Pre-computes the DFT inverse basis and performs the inverse transform via
F.conv_transpose1d. All operations are Conv1d / ConvTranspose1d (ONNX opset 15).
The inverse basis absorbs the Hann window and OLA normalisation.

Vendored from piper-plus (MIT License). See THIRD_PARTY_NOTICES.md.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class OnnxISTFT(nn.Module):
    """Inverse STFT implemented with conv_transpose1d (no torch.istft needed)."""

    def __init__(self, n_fft: int = 16, hop_length: int = 4) -> None:
        super().__init__()
        inverse_basis = self._build_inverse_basis(n_fft, hop_length)
        self.register_buffer("inverse_basis", inverse_basis)
        self.hop_length = hop_length
        self.n_fft = n_fft

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Reconstruct waveform from (magnitude, phase) spectrograms.

        Args:
            magnitude: (B, n_fft//2+1, T)
            phase: (B, n_fft//2+1, T)
        Returns:
            (B, 1, T_out) waveform.
        """
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        combined = torch.cat([real, imag], dim=1)
        return self.forward_cartesian(combined)

    def forward_cartesian(self, combined: torch.Tensor) -> torch.Tensor:
        """Reconstruct waveform from Cartesian [real; imag] spectrum.

        Args:
            combined: (B, n_fft+2, T) complex spectrum as [real; imag].
        Returns:
            (B, 1, T_out) waveform.
        """
        return F.conv_transpose1d(
            combined, self.inverse_basis, stride=self.hop_length
        )

    @staticmethod
    def _build_inverse_basis(n_fft: int, hop_length: int) -> torch.Tensor:
        """Build inverse DFT basis with Hann window and OLA normalisation.

        Returns tensor of shape (n_fft+2, 1, n_fft).
        """
        cutoff = n_fft // 2 + 1

        n_idx = np.arange(n_fft)
        k_idx = np.arange(cutoff)
        angle = 2.0 * np.pi * np.outer(n_idx, k_idx) / n_fft
        cos_table = np.cos(angle)
        sin_table = np.sin(angle)

        S_re = np.zeros((n_fft, cutoff))
        S_im = np.zeros((n_fft, cutoff))

        S_re[:, 0] = 1.0 / n_fft
        S_re[:, cutoff - 1] = cos_table[:, cutoff - 1] / n_fft
        S_im[:, cutoff - 1] = -sin_table[:, cutoff - 1] / n_fft

        for ki in range(1, cutoff - 1):
            S_re[:, ki] = 2.0 * cos_table[:, ki] / n_fft
            S_im[:, ki] = -2.0 * sin_table[:, ki] / n_fft

        S = np.hstack([S_re, S_im])

        window = np.hanning(n_fft + 1)[:n_fft]
        wss = np.sum(window**2) * hop_length / n_fft
        inverse_basis = S * (window[:, np.newaxis] / wss)
        inverse_basis = inverse_basis.T[:, np.newaxis, :]

        return torch.FloatTensor(inverse_basis)
