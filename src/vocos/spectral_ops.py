import numpy as np
import scipy
import torch
from torch import nn, view_as_real, view_as_complex
from torch.nn import functional as F


class ISTFT(nn.Module):
    """ONNX-compatible iSTFT using conv_transpose1d with precomputed inverse DFT basis.

    Replaces torch.fft.irfft + fold with a real-valued inverse DFT basis applied
    via conv_transpose1d for overlap-add. Hann window and OLA normalisation are
    absorbed into the basis. All ops are ONNX opset-15 compatible (no complex
    types, no fold/col2im, no FFT). Follows piper-plus OnnxISTFT approach.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): Unused (kept for API compat). n_fft is always used.
        padding (str, optional): Type of padding. Options are "center" or "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int = 0, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        inverse_basis = self._build_inverse_basis(n_fft, hop_length)
        self.register_buffer("inverse_basis", inverse_basis)

    def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """Reconstruct waveform from real and imaginary spectrograms.

        Args:
            real: (B, n_fft//2+1, T) real part of STFT.
            imag: (B, n_fft//2+1, T) imaginary part of STFT.

        Returns:
            Tensor: (B, T_out) reconstructed waveform.
        """
        combined = torch.cat([real, imag], dim=1)  # (B, n_fft+2, T)
        y = F.conv_transpose1d(combined, self.inverse_basis, stride=self.hop_length)

        if self.padding == "same":
            pad = (self.n_fft - self.hop_length) // 2
            y = y[:, :, pad:-pad]

        return y.squeeze(1)

    @staticmethod
    def _build_inverse_basis(n_fft: int, hop_length: int) -> torch.Tensor:
        """Build windowed inverse DFT basis with Hann window and OLA normalisation.

        Returns tensor of shape (n_fft+2, 1, n_fft) for conv_transpose1d."""
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

        S = np.hstack([S_re, S_im])  # (n_fft, 2*cutoff)
        window = np.hanning(n_fft + 1)[:n_fft]
        # Windowed-OLA normalisation: the synthesis window envelope under
        # constant overlap-add equals sum(w^2)/hop (matches torch.istft).
        wss = np.sum(window**2) / hop_length
        inverse_basis = S * (window[:, np.newaxis] / wss)
        inverse_basis = inverse_basis.T[:, np.newaxis, :]  # (2*cutoff, 1, n_fft)

        return torch.FloatTensor(inverse_basis)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        # Old checkpoints have 'window' buffer instead of 'inverse_basis'.
        # Drop it silently — inverse_basis is computed from n_fft/hop_length.
        old_key = prefix + "window"
        if old_key in state_dict:
            del state_dict[old_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


class MDCT(nn.Module):
    """
    Modified Discrete Cosine Transform (MDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.cosine(frame_len)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(-1j * torch.pi * torch.arange(frame_len) / frame_len)
        post_twiddle = torch.exp(-1j * torch.pi * n0 * (torch.arange(N) + 0.5) / N)
        # view_as_real: NCCL Backend does not support ComplexFloat data type
        # https://github.com/pytorch/pytorch/issues/71613
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply the Modified Discrete Cosine Transform (MDCT) to the input audio.

        Args:
            audio (Tensor): Input audio waveform of shape (B, T), where B is the batch size
                and T is the length of the audio.

        Returns:
            Tensor: MDCT coefficients of shape (B, L, N), where L is the number of output frames
                and N is the number of frequency bins.
        """
        if self.padding == "center":
            audio = torch.nn.functional.pad(audio, (self.frame_len // 2, self.frame_len // 2))
        elif self.padding == "same":
            # hop_length is 1/2 frame_len
            audio = torch.nn.functional.pad(audio, (self.frame_len // 4, self.frame_len // 4))
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        x = audio.unfold(-1, self.frame_len, self.frame_len // 2)
        N = self.frame_len // 2
        x = x * self.window.expand(x.shape)
        X = torch.fft.fft(x * view_as_complex(self.pre_twiddle).expand(x.shape), dim=-1)[..., :N]
        res = X * view_as_complex(self.post_twiddle).expand(X.shape) * np.sqrt(1 / N)
        return torch.real(res) * np.sqrt(2)


class IMDCT(nn.Module):
    """
    Inverse Modified Discrete Cosine Transform (IMDCT) module.

    Args:
        frame_len (int): Length of the MDCT frame.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, frame_len: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.frame_len = frame_len
        N = frame_len // 2
        n0 = (N + 1) / 2
        window = torch.from_numpy(scipy.signal.cosine(frame_len)).float()
        self.register_buffer("window", window)

        pre_twiddle = torch.exp(1j * torch.pi * n0 * torch.arange(N * 2) / N)
        post_twiddle = torch.exp(1j * torch.pi * (torch.arange(N * 2) + n0) / (N * 2))
        self.register_buffer("pre_twiddle", view_as_real(pre_twiddle))
        self.register_buffer("post_twiddle", view_as_real(post_twiddle))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply the Inverse Modified Discrete Cosine Transform (IMDCT) to the input MDCT coefficients.

        Args:
            X (Tensor): Input MDCT coefficients of shape (B, L, N), where B is the batch size,
                L is the number of frames, and N is the number of frequency bins.

        Returns:
            Tensor: Reconstructed audio waveform of shape (B, T), where T is the length of the audio.
        """
        B, L, N = X.shape
        Y = torch.zeros((B, L, N * 2), dtype=X.dtype, device=X.device)
        Y[..., :N] = X
        Y[..., N:] = -1 * torch.conj(torch.flip(X, dims=(-1,)))
        y = torch.fft.ifft(Y * view_as_complex(self.pre_twiddle).expand(Y.shape), dim=-1)
        y = torch.real(y * view_as_complex(self.post_twiddle).expand(y.shape)) * np.sqrt(N) * np.sqrt(2)
        result = y * self.window.expand(y.shape)
        output_size = (1, (L + 1) * N)
        audio = torch.nn.functional.fold(
            result.transpose(1, 2),
            output_size=output_size,
            kernel_size=(1, self.frame_len),
            stride=(1, self.frame_len // 2),
        )[:, 0, 0, :]

        if self.padding == "center":
            pad = self.frame_len // 2
        elif self.padding == "same":
            pad = self.frame_len // 4
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        audio = audio[:, pad:-pad]
        return audio
