"""MB-iSTFT vocoder: mel -> audio via multi-band inverse STFT + PQMF synthesis.

Architecture from MB-iSTFT-VITS2 (Kawamura et al., ICASSP 2023), adapted as a
standalone mel-conditioned vocoder (not end-to-end VITS). Same interface as
decoders.hifigan and decoders.vocos: audio->mel->vocoder->audio, so it
trains with the same discriminators/losses. The sub-band STFT loss (stft_loss.py)
is additional and handled in the training script.

Total upsample factor: upsample_rates * istft_hop * subbands = 256
which matches PiperMelFeatures (hop_length=256).
Presets: n16=(4,4)*4*4=256, n32=(4,2)*8*4=256, n64=(4,)*16*4=256.

See THIRD_PARTY_NOTICES.md for attribution."""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from common.features import PiperMelFeatures
from common.modules import ResBlock1, ResBlock2, get_padding, init_weights, LRELU
from decoders.mb_istft.stft_onnx import OnnxISTFT

LRELU_SLOPE = 0.1


class PQMF(nn.Module):
    """Pseudo Quadrature Mirror Filterbank for sub-band analysis/synthesis."""

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
        super().__init__()
        self.subbands = subbands
        self.taps = taps

        filter_length = taps + 1
        omega_c = np.pi * cutoff_ratio
        t = np.arange(-(taps // 2), taps // 2 + 1, dtype=np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            sinc = np.where(t == 0, omega_c / np.pi, np.sin(omega_c * t) / (np.pi * t))
        window = np.kaiser(filter_length, beta)
        prototype = sinc * window

        analysis_filter = np.zeros((subbands, 1, filter_length), dtype=np.float64)
        for k in range(subbands):
            for n in range(filter_length):
                analysis_filter[k, 0, n] = (
                    2.0 * prototype[n]
                    * np.cos(
                        (2 * k + 1) * np.pi / (2 * subbands) * (n - taps / 2)
                        + (-1) ** k * np.pi / 4
                    )
                )

        synthesis_filter = analysis_filter[:, :, ::-1].copy()

        self.register_buffer("analysis_filter", torch.from_numpy(analysis_filter).float())
        self.register_buffer("synthesis_filter", torch.from_numpy(synthesis_filter).float())

        updown = np.zeros((subbands, subbands, subbands), dtype=np.float32)
        for k in range(subbands):
            updown[k, k, 0] = 1.0
        self.register_buffer("updown_filter", torch.from_numpy(updown))

        self.pad = nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """[B, 1, T] -> [B, subbands, T // subbands]"""
        x = self.pad(x)
        x = F.conv1d(x, self.analysis_filter)
        x = F.conv1d(x, self.updown_filter, stride=self.subbands)
        return x

    def synthesis(self, x):
        """[B, subbands, T_sub] -> [B, 1, T]"""
        x = F.conv_transpose1d(x, self.updown_filter * self.subbands, stride=self.subbands)
        x = self.pad(x)
        x = F.conv1d(x, self.synthesis_filter.permute(1, 0, 2))
        return x


class MBiSTFT(nn.Module):
    """Multi-Band iSTFT generator core: mel -> (fullband, subbands).

    Two upsampling stages (4x each = 16x) followed by sub-band iSTFT (hop=4)
    and PQMF synthesis (4 bands). Total: 16 * 4 * 4 = 256x = mel hop.
    """

    def __init__(self, in_channels=80, resblock="2", init_channels=256,
                 upsample_rates=(4, 4), upsample_kernel_sizes=(16, 16),
                 resblock_kernel_sizes=(3, 5, 7),
                 resblock_dilation_sizes=((1, 2), (2, 6), (3, 12)),
                 n_fft=16, hop_length=4, subbands=4):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.subbands = subbands
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.conv_pre = weight_norm(
            Conv1d(in_channels, init_channels, 7, 1, padding=3)
        )

        RB = ResBlock1 if str(resblock) == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(
                init_channels // (2 ** i),
                init_channels // (2 ** (i + 1)),
                k, u, padding=(k - u) // 2,
            )))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = init_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(RB(ch, k, d))

        post_ch = init_channels // (2 ** len(upsample_rates))
        self.subband_conv_post = Conv1d(post_ch, subbands * (n_fft + 2), 7, padding=3)

        self.istft = OnnxISTFT(n_fft=n_fft, hop_length=hop_length)
        self.pqmf = PQMF(subbands=subbands)

        self.ups.apply(init_weights)

    def forward(self, mel):
        """mel [B, 80, T] -> (fullband [B, 1, T*256], subbands [B, 4, T*64])"""
        x = self.conv_pre(mel)

        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, LRELU_SLOPE))
            xs = 0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x, LRELU_SLOPE)
        x = self.subband_conv_post(x)

        subbands_signal = self._spectrum_to_subbands(x)
        fullband = self.pqmf.synthesis(subbands_signal)

        return fullband, subbands_signal

    def _spectrum_to_subbands(self, x):
        """[B, subbands*(n_fft+2), T] -> [B, subbands, T*hop_length]"""
        B, _, T = x.shape
        x = x.reshape(B, self.subbands, self.n_fft + 2, T)

        n_half = self.n_fft // 2 + 1
        mag = torch.exp(x[:, :, :n_half, :])
        phase = torch.sin(x[:, :, n_half:, :]) * math.pi

        mag = mag.reshape(B * self.subbands, n_half, T)
        phase = phase.reshape(B * self.subbands, n_half, T)
        sub_wav = self.istft(mag, phase)

        subbands_signal = sub_wav.reshape(B, self.subbands, -1)
        expected_T = T * self.hop_length
        return subbands_signal[..., :expected_T]

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for l in self.ups:
            remove_weight_norm(l)
        for b in self.resblocks:
            b.remove_weight_norm()


class Generator(nn.Module):
    """audio -> PiperMel -> MBiSTFT -> audio. Same interface as hifigan/vocos.

    During forward(), caches sub-band signals in self.last_subbands for training
    (sub-band STFT loss). Expose self.pqmf for GT decomposition in the trainer.
    """

    PRESETS = {
        16: dict(upsample_rates=(4, 4), upsample_kernel_sizes=(16, 16), n_fft=16, hop_length=4),
        32: dict(upsample_rates=(4, 2), upsample_kernel_sizes=(16, 8),  n_fft=32, hop_length=8),
        64: dict(upsample_rates=(4,),   upsample_kernel_sizes=(16,),    n_fft=64, hop_length=16),
    }

    def __init__(self, init_channels=256, n_fft=16):
        super().__init__()
        self.init_channels = init_channels
        self.n_fft = n_fft
        self.feat = PiperMelFeatures()
        preset = self.PRESETS[n_fft]
        self.mbistft = MBiSTFT(in_channels=80, init_channels=init_channels, **preset)
        self.pqmf = self.mbistft.pqmf
        self.last_subbands = None

    def forward(self, audio):
        mel = self.feat(audio)
        fullband, subbands = self.mbistft(mel)
        self.last_subbands = subbands
        return fullband.squeeze(1)
