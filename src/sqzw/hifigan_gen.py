"""HiFi-GAN generator matching piper's decoder configuration (so the project's
HiFi-GAN == piper's VITS decoder for a fair comparison). piper uses the lighter
ResBlock2 MRF with kernels (3,5,7), 256 initial channels, and upsample (8,8,4)
(product 256 = hop). The `Generator` wrapper is audio->audio, identical in
interface to sqzw.vocos_gen.Generator, so it trains with the same
discriminators/losses (cli/hifigan_train.py). Architecture re-implementation;
see THIRD_PARTY_NOTICES.md (HiFi-GAN; config follows piper)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm
from sqzw.features import PiperMelFeatures

LRELU = 0.1


def get_padding(k, d=1):
    return (k * d - d) // 2


def init_weights(m, mean=0.0, std=0.01):
    if "Conv" in m.__class__.__name__:
        m.weight.data.normal_(mean, std)


class ResBlock1(nn.Module):
    """HiFi-GAN V1 MRF block: per dilation, a (dilated conv -> conv) residual pair."""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1)))
            for _ in dilation])
        self.convs1.apply(init_weights); self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            x = c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU)) + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1: remove_weight_norm(l)
        for l in self.convs2: remove_weight_norm(l)


class ResBlock2(nn.Module):
    """HiFi-GAN V2 / piper MRF block: per dilation, a single dilated-conv residual (lighter)."""
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d)))
            for d in dilation])
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            x = c(F.leaky_relu(x, LRELU)) + x
        return x

    def remove_weight_norm(self):
        for l in self.convs: remove_weight_norm(l)


class HiFiGAN(nn.Module):
    """HiFi-GAN generator: mel (B, n_mels, T) -> audio (B, 1, T*hop). Defaults = piper
    (resblock '2', 256 ch, kernels (3,5,7), upsample (8,8,4)). Pass resblock='1',
    init_channels=512, etc. for the standard HiFi-GAN V1."""
    def __init__(self, in_channels=80, resblock="2", init_channels=256,
                 upsample_rates=(8, 8, 4), upsample_kernel_sizes=(16, 16, 8),
                 resblock_kernel_sizes=(3, 5, 7),
                 resblock_dilation_sizes=((1, 2), (2, 6), (3, 12))):
        super().__init__()
        RB = ResBlock1 if str(resblock) == "1" else ResBlock2
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = weight_norm(Conv1d(in_channels, init_channels, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(ConvTranspose1d(
                init_channels // (2 ** i), init_channels // (2 ** (i + 1)), k, u, padding=(k - u) // 2)))
        self.resblocks = nn.ModuleList()
        ch = init_channels
        for i in range(len(self.ups)):
            ch = init_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(RB(ch, k, d))
        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights); self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self.ups[i](F.leaky_relu(x, LRELU))
            xs = 0
            for j in range(self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = self.conv_post(F.leaky_relu(x))
        return torch.tanh(x)

    def remove_weight_norm(self):
        for l in self.ups: remove_weight_norm(l)
        for b in self.resblocks: b.remove_weight_norm()
        remove_weight_norm(self.conv_pre); remove_weight_norm(self.conv_post)


class Generator(nn.Module):
    """audio -> PiperMel -> HiFiGAN(piper config) -> audio (mirrors sqzw.vocos_gen.Generator)."""
    def __init__(self, init_channels=256):
        super().__init__()
        self.init_channels = init_channels
        self.feat = PiperMelFeatures()
        self.hifigan = HiFiGAN(in_channels=80, init_channels=init_channels)

    def forward(self, audio):
        mel = self.feat(audio)               # (B, 80, T)
        return self.hifigan(mel).squeeze(1)  # (B, T*256)
