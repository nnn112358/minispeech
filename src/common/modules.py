"""Shared building blocks for upsampling-based vocoders (HiFi-GAN, MB-iSTFT)."""
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

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
    """HiFi-GAN V2 / piper MRF block: per dilation, a single dilated-conv residual."""
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
