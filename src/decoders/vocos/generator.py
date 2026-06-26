"""Vocos vocoder Generator (mel->audio) = PiperMelFeatures + VocosBackbone +
ISTFTHead. Defaults match the original (dim=512, 8 layers, n_fft=1024).
Pass smaller values for lite variants (e.g. dim=192, num_layers=6, n_fft=512)."""
import torch.nn as nn
from vocos.models import VocosBackbone
from vocos.heads import ISTFTHead
from common.features import PiperMelFeatures


class Generator(nn.Module):
    def __init__(self, dim=512, intermediate_dim=1536, num_layers=8,
                 n_fft=1024, hop_length=256):
        super().__init__()
        self.config = dict(dim=dim, intermediate_dim=intermediate_dim,
                           num_layers=num_layers, n_fft=n_fft, hop_length=hop_length)
        self.feat = PiperMelFeatures()
        self.backbone = VocosBackbone(input_channels=80, dim=dim,
                                      intermediate_dim=intermediate_dim,
                                      num_layers=num_layers)
        self.head = ISTFTHead(dim=dim, n_fft=n_fft, hop_length=hop_length,
                              padding="same")

    def forward(self, audio):
        return self.head(self.backbone(self.feat(audio)))
