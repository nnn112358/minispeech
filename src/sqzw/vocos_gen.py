"""Vocos vocoder Generator (mel->audio) = PiperMelFeatures + VocosBackbone +
ISTFTHead. Used as the comparison baseline (trained by cli/vocos_train.py)."""
import torch.nn as nn
from vocos.models import VocosBackbone
from vocos.heads import ISTFTHead
from sqzw.features import PiperMelFeatures


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = PiperMelFeatures()
        self.backbone = VocosBackbone(input_channels=80, dim=512, intermediate_dim=1536, num_layers=8)
        self.head = ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="same")

    def forward(self, audio):
        return self.head(self.backbone(self.feat(audio)))
