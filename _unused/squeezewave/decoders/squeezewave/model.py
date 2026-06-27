# SqueezeWave model (NVIDIA WaveGlow-derived, tianrengao/SqueezeWave).
# Patched for this project:
#   (1) torch.qr -> torch.linalg.qr  (torch.qr removed in torch 2.x)
#   (2) WN conditioning: rigid Upsample1d(2) replaced by F.interpolate(spec,
#       size=audio_time) so the model accepts ANY mel-frame : audio-group ratio.
#       This lets us train/infer on PiperMelFeatures mel (center=False, 61 frames
#       per 16384-sample segment) instead of the original center=True 64-frame mel,
#       keeping MiniSpeech-output mel and SqueezeWave-input mel identical.
# Original NVIDIA BSD-3 copyright retained below.
#
# *****************************************************************************
#  Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the BSD-3 conditions are met.
#  THIS SOFTWARE IS PROVIDED "AS IS"; SEE ORIGINAL HEADER FOR FULL TEXT.
# *****************************************************************************
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels):
    n_channels_int = n_channels[0]
    in_act = input_a + input_b
    t_act = torch.tanh(in_act[:, :n_channels_int, :])
    s_act = torch.sigmoid(in_act[:, n_channels_int:, :])
    acts = t_act * s_act
    return acts


class SqueezeWaveLoss(torch.nn.Module):
    def __init__(self, sigma=1.0):
        super(SqueezeWaveLoss, self).__init__()
        self.sigma = sigma

    def forward(self, model_output):
        z, log_s_list, log_det_W_list = model_output
        for i, log_s in enumerate(log_s_list):
            if i == 0:
                log_s_total = torch.sum(log_s)
                log_det_W_total = log_det_W_list[i]
            else:
                log_s_total = log_s_total + torch.sum(log_s)
                log_det_W_total += log_det_W_list[i]
        loss = torch.sum(z * z) / (2 * self.sigma * self.sigma) - log_s_total - log_det_W_total
        return loss / (z.size(0) * z.size(1) * z.size(2))


class Invertible1x1Conv(torch.nn.Module):
    """Convolution whose weight is an orthonormal matrix; returns conv output
    and log-determinant of its weight (or the inverse conv if reverse=True)."""
    def __init__(self, c):
        super(Invertible1x1Conv, self).__init__()
        self.conv = torch.nn.Conv1d(c, c, kernel_size=1, stride=1, padding=0, bias=False)
        # Sample a random orthonormal matrix to initialize weights
        W = torch.linalg.qr(torch.FloatTensor(c, c).normal_())[0]
        # Ensure determinant is 1.0 not -1.0
        if torch.det(W) < 0:
            W[:, 0] = -1 * W[:, 0]
        W = W.view(c, c, 1)
        self.conv.weight.data = W

    def forward(self, z, reverse=False):
        batch_size, group_size, n_of_groups = z.size()
        W = self.conv.weight.squeeze()
        if reverse:
            if not hasattr(self, 'W_inverse'):
                W_inverse = W.float().inverse()
                W_inverse = Variable(W_inverse[..., None])
                self.W_inverse = W_inverse
            z = F.conv1d(z, self.W_inverse, bias=None, stride=1, padding=0)
            return z
        else:
            log_det_W = batch_size * n_of_groups * torch.logdet(W)
            z = self.conv(z)
            return z, log_det_W


class WN(torch.nn.Module):
    """WaveNet-like affine-coupling network with depthwise-separable convs.
    mel conditioning is interpolated to the audio time dimension each layer."""
    def __init__(self, n_in_channels, n_mel_channels, n_layers, n_channels, kernel_size,
                 dilation_cycle=None):
        super(WN, self).__init__()
        assert (kernel_size % 2 == 1)
        assert (n_channels % 2 == 0)
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        start = torch.nn.Conv1d(n_in_channels, n_channels, 1)
        start = torch.nn.utils.weight_norm(start, name='weight')
        self.start = start
        end = torch.nn.Conv1d(n_channels, 2 * n_in_channels, 1)
        end.weight.data.zero_()
        end.bias.data.zero_()
        self.end = end
        cond_layer = torch.nn.Conv1d(n_mel_channels, 2 * n_channels * n_layers, 1)
        self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name='weight')
        for i in range(n_layers):
            # dilation_cycle=None -> dilation 1 everywhere (original, weight-shape-identical).
            # dilation_cycle=K   -> WaveNet-style 2^(i%K): wider receptive field, SAME params.
            dilation = 1 if not dilation_cycle else 2 ** (i % dilation_cycle)
            padding = int((kernel_size * dilation - dilation) / 2)
            # depthwise separable convolution
            depthwise = torch.nn.Conv1d(n_channels, n_channels, kernel_size,
                                        dilation=dilation, padding=padding, groups=n_channels)
            pointwise = torch.nn.Conv1d(n_channels, 2 * n_channels, 1)
            bn = torch.nn.BatchNorm1d(n_channels)
            self.in_layers.append(torch.nn.Sequential(bn, depthwise, pointwise))
            res_skip_layer = torch.nn.Conv1d(n_channels, n_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name='weight')
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, forward_input):
        audio, spect = forward_input
        audio = self.start(audio)
        n_channels_tensor = torch.IntTensor([self.n_channels])
        spect = self.cond_layer(spect)
        for i in range(self.n_layers):
            spect_offset = i * 2 * self.n_channels
            spec = spect[:, spect_offset:spect_offset + 2 * self.n_channels, :]
            # PATCH: interpolate mel-conditioning to the audio time dim (robust to
            # any mel-frame:audio-group ratio; replaces rigid Upsample1d(2)).
            if cond_needs_resize(audio, spec):
                cond = F.interpolate(spec, size=audio.size(2), mode='nearest')
            else:
                cond = spec
            acts = fused_add_tanh_sigmoid_multiply(self.in_layers[i](audio), cond, n_channels_tensor)
            res_skip_acts = self.res_skip_layers[i](acts)
            audio = audio + res_skip_acts
        return self.end(audio)


def cond_needs_resize(audio, spec):
    return audio.size(2) != spec.size(2)


class SqueezeWave(torch.nn.Module):
    def __init__(self, n_mel_channels, n_flows, n_audio_channel, n_early_every,
                 n_early_size, WN_config):
        super(SqueezeWave, self).__init__()
        assert (n_audio_channel % 2 == 0)
        self.n_flows = n_flows
        self.n_audio_channel = n_audio_channel
        self.n_early_every = n_early_every
        self.n_early_size = n_early_size
        self.WN = torch.nn.ModuleList()
        self.convinv = torch.nn.ModuleList()
        n_half = int(n_audio_channel / 2)
        n_remaining_channels = n_audio_channel
        for k in range(n_flows):
            if k % self.n_early_every == 0 and k > 0:
                n_half = n_half - int(self.n_early_size / 2)
                n_remaining_channels = n_remaining_channels - self.n_early_size
            self.convinv.append(Invertible1x1Conv(n_remaining_channels))
            self.WN.append(WN(n_half, n_mel_channels, **WN_config))
        self.n_remaining_channels = n_remaining_channels

    def forward(self, forward_input):
        """forward_input[0] = mel (B, n_mel_channels, frames);
           forward_input[1] = audio (B, time)."""
        spect, audio = forward_input
        audio = audio.unfold(1, self.n_audio_channel, self.n_audio_channel).permute(0, 2, 1)
        output_audio = []
        log_s_list = []
        log_det_W_list = []
        for k in range(self.n_flows):
            if k % self.n_early_every == 0 and k > 0:
                output_audio.append(audio[:, :self.n_early_size, :])
                audio = audio[:, self.n_early_size:, :]
            audio, log_det_W = self.convinv[k](audio)
            log_det_W_list.append(log_det_W)
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]
            output = self.WN[k]((audio_0, spect))
            log_s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (torch.exp(log_s)) * audio_1 + b
            log_s_list.append(log_s)
            audio = torch.cat([audio_0, audio_1], 1)
        output_audio.append(audio)
        return torch.cat(output_audio, 1), log_s_list, log_det_W_list

    def infer(self, spect, sigma=1.0):
        l = spect.size(2) * (256 // self.n_audio_channel)
        spect = spect.to(torch.float32)
        audio = torch.FloatTensor(spect.size(0), self.n_remaining_channels, l).to(spect.device).normal_()
        audio = sigma * audio
        for k in reversed(range(self.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]
            output = self.WN[k]((audio_0, spect))
            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)
            audio = self.convinv[k](audio, reverse=True)
            if k % self.n_early_every == 0 and k > 0:
                z = torch.FloatTensor(spect.size(0), self.n_early_size, l).to(spect.device).normal_()
                audio = torch.cat((sigma * z, audio), 1)
        audio = audio.permute(0, 2, 1).contiguous().view(audio.size(0), -1).data
        return audio

    @staticmethod
    def remove_weightnorm(model):
        squeezewave = model
        for WN in squeezewave.WN:
            WN.start = torch.nn.utils.remove_weight_norm(WN.start)
            WN.in_layers = remove_batch_norm(WN.in_layers)
            WN.cond_layer = torch.nn.utils.remove_weight_norm(WN.cond_layer)
            WN.res_skip_layers = remove(WN.res_skip_layers)
        return squeezewave


def fuse_conv_and_bn(conv, bn):
    fusedconv = torch.nn.Conv1d(conv.in_channels, conv.out_channels,
                                kernel_size=conv.kernel_size, padding=conv.padding,
                                dilation=conv.dilation, bias=True, groups=conv.groups)
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    w_bn = w_bn.clone()
    fusedconv.weight.data = torch.mm(w_bn, w_conv).view(fusedconv.weight.size())
    b_conv = conv.bias if conv.bias is not None else torch.zeros(conv.weight.size(0))
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    b_bn = torch.unsqueeze(b_bn, 1)
    k = conv.kernel_size[0]
    bn_k = b_bn.expand(-1, k)
    b = torch.matmul(w_conv, torch.transpose(bn_k, 0, 1))[range(b_bn.size()[0]), range(b_bn.size()[0])]
    fusedconv.bias.data = (b_conv + b)
    return fusedconv


def remove_batch_norm(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        depthwise = fuse_conv_and_bn(old_conv[1], old_conv[0])
        pointwise = old_conv[2]
        new_conv_list.append(torch.nn.Sequential(depthwise, pointwise))
    return new_conv_list


def remove(conv_list):
    new_conv_list = torch.nn.ModuleList()
    for old_conv in conv_list:
        old_conv = torch.nn.utils.remove_weight_norm(old_conv)
        new_conv_list.append(old_conv)
    return new_conv_list
