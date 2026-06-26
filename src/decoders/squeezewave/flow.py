"""SqueezeWave-specific flow utilities: differentiable reverse flow and
multi-resolution STFT loss. Used by cli/finetune_aux.py, cli/finetune_gan.py."""
import torch
import torch.nn.functional as F

RES = [(512, 128, 512), (1024, 256, 1024), (2048, 512, 2048)]


def diff_infer(model, mel, sigma):
    """Differentiable reverse flow: GT mel -> audio (random z, no W_inverse caching).
    Used by aux/GAN fine-tune so losses backprop through the inference path."""
    nac = model.n_audio_channel
    l = mel.size(2) * (256 // nac)
    B, dev = mel.size(0), mel.device
    audio = sigma * torch.randn(B, model.n_remaining_channels, l, device=dev)
    for k in reversed(range(model.n_flows)):
        n_half = audio.size(1) // 2
        a0 = audio[:, :n_half, :]; a1 = audio[:, n_half:, :]
        output = model.WN[k]((a0, mel))
        s = output[:, n_half:, :]; b = output[:, :n_half, :]
        a1 = (a1 - b) / torch.exp(s)
        audio = torch.cat([a0, a1], 1)
        W = model.convinv[k].conv.weight.squeeze()
        Winv = torch.inverse(W.float()).unsqueeze(-1)
        audio = F.conv1d(audio, Winv)
        if k % model.n_early_every == 0 and k > 0:
            z = sigma * torch.randn(B, model.n_early_size, l, device=dev)
            audio = torch.cat([z, audio], 1)
    return audio.permute(0, 2, 1).contiguous().view(B, -1)


def mrstft_loss(gt, gen, dev):
    """Multi-resolution STFT loss -> (spectral_convergence, log_mag_l1), averaged."""
    L = min(gt.shape[-1], gen.shape[-1])
    gt = gt[..., :L]; gen = gen[..., :L]
    sc_t, lm_t = 0.0, 0.0
    for n_fft, hop, win in RES:
        w = torch.hann_window(win, device=dev)
        Y = torch.stft(gt, n_fft, hop, win, w, return_complex=True).abs()
        Yh = torch.stft(gen, n_fft, hop, win, w, return_complex=True).abs()
        sc_t = sc_t + torch.linalg.norm(Y - Yh) / (torch.linalg.norm(Y) + 1e-8)
        lm_t = lm_t + (torch.log(Y + 1e-5) - torch.log(Yh + 1e-5)).abs().mean()
    return sc_t / len(RES), lm_t / len(RES)
