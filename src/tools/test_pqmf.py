"""Investigate PQMF precision limits and try improvements."""
import sys
sys.path.insert(0, '/home/nnn/mini-jtts/src')
import torch
import numpy as np
from scipy import signal

def test_pqmf_custom(analysis_filter, subbands=4, taps=62, length=16000):
    """Test with custom filter coefficients."""
    filter_length = taps + 1
    synthesis_filter = analysis_filter[:, :, ::-1].copy()

    af = torch.from_numpy(analysis_filter).float()
    sf = torch.from_numpy(synthesis_filter).float()

    updown = np.zeros((subbands, subbands, subbands), dtype=np.float32)
    for k in range(subbands):
        updown[k, k, 0] = 1.0
    uf = torch.from_numpy(updown)

    pad = torch.nn.ConstantPad1d(taps // 2, 0.0)

    torch.manual_seed(42)
    x = torch.randn(1, 1, length)

    # Analysis
    xa = pad(x)
    xa = torch.nn.functional.conv1d(xa, af)
    xa = torch.nn.functional.conv1d(xa, uf, stride=subbands)

    # Synthesis
    xs = torch.nn.functional.conv_transpose1d(xa, uf * subbands, stride=subbands)
    xs = pad(xs)
    xs = torch.nn.functional.conv1d(xs, sf.permute(1, 0, 2))

    trim = taps
    x_t = x[..., trim:-trim]
    y_t = xs[..., trim:trim + x_t.shape[-1]]
    error = y_t - x_t
    snr = 10 * torch.log10((x_t**2).mean() / ((error**2).mean() + 1e-30))
    return snr.item()

def make_pqmf_filter(subbands=4, taps=62, cutoff_ratio=0.142, beta=9.0):
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
    return analysis_filter

def make_pqmf_scipy(subbands=4, taps=62, beta=9.0):
    """Use scipy.signal.firwin for prototype."""
    filter_length = taps + 1
    cutoff = 1.0 / (2 * subbands)  # Normalized to Nyquist
    prototype = signal.firwin(filter_length, cutoff, window=('kaiser', beta))

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
    return analysis_filter

# Test 1: Current implementation
print("=== Current (cutoff=0.142, kaiser beta=9) ===")
af = make_pqmf_filter(cutoff_ratio=0.142)
print(f"  PR-SNR: {test_pqmf_custom(af):.1f} dB")

# Test 2: scipy firwin
print("\n=== scipy.signal.firwin ===")
for beta in [9, 10, 12, 14]:
    af = make_pqmf_scipy(beta=beta)
    snr = test_pqmf_custom(af)
    print(f"  beta={beta}: PR-SNR={snr:.1f} dB")

# Test 3: scipy firwin with different taps
print("\n=== scipy firwin, beta=9, varying taps ===")
for taps in [62, 94, 126, 190, 254]:
    fl = taps + 1
    cutoff = 1.0 / 8  # 4 subbands
    prototype = signal.firwin(fl, cutoff, window=('kaiser', 9))
    af = np.zeros((4, 1, fl), dtype=np.float64)
    for k in range(4):
        for n in range(fl):
            af[k, 0, n] = 2.0 * prototype[n] * np.cos(
                (2*k+1)*np.pi/8*(n - taps/2) + (-1)**k*np.pi/4)
    snr = test_pqmf_custom(af, taps=taps)
    print(f"  taps={taps}: PR-SNR={snr:.1f} dB")

# Test 4: scipy remez (equiripple) prototype
print("\n=== scipy.signal.remez (equiripple) ===")
for taps in [62, 94, 126]:
    fl = taps + 1
    try:
        prototype = signal.remez(fl, [0, 0.12, 0.13, 0.5], [1, 0], fs=1.0)
        af = np.zeros((4, 1, fl), dtype=np.float64)
        for k in range(4):
            for n in range(fl):
                af[k, 0, n] = 2.0 * prototype[n] * np.cos(
                    (2*k+1)*np.pi/8*(n - taps/2) + (-1)**k*np.pi/4)
        snr = test_pqmf_custom(af, taps=taps)
        print(f"  taps={taps}: PR-SNR={snr:.1f} dB")
    except Exception as e:
        print(f"  taps={taps}: failed ({e})")

# Test 5: Use pre-computed optimal cutoff with scipy firwin
print("\n=== Optimal cutoff search with scipy firwin ===")
best_snr, best_params = -999, None
for beta in [8, 9, 10, 11, 12, 14]:
    for taps in [62, 94, 126]:
        fl = taps + 1
        for cut_mul in np.arange(0.90, 1.10, 0.01):
            cutoff = cut_mul / 8
            try:
                prototype = signal.firwin(fl, cutoff, window=('kaiser', beta))
                af = np.zeros((4, 1, fl), dtype=np.float64)
                for k in range(4):
                    for n in range(fl):
                        af[k, 0, n] = 2.0 * prototype[n] * np.cos(
                            (2*k+1)*np.pi/8*(n - taps/2) + (-1)**k*np.pi/4)
                snr = test_pqmf_custom(af, taps=taps)
                if snr > best_snr:
                    best_snr = snr
                    best_params = (taps, beta, cutoff)
            except:
                pass
if best_params:
    print(f"  Best: taps={best_params[0]}, beta={best_params[1]}, cutoff={best_params[2]:.4f}")
    print(f"  PR-SNR: {best_snr:.1f} dB")
