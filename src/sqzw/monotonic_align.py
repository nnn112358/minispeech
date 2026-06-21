"""Self-contained Monotonic Alignment Search (numba), so training needs no
external aligner (VITS/MFA). Core DP vendored from the Glow-TTS / VITS
monotonic_align (MIT) — see THIRD_PARTY_NOTICES.md."""
import numba
import numpy as np
import torch


@numba.jit(nopython=True, cache=True)
def _maximum_path_each(path, value, t_y, t_x):
    max_neg_val = -1e9
    index = t_x - 1
    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            v_cur = max_neg_val if x == y else value[y - 1, x]
            if x == 0:
                v_prev = 0.0 if y == 0 else max_neg_val
            else:
                v_prev = value[y - 1, x - 1]
            value[y, x] += max(v_prev, v_cur)
    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y - 1, index] < value[y - 1, index - 1]):
            index = index - 1


@numba.jit(nopython=True, cache=True)
def _maximum_path_c(paths, values, t_ys, t_xs):
    for i in range(paths.shape[0]):
        _maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])


def maximum_path(neg_cent, mask):
    """neg_cent, mask: (B, t_y, t_x) with t_y=frames (mel), t_x=tokens (phonemes).
    Returns a hard monotonic alignment path of the same shape (1 where aligned)."""
    device, dtype = neg_cent.device, neg_cent.dtype
    neg = (neg_cent * mask).detach().cpu().numpy().astype(np.float32)
    path = np.zeros_like(neg, dtype=np.int32)
    t_y = mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32)   # mel frames per item
    t_x = mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32)   # phonemes per item
    _maximum_path_c(path, neg, t_y, t_x)
    return torch.from_numpy(path).to(device=device, dtype=dtype)
