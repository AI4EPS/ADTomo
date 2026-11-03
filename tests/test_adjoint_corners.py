"""The adjoint matches finite differences at every corner of the source cell, where the gradient comes from the
Simpson-rule start: du_c/df_k = (d_c / 6) * (w_k + 4 / N + delta_ck), N = 4 corners in 2D and 8 in 3D.
Run with pytest after building the ops (python setup.py build_ext --inplace)."""

import itertools
import math

import torch  # first: loads libtorch for the ops

import eikonal2d_op
import eikonal3d_op


def max_relative_error(op, f, src, h=0.5, e=1e-6):
    w = torch.rand_like(f)  # loss = sum(w * u)
    u = op.forward(f, h, *src)
    g = op.backward(w, u, f, h, *src)
    worst = 0.0
    for c in itertools.product(*[(math.floor(s), math.floor(s) + 1) for s in src]):
        fp, fm = f.clone(), f.clone()
        fp[c] += e
        fm[c] -= e
        fd = ((w * op.forward(fp, h, *src)).sum() - (w * op.forward(fm, h, *src)).sum()).item() / (2 * e)
        worst = max(worst, abs(g[c].item() - fd) / abs(fd))
    return worst


def test_2d_corners():
    torch.manual_seed(0)
    assert max_relative_error(eikonal2d_op, torch.rand(30, 25, dtype=torch.float64) + 0.5, (12.3, 7.8)) < 1e-5


def test_3d_corners():
    torch.manual_seed(0)
    assert max_relative_error(eikonal3d_op, torch.rand(20, 18, 16, dtype=torch.float64) + 0.5, (8.37, 6.81, 5.46)) < 1e-5
