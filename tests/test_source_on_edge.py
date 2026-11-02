"""A source on the last grid node must give the mirror image of a source on the first node, in forward and backward.

The source cell index is floor(x) limited to the last cell; a source on the last node line (2D) or plane (3D) used to
pick a cell beyond the grid. Run with pytest after building the ops (python setup.py build_ext --inplace)."""

import torch  # first: loads libtorch for the ops

import eikonal2d_op
import eikonal3d_op

torch.manual_seed(0)


def test_2d_source_on_last_node():
    f = torch.rand(21, 16, dtype=torch.float64) + 0.5  # slowness on (m + 1, n + 1) nodes
    g = torch.rand_like(f)  # upstream gradient for backward
    m, n = f.shape[0] - 1, f.shape[1] - 1
    for y in (3.4, float(n)):  # source on the last x line, and on the last corner
        u = eikonal2d_op.forward(f, 0.5, float(m), y)
        u_flip = eikonal2d_op.forward(f.flip(0), 0.5, 0.0, y).flip(0)
        assert torch.allclose(u, u_flip, atol=1e-10)
        gf = eikonal2d_op.backward(g, u, f, 0.5, float(m), y)
        gf_flip = eikonal2d_op.backward(g.flip(0), u_flip.flip(0), f.flip(0), 0.5, 0.0, y).flip(0)
        assert torch.allclose(gf, gf_flip, atol=1e-10)


def test_3d_source_on_last_node():
    f = torch.rand(12, 10, 9, dtype=torch.float64) + 0.5  # slowness on (m, n, l) nodes
    g = torch.rand_like(f)
    m, n, l = f.shape
    for y in (4.3, float(n - 1)):  # source on the last x and z planes, and on the last corner
        u = eikonal3d_op.forward(f, 0.5, float(m - 1), y, float(l - 1))
        u_flip = eikonal3d_op.forward(f.flip(0, 2), 0.5, 0.0, y, 0.0).flip(0, 2)
        assert torch.allclose(u, u_flip, atol=1e-10)
        gf = eikonal3d_op.backward(g, u, f, 0.5, float(m - 1), y, float(l - 1))
        gf_flip = eikonal3d_op.backward(g.flip(0, 2), u_flip.flip(0, 2), f.flip(0, 2), 0.5, 0.0, y, 0.0).flip(0, 2)
        assert torch.allclose(gf, gf_flip, atol=1e-10)
