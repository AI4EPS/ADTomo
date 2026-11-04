import torch

from adtomo.eikonal2d import Eikonal2DFunction
from adtomo.eikonal3d import Eikonal3DFunction
from adtomo.interp import interp

torch.manual_seed(0)


def test_gradcheck_table_and_position():
    """Exact gradients w.r.t. the table and the event position, inside the grid and up to its edges, in 2D and 3D."""
    for shape in [(5, 6), (4, 5, 6)]:
        table = torch.rand(shape, dtype=torch.float64, requires_grad=True)
        x = [(torch.rand(7, dtype=torch.float64) * (n - 1)).requires_grad_() for n in shape]
        assert torch.autograd.gradcheck(lambda t, *q: interp(t, *q), (table, *x))


def test_outside_grid():
    """Beyond each edge and corner: exact gradient (gradcheck), never zero, and time and gradient continuous across
    the edge, in 2D and 3D."""
    for shape in [(5, 6), (4, 5, 6)]:
        table = torch.rand(shape, dtype=torch.float64, requires_grad=True)
        n = torch.tensor(shape, dtype=torch.float64)
        q = torch.rand(40, len(shape), dtype=torch.float64) * (n - 1)
        side = torch.randint(0, 3, q.shape)  # per axis: inside, below 0, or above n - 1, by up to 2 nodes
        q = torch.where(side == 1, -0.1 - 2 * torch.rand(q.shape, dtype=torch.float64), q)
        q = torch.where(side == 2, n - 1 + 0.1 + 2 * torch.rand(q.shape, dtype=torch.float64), q)
        x = [q[:, d].clone().requires_grad_() for d in range(len(shape))]
        assert torch.autograd.gradcheck(lambda t, *p: interp(t, *p), (table, *x))
        t = interp(table, *x)
        g = torch.stack(torch.autograd.grad(t.sum(), x), 1)
        assert (g.norm(dim=1) > 1e-6).all()
        for d in range(len(shape)):  # across the edges of axis d, the other axes inside
            for edge in (0.0, shape[d] - 1.0):
                pair = []
                for e in (edge - 1e-7, edge + 1e-7):
                    p = [torch.full((1,), 1.5, dtype=torch.float64, requires_grad=True) for _ in shape]
                    p[d] = torch.full((1,), e, dtype=torch.float64, requires_grad=True)
                    tp = interp(table, *p)
                    pair.append((tp.item(), torch.cat(torch.autograd.grad(tp.sum(), p))))
                assert abs(pair[1][0] - pair[0][0]) < 1e-6 and (pair[1][1] - pair[0][1]).abs().max() < 1e-5


def test_exact_for_linear_and_quadratic():
    """Catmull-Rom reproduces quadratic tables exactly, also in the edge cells (linear extrapolation beyond them)."""
    g = [torch.arange(n, dtype=torch.float64) for n in (6, 7, 8)]
    X, Y, Z = torch.meshgrid(*g, indexing="ij")
    q = [torch.rand(50, dtype=torch.float64) * (n - 1) for n in (6, 7, 8)]
    linear = lambda x, y, z: 0.3 * x - 0.2 * y + 0.5 * z + 1.0
    assert torch.allclose(interp(linear(X, Y, Z), *q), linear(*q))
    quad = lambda x, y: 0.1 * x**2 + 0.05 * x * y - 0.2 * y**2 + x
    inner = [qi.clamp(1, 4) for qi in q[:2]]  # away from the edge cells, where the padding is only linear
    assert torch.allclose(interp(quad(X[:, :, 0], Y[:, :, 0]), *inner), quad(*inner))


def test_gradient_continuous_across_nodes():
    """The derivative is the same on both sides of a node (a bilinear one jumps there)."""
    table = torch.rand(6, 7, dtype=torch.float64)
    for node in (2.0, 3.0):
        side = []
        for x0 in (node - 1e-7, node + 1e-7):
            x = torch.tensor([x0], dtype=torch.float64, requires_grad=True)
            interp(table, x, torch.tensor([2.5], dtype=torch.float64)).sum().backward()
            side.append(x.grad.item())
        assert abs(side[1] - side[0]) < 1e-5


def test_gradcheck_through_eikonal():
    """Travel time at an event from the eikonal op and the interpolation, exact w.r.t. slowness and event position."""
    h = 1.0
    f2 = (1.0 + 0.2 * torch.rand(6, 7, dtype=torch.float64)).requires_grad_()
    e2 = torch.tensor([3.3, 4.6], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda f, e: interp(Eikonal2DFunction.apply(f, h, 1.0, 2.0), e[0:1], e[1:2]), (f2, e2))
    f3 = (1.0 + 0.2 * torch.rand(5, 5, 5, dtype=torch.float64)).requires_grad_()
    e3 = torch.tensor([2.4, 1.7, 3.2], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda f, e: interp(Eikonal3DFunction.apply(f, h, 1.0, 1.0, 1.0), e[0:1], e[1:2], e[2:3]), (f3, e3))


if __name__ == "__main__":
    test_gradcheck_table_and_position()
    test_outside_grid()
    test_exact_for_linear_and_quadratic()
    test_gradient_continuous_across_nodes()
    test_gradcheck_through_eikonal()
    print("ok")
