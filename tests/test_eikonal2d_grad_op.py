import eikonal2d_op
import matplotlib.pyplot as plt
import torch

import numpy as np
import matplotlib.pyplot as plt
import random


class Eikonal2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, h, ix, jx):
        u = eikonal2d_op.forward(f, h, ix, jx)
        ctx.save_for_backward(u, f)
        ctx.h = h
        ctx.ix = ix
        ctx.jx = jx
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, f = ctx.saved_tensors
        grad_f = eikonal2d_op.backward(grad_output.contiguous(), u, f, ctx.h, ctx.ix, ctx.jx)
        return grad_f, None, None, None


class Eikonal2D(torch.nn.Module):
    def __init__(self, h, ix, jx):
        super(Eikonal2D, self).__init__()
        self.h = h
        self.ix = ix
        self.jx = jx

    def forward(self, f):
        return Eikonal2DFunction.apply(f, self.h, self.ix, self.jx)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define the grid size
    m, n = 3, 3

    # Create initial speed function (f)
    f_ = torch.ones((m, n), dtype=torch.float64)
    f = torch.nn.Parameter(f_, requires_grad=True)

    # Set a point source at the center
    ix, jx = (0.0, 0.0)

    # Define the grid spacing
    h = 0.5

    # Create the Eikonal solver
    eikonal_solver = Eikonal2D(h, ix, jx)

    # Solve the Eikonal equation
    u = eikonal_solver(f)

    # Gradient test
    m_ = f
    v_ = torch.randn(m, n, dtype=torch.float64)

    ms_ = []
    ys_ = []
    s_ = []
    w_ = []
    gs_ = [10 ** (-i) for i in range(1, 6)]

    def scalar_function(f):
        u = eikonal_solver(f)
        return torch.sum(u)

    y_ = scalar_function(m_)
    print("y_:", y_)

    # dy_ = torch.autograd.grad(y_, m_, create_graph=True)[0]
    y_.backward()
    dy_ = f.grad

    y_ = y_.item()

    print("dy_:", dy_)

    for g_ in gs_:
        ms_.append(m_ + g_ * v_)
        ys_.append(scalar_function(ms_[-1]).item())
        s_.append(ys_[-1] - y_)
        w_.append(s_[-1] - g_ * torch.sum(v_ * dy_))
        # w_.append(-g_ * torch.sum(v_ * dy_))

    plt.figure()
    plt.loglog(gs_, np.abs(s_), "*-", label="finite difference")
    plt.loglog(gs_, np.abs(w_), "+-", label="automatic differentiation")
    plt.loglog(gs_, [g**2 * 0.5 * abs(w_[0]) / gs_[0] ** 2 for g in gs_], "--", label="$\\mathcal{O}(\\gamma^2)$")
    plt.loglog(gs_, [g * 0.5 * abs(s_[0]) / gs_[0] for g in gs_], "--", label="$\\mathcal{O}(\\gamma)$")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.xlabel("$\\gamma$")
    plt.ylabel("Error")
    plt.savefig("gradtest2d.png")
    plt.close()
