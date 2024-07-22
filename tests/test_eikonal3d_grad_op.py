import eikonal3d_op
import matplotlib.pyplot as plt
import torch

import numpy as np
import matplotlib.pyplot as plt
import random


class Eikonal3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, h, x, y, z):
        u = eikonal3d_op.forward(f, h, x, y, z)
        ctx.save_for_backward(u, f)
        ctx.x = x
        ctx.y = y
        ctx.z = z
        ctx.h = h
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, f = ctx.saved_tensors
        grad_f = eikonal3d_op.backward(grad_output.contiguous(), u, f, ctx.h, ctx.x, ctx.y, ctx.z)
        return grad_f, None, None, None, None


class Eikonal3D(torch.nn.Module):
    def __init__(self, h, x, y, z):
        super(Eikonal3D, self).__init__()
        self.h = h
        self.x = x
        self.y = y
        self.z = z

    def forward(self, f):
        return Eikonal3DFunction.apply(f, self.h, float(self.x), float(self.y), float(self.z))


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define the grid size
    m, n, l = 3, 3, 2

    # Create initial conditions (u0) and speed function (f)
    f_ = torch.ones((m, n, l), dtype=torch.float64)
    f = torch.nn.Parameter(f_, requires_grad=True)

    # Set a point source at the center
    x, y, z = 0, 0, 0

    # Define the grid spacing
    h = 0.5

    # Create the Eikonal solver
    eikonal_solver = Eikonal3D(h, x, y, z)

    # Solve the Eikonal equation
    u = eikonal_solver(f)

    # Gradient test
    m_ = f
    v_ = torch.randn(m, n, l, dtype=torch.float64)

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

    print("dy_:", dy_[:, :, z])

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
    plt.savefig("gradtest3d.png")
    plt.close()
