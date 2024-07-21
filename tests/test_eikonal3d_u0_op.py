import eikonal3d_op
import matplotlib.pyplot as plt
import torch


class Eikonal3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u0, f, h):
        u = eikonal3d_op.forward(u0, f, h)
        ctx.save_for_backward(u, u0, f)
        ctx.h = h
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, u0, f = ctx.saved_tensors
        grad_u0, grad_f = eikonal3d_op.backward(grad_output, u, u0, f, ctx.h)
        return grad_u0, grad_f, None


class Eikonal3D(torch.nn.Module):
    def __init__(self, h):
        super(Eikonal3D, self).__init__()
        self.h = h

    def forward(self, u0, f):
        return Eikonal3DFunction.apply(u0, f, self.h)


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define the grid size
    m, n, l = 20, 15, 10

    # Create initial conditions (u0) and speed function (f)
    u0_ = torch.ones((m, n, l), dtype=torch.float64) * 1000.0
    f_ = torch.ones((m, n, l), dtype=torch.float64)

    # u0_[0, 0, 0] = 0.0
    u0_[1, 1, 0] = 0.0
    # f_[m // 3 : 2 * m // 3, n // 3 : 2 * n // 3, 0] /= 5.0

    u0 = torch.nn.Parameter(u0_, requires_grad=True)
    f = torch.nn.Parameter(f_, requires_grad=True)

    # Define the grid spacing
    h = 0.5

    # Create the Eikonal solver
    eikonal_solver = Eikonal3D(h)

    # Solve the Eikonal equation
    u = eikonal_solver(u0, f)

    # Print the result
    print("Solution u:")
    # print(u[:, :, 0])

    # Compute some loss (e.g., mean of u)
    # loss = u.mean()
    loss = u[m - 1, n - 1, 0]

    # Backward pass
    loss.backward()

    # Print gradients
    print("\nGradient of u0:")
    # print(u0.grad.detach().numpy()[:, :, 0])

    print("\nGradient of f:")
    # print(f.grad.detach().numpy()[:, :, 0])

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    im = ax[0].imshow(f.detach().numpy()[:, :, 0], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Speed function f")
    im = ax[1].imshow(u.detach().numpy()[:, :, 0] / h, cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Solution u")
    # im = ax[1].imshow(u0.grad.detach().numpy()[:, :, 0], cmap="viridis")
    # fig.colorbar(im, ax=ax[1])
    # ax[1].set_title("Initial condition u0")
    im = ax[2].imshow(f.grad.detach().numpy()[:, :, 0] / h, cmap="viridis")
    fig.colorbar(im, ax=ax[2])
    ax[2].set_title("Gradient of f")

    plt.savefig("eikonal3d.png")
