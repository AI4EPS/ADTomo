import eikonal3d_op
import matplotlib.pyplot as plt
import torch


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
        grad_f = eikonal3d_op.backward(grad_output, u, f, ctx.h, ctx.x, ctx.y, ctx.z)
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
    m, n, l = 20, 15, 10

    # Create initial conditions (u0) and speed function (f)
    f_ = torch.ones((m, n, l), dtype=torch.float64)

    x, y, z = 1, 1, 0
    # u0_[0, 0, 0] = 0.0
    # u0_[1, 1, 0] = 0.0
    # f_[m // 3 : 2 * m // 3, n // 3 : 2 * n // 3, 0] /= 5.0

    f = torch.nn.Parameter(f_, requires_grad=True)

    # Define the grid spacing
    h = 0.5

    # Create the Eikonal solver
    eikonal_solver = Eikonal3D(h, x, y, z)

    # Solve the Eikonal equation
    u = eikonal_solver(f)

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
