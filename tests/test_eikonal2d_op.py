import eikonal2d_op
import matplotlib.pyplot as plt
import torch


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
        grad_f = eikonal2d_op.backward(grad_output, u, f, ctx.h, ctx.ix, ctx.jx)
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
    m, n = 10, 10

    # Create initial speed function (f)
    f_ = torch.ones((m, n), dtype=torch.float64)
    # f_[m // 3 : 2 * m // 3, n // 3 : 2 * n // 3] /= 5.0

    f = torch.nn.Parameter(f_, requires_grad=True)

    # Set a point source at the center
    ix, jx = (0, 0)
    # ix, jx = (0.5, 0.5)
    # ix, jx = (1.0, 1.0)

    # Define the grid spacing
    h = 0.5

    # Create the Eikonal solver
    eikonal_solver = Eikonal2D(h, ix, jx)

    # Solve the Eikonal equation
    u = eikonal_solver(f)

    # Print the result
    print("Solution u:")
    print(u[:, :])

    # Compute some loss (e.g., mean of u)
    # loss = u.mean()
    # loss = u[m - 1, n - 1]
    loss = u[0, n - 1]
    # loss = u.mean()
    # loss = u[0, 1]
    # loss = u[1, 1]

    # Backward pass
    loss.backward()

    # Print gradients
    print("\nGradient of f:")
    print(f.grad)

    # Plot the solution
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    im = ax[0].imshow(f.detach().numpy(), cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Speed function f")
    im = ax[1].imshow(u.detach().numpy() / h, cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Solution u")
    ax[1].axis("off")
    im = ax[2].imshow(f.grad.numpy() / h, cmap="viridis")
    fig.colorbar(im, ax=ax[2])
    ax[2].set_title("Gradient of f")
    ax[2].axis("off")
    plt.savefig("solution2d.png")
