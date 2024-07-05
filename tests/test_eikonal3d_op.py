import eikonal3d_op
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
    m, n, l = 10, 10, 10

    # Create initial conditions (u0) and speed function (f)
    u0 = torch.ones((m, n, l), dtype=torch.float64) * 1000.0
    f = torch.ones((m, n, l), dtype=torch.float64, requires_grad=True)

    # Set a point source at the center
    center = m // 2
    u0[0, 0, 0] = 0.0

    # Define the grid spacing
    h = 1.0

    # Create the Eikonal solver
    eikonal_solver = Eikonal3D(h)

    # Solve the Eikonal equation
    u = eikonal_solver(u0, f)

    # Print the result
    print("Solution u:")
    print(u[:, :, 0])

    # Compute some loss (e.g., mean of u)
    loss = u.mean()

    # Backward pass
    loss.backward()

    # Print gradients
    print("\nGradient of u0:")
    # print(u0.grad)

    print("\nGradient of f:")
    # print(f.grad)
