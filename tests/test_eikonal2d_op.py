import eikonal2d_op
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


# eikonal = Eikonal(h=1, ix=0, jx=0)
# f = torch.ones(11, 11, dtype=torch.double, requires_grad=True)
# u = eikonal(f)

# print(u)
