import itertools
import math

import torch


def interp(table, *x):
    """Catmull-Rom (bicubic in 2D, tricubic in 3D) interpolation of a travel-time table at grid coordinates x (in
    nodes, one tensor per axis). The time and its derivatives are continuous across the nodes: a bilinear time has a
    kink at every node, where L-BFGS stops (as in ADLoc and GaMMA). One node beyond each edge comes from linear
    extrapolation. Outside the grid the time continues linearly from the nearest grid point c, T(c) + grad T(c) . (x - c):
    continuous, with an exact gradient that never vanishes, so an event pushed outside is pulled back."""
    shape = table.shape
    for d in range(table.dim()):  # pad one node on each side: 2 T[0] - T[1] and 2 T[-1] - T[-2]
        lo = 2 * table.narrow(d, 0, 1) - table.narrow(d, 1, 1)
        hi = 2 * table.narrow(d, -1, 1) - table.narrow(d, -2, 1)
        table = torch.cat([lo, table, hi], d)
    # Outside the grid the time continues linearly from the nearest grid point c (the event clamped to the grid; c = x
    # inside) with the slope at c:
    #     T(x) = T(c) + sum_d dT/dx_d(c) (x_d - c_d)
    # (inside, x - c = 0: the bicubic time). Autograd differentiates this exactly: c follows the event along an axis
    # where the event is inside the grid (clamp passes the gradient) and stays fixed along an axis where it is outside
    # (clamp blocks it), which adds the cross terms d2T/dx_d dx_e (c) (x_e - c_e) for an event above the top, say,
    # whose c slides along the top face. Time and gradient are continuous across the edge, and the gradient is the
    # edge's, never zero, so an event pushed outside is pulled back toward the data.
    index, weight, dweight, beyond = [], [], [], []
    for xd, n in zip(x, shape):
        xc = xd.clamp(0, n - 1)
        i = torch.floor(xc.detach()).clamp(0, n - 2)
        f = xc - i
        f2, f3 = f * f, f * f * f
        weight.append([(-f3 + 2 * f2 - f) / 2, (3 * f3 - 5 * f2 + 2) / 2, (-3 * f3 + 4 * f2 + f) / 2, (f3 - f2) / 2])
        dweight.append([(-3 * f2 + 4 * f - 1) / 2, (9 * f2 - 10 * f) / 2, (-9 * f2 + 8 * f + 1) / 2, (3 * f2 - 2 * f) / 2])
        index.append(i.long())  # node i - 1 + o of the table is node i + o of the padded table
        beyond.append(xd - xc)  # zero inside the grid
    t = 0.0
    for offset in itertools.product(range(4), repeat=len(x)):
        v = table[tuple(index[d] + o for d, o in enumerate(offset))]
        w = [weight[d][o] for d, o in enumerate(offset)]
        t = t + math.prod(w) * v
        for d, o in enumerate(offset):  # + dT/dx_d (c) * (x_d - c_d)
            t = t + beyond[d] * math.prod(w[:d] + [dweight[d][o]] + w[d + 1 :]) * v
    return t
