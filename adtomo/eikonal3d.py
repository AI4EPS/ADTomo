import eik3d_cpp
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader


class Eik3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, u0, m, n, l, h, tol, verb, f):
        # print("Calling forward")
        eik3d_cpp.forward(u, f, tol, verb, m, n, l, h)
        ctx.save_for_backward(u, f, u0)
        ctx.tol = tol
        ctx.verb = verb
        ctx.m = m
        ctx.n = n
        ctx.l = l
        ctx.h = h
        return u

    @staticmethod
    def backward(ctx, grad_output):
        # print("Calling backward")
        u, f, u0 = ctx.saved_tensors
        tol = ctx.tol
        verb = ctx.verb
        m = ctx.m
        n = ctx.n
        l = ctx.l
        h = ctx.h
        grad_u = torch.ones(m * n * l, dtype=torch.double)
        grad_u = grad_u * grad_output
        grad_f = torch.zeros(m * n * l, dtype=torch.double)
        grad_u0 = torch.zeros(m * n * l, dtype=torch.double)
        eik3d_cpp.backward(grad_u0, grad_f, grad_u, u, u0, f, h, m, n, l)
        grad_m = None
        grad_n = None
        grad_l = None
        grad_h = None
        grad_tol = None
        grad_verb = None
        return grad_output, grad_u0, grad_m, grad_n, grad_l, grad_h, grad_tol, grad_verb, grad_f


class Eikonal3D(torch.nn.Module):
    def __init__(
        self,
        m,
        n,
        l,
        h,
        tol,
        dx,
        dy,
        dz,
        num_event,
        num_station,
        f=None,
        station_loc=None,
        event_loc=None,
        ratio=None,
        ratio_s=None,
        smooth_hori=None,
        smooth_vert=None,
        lambda_p=None,
    ):
        super().__init__()
        self.m = m
        self.n = n
        self.l = l
        self.h = h
        self.tol = tol

        self.dx = dx
        self.dy = dy
        self.dz = dz

        if f is None:
            exit("field has to be prescribed")
        else:
            # assume f precribed in layers
            self.f = torch.clone(f)
            self.f = torch.flatten(self.f)
            self.f = torch.nn.Parameter(self.f, requires_grad=True)

        self.num_station = num_station
        self.num_event = num_event
        dtype = torch.double

        if event_loc is not None:
            self.event_loc = nn.Embedding(num_event, 3)
            self.event_loc.weight = torch.nn.Parameter(torch.tensor(event_loc, dtype=dtype), requires_grad=False)

        if station_loc is not None:
            self.station_loc = nn.Embedding(num_station, 3)
            self.station_loc.weight = torch.nn.Parameter(torch.tensor(station_loc, dtype=dtype), requires_grad=False)

        print("Initialized Eikonal 3D object")

        # Not used yet
        if ratio_s is None:
            self.ratio_s = torch.ones(m, n, l, dtype=torch.double)
            # exit("no ratio")
            # self.ratio_s = (torch.ones(m,n,l,dtype=torch.double) * torch.sqrt(3.0,dtype=torch.double))
        else:
            # check for size
            # if ratio_s.size != self.ftest.size:
            #     print('ratio_s and f size different',ratio_s.size,self.ftest.size)
            #     exit('1')
            self.ratio_s = ratio_s
            self.ratio_s = torch.reshape(self.ratio_s, (m, n, l))
            self.ratio_sf = torch.clone(ratio_s)
            self.ratio_sf = torch.flatten(self.ratio_sf)

        # smoothing parameters -- if none, they are not used, set dummy value for now
        if smooth_hori is None:
            self.smooth_hori = 1
        else:
            self.smooth_hori = smooth_hori

        if smooth_vert is None:
            smooth_vert = 1
        else:
            self.smooth_vert = smooth_vert
        if lambda_p is None:
            self.lambda_p = 1
        else:
            self.lambda_p = lambda_p

    def init_u(self, ix, iy, iz, phase):
        h = self.h
        m = self.m
        n = self.n
        l = self.l
        nl = n * l
        ixu = int(np.ceil(ix))
        ixd = int(np.floor(ix))
        iyu = int(np.ceil(iy))
        iyd = int(np.floor(iy))
        izu = int(np.ceil(iz))
        izd = int(np.floor(iz))

        if phase == "P":
            self.u0 = torch.ones(self.m * self.n * self.l, dtype=torch.double) * 1000.0
            self.u0[ixu * nl + iyu * l + izu] = (
                np.sqrt((ix - ixu) ** 2 + (iy - iyu) ** 2 + (iz - izu) ** 2) * h / self.f[ixu * nl + iyu * l + izu]
            )
            self.u0[ixu * nl + iyu * l + izd] = (
                np.sqrt((ix - ixu) ** 2 + (iy - iyu) ** 2 + (iz - izd) ** 2) * h / self.f[ixu * nl + iyu * l + izd]
            )
            self.u0[ixu * nl + iyd * l + izu] = (
                np.sqrt((ix - ixu) ** 2 + (iy - iyd) ** 2 + (iz - izu) ** 2) * h / self.f[ixu * nl + iyd * l + izu]
            )
            self.u0[ixu * nl + iyd * l + izd] = (
                np.sqrt((ix - ixu) ** 2 + (iy - iyd) ** 2 + (iz - izd) ** 2) * h / self.f[ixu * nl + iyd * l + izd]
            )
            self.u0[ixd * nl + iyu * l + izu] = (
                np.sqrt((ix - ixd) ** 2 + (iy - iyu) ** 2 + (iz - izu) ** 2) * h / self.f[ixd * nl + iyu * l + izu]
            )
            self.u0[ixd * nl + iyu * l + izd] = (
                np.sqrt((ix - ixd) ** 2 + (iy - iyu) ** 2 + (iz - izd) ** 2) * h / self.f[ixd * nl + iyu * l + izd]
            )
            self.u0[ixd * nl + iyd * l + izu] = (
                np.sqrt((ix - ixd) ** 2 + (iy - iyd) ** 2 + (iz - izu) ** 2) * h / self.f[ixd * nl + iyd * l + izu]
            )
            self.u0[ixd * nl + iyd * l + izd] = (
                np.sqrt((ix - ixd) ** 2 + (iy - iyd) ** 2 + (iz - izd) ** 2) * h / self.f[ixd * nl + iyd * l + izd]
            )

        elif phase == "S":
            self.u0 = torch.ones(self.m * self.n * self.l, dtype=torch.double) * 1000.0
            self.u0[ixu * nl + iyu * l + izu] = (
                np.sqrt((ix - ixu) ** 2 + (iy - iyu) ** 2 + (iz - izu) ** 2)
                * h
                / (self.f[ixu * nl + iyu * l + izu] * self.ratio_s[ixu * nl + iyu * l + izu])
            )
            self.u0[ixu * nl + iyu * l + izd] = (
                np.sqrt((ix - ixu) ** 2 + (iy - iyu) ** 2 + (iz - izd) ** 2)
                * h
                / (self.f[ixu * nl + iyu * l + izd] * self.ratio_s[ixu * nl + iyu * l + izd])
            )
            self.u0[ixu * nl + iyd * l + izu] = (
                np.sqrt((ix - ixu) ** 2 + (iy - iyd) ** 2 + (iz - izu) ** 2)
                * h
                / (self.f[ixu * nl + iyd * l + izu] * self.ratio_s[ixu * nl + iyd * l + izu])
            )
            self.u0[ixu * nl + iyd * l + izd] = (
                np.sqrt((ix - ixu) ** 2 + (iy - iyd) ** 2 + (iz - izd) ** 2)
                * h
                / (self.f[ixu * nl + iyd * l + izd] * self.ratio_s[ixu * nl + iyd * l + izd])
            )
            self.u0[ixd * nl + iyu * l + izu] = (
                np.sqrt((ix - ixd) ** 2 + (iy - iyu) ** 2 + (iz - izu) ** 2)
                * h
                / (self.f[ixd * nl + iyu * l + izu] * self.ratio_s[ixd * nl + iyu * l + izu])
            )
            self.u0[ixd * nl + iyu * l + izd] = (
                np.sqrt((ix - ixd) ** 2 + (iy - iyu) ** 2 + (iz - izd) ** 2)
                * h
                / (self.f[ixd * nl + iyu * l + izd] * self.ratio_s[ixd * nl + iyu * l + izd])
            )
            self.u0[ixd * nl + iyd * l + izu] = (
                np.sqrt((ix - ixd) ** 2 + (iy - iyd) ** 2 + (iz - izu) ** 2)
                * h
                / (self.f[ixd * nl + iyd * l + izu] * self.ratio_s[ixd * nl + iyd * l + izu])
            )
            self.u0[ixd * nl + iyd * l + izd] = (
                np.sqrt((ix - ixd) ** 2 + (iy - iyd) ** 2 + (iz - izd) ** 2)
                * h
                / (self.f[ixd * nl + iyd * l + izd] * self.ratio_s[ixd * nl + iyd * l + izd])
            )
        else:
            print("Unknown phase type. Should be P or S")
            exit("unknown phase")

    def forward(self, x, y, z, g_x, g_y, g_z, phase):

        self.init_u(x, y, z, phase)
        u = torch.clone(self.u0)

        if phase == "P":
            self.u = Eik3DFunction.apply(u, self.u0, self.m, self.n, self.l, self.h, self.tol, True, 1 / self.f)
        elif phase == "S":
            self.u = Eik3DFunction.apply(
                u, self.u0, self.m, self.n, self.l, self.h, self.tol, True, 1 / (self.f * self.ratio_s)
            )
        else:
            exit("S . T")

        nl = self.n * self.l
        l = self.l

        grid_x = torch.tensor(g_x, dtype=torch.double)
        grid_y = torch.tensor(g_y, dtype=torch.double)
        grid_z = torch.tensor(g_z, dtype=torch.double)

        x1 = (torch.floor(grid_x)).to(torch.int)
        y1 = (torch.floor(grid_y)).to(torch.int)
        z1 = (torch.floor(grid_z)).to(torch.int)
        x2 = (torch.ceil(grid_x)).to(torch.int)
        y2 = (torch.ceil(grid_y)).to(torch.int)
        z2 = (torch.ceil(grid_z)).to(torch.int)

        f111_val_ind = x1 * nl + y1 * l + z1
        f211_val_ind = x2 * nl + y1 * l + z1

        f121_val_ind = x1 * nl + y2 * l + z1
        f221_val_ind = x2 * nl + y2 * l + z1

        f112_val_ind = x1 * nl + y1 * l + z2
        f212_val_ind = x2 * nl + y1 * l + z2

        f122_val_ind = x1 * nl + y2 * l + z2
        f222_val_ind = x2 * nl + y2 * l + z2

        f111_ = torch.take_along_dim(self.u, f111_val_ind.to(torch.int64))
        f211_ = torch.take_along_dim(self.u, f211_val_ind.to(torch.int64))

        f121_ = torch.take_along_dim(self.u, f121_val_ind.to(torch.int64))
        f221_ = torch.take_along_dim(self.u, f221_val_ind.to(torch.int64))

        f112_ = torch.take_along_dim(self.u, f112_val_ind.to(torch.int64))
        f212_ = torch.take_along_dim(self.u, f212_val_ind.to(torch.int64))

        f122_ = torch.take_along_dim(self.u, f122_val_ind.to(torch.int64))
        f222_ = torch.take_along_dim(self.u, f222_val_ind.to(torch.int64))

        t_xyz = (
            (z2 - grid_z) * (y2 - grid_y) * ((x2 - grid_x) * f111_ + (grid_x - x1) * f211_)
            + (z2 - grid_z) * (grid_y - y1) * ((x2 - grid_x) * f121_ + (grid_x - x1) * f221_)
            + (grid_z - z1) * (y2 - grid_y) * ((x2 - grid_x) * f112_ + (grid_x - x1) * f212_)
            + (grid_z - z1) * (grid_y - y1) * ((x2 - grid_x) * f122_ + (grid_x - x1) * f222_)
        )

        return t_xyz
