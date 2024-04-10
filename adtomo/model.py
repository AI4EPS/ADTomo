import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import eik2d_cpp

class Eik2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,u,m,n,h,f,m1,n1):
        eik2d_cpp.forward(u.contiguous(),f.contiguous(),m,n,h,m1,n1)
        ctx.save_for_backward(u,f) # save tensors
        ctx.m = m
        ctx.n = n
        ctx.h = h
        ctx.m1 = m1
        ctx.n1 = n1
        return u

    @staticmethod 
    def backward(ctx,grad_output): # grad_output = delta L / delta u
        u,f = ctx.saved_tensors
        m = ctx.m 
        n = ctx.n 
        h = ctx.h 
        m1 = ctx.m1 
        n1 = ctx.n1 
        grad_u = torch.ones(m*n, dtype=torch.double) 
        grad_u = grad_u * grad_output
        grad_f = torch.zeros(m*n, dtype=torch.double)
        eik2d_cpp.backward(grad_f.contiguous(),grad_u.contiguous(),u.contiguous(),f.contiguous(),m,n,h,m1,n1)
        gm = None
        gn = None
        gh = None
        gm1 = None
        gn1 = None
        return grad_output, gm, gn, gh, grad_f, gm1, gn1

class Eikonal2D(torch.nn.Module):
    def __init__(self,
        m,
        n,
        h,
        tol,
        num_event,
        num_station,
        dx,
        dy,
        f=None,
        station_loc=None,
        event_loc=None):
        super().__init__()

        self.m = m
        self.n = n
        self.h = h
        self.tol = tol

        self.num_event = num_event
        self.num_station = num_station

        self.dx = dx
        self.dy = dy

        if f is None:
            self.f = torch.nn.Parameter(torch.ones(m*n,dtype=torch.double),requires_grad=True)
        else:
            self.f = f

        dtype = torch.double
        
        if event_loc is not None:
            self.event_loc = nn.Embedding(num_event,3)
            self.event_loc.weight = torch.nn.Parameter(torch.tensor(event_loc, dtype=dtype),requires_grad=False)

        if station_loc is not None:
            self.station_loc = nn.Embedding(num_station,3)
            self.station_loc.weight = torch.nn.Parameter(torch.tensor(station_loc, dtype=dtype), requires_grad=False)


        print('Initialized Eikonal2D object')

    def forward(self,m1,n1,gridx,gridy):

        u = torch.ones(self.m*self.n, dtype=torch.double)
        self.u = Eik2DFunction.apply(u,self.m,self.n,self.h,self.f,m1,n1)

        # x, y are vectors
        grid_x = torch.tensor(gridx)
        grid_y = torch.tensor(gridy)

        # indices of boundary of cells :: on grid
        x_upper = torch.ceil(grid_x)
        x_lower = torch.floor(grid_x)
        y_upper = torch.ceil(grid_y)
        y_lower = torch.floor(grid_y)

        # indices of bdry of cells :: stored 1D
        g_uu = x_upper*self.n + y_upper
        g_ul = x_upper*self.n + y_lower
        g_lu = x_lower*self.n + y_upper
        g_ll = x_lower*self.n + y_lower

        # Gather based on indices :: those are values on the grid
        f_x2y2 = torch.take_along_dim(self.u,g_uu.to(torch.int64))
        f_x2y1 = torch.take_along_dim(self.u,g_ul.to(torch.int64))
        f_x1y1 = torch.take_along_dim(self.u,g_ll.to(torch.int64))
        f_x1y2 = torch.take_along_dim(self.u,g_lu.to(torch.int64))

        xh = x_upper - grid_x #(x2 - x)
        xl = grid_x - x_lower #(x - x1)
        yh = y_upper - grid_y #(y2 - y)
        yl = grid_y - y_lower #(y - y1)

        # weights
        w11 = xh*yh
        w12 = xh*yl
        w21 = xl*yh
        w22 = xl*yl

        # Assemble
        t_xyz = (w11*f_x1y1 + w12*f_x2y1 + w21*f_x2y1 + w22*f_x2y2) * (1/(self.dx*self.dy))

        return t_xyz

