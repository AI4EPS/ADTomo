"""3-D eikonal tomography."""

import eikonal3d_op
import torch
import torch.nn as nn

from .grid import R_EARTH


class _Eikonal3D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slowness, spacing, x, y, z):
        traveltime = eikonal3d_op.forward(slowness, spacing, x, y, z)
        ctx.save_for_backward(traveltime, slowness)
        ctx.spacing = spacing
        ctx.source = (x, y, z)
        return traveltime

    @staticmethod
    def backward(ctx, grad_output):
        traveltime, slowness = ctx.saved_tensors
        grad_slowness = eikonal3d_op.backward(grad_output.contiguous(), traveltime, slowness, ctx.spacing, *ctx.source)
        return grad_slowness, None, None, None, None


def predict_travel_times(model, grid, phase, events_spherical):
    """Compute P or S travel times at live event positions."""
    velocity = grid.sample_model({"P": model.vp, "S": model.vs}[phase.upper()])
    if torch.any(velocity <= 0):
        raise ValueError("velocity must stay positive")
    traveltime = _Eikonal3D.apply(1.0 / velocity, grid.spacing, *grid.station_index.tolist())
    return grid.sample_events(traveltime, events_spherical)


class Tomography(nn.Module):
    """Arrival-time objective with trainable velocity and event parameters."""

    def __init__(self, model, event_loc, beta_vp=0.0, beta_vs=0.0, alpha_vp=0.0, alpha_vs=0.0):
        super().__init__()
        self.model = model
        self.register_buffer("vp0", model.vp.detach().clone())
        self.register_buffer("vs0", model.vs.detach().clone())
        event_loc = torch.as_tensor(event_loc, dtype=torch.float64).detach().reshape(-1, 3).contiguous()
        self.event_loc_hori = nn.Parameter(event_loc[:, :2].clone())
        self.event_loc_vert = nn.Parameter(event_loc[:, 2].clone())
        self.event_time_correction = nn.Parameter(torch.zeros(len(event_loc), dtype=torch.float64))
        self.beta_vp, self.beta_vs = beta_vp, beta_vs
        self.alpha_vp, self.alpha_vs = alpha_vp, alpha_vs

        depth = model.depth.detach().to(dtype=torch.float64)
        phi = torch.deg2rad(model.lat.detach().to(dtype=torch.float64))
        lam = torch.deg2rad(model.lon.detach().to(dtype=torch.float64))
        radius = R_EARTH - depth
        cos_lat = torch.cos(phi)

        def dual(axis):
            spacing = axis[1:] - axis[:-1]
            weights = torch.empty_like(axis)
            weights[0], weights[-1] = 0.5 * spacing[0], 0.5 * spacing[-1]
            weights[1:-1] = 0.5 * (spacing[:-1] + spacing[1:])
            return weights

        volume = (
            radius[:, None, None].square()
            * cos_lat[None, :, None]
            * dual(depth)[:, None, None]
            * dual(phi)[None, :, None]
            * dual(lam)[None, None, :]
        )
        for name, value in zip(
            ("depth", "phi", "lam", "radius", "cos_lat", "volume"),
            (depth, phi, lam, radius, cos_lat, volume),
        ):
            self.register_buffer(name, value)

    @property
    def event_loc(self):
        return torch.cat((self.event_loc_hori, self.event_loc_vert[:, None]), dim=1)

    def damping(self, field):
        return (field.square() * self.volume).sum() / self.volume.sum()

    def smoothness(self, field):
        def grad2(field, axis, dim):
            field = field.transpose(0, dim)
            gradient = (field[1:] - field[:-1]) / (axis[1:] - axis[:-1]).reshape(
                (-1,) + (1,) * (field.ndim - 1)
            )
            gradient = torch.cat((gradient[:1], gradient, gradient[-1:]), dim=0)
            return (0.5 * (gradient[:-1].square() + gradient[1:].square())).transpose(0, dim)

        grad_depth = grad2(field, self.depth, 0)
        grad_lat = grad2(field, self.phi, 1) / self.radius[:, None, None].square()
        grad_lon = grad2(field, self.lam, 2) / (
            self.radius[:, None, None] * self.cos_lat[None, :, None]
        ).square()
        return ((grad_depth + grad_lat + grad_lon) * self.volume).sum() / self.volume.sum()

    def forward(self, station_groups, data_scale=None, regularization_scale=1.0):
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, event_indices, observed_phase_dt in phase_groups:
                travel_time = predict_travel_times(self.model, grid, phase, self.event_loc[event_indices])
                residuals.append(travel_time + self.event_time_correction[event_indices] - observed_phase_dt)
        residual = torch.cat(residuals)
        data_sum = residual.square().sum()
        data_loss = data_sum / residual.numel() if data_scale is None else data_scale * data_sum
        regularization_loss = data_loss.new_zeros(())
        if self.alpha_vp != 0.0 or self.beta_vp != 0.0:
            dvp = self.model.vp - self.vp0
            if self.beta_vp != 0.0:
                regularization_loss = regularization_loss + self.beta_vp * self.smoothness(dvp)
            if self.alpha_vp != 0.0:
                regularization_loss = regularization_loss + self.alpha_vp * self.damping(dvp)
        if self.alpha_vs != 0.0 or self.beta_vs != 0.0:
            dvs = self.model.vs - self.vs0
            if self.beta_vs != 0.0:
                regularization_loss = regularization_loss + self.beta_vs * self.smoothness(dvs)
            if self.alpha_vs != 0.0:
                regularization_loss = regularization_loss + self.alpha_vs * self.damping(dvs)
        loss = data_loss + regularization_scale * regularization_loss
        self.data_sum = data_sum.detach()
        return loss
