"""2-D eikonal tomography."""

import eikonal2d_op
import torch
import torch.nn as nn


class _Eikonal2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, slowness, spacing, x, y):
        traveltime = eikonal2d_op.forward(slowness, spacing, x, y)
        ctx.save_for_backward(traveltime, slowness)
        ctx.spacing = spacing
        ctx.source = (x, y)
        return traveltime

    @staticmethod
    def backward(ctx, grad_output):
        traveltime, slowness = ctx.saved_tensors
        grad_slowness = eikonal2d_op.backward(grad_output.contiguous(), traveltime, slowness, ctx.spacing, *ctx.source)
        return grad_slowness, None, None, None


def predict_travel_times_2d(model, grid, phase, events_spherical):
    """Compute P or S travel times at live event positions."""
    velocity = grid.sample_model({"P": model.vp, "S": model.vs}[phase.upper()], model.depth)
    if torch.any(velocity <= 0):
        raise ValueError("velocity must stay positive")
    # eikonal2d_op works on an (x, y) layout; the grid stores fields as (y, x).
    traveltime = _Eikonal2D.apply((1.0 / velocity).T.contiguous(), grid.spacing, *grid.station_index.tolist()).T
    return grid.sample_events(traveltime, events_spherical)


class Tomography2D(nn.Module):
    """Arrival-time objective for a depth-only velocity model."""

    def __init__(self, model, event_loc, beta_vp=0.0, beta_vs=0.0, alpha_vp=0.0, alpha_vs=0.0):
        super().__init__()
        self.model = model
        self.register_buffer("vp0", model.vp.detach().clone())
        self.register_buffer("vs0", model.vs.detach().clone())
        event_loc = torch.as_tensor(event_loc, dtype=torch.float64).detach().reshape(-1, 3).contiguous()
        self.event_loc_hori = nn.Parameter(event_loc[:, :2].clone())
        self.event_loc_vert = nn.Parameter(event_loc[:, 2].clone())
        self.event_time_correction = nn.Parameter(torch.zeros(len(event_loc), dtype=torch.float64))
        self.beta_vp = beta_vp
        self.beta_vs = beta_vs
        self.alpha_vp = alpha_vp
        self.alpha_vs = alpha_vs
        self.register_buffer("dz", model.depth[1:] - model.depth[:-1])

    @property
    def event_loc(self):
        return torch.cat((self.event_loc_hori, self.event_loc_vert[:, None]), dim=1)

    def _smoothness(self, field):
        gradient = (field[1:] - field[:-1]) / self.dz
        return (gradient.square() * self.dz).sum() / self.dz.sum()

    def forward(self, station_groups, data_scale=None, regularization_scale=1.0):
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, event_indices, observed_phase_dt in phase_groups:
                travel_time = predict_travel_times_2d(self.model, grid, phase, self.event_loc[event_indices])
                residuals.append(travel_time + self.event_time_correction[event_indices] - observed_phase_dt)
        if residuals:
            residual = torch.cat(residuals)
        else:
            residual = sum(parameter.sum() for parameter in self.parameters()).reshape(1) * 0.0
        data_sum = residual.square().sum()
        data_loss = data_sum / residual.numel() if data_scale is None else data_scale * data_sum
        regularization_loss = data_loss.new_zeros(())
        if self.alpha_vp != 0.0 or self.beta_vp != 0.0:
            dvp = self.model.vp - self.vp0
            if self.beta_vp != 0.0:
                regularization_loss = regularization_loss + self.beta_vp * self._smoothness(dvp)
            if self.alpha_vp != 0.0:
                regularization_loss = regularization_loss + self.alpha_vp * dvp.square().mean()
        if self.alpha_vs != 0.0 or self.beta_vs != 0.0:
            dvs = self.model.vs - self.vs0
            if self.beta_vs != 0.0:
                regularization_loss = regularization_loss + self.beta_vs * self._smoothness(dvs)
            if self.alpha_vs != 0.0:
                regularization_loss = regularization_loss + self.alpha_vs * dvs.square().mean()
        loss = data_loss + regularization_scale * regularization_loss
        self.data_sum = data_sum.detach()
        return loss
