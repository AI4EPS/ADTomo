"""2-D eikonal tomography for a spherical 1-D model: travel-time prediction and the objective.

A model ``v(depth)`` keeps every station-event ray in the great-circle plane
through Earth's center, so each station's forward problem is a 2-D Cartesian
eikonal solve on that section (:class:`~adtomo.grid.ForwardGrid2D`) with the
station as the source.
"""

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
    """P or S travel times from a station's fixed section to live event positions."""
    velocity = grid.sample_model({"P": model.vp, "S": model.vs}[phase.upper()], model.depth)
    if torch.any(velocity <= 0):
        raise ValueError("velocity must stay positive")
    # eikonal2d_op works on an (x, y) layout; the grid stores fields as (y, x).
    traveltime = _Eikonal2D.apply((1.0 / velocity).T.contiguous(), grid.spacing, *grid.station_index.tolist()).T
    return grid.sample_events(traveltime, events_spherical)


def smoothness_1d(field, depth):
    """Mean squared physical depth gradient of a depth-only field."""
    return ((field[1:] - field[:-1]) / (depth[1:] - depth[:-1])).square().mean()


class Tomography2D(nn.Module):
    """Arrival-time objective over a 1-D velocity model and trainable event parameters.

    Same interface and conventions as :class:`~adtomo.tomography3d.Tomography`.
    ``huber_delta`` (s) switches the data term to the Huber loss: ``r^2`` for
    ``|r| <= delta``, ``delta * (2 |r| - delta)`` beyond, so outliers count linearly.
    """

    def __init__(self, model, event_loc, beta_vp=0.0, beta_vs=0.0, alpha_vp=0.0, alpha_vs=0.0, huber_delta=None):
        super().__init__()
        self.model = model
        self.register_buffer("vp0", model.vp.detach().clone())
        self.register_buffer("vs0", model.vs.detach().clone())
        event_loc = torch.as_tensor(event_loc, dtype=torch.float64).detach().reshape(-1, 3).contiguous()
        self.event_loc = nn.Parameter(event_loc.clone())
        self.event_time_correction = nn.Parameter(torch.zeros(len(event_loc), dtype=torch.float64))
        self.beta_vp = beta_vp
        self.beta_vs = beta_vs
        self.alpha_vp = alpha_vp
        self.alpha_vs = alpha_vs
        self.huber_delta = huber_delta

    def forward(self, station_groups, data_scale=None, regularization_scale=1.0):
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, event_indices, observed_phase_dt in phase_groups:
                travel_time = predict_travel_times_2d(self.model, grid, phase, self.event_loc[event_indices])
                residuals.append(travel_time + self.event_time_correction[event_indices] - observed_phase_dt)
        residual = torch.cat(residuals)
        if self.huber_delta is None:
            data_sum = residual.square().sum()
        else:
            absolute = residual.abs()
            data_sum = torch.where(absolute <= self.huber_delta, residual.square(), self.huber_delta * (2.0 * absolute - self.huber_delta)).sum()
        data_loss = data_sum / residual.numel() if data_scale is None else data_scale * data_sum
        zero = data_loss.new_zeros(())
        smooth_vp = smooth_vs = damp_vp = damp_vs = zero
        regularization_loss = zero
        if self.alpha_vp != 0.0 or self.beta_vp != 0.0:
            dvp = self.model.vp - self.vp0
            if self.beta_vp != 0.0:
                smooth_vp = smoothness_1d(dvp, self.model.depth)
                regularization_loss = regularization_loss + self.beta_vp * smooth_vp
            if self.alpha_vp != 0.0:
                damp_vp = dvp.square().mean()
                regularization_loss = regularization_loss + self.alpha_vp * damp_vp
        if self.alpha_vs != 0.0 or self.beta_vs != 0.0:
            dvs = self.model.vs - self.vs0
            if self.beta_vs != 0.0:
                smooth_vs = smoothness_1d(dvs, self.model.depth)
                regularization_loss = regularization_loss + self.beta_vs * smooth_vs
            if self.alpha_vs != 0.0:
                damp_vs = dvs.square().mean()
                regularization_loss = regularization_loss + self.alpha_vs * damp_vs
        loss = data_loss + regularization_scale * regularization_loss
        self.data_sum = data_sum.detach()
        self.data_loss = data_loss.detach()
        self.smooth_vp = smooth_vp.detach()
        self.smooth_vs = smooth_vs.detach()
        self.damp_vp = damp_vp.detach()
        self.damp_vs = damp_vs.detach()
        self.regularization_loss = regularization_loss.detach()
        self.total_loss = loss.detach()
        return loss
