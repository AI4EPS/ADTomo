"""3-D eikonal tomography: travel-time prediction and the arrival-time objective."""

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
    """P or S travel times from a station's fixed 3-D grid to live event positions."""
    velocity = grid.sample_model({"P": model.vp, "S": model.vs}[phase.upper()])
    if torch.any(velocity <= 0):
        raise ValueError("velocity must stay positive")
    traveltime = _Eikonal3D.apply(1.0 / velocity, grid.spacing, *grid.station_index.tolist())
    return grid.sample_events(traveltime, events_spherical)


class Tomography(nn.Module):
    """Arrival-time objective over a 3-D velocity model and trainable event parameters.

    ``event_loc`` (``(N, 3)`` lon/lat/depth) and ``event_time_correction``
    (``N`` s) start from the catalog, so ``t_pred - t0_initial = dt0 + T``
    against ``observed_phase_dt = phase_time - t0_initial``. Station groups are
    ``[(grid, [(phase, event_indices, observed_phase_dt), ...]), ...]`` with
    ``event_indices`` indexing ``event_loc``.
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
        dz = model.depth[1:] - model.depth[:-1]
        dphi = torch.deg2rad(model.lat[1:] - model.lat[:-1])
        dlambda = torch.deg2rad(model.lon[1:] - model.lon[:-1])
        radius = R_EARTH - model.depth[:-1]
        cos_lat = torch.cos(torch.deg2rad(model.lat[:-1]))
        volume = (
            radius[:, None, None].square()
            * cos_lat[None, :, None]
            * dz[:, None, None]
            * dphi[None, :, None]
            * dlambda[None, None, :]
        )
        for name, value in zip(("dz", "dphi", "dlambda", "radius", "cos_lat", "volume"), (dz, dphi, dlambda, radius, cos_lat, volume)):
            self.register_buffer(name, value)

    def _smoothness(self, field):
        reference = field[:-1, :-1, :-1]
        grad_depth = (field[1:, :-1, :-1] - reference) / self.dz[:, None, None]
        grad_lat = (field[:-1, 1:, :-1] - reference) / (self.radius[:, None, None] * self.dphi[None, :, None])
        grad_lon = (field[:-1, :-1, 1:] - reference) / (
            self.radius[:, None, None] * self.cos_lat[None, :, None] * self.dlambda[None, None, :]
        )
        grad2 = grad_depth.square() + grad_lat.square() + grad_lon.square()
        return (grad2 * self.volume).sum() / self.volume.sum()

    def forward(self, station_groups, data_scale=None, regularization_scale=1.0):
        """Data loss plus regularization.

        ``data_scale`` replaces the mean over residuals (use ``1 /
        total_observations`` when gradients are summed over ranks, with
        ``regularization_scale = 1 / world_size`` so the regularization is
        counted once).
        """
        residuals = []
        for grid, phase_groups in station_groups:
            for phase, event_indices, observed_phase_dt in phase_groups:
                travel_time = predict_travel_times(self.model, grid, phase, self.event_loc[event_indices])
                residuals.append(travel_time + self.event_time_correction[event_indices] - observed_phase_dt)
        if residuals:
            residual = torch.cat(residuals)
        else:
            residual = sum(parameter.sum() for parameter in self.parameters()).reshape(1) * 0.0
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
                smooth_vp = self._smoothness(dvp)
                regularization_loss = regularization_loss + self.beta_vp * smooth_vp
            if self.alpha_vp != 0.0:
                damp_vp = dvp.square().mean()
                regularization_loss = regularization_loss + self.alpha_vp * damp_vp
        if self.alpha_vs != 0.0 or self.beta_vs != 0.0:
            dvs = self.model.vs - self.vs0
            if self.beta_vs != 0.0:
                smooth_vs = self._smoothness(dvs)
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
