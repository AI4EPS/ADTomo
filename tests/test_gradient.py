import math
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adtomo import (
    ForwardGrid,
    ForwardGrid2D,
    Tomography,
    Tomography2D,
    VelocityModel,
    VelocityModel1D,
    predict_travel_times,
    predict_travel_times_2d,
    smoothness,
)
from adtomo.tomography3d import _Eikonal3D


def solve_eikonal3d(velocity, source, spacing):
    return _Eikonal3D.apply((1.0 / velocity).contiguous(), float(spacing), *[float(v) for v in source])


FIGURES = Path("figures")
FIGURES.mkdir(exist_ok=True)

# Top-boundary source regression: source z=0 uses iz0=0 and iz1=1.
taylor_velocity = torch.full((4, 5, 6), 5.0, dtype=torch.float64, requires_grad=True)
taylor_direction = torch.linspace(-0.2, 0.2, taylor_velocity.numel(), dtype=torch.float64).reshape_as(taylor_velocity)
top_source = (1.2, 1.3, 0.0)
taylor_objective = solve_eikonal3d(taylor_velocity, top_source, 1.0)[3, 4, 5]
taylor_objective.backward()
taylor_derivative = (taylor_velocity.grad * taylor_direction).sum().item()
kernel_epsilons = (1e-2, 5e-3, 2.5e-3, 1.25e-3)
kernel_changes = []
kernel_remainders = []
for epsilon in kernel_epsilons:
    perturbed = solve_eikonal3d(taylor_velocity.detach() + epsilon * taylor_direction, top_source, 1.0)[3, 4, 5]
    kernel_changes.append(abs(perturbed.item() - taylor_objective.item()))
    kernel_remainders.append(abs(perturbed.item() - taylor_objective.item() - epsilon * taylor_derivative))
kernel_change_slopes = [
    math.log(right / left) / math.log(next_epsilon / epsilon)
    for left, right, epsilon, next_epsilon in zip(
        kernel_changes, kernel_changes[1:], kernel_epsilons, kernel_epsilons[1:]
    )
]
kernel_slopes = [
    math.log(right / left) / math.log(next_epsilon / epsilon)
    for left, right, epsilon, next_epsilon in zip(
        kernel_remainders, kernel_remainders[1:], kernel_epsilons, kernel_epsilons[1:]
    )
]
assert 0.8 < statistics.median(kernel_change_slopes) < 1.2
assert 1.8 < statistics.median(kernel_slopes) < 2.2

lon = torch.arange(-120.8, -119.19, 0.1, dtype=torch.float64)
lat = torch.arange(34.2, 35.81, 0.1, dtype=torch.float64)
depth = torch.arange(-15.0, 50.1, 5.0, dtype=torch.float64)
vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
station_spherical = torch.tensor([-120.0, 35.0, 0.0], dtype=torch.float64)
events_spherical = torch.tensor([[-119.9, 35.1, 10.0]], dtype=torch.float64)

# Full global-Vp chain: sampling, solver, event interpolation, loss.
taylor_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
taylor_grid = ForwardGrid(station_spherical, events_spherical, taylor_model, spacing=5.0)
assert taylor_grid.z[0].item() == 0.0
assert taylor_grid.station_index[2].item() == 0.0
global_direction = torch.linspace(-0.01, 0.01, taylor_model.vp.numel(), dtype=torch.float64).reshape_as(taylor_model.vp)


def full_phase_time_loss(candidate):
    predicted_phase_dt = predict_travel_times(candidate, taylor_grid, "P", events_spherical)
    return (predicted_phase_dt - 3.0).square().sum()


full_loss = full_phase_time_loss(taylor_model)
full_loss.backward()
assert taylor_model.vp.grad is not None
assert torch.isfinite(taylor_model.vp.grad).all()
assert taylor_model.vp.grad.abs().sum() > 0
full_derivative = (taylor_model.vp.grad * global_direction).sum().item()
base_vp = taylor_model.vp.detach()
epsilons = (1e-2, 5e-3, 2.5e-3, 1.25e-3)
full_remainders = []
for epsilon in epsilons:
    perturbed_model = VelocityModel(lon, lat, depth, base_vp + epsilon * global_direction, vp / 1.73, trainable=False)
    full_remainders.append(abs(full_phase_time_loss(perturbed_model).item() - full_loss.item() - epsilon * full_derivative))
assert all(math.isfinite(remainder) and remainder > 0 for remainder in full_remainders)
assert all(right < left for left, right in zip(full_remainders, full_remainders[1:]))
full_slopes = [
    math.log(right / left) / math.log(next_epsilon / epsilon)
    for left, right, epsilon, next_epsilon in zip(full_remainders, full_remainders[1:], epsilons, epsilons[1:])
]
assert 1.7 < sorted(full_slopes)[len(full_slopes) // 2] < 2.3

# Regularized tomography objective: smoothness applies to perturbations only.
objective_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
objective_grid = ForwardGrid(station_spherical, events_spherical, objective_model, spacing=5.0)
station_groups = [
    (
        objective_grid,
        [
            ("P", torch.tensor([0]), torch.tensor([3.0], dtype=torch.float64)),
            ("S", torch.tensor([0]), torch.tensor([5.0], dtype=torch.float64)),
        ],
    )
]
data_only = Tomography(objective_model, events_spherical)
data_only_loss = data_only(station_groups)
direct_residual = torch.cat(
    [
        predict_travel_times(objective_model, objective_grid, "P", events_spherical) - 3.0,
        predict_travel_times(objective_model, objective_grid, "S", events_spherical) - 5.0,
    ]
)
assert torch.allclose(data_only_loss, direct_residual.square().mean())
assert data_only.damp_vp.item() == 0.0
assert data_only.damp_vs.item() == 0.0
assert smoothness(
    torch.full_like(objective_model.vp, 0.2), objective_model.lon, objective_model.lat, objective_model.depth
).item() == 0.0

# A linear physical depth gradient has the same penalty on coarse and fine depth axes.
depth_gradient = 0.02
depth_coarse = torch.tensor([0.0, 5.0, 10.0], dtype=torch.float64)
depth_fine = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0], dtype=torch.float64)
lon_small = torch.tensor([-120.0, -119.9], dtype=torch.float64)
lat_small = torch.tensor([35.0, 35.1], dtype=torch.float64)
field_coarse = (depth_gradient * depth_coarse)[:, None, None].expand(-1, len(lat_small), len(lon_small))
field_fine = (depth_gradient * depth_fine)[:, None, None].expand(-1, len(lat_small), len(lon_small))
coarse_smoothness = smoothness(field_coarse, lon_small, lat_small, depth_coarse)
fine_smoothness = smoothness(field_fine, lon_small, lat_small, depth_fine)
assert torch.allclose(coarse_smoothness, torch.tensor(depth_gradient**2, dtype=torch.float64))
assert torch.allclose(fine_smoothness, coarse_smoothness)

constant_model = VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=True)
constant_tomography = Tomography(constant_model, events_spherical, alpha_vp=0.5, alpha_vs=0.25)
with torch.no_grad():
    constant_model.vp += 0.2
    constant_model.vs -= 0.1
constant_tomography(station_groups)
assert constant_tomography.smooth_vp.item() == 0.0
assert constant_tomography.smooth_vs.item() == 0.0
assert constant_tomography.damp_vp.item() > 0.0
assert constant_tomography.damp_vs.item() > 0.0

regularized = Tomography(objective_model, events_spherical, beta_vp=0.5, beta_vs=0.25, alpha_vp=0.125, alpha_vs=0.0625)
with torch.no_grad():
    objective_model.vp[2, 3, 4] += 0.2
    objective_model.vs[2, 3, 4] -= 0.1
regularized_loss = regularized(station_groups)
assert regularized.smooth_vp.item() > 0.0
assert regularized.smooth_vs.item() > 0.0
assert regularized.damp_vp.item() > 0.0
assert regularized.damp_vs.item() > 0.0
assert torch.allclose(
    regularized_loss,
    regularized.data_loss
    + 0.5 * regularized.smooth_vp
    + 0.25 * regularized.smooth_vs
    + 0.125 * regularized.damp_vp
    + 0.0625 * regularized.damp_vs,
)
regularized_loss.backward()
assert torch.isfinite(objective_model.vp.grad).all() and objective_model.vp.grad.abs().sum() > 0
assert torch.isfinite(objective_model.vs.grad).all() and objective_model.vs.grad.abs().sum() > 0

# The node-based spherical smoothing term itself has a second-order Taylor
# remainder, including all three spherical metric factors.
smooth_field = (torch.arange(objective_model.vp.numel(), dtype=torch.float64).reshape_as(objective_model.vp) / 1000.0).requires_grad_()
smooth_value = smoothness(smooth_field, objective_model.lon, objective_model.lat, objective_model.depth)
smooth_value.backward()
smooth_direction = torch.linspace(-0.01, 0.01, smooth_field.numel(), dtype=torch.float64).reshape_as(smooth_field)
smooth_derivative = (smooth_field.grad * smooth_direction).sum().item()
smooth_remainders = []
for epsilon in epsilons:
    candidate = (smooth_field.detach() + epsilon * smooth_direction).requires_grad_(False)
    smooth_remainders.append(abs(smoothness(candidate, objective_model.lon, objective_model.lat, objective_model.depth).item() - smooth_value.item() - epsilon * smooth_derivative))
assert all(math.isfinite(remainder) and remainder > 0 for remainder in smooth_remainders)
smooth_slopes = [
    math.log(right / left) / math.log(next_epsilon / epsilon)
    for left, right, epsilon, next_epsilon in zip(smooth_remainders, smooth_remainders[1:], epsilons, epsilons[1:])
]
assert 1.7 < sorted(smooth_slopes)[len(smooth_slopes) // 2] < 2.3

epsilons_tensor = torch.tensor(epsilons, dtype=torch.float64)
reference = full_remainders[0] * (epsilons_tensor / epsilons_tensor[0]).square()
figure = plt.figure(figsize=(5, 4))
plt.loglog(epsilons, full_remainders, "o-", label="Full-chain remainder")
plt.loglog(epsilons, reference, "--", label=r"$O(\epsilon^2)$")
plt.xlabel(r"$\epsilon$")
plt.ylabel("Taylor remainder")
plt.title("Global Vp Taylor test")
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.tight_layout()
figure.savefig(FIGURES / "taylor_remainders.png", dpi=200)
plt.show()

# Full 1-D Vp chain through ForwardGrid2D: sampling, 2-D eikonal solve,
# event interpolation, loss.
radial_depth = torch.arange(-5.0, 20.1, 1.0, dtype=torch.float64)
radial_vp = torch.full_like(radial_depth, 5.0)
radial_station = torch.tensor([-122.80, 38.80, 0.0], dtype=torch.float64)
radial_events = torch.tensor([[-122.79, 38.81, 6.0]], dtype=torch.float64)
radial_direction = torch.linspace(-0.01, 0.01, radial_vp.numel(), dtype=torch.float64)


def radial_phase_time_loss(vp):
    candidate = VelocityModel1D(radial_depth, vp, vp / 1.73, trainable=False)
    grid = ForwardGrid2D(radial_station, radial_events, candidate, spacing=0.5)
    predicted_phase_dt = predict_travel_times_2d(candidate, grid, "P", radial_events)
    return (predicted_phase_dt - 3.0).square().sum()


radial_model = VelocityModel1D(radial_depth, radial_vp, radial_vp / 1.73, trainable=True)
radial_grid = ForwardGrid2D(radial_station, radial_events, radial_model, spacing=0.5)
radial_loss = (predict_travel_times_2d(radial_model, radial_grid, "P", radial_events) - 3.0).square().sum()
radial_loss.backward()
assert radial_model.vp.grad is not None and torch.isfinite(radial_model.vp.grad).all()
radial_derivative = (radial_model.vp.grad * radial_direction).sum().item()
radial_base_vp = radial_model.vp.detach()
radial_remainders = []
for epsilon in epsilons:
    radial_remainders.append(
        abs(radial_phase_time_loss(radial_base_vp + epsilon * radial_direction).item() - radial_loss.item() - epsilon * radial_derivative)
    )
assert all(right < left for left, right in zip(radial_remainders, radial_remainders[1:]))
radial_slopes = [
    math.log(right / left) / math.log(next_epsilon / epsilon)
    for left, right, epsilon, next_epsilon in zip(radial_remainders, radial_remainders[1:], epsilons, epsilons[1:])
]
assert 1.7 < sorted(radial_slopes)[len(radial_slopes) // 2] < 2.3

# Event relocation: the autograd gradient of the objective with respect to an
# event's longitude must match a finite-difference check.
joint_model = VelocityModel1D(radial_depth, radial_vp.clone(), (radial_vp / 1.73).clone(), trainable=False)
joint_initial_loc = radial_events[0] + torch.tensor([0.01, -0.01, -1.0], dtype=torch.float64)
joint_grid = ForwardGrid2D(radial_station, joint_initial_loc[None], joint_model, spacing=0.5)
joint_groups = [(joint_grid, [("P", torch.tensor([0]), torch.tensor([3.0], dtype=torch.float64))])]
joint_tomography = Tomography2D(joint_model, joint_initial_loc[None])
joint_loss = joint_tomography(joint_groups)
joint_loss.backward()
assert joint_tomography.event_loc.grad is not None and torch.isfinite(joint_tomography.event_loc.grad).all()
assert joint_tomography.event_time_correction.grad is not None
assert torch.isfinite(joint_tomography.event_time_correction.grad).all()

joint_finite_difference_step = 1e-5  # degrees


def joint_data_loss(delta_lon):
    with torch.no_grad():
        joint_tomography.event_loc[0, 0] = joint_initial_loc[0] + delta_lon
    loss = joint_tomography(joint_groups).item()
    with torch.no_grad():
        joint_tomography.event_loc[0, 0] = joint_initial_loc[0]
    return loss


joint_plus = joint_data_loss(joint_finite_difference_step)
joint_minus = joint_data_loss(-joint_finite_difference_step)
joint_finite_difference = (joint_plus - joint_minus) / (2.0 * joint_finite_difference_step)
joint_adjoint = joint_tomography.event_loc.grad[0, 0].item()
assert abs(joint_adjoint - joint_finite_difference) / (abs(joint_finite_difference) + 1e-12) < 1e-3

print("test_gradient.py: passed")
