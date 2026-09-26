"""Visual Taylor-convergence checks for the retained Eikonal adjoints."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adtomo.tomography2d import _Eikonal2D
from adtomo.tomography3d import _Eikonal3D


FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)


def solve_2d(velocity_yx, source_xy, spacing):
    """Solve the 2-D kernel while keeping Python fields in (y, x) order."""
    slowness_xy = (1.0 / velocity_yx).T.contiguous()
    return _Eikonal2D.apply(slowness_xy, float(spacing), *source_xy).T


def solve_3d(velocity_zyx, source_xyz, spacing):
    """Solve the 3-D kernel while keeping Python fields in (z, y, x) order."""
    slowness_zyx = (1.0 / velocity_zyx).contiguous()
    return _Eikonal3D.apply(slowness_zyx, float(spacing), *source_xyz)


def taylor_errors(solve, velocity, direction, source, target, spacing):
    value = solve(velocity, source, spacing)[target]
    value.backward()
    derivative = (velocity.grad * direction).sum().item()
    epsilons = torch.logspace(-1, -7, 7, dtype=torch.float64)
    first_order = []
    second_order = []
    for epsilon in epsilons:
        perturbed_value = solve(velocity.detach() + epsilon * direction, source, spacing)[target].item()
        change = perturbed_value - value.item()
        first_order.append(abs(change))
        second_order.append(abs(change - epsilon.item() * derivative))
    first_order = torch.tensor(first_order, dtype=torch.float64)
    second_order = torch.tensor(second_order, dtype=torch.float64)
    assert torch.isfinite(first_order).all() and torch.isfinite(second_order).all()
    assert (first_order > 0).all() and (second_order > 0).all()
    return epsilons, first_order, second_order


def plot_taylor_panel(axis, epsilons, first_order, second_order, title):
    first_reference = first_order[0] * epsilons / epsilons[0]
    second_reference = second_order[0] * (epsilons / epsilons[0]).square()
    axis.loglog(epsilons, first_order, "o-", label="E1")
    axis.loglog(epsilons, first_reference, "--", label=r"$O(\epsilon)$")
    axis.loglog(epsilons, second_order, "s-", label="E2")
    axis.loglog(epsilons, second_reference, ":", label=r"$O(\epsilon^2)$")
    axis.set(xlabel="epsilon", ylabel="error", title=title)
    axis.invert_xaxis()
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()


velocity_2d = torch.full((25, 31), 5.0, dtype=torch.float64, requires_grad=True)
direction_2d = torch.linspace(-0.02, 0.02, velocity_2d.numel(), dtype=torch.float64).reshape_as(velocity_2d)
epsilons_2d, first_2d, second_2d = taylor_errors(
    solve_2d, velocity_2d, direction_2d, (12.2, 9.3), (20, 25), 1.0
)

velocity_3d = torch.full((10, 12, 14), 5.0, dtype=torch.float64, requires_grad=True)
direction_3d = torch.linspace(-0.02, 0.02, velocity_3d.numel(), dtype=torch.float64).reshape_as(velocity_3d)
epsilons_3d, first_3d, second_3d = taylor_errors(
    solve_3d, velocity_3d, direction_3d, (6.2, 5.3, 0.0), (8, 10, 12), 1.0
)

figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
plot_taylor_panel(axes[0], epsilons_2d, first_2d, second_2d, "2-D Eikonal gradient")
plot_taylor_panel(axes[1], epsilons_3d, first_3d, second_3d, "3-D Eikonal gradient")
figure.savefig(FIGURES / "gradient_test.png", dpi=200)
plt.show()
