"""Visual Taylor-convergence checks for the retained Eikonal adjoints."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from adtomo.tomography2d import _Eikonal2D
from adtomo.tomography3d import _Eikonal3D


FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)
epsilons = torch.logspace(-1, -7, 7, dtype=torch.float64)

# 2-D: Python stores (y, x), while the retained kernel receives (x, y).
velocity_2d = torch.full((25, 31), 5.0, dtype=torch.float64, requires_grad=True)
direction_2d = torch.linspace(-0.02, 0.02, velocity_2d.numel(), dtype=torch.float64).reshape_as(velocity_2d)
value_2d = _Eikonal2D.apply((1.0 / velocity_2d).T.contiguous(), 1.0, 12.2, 9.3).T[20, 25]
value_2d.backward()
derivative_2d = (velocity_2d.grad * direction_2d).sum().item()
first_2d = []
second_2d = []
for epsilon in epsilons:
    perturbed = velocity_2d.detach() + epsilon * direction_2d
    value = _Eikonal2D.apply((1.0 / perturbed).T.contiguous(), 1.0, 12.2, 9.3).T[20, 25].item()
    change = value - value_2d.item()
    first_2d.append(abs(change))
    second_2d.append(abs(change - epsilon.item() * derivative_2d))
first_2d = torch.tensor(first_2d, dtype=torch.float64)
second_2d = torch.tensor(second_2d, dtype=torch.float64)

# 3-D: Python and the kernel both use the local (z, y, x) storage here.
velocity_3d = torch.full((10, 12, 14), 5.0, dtype=torch.float64, requires_grad=True)
direction_3d = torch.linspace(-0.02, 0.02, velocity_3d.numel(), dtype=torch.float64).reshape_as(velocity_3d)
value_3d = _Eikonal3D.apply(1.0 / velocity_3d, 1.0, 6.2, 5.3, 0.0)[8, 10, 12]
value_3d.backward()
derivative_3d = (velocity_3d.grad * direction_3d).sum().item()
first_3d = []
second_3d = []
for epsilon in epsilons:
    perturbed = velocity_3d.detach() + epsilon * direction_3d
    value = _Eikonal3D.apply(1.0 / perturbed, 1.0, 6.2, 5.3, 0.0)[8, 10, 12].item()
    change = value - value_3d.item()
    first_3d.append(abs(change))
    second_3d.append(abs(change - epsilon.item() * derivative_3d))
first_3d = torch.tensor(first_3d, dtype=torch.float64)
second_3d = torch.tensor(second_3d, dtype=torch.float64)
assert torch.isfinite(first_2d).all() and torch.isfinite(second_2d).all()
assert torch.isfinite(first_3d).all() and torch.isfinite(second_3d).all()
assert (first_2d > 0).all() and (second_2d > 0).all()
assert (first_3d > 0).all() and (second_3d > 0).all()

figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for axis, eps, first, second, title in (
    (axes[0], epsilons, first_2d, second_2d, "2-D Eikonal gradient"),
    (axes[1], epsilons, first_3d, second_3d, "3-D Eikonal gradient"),
):
    axis.loglog(eps, first, "o-", label="E1")
    axis.loglog(eps, first[0] * eps / eps[0], "--", label=r"$O(\epsilon)$")
    axis.loglog(eps, second, "s-", label="E2")
    axis.loglog(eps, second[0] * (eps / eps[0]).square(), ":", label=r"$O(\epsilon^2)$")
    axis.set(xlabel="epsilon", ylabel="error", title=title)
    axis.invert_xaxis()
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
figure.savefig(FIGURES / "gradient_test.png", dpi=200)
plt.show()
