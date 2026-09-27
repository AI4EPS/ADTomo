"""Taylor checks for the 2-D and 3-D Eikonal adjoints."""

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "1"

import matplotlib.pyplot as plt
import torch

from adtomo.tomography2d import _Eikonal2D
from adtomo.tomography3d import _Eikonal3D


FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)
epsilons = torch.tensor((1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3), dtype=torch.float64)


velocity_2d = torch.full((25, 31), 5.0, dtype=torch.float64, requires_grad=True)
direction_2d = torch.linspace(-0.02, 0.02, velocity_2d.numel(), dtype=torch.float64).reshape_as(velocity_2d)
value_2d = _Eikonal2D.apply((1.0 / velocity_2d).T.contiguous(), 1.0, 12.2, 9.3).T.sum()
gradient_2d = torch.autograd.grad(value_2d, velocity_2d)[0]
first_2d, second_2d = [], []
for epsilon in epsilons:
    perturbed = velocity_2d.detach() + epsilon * direction_2d
    value = _Eikonal2D.apply((1.0 / perturbed).T.contiguous(), 1.0, 12.2, 9.3).T.sum()
    change = value - value_2d.detach()
    first_2d.append(abs(change.item()))
    second_2d.append(abs((change - epsilon * (gradient_2d * direction_2d).sum()).item()))
first_2d = torch.tensor(first_2d, dtype=torch.float64)
second_2d = torch.tensor(second_2d, dtype=torch.float64)


velocity_3d = torch.full((10, 12, 14), 5.0, dtype=torch.float64, requires_grad=True)
direction_3d = torch.linspace(-0.02, 0.02, velocity_3d.numel(), dtype=torch.float64).reshape_as(velocity_3d)
value_3d = _Eikonal3D.apply(1.0 / velocity_3d, 1.0, 6.2, 5.3, 0.0).sum()
gradient_3d = torch.autograd.grad(value_3d, velocity_3d)[0]
first_3d, second_3d = [], []
for epsilon in epsilons:
    perturbed = velocity_3d.detach() + epsilon * direction_3d
    value = _Eikonal3D.apply(1.0 / perturbed, 1.0, 6.2, 5.3, 0.0).sum()
    change = value - value_3d.detach()
    first_3d.append(abs(change.item()))
    second_3d.append(abs((change - epsilon * (gradient_3d * direction_3d).sum()).item()))
first_3d = torch.tensor(first_3d, dtype=torch.float64)
second_3d = torch.tensor(second_3d, dtype=torch.float64)


boundary_velocity = torch.full((4, 5, 6), 5.0, dtype=torch.float64, requires_grad=True)
boundary_field = _Eikonal3D.apply(1.0 / boundary_velocity, 1.0, 5.0, 4.0, 3.0)
assert torch.isfinite(boundary_field).all()
boundary_gradient = torch.autograd.grad(boundary_field.square().sum(), boundary_velocity)[0]
assert torch.isfinite(boundary_gradient).all()

for first, second in ((first_2d, second_2d), (first_3d, second_3d)):
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert (first > 0).all() and (second > 0).all()
    assert all(left > right for left, right in zip(first.tolist(), first.tolist()[1:]))
    assert all(left > right for left, right in zip(second.tolist(), second.tolist()[1:]))


figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
for axis, first, second, title in (
    (axes[0], first_2d, second_2d, "2-D Eikonal gradient"),
    (axes[1], first_3d, second_3d, "3-D Eikonal gradient"),
):
    axis.loglog(epsilons, first, "o-", label="first-order change")
    axis.loglog(epsilons, first[0] * epsilons / epsilons[0], "--", label=r"$O(\epsilon)$")
    axis.loglog(epsilons, second, "s-", label="Taylor remainder")
    axis.loglog(epsilons, second[0] * (epsilons / epsilons[0]).square(), ":", label=r"$O(\epsilon^2)$")
    axis.set_xlim(epsilons[0] * 1.2, epsilons[-1] / 1.2)
    axis.set(xlabel="epsilon", ylabel="absolute error", title=title)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
figure.savefig(FIGURES / "gradient_test.png", dpi=200)
plt.show()

print("test_gradient.py: passed")
