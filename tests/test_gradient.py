"""Finite-difference tests for the 2-D and 3-D Eikonal gradients."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adtomo.tomography2d import _Eikonal2D
from adtomo.tomography3d import _Eikonal3D


EPSILONS = (1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3)
FIGURE = Path(__file__).parent / "figures" / "gradient_test.png"


def solve_2d(slowness):
    return _Eikonal2D.apply(slowness, 0.5, 0.0, 0.0).sum()


def solve_3d(slowness):
    return _Eikonal3D.apply(slowness, 0.5, 0.0, 0.0, 0.0).sum()


def gradient_errors(solver, shape):
    model = 1.0 + 0.05 * torch.randn(shape,dtype=torch.float64)
    model.requires_grad_(True)
    direction = torch.randn_like(model)
    direction /= direction.norm()

    objective = solver(model)
    gradient = torch.autograd.grad(objective, model)[0]
    derivative_ad = (gradient * direction).sum().item()

    forward_errors = []
    central_errors = []
    for epsilon in EPSILONS:
        plus = solver(model.detach() + epsilon * direction).item()
        minus = solver(model.detach() - epsilon * direction).item()
        derivative_forward = (plus - objective.item()) / epsilon
        derivative_central = (plus - minus) / (2.0 * epsilon)
        scale = abs(derivative_ad)
        forward_errors.append(abs(derivative_forward - derivative_ad) )
        central_errors.append(abs(derivative_central - derivative_ad) )

    return forward_errors, central_errors


def test_gradients():
    torch.manual_seed(0)
    cases = (("2-D", solve_2d, (13, 13)), ("3-D", solve_3d, (9, 8, 4)))
    results = [(name, *gradient_errors(solver, shape)) for name, solver, shape in cases]

    FIGURE.parent.mkdir(exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for axis, (name, forward_errors, central_errors) in zip(axes, results):
        first_order = [forward_errors[0] * epsilon / EPSILONS[0] for epsilon in EPSILONS]
        second_order = [central_errors[0] * (epsilon / EPSILONS[0]) ** 2 for epsilon in EPSILONS]
        axis.loglog(EPSILONS, forward_errors, "o-", label="forward difference")
        axis.loglog(EPSILONS, central_errors, "s-", label="central difference")
        axis.loglog(EPSILONS, first_order, "--", label=r"$O(\epsilon)$")
        axis.loglog(EPSILONS, second_order, ":", label=r"$O(\epsilon^2)$")
        axis.invert_xaxis()
        axis.set(title=name, xlabel=r"$\epsilon$", ylabel="relative error")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()

    figure.savefig(FIGURE, dpi=200)
    plt.show()
    plt.close(figure)


if __name__ == "__main__":
    test_gradients()
