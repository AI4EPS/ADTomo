"""Directional Taylor checks for the retained 2-D and 3-D Eikonal adjoints."""

import math

import torch

from adtomo.tomography2d import _Eikonal2D
from adtomo.tomography3d import _Eikonal3D


def solve_2d(velocity_yx, source_xy, spacing):
    """Solve the 2-D kernel while keeping public Python axes as (y, x)."""
    slowness_xy = (1.0 / velocity_yx).T.contiguous()
    return _Eikonal2D.apply(slowness_xy, float(spacing), *source_xy).T


def solve_3d(velocity_zyx, source_xyz, spacing):
    """Solve the 3-D kernel while keeping public Python axes as (z, y, x)."""
    slowness_zyx = (1.0 / velocity_zyx).contiguous()
    return _Eikonal3D.apply(slowness_zyx, float(spacing), *source_xyz)


def taylor_remainders(solve, velocity, direction, source, target, spacing):
    value = solve(velocity, source, spacing)[target]
    value.backward()
    derivative = (velocity.grad * direction).sum().item()
    remainders = []
    epsilons = (1e-2, 5e-3, 2.5e-3, 1.25e-3)
    for epsilon in epsilons:
        perturbed = solve(velocity.detach() + epsilon * direction, source, spacing)[target]
        remainders.append(abs(perturbed.item() - value.item() - epsilon * derivative))
    slopes = [
        math.log(right / left) / math.log(next_epsilon / epsilon)
        for left, right, epsilon, next_epsilon in zip(remainders, remainders[1:], epsilons, epsilons[1:])
    ]
    assert all(math.isfinite(remainder) and remainder > 0.0 for remainder in remainders)
    assert all(right < left for left, right in zip(remainders, remainders[1:]))
    assert 1.7 < sorted(slopes)[len(slopes) // 2] < 2.3


velocity_2d = torch.full((25, 31), 5.0, dtype=torch.float64, requires_grad=True)
direction_2d = torch.linspace(-0.02, 0.02, velocity_2d.numel(), dtype=torch.float64).reshape_as(velocity_2d)
taylor_remainders(solve_2d, velocity_2d, direction_2d, (12.2, 9.3), (20, 25), 1.0)

velocity_3d = torch.full((10, 12, 14), 5.0, dtype=torch.float64, requires_grad=True)
direction_3d = torch.linspace(-0.02, 0.02, velocity_3d.numel(), dtype=torch.float64).reshape_as(velocity_3d)
taylor_remainders(solve_3d, velocity_3d, direction_3d, (6.2, 5.3, 0.0), (8, 10, 12), 1.0)

print("test_gradient.py: passed")
