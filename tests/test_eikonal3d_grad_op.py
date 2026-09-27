import torch

from adtomo.tomography3d import _Eikonal3D


torch.manual_seed(0)
shape = (4, 5, 6)
spacing = 1.0

boundary_slowness = torch.full(shape, 0.2, dtype=torch.float64, requires_grad=True)
boundary = _Eikonal3D.apply(boundary_slowness, spacing, 5.0, 4.0, 3.0)
assert torch.isfinite(boundary).all()
boundary.square().sum().backward()
assert boundary_slowness.grad is not None
assert torch.isfinite(boundary_slowness.grad).all()

base = torch.full(shape, 0.2, dtype=torch.float64, requires_grad=True)
direction = torch.linspace(-0.01, 0.01, base.numel(), dtype=torch.float64).reshape(shape)
source = (2.2, 2.1, 1.3)
value = _Eikonal3D.apply(base, spacing, *source).sum()
gradient = torch.autograd.grad(value, base)[0]
assert torch.isfinite(gradient).all()
assert gradient.abs().sum() > 0

remainders = []
for epsilon in (1e-2, 5e-3, 2.5e-3, 1.25e-3):
    perturbed = base.detach() + epsilon * direction
    trial = _Eikonal3D.apply(perturbed, spacing, *source).sum()
    remainder = trial - value.detach() - epsilon * (gradient * direction).sum()
    remainders.append(abs(remainder.item()))

assert all(torch.isfinite(torch.tensor(remainders)))
assert all(remainder > 0.0 for remainder in remainders)
assert all(left > right for left, right in zip(remainders, remainders[1:]))

print("test_eikonal3d_grad_op.py: passed")
