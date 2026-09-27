"""Trainable-parameter selection and the (optionally distributed) optimization loop."""

import os

import torch
import torch.distributed as dist


_TRAINABLE = ("vp", "vs", "event_loc", "event_time")


def init_distributed():
    """Return ``(rank, world_size)``, joining the process group when launched by ``torchrun``."""
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return 0, 1
    dist.init_process_group("gloo")
    return dist.get_rank(), dist.get_world_size()


def set_trainable(tomography, names):
    """Enable gradients for the named parameters and return them."""
    names = set(names)
    if not names or names - set(_TRAINABLE):
        raise ValueError(f"trainable names must be a non-empty subset of {_TRAINABLE}, got {sorted(names)}")
    tomography.model.vp.requires_grad_("vp" in names)
    tomography.model.vs.requires_grad_("vs" in names)
    if hasattr(tomography, "event_loc_hori"):
        tomography.event_loc_hori.requires_grad_("event_loc" in names)
        tomography.event_loc_vert.requires_grad_("event_loc" in names)
    else:
        tomography.event_loc.requires_grad_("event_loc" in names)
    tomography.event_time_correction.requires_grad_("event_time" in names)
    return [parameter for parameter in tomography.parameters() if parameter.requires_grad]


def optimize(tomography, groups, parameters, total_observations, optimizer="lbfgs", iterations=30, learning_rate=None, learning_rates=None, log=print):
    """Minimize ``tomography(groups)`` with L-BFGS or Adam; returns the data loss after every iteration.

    Under ``torchrun`` each rank passes its own station ``groups``; gradients
    and the loss are summed across ranks inside the closure. Forward-grid
    padding is fixed by the caller, so an event leaving the grid propagates
    the existing ``ValueError``.
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if learning_rate is None:
        learning_rate = 1.0 if optimizer == "lbfgs" else 0.01
    if optimizer == "lbfgs":
        if learning_rates is not None:
            raise ValueError("learning_rates are only used with Adam")
        optimizer = torch.optim.LBFGS(parameters, lr=learning_rate, max_iter=20, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)
    elif optimizer == "adam":
        if learning_rates is None:
            optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        else:
            if len(learning_rates) != len(parameters):
                raise ValueError("learning_rates must have one value per trainable parameter")
            optimizer = torch.optim.Adam(
                [{"params": [parameter], "lr": rate} for parameter, rate in zip(parameters, learning_rates)]
            )
    else:
        raise ValueError(f"optimizer must be 'lbfgs' or 'adam', got {optimizer!r}")

    def objective():
        return tomography(groups, data_scale=1.0 / total_observations, regularization_scale=1.0 / world_size)

    def closure():
        optimizer.zero_grad()
        loss = objective()
        loss.backward()
        loss = loss.detach().clone()
        if world_size > 1:
            for parameter in parameters:
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        return loss

    def data_loss():
        with torch.no_grad():
            objective()
        data_sum = tomography.data_sum.clone()
        if world_size > 1:
            dist.all_reduce(data_sum, op=dist.ReduceOp.SUM)
        return (data_sum / total_observations).item()

    history = [data_loss()]
    for iteration in range(iterations):
        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(closure)
        else:
            closure()
            optimizer.step()
        history.append(data_loss())
        if rank == 0 and log is not None:
            log(f"iteration {iteration + 1}/{iterations} data={history[-1]:.6e}")
    return history
