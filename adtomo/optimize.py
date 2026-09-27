"""Parameter selection and optimization."""

import os

import torch
import torch.distributed as dist


def init_distributed():
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
        return 0, 1
    dist.init_process_group("gloo")
    return dist.get_rank(), dist.get_world_size()


def set_trainable(tomography, names):
    names = set(names)
    tomography.model.vp.requires_grad_("vp" in names)
    tomography.model.vs.requires_grad_("vs" in names)
    tomography.event_loc_hori.requires_grad_("event_loc" in names)
    tomography.event_loc_vert.requires_grad_("event_loc" in names)
    tomography.event_time_correction.requires_grad_("event_time" in names)
    return [parameter for parameter in tomography.parameters() if parameter.requires_grad]


def optimize(tomography, groups, parameters, total_observations, iterations=30, *, learning_rates, log=print):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    if len(learning_rates) != len(parameters):
        raise ValueError("learning_rates must have one value per trainable parameter")
    optimizer = torch.optim.Adam(
        [{"params": [parameter], "lr": rate} for parameter, rate in zip(parameters, learning_rates)]
    )

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
        closure()
        optimizer.step()
        history.append(data_loss())
        if rank == 0 and log is not None:
            log(f"iteration {iteration + 1}/{iterations} data={history[-1]:.6e}")
    return history
