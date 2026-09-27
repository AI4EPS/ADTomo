"""Jointly invert 3-D Vp/Vs, event locations, and origin times."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist

from adtomo import Tomography, VelocityModel, build_station_groups


DATA = Path(os.environ["INPUT_DIR"]).expanduser().resolve()
OUTPUT = Path(os.environ["OUTPUT_DIR"]).expanduser().resolve()
TRUE_DATA = Path("data")
FIGURES = OUTPUT / "figures"


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


def optimize(tomography, groups, parameters, total_observations, iterations, learning_rates):
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
        if rank == 0:
            print(f"iteration {iteration + 1}/{iterations} data={history[-1]:.6e}")
    return history


rank, world_size = init_distributed()
try:
    initial = torch.load(DATA / "model.pt", weights_only=True)
    true = torch.load(TRUE_DATA / "model_true.pt", weights_only=True)
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events_initial = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    events_true = pd.read_csv(TRUE_DATA / "events_true.csv", dtype={"event_id": str})
    picks = pd.read_csv(DATA / "picks.csv", dtype={"event_id": str, "station_id": str})

    model = VelocityModel(**initial)
    tomography = Tomography(
        model,
        events_initial[["longitude", "latitude", "depth_km"]].to_numpy(),
        alpha_vp=float(os.environ["ALPHA_VP"]),
        alpha_vs=float(os.environ["ALPHA_VS"]),
        beta_vp=float(os.environ["BETA_VP"]),
        beta_vs=float(os.environ["BETA_VS"]),
    )
    trainable = [name.strip() for name in os.environ["TRAINABLE"].split(",") if name.strip()]
    parameters = set_trainable(tomography, trainable)
    learning_rates = []
    for parameter in parameters:
        if parameter is tomography.model.vp:
            learning_rates.append(float(os.environ["LR_VP"]))
        elif parameter is tomography.model.vs:
            learning_rates.append(float(os.environ["LR_VS"]))
        elif parameter is tomography.event_loc_hori:
            learning_rates.append(float(os.environ["LR_LOC_HORI"]))
        elif parameter is tomography.event_loc_vert:
            learning_rates.append(float(os.environ["LR_LOC_VERT"]))
        elif parameter is tomography.event_time_correction:
            learning_rates.append(float(os.environ["LR_ORIGIN_TIME"]))
    groups = build_station_groups(
        stations,
        events_initial,
        picks,
        model,
        "3d",
        float(os.environ["SPACING_KM"]),
        padding=float(os.environ["GRID_PADDING_KM"]),
        padding_above=float(os.environ["GRID_PADDING_KM"]),
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        print(f"trainable: {', '.join(trainable)}")
    history = optimize(
        tomography,
        groups,
        parameters,
        len(picks),
        int(os.environ["ITERATIONS"]),
        learning_rates=learning_rates,
    )

    if rank == 0:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        FIGURES.mkdir(parents=True, exist_ok=True)
        extent = [model.lon[0], model.lon[-1], model.lat[0], model.lat[-1]]
        latitude_ticks = torch.linspace(model.lat[0], model.lat[-1], 4).tolist()
        target_depths = (0.0, 4.0, 8.0, 12.0)
        depth_indices = [int(torch.argmin((model.depth - depth).abs())) for depth in target_depths]
        limit = max(
            *[
                ((getattr(model, phase).detach()[depth_index] - initial[phase][depth_index]) / initial[phase][depth_index]).abs().max().item()
                for depth_index in depth_indices
                for phase in ("vp", "vs")
            ],
            1e-6,
        )
        figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        for row, phase in enumerate(("vp", "vs")):
            for column, depth_index in enumerate(depth_indices):
                background = initial[phase][depth_index]
                recovered = (getattr(model, phase).detach()[depth_index] - background) / background
                image = axes[row, column].imshow(
                    recovered,
                    origin="lower",
                    extent=extent,
                    cmap="seismic",
                    vmin=-limit,
                    vmax=limit,
                )
                axes[row, column].set(
                    title=f"{phase.upper()} recovered at {model.depth[depth_index]:.1f} km",
                    xlabel="longitude (deg)",
                    ylabel="latitude (deg)",
                    ylim=(model.lat[0], model.lat[-1]),
                    yticks=latitude_ticks,
                )
                figure.colorbar(image, ax=axes[row, column], label="relative velocity")
        figure.savefig(FIGURES / "tomography.png", dpi=180)
        plt.close(figure)

        relocated = events_initial.copy()
        relocated[["longitude", "latitude", "depth_km"]] = tomography.event_loc.detach().cpu().numpy()
        correction = tomography.event_time_correction.detach().cpu().numpy()
        initial_time_error = (
            pd.to_datetime(events_initial["event_time"])
            - pd.to_datetime(events_true["event_time"])
        ).dt.total_seconds().to_numpy()
        recovered_time_error = initial_time_error + correction
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        axes[0, 0].semilogy(range(len(history)), history, "o-", color="tab:blue")
        axes[0, 0].set(title="Data loss", xlabel="iteration", ylabel="mean squared residual")
        axes[0, 0].grid(alpha=0.3)
        axes[0, 1].scatter(events_true.longitude, events_true.latitude, marker="*", s=45, color="k", label="true")
        axes[0, 1].scatter(events_initial.longitude, events_initial.latitude, marker="x", color="tab:gray", label="initial")
        axes[0, 1].scatter(relocated.longitude, relocated.latitude, facecolors="none", edgecolors="tab:red", label="inverted")
        axes[0, 1].set(
            title="Epicenters",
            xlabel="longitude (deg)",
            ylabel="latitude (deg)",
            ylim=(model.lat[0], model.lat[-1]),
            yticks=latitude_ticks,
        )
        axes[0, 1].legend()
        axes[1, 0].plot(events_initial.depth_km - events_true.depth_km, "x", color="tab:gray", label="initial")
        axes[1, 0].plot(relocated.depth_km - events_true.depth_km, "o", color="tab:red", label="inverted")
        axes[1, 0].axhline(0.0, color="k", lw=0.8)
        axes[1, 0].set(title="Depth error", xlabel="event index", ylabel="depth error (km)")
        axes[1, 0].set_ylim(-1.0, 1.0)
        axes[1, 0].set_yticks((-1.0, -0.5, 0.0, 0.5, 1.0))
        axes[1, 0].legend()
        axes[1, 1].plot(initial_time_error, "x", color="tab:gray", label="initial")
        axes[1, 1].plot(recovered_time_error, ".", color="tab:green", label="recovered")
        axes[1, 1].axhline(0.0, color="k", lw=0.8)
        axes[1, 1].set(title="Origin-time error vs true", xlabel="event index", ylabel="error (s)")
        axes[1, 1].set_ylim(-0.5, 0.5)
        axes[1, 1].set_yticks((-0.5, -0.25, 0.0, 0.25, 0.5))
        axes[1, 1].legend()
        figure.savefig(FIGURES / "relocation.png", dpi=180)
        plt.close(figure)
        event_shift = tomography.event_loc.detach() - torch.tensor(
            events_initial[["longitude", "latitude", "depth_km"]].to_numpy(), dtype=torch.float64
        )
        velocity_change = max(
            ((model.vp.detach() - initial["vp"]) / initial["vp"]).abs().max().item(),
            ((model.vs.detach() - initial["vs"]) / initial["vs"]).abs().max().item(),
        )
        print(f"max event shift lon/lat/depth = {event_shift.abs().amax(dim=0).tolist()}")
        print(f"max recovered origin-time error = {abs(recovered_time_error).max():.3e} s")
        print(f"max recovered relative velocity change = {velocity_change:.3e}")
        print(f"joint 3-D data loss {history[0]:.3e} -> {history[-1]:.3e}")
        recovered = {
            "lon": model.lon.detach(),
            "lat": model.lat.detach(),
            "depth": model.depth.detach(),
            "vp": model.vp.detach(),
            "vs": model.vs.detach(),
        }
        torch.save(recovered, OUTPUT / "model.pt")
        relocated["event_time"] = (
            pd.to_datetime(events_initial["event_time"])
            + pd.to_timedelta(correction, unit="s")
        ).dt.strftime("%Y-%m-%dT%H:%M:%S.%f").str[:-3]
        relocated.to_csv(OUTPUT / "events.csv", index=False)
        stations.to_csv(OUTPUT / "stations.csv", index=False)
        picks.to_csv(OUTPUT / "picks.csv", index=False)
    if world_size > 1:
        dist.barrier()
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
