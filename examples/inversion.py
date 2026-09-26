"""Run the synthetic inversion workflow: 1-D model, relocation, then 3-D model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist

from adtomo import (
    Tomography,
    Tomography2D,
    VelocityModel,
    VelocityModel1D,
    build_station_groups,
    init_distributed,
    optimize,
    set_trainable,
)


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing-1d", type=float, default=1.0)
    parser.add_argument("--spacing-3d", type=float, default=2.0)
    parser.add_argument("--grid-padding", type=float, default=20.0)
    parser.add_argument("--iterations-1d", type=int, default=20)
    parser.add_argument("--iterations-relocation", type=int, default=20)
    parser.add_argument("--iterations-3d", type=int, default=30)
    for dimension in ("1d", "3d"):
        parser.add_argument(f"--alpha-vp-{dimension}", type=float, default=0.0)
        parser.add_argument(f"--alpha-vs-{dimension}", type=float, default=0.0)
        parser.add_argument(f"--beta-vp-{dimension}", type=float, default=0.0)
        parser.add_argument(f"--beta-vs-{dimension}", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    return parser.parse_args()


def load_data(data_dir):
    initial = torch.load(data_dir / "model_initial.pt", weights_only=True)
    true = torch.load(data_dir / "model_true.pt", weights_only=True)
    stations = pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(data_dir / "events.csv", dtype={"event_id": str})
    initial_path = data_dir / "events_initial.csv"
    events_initial = pd.read_csv(initial_path, dtype={"event_id": str}) if initial_path.is_file() else events.copy()
    picks = pd.read_csv(data_dir / "picks.csv", dtype={"event_id": str, "station_id": str})
    return initial, true, stations, events, events_initial, picks


def catalog_locations(events):
    return events[["longitude", "latitude", "depth_km"]].to_numpy()


def relocated_catalog(events_initial, tomography):
    catalog = events_initial.copy()
    event_loc = tomography.event_loc.detach().cpu().numpy()
    correction = tomography.event_time_correction.detach().cpu().numpy()
    catalog[["longitude", "latitude", "depth_km"]] = event_loc
    catalog["event_time"] = [
        (pd.Timestamp(time) + pd.Timedelta(seconds=float(shift))).isoformat(timespec="milliseconds")
        for time, shift in zip(events_initial.event_time, correction)
    ]
    catalog["dt0_s"] = correction
    return catalog


def plot_1d(true_model, initial_model, inverted_model, path):
    figure, axis = plt.subplots(figsize=(5, 6), constrained_layout=True)
    for phase, color in (("vp", "tab:red"), ("vs", "tab:blue")):
        axis.plot(getattr(true_model, phase).detach(), true_model.depth, "-", color=color, label=f"{phase.upper()} true mean")
        axis.plot(getattr(initial_model, phase).detach(), initial_model.depth, ":", color=color, label=f"{phase.upper()} initial")
        axis.plot(getattr(inverted_model, phase).detach(), inverted_model.depth, "--", color=color, label=f"{phase.upper()} inverted")
    axis.invert_yaxis()
    axis.set(title="1-D velocity inversion", xlabel="velocity (km/s)", ylabel="depth (km)")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_relocation(events, events_initial, relocated, path):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].scatter(events.longitude, events.latitude, marker="*", s=45, color="k", label="true")
    axes[0].scatter(events_initial.longitude, events_initial.latitude, marker="x", color="tab:gray", label="initial")
    axes[0].scatter(relocated.longitude, relocated.latitude, facecolors="none", edgecolors="tab:red", label="relocated")
    axes[0].set(title="Epicenters", xlabel="longitude (deg)", ylabel="latitude (deg)")
    axes[0].legend()
    axes[1].plot(events_initial.depth_km - events.depth_km, "x", color="tab:gray", label="initial")
    axes[1].plot(relocated.depth_km - events.depth_km, "o", color="tab:red", label="relocated")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set(title="Depth error", xlabel="event index", ylabel="error (km)")
    axes[1].legend()
    axes[2].plot(relocated.dt0_s, ".", color="tab:green")
    axes[2].axhline(0.0, color="k", lw=0.8)
    axes[2].set(title="Origin-time correction", xlabel="event index", ylabel="correction (s)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_3d(true_model, initial_model, inverted_model, path):
    depth_index = int(torch.argmin((inverted_model.depth - 15.0).abs()))
    extent = [inverted_model.lon[0], inverted_model.lon[-1], inverted_model.lat[0], inverted_model.lat[-1]]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row, phase in enumerate(("vp", "vs")):
        initial = getattr(initial_model, phase).detach()[depth_index]
        truth = (getattr(true_model, phase).detach()[depth_index] - initial) / initial
        recovered = (getattr(inverted_model, phase).detach()[depth_index] - initial) / initial
        limit = max(truth.abs().max().item(), recovered.abs().max().item(), 1e-6)
        for axis, title, field in zip(axes[row], ("true perturbation", "recovered perturbation"), (truth, recovered)):
            image = axis.imshow(field, origin="lower", extent=extent, cmap="seismic", vmin=-limit, vmax=limit)
            axis.set(title=f"{phase.upper()} {title}", xlabel="longitude (deg)", ylabel="latitude (deg)")
            figure.colorbar(image, ax=axis, label="relative velocity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    rank, world_size = init_distributed()
    try:
        initial, true, stations, events, events_initial, picks = load_data(args.data_dir)
        initial_model_3d = VelocityModel(**initial)
        true_model_3d = VelocityModel(**true, trainable=False)
        event_loc = catalog_locations(events_initial)
        args.figures_dir.mkdir(parents=True, exist_ok=True)

        # 1-D velocity inversion.
        model_1d = VelocityModel1D.from_3d(initial_model_3d)
        initial_model_1d = VelocityModel1D.from_3d(initial_model_3d, trainable=False)
        true_model_1d = VelocityModel1D.from_3d(true_model_3d, trainable=False)
        tomography_1d = Tomography2D(
            model_1d,
            event_loc,
            beta_vp=args.beta_vp_1d,
            beta_vs=args.beta_vs_1d,
            alpha_vp=args.alpha_vp_1d,
            alpha_vs=args.alpha_vs_1d,
        )
        set_trainable(tomography_1d, ["vp", "vs"])
        groups_1d = build_station_groups(
            stations, events_initial, picks, model_1d, "1d", args.spacing_1d,
            padding=args.grid_padding, padding_above=args.grid_padding,
            rank=rank, world_size=world_size,
        )
        history_1d = optimize(tomography_1d, groups_1d, [model_1d.vp, model_1d.vs], len(picks), "lbfgs", args.iterations_1d)
        if rank == 0:
            plot_1d(true_model_1d, initial_model_1d, model_1d, args.figures_dir / "inversion_1d.png")

        # Event relocation using the inverted 1-D model and the same 1-D spacing.
        tomography_relocation = Tomography2D(model_1d, event_loc)
        set_trainable(tomography_relocation, ["event_loc", "event_time"])
        groups_relocation = build_station_groups(
            stations, events_initial, picks, model_1d, "1d", args.spacing_1d,
            padding=args.grid_padding, padding_above=args.grid_padding,
            rank=rank, world_size=world_size,
        )
        history_relocation = optimize(
            tomography_relocation, groups_relocation,
            [tomography_relocation.event_loc, tomography_relocation.event_time_correction],
            len(picks), "lbfgs", args.iterations_relocation,
        )
        relocated = relocated_catalog(events_initial, tomography_relocation)
        if rank == 0:
            plot_relocation(events, events_initial, relocated, args.figures_dir / "relocation.png")

        # 3-D velocity inversion from the inverted 1-D background and relocated catalog.
        vp_3d = model_1d.vp.detach()[:, None, None].expand_as(initial["vp"]).clone()
        vs_3d = model_1d.vs.detach()[:, None, None].expand_as(initial["vs"]).clone()
        model_3d = VelocityModel(initial["lon"], initial["lat"], initial["depth"], vp_3d, vs_3d)
        tomography_3d = Tomography(
            model_3d,
            catalog_locations(relocated),
            beta_vp=args.beta_vp_3d,
            beta_vs=args.beta_vs_3d,
            alpha_vp=args.alpha_vp_3d,
            alpha_vs=args.alpha_vs_3d,
        )
        set_trainable(tomography_3d, ["vp", "vs"])
        groups_3d = build_station_groups(
            stations, relocated, picks, model_3d, "3d", args.spacing_3d,
            padding=args.grid_padding, padding_above=args.grid_padding,
            rank=rank, world_size=world_size,
        )
        history_3d = optimize(tomography_3d, groups_3d, [model_3d.vp, model_3d.vs], len(picks), "lbfgs", args.iterations_3d)
        if rank == 0:
            initial_3d = VelocityModel(initial["lon"], initial["lat"], initial["depth"], vp_3d, vs_3d, trainable=False)
            plot_3d(true_model_3d, initial_3d, model_3d, args.figures_dir / "inversion_3d.png")
            print(f"1-D data loss {history_1d[0]:.3e} -> {history_1d[-1]:.3e}")
            print(f"relocation data loss {history_relocation[0]:.3e} -> {history_relocation[-1]:.3e}")
            print(f"3-D data loss {history_3d[0]:.3e} -> {history_3d[-1]:.3e}")
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
