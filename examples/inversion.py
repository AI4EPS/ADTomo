"""Invert the synthetic dataset with a 1-D or 3-D velocity model.

    python inversion.py --model 3d --trainable vp,vs
    python inversion.py --model 1d --trainable event_loc,event_time
    torchrun --standalone --nproc_per_node=4 inversion.py --model 3d --trainable vp,vs,event_loc,event_time

Both modes read the same data/ files (model_initial.pt, stations.csv, events.csv,
events_initial.csv, picks.csv); the 1-D mode starts from the horizontal mean of
the 3-D initial model. Parameters are toggled with requires_grad, as in AI4EPS/ADTomo.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist

from adtomo import (
    TRAINABLE,
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=("1d", "3d"), default="3d")
    parser.add_argument("--trainable", default="vp,vs", help=f"comma-separated subset of {','.join(TRAINABLE)}")
    parser.add_argument("--optimizer", choices=("lbfgs", "adam"), default="lbfgs")
    parser.add_argument("--iterations", type=int, default=30, help="L-BFGS steps (20 inner iterations each) or Adam iterations")
    parser.add_argument("--learning-rate", type=float, default=None, help="default 1.0 for lbfgs, 0.01 for adam")
    parser.add_argument("--spacing", type=float, default=None, help="forward-grid spacing in km (default 4 for 3d, 2 for 1d)")
    parser.add_argument("--grid-padding", type=float, default=10.0, help="km around (and above) the initial events so relocated events stay inside the fixed grids")
    parser.add_argument("--beta-vp", type=float, default=0.0)
    parser.add_argument("--beta-vs", type=float, default=0.0)
    parser.add_argument("--alpha-vp", type=float, default=0.0)
    parser.add_argument("--alpha-vs", type=float, default=0.0)
    parser.add_argument("--huber-delta", type=float, default=None, help="seconds; omit for squared-error loss")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.trainable = args.trainable.split(",")
    args.spacing = args.spacing or (4.0 if args.model == "3d" else 2.0)
    return args


def load_data(data_dir):
    initial = torch.load(data_dir / "model_initial.pt", weights_only=True)
    true = torch.load(data_dir / "model_true.pt", weights_only=True)
    stations = pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(data_dir / "events.csv", dtype={"event_id": str})
    initial_path = data_dir / "events_initial.csv"
    events_initial = pd.read_csv(initial_path, dtype={"event_id": str}) if initial_path.is_file() else events
    picks = pd.read_csv(data_dir / "picks.csv", dtype={"event_id": str, "station_id": str})
    return initial, true, stations, events, events_initial, picks


def inverted_catalog(events_initial, tomography):
    catalog = events_initial.copy()
    event_loc = tomography.event_loc.detach().numpy()
    correction = tomography.event_time_correction.detach().numpy()
    catalog["longitude"], catalog["latitude"], catalog["depth_km"] = event_loc[:, 0], event_loc[:, 1], event_loc[:, 2]
    catalog["event_time"] = [
        (pd.Timestamp(time) + pd.Timedelta(seconds=float(shift))).isoformat(timespec="milliseconds")
        for time, shift in zip(events_initial.event_time, correction)
    ]
    catalog["dt0_s"] = correction
    return catalog


def plot_velocity(true_model, initial_model, model, path):
    if model.vp.dim() == 1:
        figure, axis = plt.subplots(figsize=(4.5, 6), constrained_layout=True)
        for phase, color in (("vp", "tab:red"), ("vs", "tab:blue")):
            axis.plot(getattr(true_model, phase).detach(), model.depth, "-", color=color, label=f"{phase.upper()} true (horizontal mean)")
            axis.plot(getattr(initial_model, phase).detach(), model.depth, ":", color=color, label=f"{phase.upper()} initial")
            axis.plot(getattr(model, phase).detach(), model.depth, "--", color=color, label=f"{phase.upper()} inverted")
        axis.invert_yaxis()
        axis.set(title="1-D velocity profiles", xlabel="velocity (km/s)", ylabel="depth (km)")
        axis.legend()
        axis.grid(alpha=0.3)
    else:
        depth_index = int(torch.argmin((model.depth - 15.0).abs()))
        extent = [model.lon[0], model.lon[-1], model.lat[0], model.lat[-1]]
        figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for row, phase in enumerate(("vp", "vs")):
            initial = getattr(initial_model, phase).detach()[depth_index]
            truth = (getattr(true_model, phase).detach()[depth_index] - initial) / initial
            recovered = (getattr(model, phase).detach()[depth_index] - initial) / initial
            limit = max(truth.abs().max().item(), recovered.abs().max().item(), 1e-6)
            for axis, title, field in zip(axes[row], ("true checkerboard", "recovered perturbation"), (truth, recovered)):
                image = axis.imshow(field, origin="lower", extent=extent, cmap="seismic", vmin=-limit, vmax=limit)
                axis.set(title=f"{phase.upper()} {title}", xlabel="longitude (deg)", ylabel="latitude (deg)")
                figure.colorbar(image, ax=axis, label="relative velocity")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_events(events, events_initial, inverted, history, path):
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    axes[0, 0].semilogy(history, "o-", color="tab:blue")
    axes[0, 0].set(title="Data loss", xlabel="iteration", ylabel="data loss")
    axes[0, 0].grid(alpha=0.3)
    axes[0, 1].scatter(events.longitude, events.latitude, marker="*", s=60, color="k", label="true")
    axes[0, 1].scatter(events_initial.longitude, events_initial.latitude, marker="x", color="tab:gray", label="initial")
    axes[0, 1].scatter(inverted.longitude, inverted.latitude, marker="o", facecolors="none", edgecolors="tab:red", label="inverted")
    axes[0, 1].set(title="Epicenters", xlabel="longitude (deg)", ylabel="latitude (deg)")
    axes[0, 1].legend()
    axes[1, 0].plot(events_initial.depth_km - events.depth_km, "x", color="tab:gray", label="initial")
    axes[1, 0].plot(inverted.depth_km - events.depth_km, "o", color="tab:red", label="inverted")
    axes[1, 0].axhline(0.0, color="k", lw=0.8)
    axes[1, 0].set(title="Depth error", xlabel="event index", ylabel="depth error (km)")
    axes[1, 0].legend()
    axes[1, 1].plot(inverted.dt0_s, ".", color="tab:green", label="origin-time correction")
    axes[1, 1].axhline(0.0, color="k", lw=0.8)
    axes[1, 1].set(title="Origin-time correction", xlabel="event index", ylabel="time correction (s)")
    axes[1, 1].legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    rank, world_size = init_distributed()
    try:
        initial, true, stations, events, events_initial, picks = load_data(args.data_dir)
        event_loc = events_initial[["longitude", "latitude", "depth_km"]].to_numpy()
        objective_options = {
            "beta_vp": args.beta_vp,
            "beta_vs": args.beta_vs,
            "alpha_vp": args.alpha_vp,
            "alpha_vs": args.alpha_vs,
            "huber_delta": args.huber_delta,
        }

        if args.model == "1d":
            model = VelocityModel1D.from_3d(VelocityModel(**initial))
            initial_model = VelocityModel1D.from_3d(VelocityModel(**initial), trainable=False)
            true_model = VelocityModel1D.from_3d(VelocityModel(**true), trainable=False)
            tomography = Tomography2D(model, event_loc, **objective_options)
        else:
            model = VelocityModel(**initial)
            initial_model = VelocityModel(**initial, trainable=False)
            true_model = VelocityModel(**true, trainable=False)
            tomography = Tomography(model, event_loc, **objective_options)

        parameters = set_trainable(tomography, args.trainable)
        groups = build_station_groups(
            stations, events_initial, picks, model, args.model, args.spacing,
            padding=args.grid_padding, padding_above=args.grid_padding, rank=rank, world_size=world_size,
        )
        if rank == 0:
            print(f"{args.model} inversion: {len(stations)} stations, {len(events)} events, {len(picks)} picks; world_size={world_size}; optimizer={args.optimizer}")
            print("Optimizing parameters:\n" + "\n".join(f"  {name}: {tuple(parameter.shape)}" for name, parameter in tomography.named_parameters() if parameter.requires_grad))

        history = optimize(tomography, groups, parameters, len(picks), args.optimizer, args.iterations, args.learning_rate)

        if rank == 0:
            args.figures_dir.mkdir(exist_ok=True)
            inverted = inverted_catalog(events_initial, tomography)
            plot_velocity(true_model, initial_model, model, args.figures_dir / "inversion.png")
            plot_events(events, events_initial, inverted, history, args.figures_dir / "events.png")
            print(f"saved figures to {args.figures_dir}; data loss {history[0]:.3e} -> {history[-1]:.3e}")
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
