"""Jointly invert 3-D Vp/Vs and event locations/origin times."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist

from adtomo import Tomography, VelocityModel, build_station_groups, init_distributed, optimize, set_trainable


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing-km", type=float, default=2.0)
    parser.add_argument("--grid-padding-km", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--alpha-vp", type=float, default=0.0)
    parser.add_argument("--alpha-vs", type=float, default=0.0)
    parser.add_argument("--beta-vp", type=float, default=0.0)
    parser.add_argument("--beta-vs", type=float, default=0.0)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    rank, world_size = init_distributed()
    try:
        # Inversion inputs: model_initial.pt, stations.csv, events_initial.csv, picks.csv.
        initial = torch.load(args.data_dir / "model_initial.pt", weights_only=True)
        stations = pd.read_csv(args.data_dir / "stations.csv", dtype={"station_id": str})
        events_initial = pd.read_csv(args.data_dir / "events_initial.csv", dtype={"event_id": str})
        picks = pd.read_csv(args.data_dir / "picks.csv", dtype={"event_id": str, "station_id": str})

        # Synthetic truth is optional and is used only for diagnostic figures.
        true = torch.load(args.data_dir / "model_true.pt", weights_only=True) if (args.data_dir / "model_true.pt").is_file() else None
        events_true = pd.read_csv(args.data_dir / "events.csv", dtype={"event_id": str}) if (args.data_dir / "events.csv").is_file() else None
        model = VelocityModel(**initial)
        true_model = VelocityModel(**true, trainable=False) if true is not None else None
        event_loc = events_initial[["longitude", "latitude", "depth_km"]].to_numpy()
        tomography = Tomography(
            model,
            event_loc,
            beta_vp=args.beta_vp,
            beta_vs=args.beta_vs,
            alpha_vp=args.alpha_vp,
            alpha_vs=args.alpha_vs,
        )
        set_trainable(tomography, ["vp", "vs", "event_loc", "event_time"])
        groups = build_station_groups(
            stations,
            events_initial,
            picks,
            model,
            "3d",
            args.spacing_km,
            padding=args.grid_padding_km,
            padding_above=args.grid_padding_km,
            rank=rank,
            world_size=world_size,
        )
        history = optimize(
            tomography,
            groups,
            [model.vp, model.vs, tomography.event_loc, tomography.event_time_correction],
            len(picks),
            "lbfgs",
            args.iterations,
        )

        if rank == 0:
            args.figures_dir.mkdir(parents=True, exist_ok=True)
            depth_index = int(torch.argmin((model.depth - 15.0).abs()))
            extent = [model.lon[0], model.lon[-1], model.lat[0], model.lat[-1]]
            figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            for row, phase in enumerate(("vp", "vs")):
                recovered = getattr(model, phase).detach()[depth_index]
                if true_model is not None:
                    initial_field = initial[phase][depth_index]
                    field = (recovered - initial_field) / initial_field
                    title = "recovered perturbation"
                    limit = max(field.abs().max().item(), 1e-6)
                else:
                    field = recovered
                    title = "recovered velocity"
                    limit = None
                image = axes[row, 0].imshow(field, origin="lower", extent=extent, cmap="seismic" if true_model is not None else "viridis", vmin=-limit if limit is not None else None, vmax=limit if limit is not None else None)
                axes[row, 0].set(title=f"{phase.upper()} {title}", xlabel="longitude (deg)", ylabel="latitude (deg)")
                figure.colorbar(image, ax=axes[row, 0], label="relative velocity" if true_model is not None else "velocity (km/s)")
                if true_model is not None:
                    truth = (getattr(true_model, phase).detach()[depth_index] - initial[phase][depth_index]) / initial[phase][depth_index]
                    image = axes[row, 1].imshow(truth, origin="lower", extent=extent, cmap="seismic", vmin=-limit, vmax=limit)
                    axes[row, 1].set(title=f"{phase.upper()} true perturbation", xlabel="longitude (deg)", ylabel="latitude (deg)")
                    figure.colorbar(image, ax=axes[row, 1], label="relative velocity")
                else:
                    axes[row, 1].axis("off")
            figure.savefig(args.figures_dir / "inversion_3d.png", dpi=180)
            plt.close(figure)

            relocated = events_initial.copy()
            relocated[["longitude", "latitude", "depth_km"]] = tomography.event_loc.detach().cpu().numpy()
            correction = tomography.event_time_correction.detach().cpu().numpy()
            figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            axes[0, 0].semilogy(range(len(history)), history, "o-", color="tab:blue")
            axes[0, 0].set(title="Data loss", xlabel="iteration", ylabel="mean squared residual")
            axes[0, 0].grid(alpha=0.3)
            if events_true is not None:
                axes[0, 1].scatter(events_true.longitude, events_true.latitude, marker="*", s=45, color="k", label="true")
            axes[0, 1].scatter(events_initial.longitude, events_initial.latitude, marker="x", color="tab:gray", label="initial")
            axes[0, 1].scatter(relocated.longitude, relocated.latitude, facecolors="none", edgecolors="tab:red", label="inverted")
            axes[0, 1].set(title="Epicenters", xlabel="longitude (deg)", ylabel="latitude (deg)")
            axes[0, 1].legend()
            if events_true is not None:
                axes[1, 0].plot(events_initial.depth_km - events_true.depth_km, "x", color="tab:gray", label="initial")
                axes[1, 0].plot(relocated.depth_km - events_true.depth_km, "o", color="tab:red", label="inverted")
            else:
                axes[1, 0].plot(relocated.depth_km - events_initial.depth_km, "o", color="tab:red", label="change")
            axes[1, 0].axhline(0.0, color="k", lw=0.8)
            axes[1, 0].set(title="Depth error", xlabel="event index", ylabel="depth error (km)")
            axes[1, 0].legend()
            axes[1, 1].plot(correction, ".", color="tab:green")
            axes[1, 1].axhline(0.0, color="k", lw=0.8)
            axes[1, 1].set(title="Origin-time correction", xlabel="event index", ylabel="correction (s)")
            figure.savefig(args.figures_dir / "relocation_3d.png", dpi=180)
            plt.close(figure)
            print(f"3-D data loss {history[0]:.3e} -> {history[-1]:.3e}")
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
