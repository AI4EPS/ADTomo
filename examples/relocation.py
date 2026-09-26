"""Relocate events with a fixed horizontally averaged 1-D velocity model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist

from adtomo import Tomography2D, VelocityModel, VelocityModel1D, build_station_groups, init_distributed, optimize, set_trainable


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing-km", type=float, default=1.0)
    parser.add_argument("--grid-padding-km", type=float, default=20.0)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()

    rank, world_size = init_distributed()
    try:
        # The relocation uses exactly the common inversion input format.
        initial = torch.load(args.data_dir / "model_initial.pt", weights_only=True)
        stations = pd.read_csv(args.data_dir / "stations.csv", dtype={"station_id": str})
        events_initial = pd.read_csv(args.data_dir / "events_initial.csv", dtype={"event_id": str})
        picks = pd.read_csv(args.data_dir / "picks.csv", dtype={"event_id": str, "station_id": str})

        # Truth is optional and affects only the comparison figure.
        events_true = pd.read_csv(args.data_dir / "events.csv", dtype={"event_id": str}) if (args.data_dir / "events.csv").is_file() else None
        model = VelocityModel1D.from_3d(VelocityModel(**initial), trainable=False)
        event_loc = events_initial[["longitude", "latitude", "depth_km"]].to_numpy()
        tomography = Tomography2D(model, event_loc)
        set_trainable(tomography, ["event_loc", "event_time"])
        groups = build_station_groups(
            stations,
            events_initial,
            picks,
            model,
            "1d",
            args.spacing_km,
            padding=args.grid_padding_km,
            padding_above=args.grid_padding_km,
            rank=rank,
            world_size=world_size,
        )
        history = optimize(
            tomography,
            groups,
            [tomography.event_loc, tomography.event_time_correction],
            len(picks),
            "lbfgs",
            args.iterations,
            learning_rate=0.1,
        )

        if rank == 0:
            args.figures_dir.mkdir(parents=True, exist_ok=True)
            relocated = events_initial.copy()
            relocated[["longitude", "latitude", "depth_km"]] = tomography.event_loc.detach().cpu().numpy()
            correction = tomography.event_time_correction.detach().cpu().numpy()
            relocated["event_time"] = [
                (pd.Timestamp(time) + pd.Timedelta(seconds=float(shift))).isoformat(timespec="milliseconds")
                for time, shift in zip(events_initial.event_time, correction)
            ]

            figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
            if events_true is not None:
                axes[0].scatter(events_true.longitude, events_true.latitude, marker="*", s=45, color="k", label="true")
            axes[0].scatter(events_initial.longitude, events_initial.latitude, marker="x", color="tab:gray", label="initial")
            axes[0].scatter(relocated.longitude, relocated.latitude, facecolors="none", edgecolors="tab:red", label="relocated")
            axes[0].set(title="Epicenters", xlabel="longitude (deg)", ylabel="latitude (deg)")
            axes[0].legend()
            if events_true is not None:
                axes[1].plot(events_initial.depth_km - events_true.depth_km, "x", color="tab:gray", label="initial")
                axes[1].plot(relocated.depth_km - events_true.depth_km, "o", color="tab:red", label="relocated")
            else:
                axes[1].plot(relocated.depth_km - events_initial.depth_km, "o", color="tab:red", label="relocation")
            axes[1].axhline(0.0, color="k", lw=0.8)
            axes[1].set(title="Depth error", xlabel="event index", ylabel="depth error (km)")
            axes[1].legend()
            axes[2].plot(correction, ".", color="tab:green")
            axes[2].axhline(0.0, color="k", lw=0.8)
            axes[2].set(title="Origin-time correction", xlabel="event index", ylabel="correction (s)")
            figure.savefig(args.figures_dir / "relocation.png", dpi=180)
            plt.close(figure)
            print(f"relocation data loss {history[0]:.3e} -> {history[-1]:.3e}")
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
