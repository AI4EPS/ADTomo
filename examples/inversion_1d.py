"""Invert 1-D Vp/Vs using the common synthetic/catalog input files."""

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

        # Synthetic truth is read only for the optional comparison figure.
        true = torch.load(args.data_dir / "model_true.pt", weights_only=True) if (args.data_dir / "model_true.pt").is_file() else None
        model = VelocityModel1D.from_3d(VelocityModel(**initial))
        initial_model = VelocityModel1D.from_3d(VelocityModel(**initial), trainable=False)
        true_model = VelocityModel1D.from_3d(VelocityModel(**true), trainable=False) if true is not None else None
        event_loc = events_initial[["longitude", "latitude", "depth_km"]].to_numpy()

        tomography = Tomography2D(
            model,
            event_loc,
            beta_vp=args.beta_vp,
            beta_vs=args.beta_vs,
            alpha_vp=args.alpha_vp,
            alpha_vs=args.alpha_vs,
        )
        set_trainable(tomography, ["vp", "vs"])
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
        history = optimize(tomography, groups, [model.vp, model.vs], len(picks), "lbfgs", args.iterations)

        if rank == 0:
            args.figures_dir.mkdir(parents=True, exist_ok=True)
            figure, axis = plt.subplots(figsize=(5, 6), constrained_layout=True)
            for phase, color in (("vp", "tab:red"), ("vs", "tab:blue")):
                if true_model is not None:
                    axis.plot(getattr(true_model, phase).detach(), true_model.depth, "-", color=color, label=f"{phase.upper()} true mean")
                axis.plot(getattr(initial_model, phase).detach(), initial_model.depth, ":", color=color, label=f"{phase.upper()} initial")
                axis.plot(getattr(model, phase).detach(), model.depth, "--", color=color, label=f"{phase.upper()} inverted")
            axis.invert_yaxis()
            axis.set(title="1-D velocity inversion", xlabel="velocity (km/s)", ylabel="depth (km)")
            axis.grid(alpha=0.3)
            axis.legend()
            figure.savefig(args.figures_dir / "inversion_1d.png", dpi=180)
            plt.close(figure)
            print(f"1-D data loss {history[0]:.3e} -> {history[-1]:.3e}")
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
