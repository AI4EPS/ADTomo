"""Generate reproducible events, optionally a noisy initial catalog, and plot the geometry.

``events.csv`` is the true catalog used to forward-model picks; ``events_initial.csv``
is the catalog the inversion starts from (identical unless noise is requested).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
KM_PER_DEGREE = 111.19


def plot_geometry(model, stations, events, initial, args):
    figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
    axis.plot([model["lon"][0], model["lon"][-1], model["lon"][-1], model["lon"][0], model["lon"][0]], [model["lat"][0], model["lat"][0], model["lat"][-1], model["lat"][-1], model["lat"][0]], "k-", label="model boundary")
    axis.plot([args.lon_min, args.lon_max, args.lon_max, args.lon_min, args.lon_min], [args.lat_min, args.lat_min, args.lat_max, args.lat_max, args.lat_min], "--", color="gray", label="acquisition region")
    scatter = axis.scatter(events.longitude, events.latitude, c=events.depth_km, cmap="viridis", s=22, label="true events")
    if args.horizontal_noise_km > 0 or args.depth_noise_km > 0:
        axis.scatter(initial.longitude, initial.latitude, marker="x", color="tab:orange", s=18, label="initial (noisy) events")
    axis.scatter(stations.longitude, stations.latitude, marker="^", color="tab:red", edgecolor="black", s=55, label="surface stations")
    axis.set(xlabel="longitude (deg)", ylabel="latitude (deg)", title="Synthetic acquisition geometry")
    axis.legend(loc="best")
    figure.colorbar(scatter, ax=axis, label="event depth (km)")
    figure.savefig(FIGURES / "geometry.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-events", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--lon-min", type=float, default=-120.6)
    parser.add_argument("--lon-max", type=float, default=-118.0)
    parser.add_argument("--lat-min", type=float, default=33.8)
    parser.add_argument("--lat-max", type=float, default=36.1)
    parser.add_argument("--depth-min", type=float, default=2.0)
    parser.add_argument("--depth-max", type=float, default=30.0)
    parser.add_argument("--horizontal-noise-km", type=float, default=2.0, help="std of horizontal initial-location noise (km)")
    parser.add_argument("--depth-noise-km", type=float, default=2.0, help="std of initial depth noise (km)")
    parser.add_argument("--time-noise-s", type=float, default=0.5, help="std of initial origin-time noise (s)")
    args = parser.parse_args()

    model_path, stations_path = DATA / "model_initial.pt", DATA / "stations.csv"
    model = torch.load(model_path, weights_only=True)
    stations = pd.read_csv(stations_path, dtype={"station_id": str})

    rng = np.random.default_rng(args.seed)
    origin = pd.Timestamp("2026-09-13T12:00:00.000")
    event_times = [origin + pd.Timedelta(seconds=30 * i) for i in range(args.num_events)]
    events = pd.DataFrame({
        "event_id": [f"EV{i:03d}" for i in range(args.num_events)],
        "event_time": [time.isoformat(timespec="milliseconds") for time in event_times],
        "longitude": rng.uniform(args.lon_min, args.lon_max, args.num_events),
        "latitude": rng.uniform(args.lat_min, args.lat_max, args.num_events),
        "depth_km": rng.uniform(args.depth_min, args.depth_max, args.num_events),
    })

    initial = events.copy()
    noise = lambda scale: rng.normal(0.0, scale, args.num_events)
    initial["longitude"] = events.longitude + noise(args.horizontal_noise_km) / (KM_PER_DEGREE * np.cos(np.deg2rad(events.latitude)))
    initial["latitude"] = events.latitude + noise(args.horizontal_noise_km) / KM_PER_DEGREE
    initial["depth_km"] = np.clip(events.depth_km + noise(args.depth_noise_km), float(model["depth"][0]) + 1e-3, float(model["depth"][-1]) - 1e-3)
    initial["event_time"] = [
        (time + pd.Timedelta(seconds=shift)).isoformat(timespec="milliseconds") for time, shift in zip(event_times, noise(args.time_noise_s))
    ]

    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    events.to_csv(DATA / "events.csv", index=False)
    initial.to_csv(DATA / "events_initial.csv", index=False)
    plot_geometry(model, stations, events, initial, args)
    print(f"saved {len(events)} events to {DATA / 'events.csv'} and the initial catalog to {DATA / 'events_initial.csv'}")
    print(f"longitude=[{events.longitude.min():.4f}, {events.longitude.max():.4f}], latitude=[{events.latitude.min():.4f}, {events.latitude.max():.4f}], depth=[{events.depth_km.min():.2f}, {events.depth_km.max():.2f}] km")
    print(
        f"initial-catalog noise: horizontal={args.horizontal_noise_km} km, "
        f"depth={args.depth_noise_km} km, origin time={args.time_noise_s} s"
    )


if __name__ == "__main__":
    main()
