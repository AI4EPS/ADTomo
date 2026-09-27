"""Generate true events and the noisy initial catalog."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


# Edit the experiment here.
NUM_EVENTS = 1000
SEED = 2
LON_MIN, LON_MAX = -120.65, -120.10
LAT_MIN, LAT_MAX = 33.85, 34.40
DEPTH_MIN_KM, DEPTH_MAX_KM = 0.0, 13.0
HORIZONTAL_NOISE_DEG = 0.025
DEPTH_NOISE_KM = 0.5
TIME_NOISE_S = 0.1

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

model = torch.load(DATA / "model.pt", weights_only=True)
stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
rng = np.random.default_rng(SEED)
origin = pd.Timestamp("2026-09-13T12:00:00.000")
event_times = [origin + pd.Timedelta(seconds=30 * i) for i in range(NUM_EVENTS)]
events = pd.DataFrame({
    "event_id": [f"EV{i:03d}" for i in range(NUM_EVENTS)],
    "event_time": [time.isoformat(timespec="milliseconds") for time in event_times],
    "longitude": rng.uniform(LON_MIN, LON_MAX, NUM_EVENTS),
    "latitude": rng.uniform(LAT_MIN, LAT_MAX, NUM_EVENTS),
    "depth_km": rng.uniform(DEPTH_MIN_KM, DEPTH_MAX_KM, NUM_EVENTS),
})

initial = events.copy()
initial["longitude"] = events.longitude + rng.normal(0.0, HORIZONTAL_NOISE_DEG, NUM_EVENTS)
initial["latitude"] = events.latitude + rng.normal(0.0, HORIZONTAL_NOISE_DEG, NUM_EVENTS)
initial["depth_km"] = np.clip(
    events.depth_km + rng.normal(0.0, DEPTH_NOISE_KM, NUM_EVENTS),
    float(model["depth"][0]) + 1e-3,
    float(model["depth"][-1]) - 1e-3,
)
initial["event_time"] = [
    (time + pd.Timedelta(seconds=shift)).isoformat(timespec="milliseconds")
    for time, shift in zip(event_times, rng.normal(0.0, TIME_NOISE_S, NUM_EVENTS))
]

DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
events.to_csv(DATA / "events_true.csv", index=False)
initial.to_csv(DATA / "events.csv", index=False)

figure, axis = plt.subplots(figsize=(7, 6), constrained_layout=True)
axis.plot(
    [model["lon"][0], model["lon"][-1], model["lon"][-1], model["lon"][0], model["lon"][0]],
    [model["lat"][0], model["lat"][0], model["lat"][-1], model["lat"][-1], model["lat"][0]],
    "k-",
    label="model boundary",
)
axis.plot(
    [LON_MIN, LON_MAX, LON_MAX, LON_MIN, LON_MIN],
    [LAT_MIN, LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN],
    "--",
    color="gray",
    label="acquisition region",
)
scatter = axis.scatter(events.longitude, events.latitude, c=events.depth_km, cmap="viridis", s=22, label="true events")
if HORIZONTAL_NOISE_DEG > 0 or DEPTH_NOISE_KM > 0:
    axis.scatter(initial.longitude, initial.latitude, marker="x", color="tab:orange", s=18, label="initial events")
axis.scatter(stations.longitude, stations.latitude, marker="^", color="tab:red", edgecolor="black", s=55, label="stations")
axis.set(xlabel="longitude (deg)", ylabel="latitude (deg)", title="Synthetic acquisition geometry")
axis.legend(loc="best")
figure.colorbar(scatter, ax=axis, label="event depth (km)")
figure.savefig(FIGURES / "geometry.png", dpi=180)
plt.close(figure)

print(f"saved {len(events)} true events to {DATA / 'events_true.csv'} and input events to {DATA / 'events.csv'}")
print(f"longitude=[{events.longitude.min():.4f}, {events.longitude.max():.4f}], latitude=[{events.latitude.min():.4f}, {events.latitude.max():.4f}], depth=[{events.depth_km.min():.2f}, {events.depth_km.max():.2f}] km")
print(f"initial-catalog noise: horizontal={HORIZONTAL_NOISE_DEG} deg, depth={DEPTH_NOISE_KM} km, origin time={TIME_NOISE_S} s")
