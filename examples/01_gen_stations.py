"""Generate reproducible stations."""

from pathlib import Path

import numpy as np
import pandas as pd


# Edit the experiment here.
NUM_STATIONS = 64
SEED = 1
LON_MIN, LON_MAX = -120.65, -120.10
LAT_MIN, LAT_MAX = 33.85, 34.40
DEPTH_MIN_KM, DEPTH_MAX_KM = -2.0, 2.0

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

rng = np.random.default_rng(SEED)
stations = pd.DataFrame({
    "station_id": [f"STA{i:03d}" for i in range(NUM_STATIONS)],
    "longitude": rng.uniform(LON_MIN, LON_MAX, NUM_STATIONS),
    "latitude": rng.uniform(LAT_MIN, LAT_MAX, NUM_STATIONS),
    "depth_km": rng.uniform(DEPTH_MIN_KM, DEPTH_MAX_KM, NUM_STATIONS),
})
DATA.mkdir(exist_ok=True)
stations.to_csv(DATA / "stations.csv", index=False)

print(f"saved {len(stations)} stations to {DATA / 'stations.csv'}")
print(f"longitude=[{stations.longitude.min():.4f}, {stations.longitude.max():.4f}], latitude=[{stations.latitude.min():.4f}, {stations.latitude.max():.4f}]")
print(f"station depth=[{stations.depth_km.min():.3f}, {stations.depth_km.max():.3f}] km")
