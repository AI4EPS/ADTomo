"""Forward model synthetic P and S picks from the true 3-D model and the true events."""

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch

from adtomo import ForwardGrid, VelocityModel, predict_travel_times


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spacing", type=float, default=2.0, help="forward-grid spacing in km")
    args = parser.parse_args()

    started = perf_counter()
    model = VelocityModel(**torch.load(DATA / "model_true.pt", weights_only=True), trainable=False)
    stations = pd.read_csv(DATA / "stations.csv", dtype={"station_id": str})
    events = pd.read_csv(DATA / "events.csv", dtype={"event_id": str})
    events_spherical = torch.tensor(events[["longitude", "latitude", "depth_km"]].to_numpy(), dtype=torch.float64)

    picks = []
    with torch.no_grad():
        for station in stations.itertuples(index=False):
            grid = ForwardGrid([station.longitude, station.latitude, station.depth_km], events_spherical, model, spacing=args.spacing)
            for phase in ("P", "S"):
                travel_times = predict_travel_times(model, grid, phase, events_spherical)
                for event, travel_time in zip(events.itertuples(index=False), travel_times.tolist()):
                    picks.append({
                        "event_id": event.event_id,
                        "station_id": station.station_id,
                        "phase_type": phase,
                        "phase_time": (pd.Timestamp(event.event_time) + pd.Timedelta(seconds=travel_time)).isoformat(timespec="milliseconds"),
                    })

    picks = pd.DataFrame(picks)
    picks.to_csv(DATA / "picks.csv", index=False)
    p_count = (picks.phase_type == "P").sum()
    print(f"saved {len(picks)} picks to {DATA / 'picks.csv'}")
    print(f"stations={len(stations)} events={len(events)} P={p_count} S={len(picks) - p_count} runtime={perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
