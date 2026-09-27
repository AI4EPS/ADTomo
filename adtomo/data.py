"""Build station groups from catalog tables."""

import numpy as np
import pandas as pd
import torch

from .grid import ForwardGrid, ForwardGrid2D


def build_station_groups(stations, events, picks, model, dimension, spacing, padding=None, padding_above=0.0, rank=0, world_size=1):
    Grid = {"1d": ForwardGrid2D, "3d": ForwardGrid}[dimension]
    event_index = pd.Index(events.event_id)
    pick_event_indices = event_index.get_indexer(picks.event_id)
    if np.any(pick_event_indices < 0):
        unknown = pd.unique(picks.loc[pick_event_indices < 0, "event_id"])
        raise ValueError(f"picks reference unknown event_id values: {unknown.tolist()}")
    picks = picks.assign(_event_index=pick_event_indices)
    origin_time = pd.to_datetime(events.event_time, format="ISO8601").to_numpy()
    events_spherical = torch.tensor(events[["longitude", "latitude", "depth_km"]].to_numpy(), dtype=torch.float64)
    stations_by_id = stations.set_index("station_id")
    groups = []
    for station_id in list(pd.unique(picks.station_id))[rank::world_size]:
        station = stations_by_id.loc[station_id]
        station_picks = picks[picks.station_id == station_id]
        station_events = events_spherical[pd.unique(station_picks["_event_index"])]
        grid = Grid([station.longitude, station.latitude, station.depth_km], station_events, model, spacing, padding, padding_above)
        phase_groups = []
        for phase, phase_picks in station_picks.groupby("phase_type", sort=False):
            indices = phase_picks["_event_index"].to_numpy()
            phase_dt = (pd.to_datetime(phase_picks.phase_time, format="ISO8601").to_numpy() - origin_time[indices]) / np.timedelta64(1, "s")
            phase_groups.append((phase, torch.tensor(indices), torch.tensor(phase_dt, dtype=torch.float64)))
        groups.append((grid, phase_groups))
    return groups
