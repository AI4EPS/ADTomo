import argparse
import json
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from pyproj import Proj
from torch import nn
from torch.utils.data import DataLoader

from adtomo import PhaseDataset, Eikonal2D, optimize

parser = argparse.ArgumentParser(description="PyTorch Detection Training", add_help=True)
parser.add_argument("--config", default="tests/2D/config.json", type=str, help="config file")
parser.add_argument("--stations", type=str, default="tests/2D/stations.json", help="station json")
parser.add_argument("--picks", type=str, default="tests/2D/picks.csv", help="picks csv")
parser.add_argument("--events", type=str, default="tests/2D/events.csv", help="events csv")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 4)")
parser.add_argument("--opt", default="lbfgs", type=str, help="optimizer")
args = parser.parse_args()

with open(args.config, "r") as fp:
    config = json.load(fp)

# set domain dimensions and solver args
m = config["m"]
n = config["n"]
h = config["h"]
tol = np.double(config["tol"])

# domain specifcations :: dx, dy help locate in what grid cell an event is
proj = Proj(f"+proj=sterea +lon_0={config['longitude0']} +lat_0={config['latitude0']} +units=km")
x_edge, y_edge = proj(longitude=config["maxlongitude"], latitude=config["maxlatitude"]) # compute edges of bounding box in km
dx = x_edge / m # dx in km
dy = y_edge / n # dy in km
dlon = np.abs(config['minlongitude'] - config['maxlongitude'])
dlat = np.abs(config['minlatitude'] - config['maxlatitude'])

## Stations
with open(args.stations, "r") as fp:
    stations = json.load(fp)
# Process stations
stations = pd.DataFrame.from_dict(stations, orient="index")
stations["station_id"] = stations.index
stations[["x_km", "y_km"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
stations["z_km"] = stations["depth_km"]
stations.reset_index(inplace=True, drop=True)
stations["st_index"] = stations.index.values  # reindex starts from 0 continuously
stations["station_index"] = stations.index.values
stations.set_index("st_index", inplace=True)

picks = pd.read_csv(args.picks, parse_dates=["phase_time"])

# Process events
events = pd.read_csv(args.events, parse_dates=["time"])
events[["x_km", "y_km"]] = events.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
events["z_km"] = events["depth_km"]

num_event = len(events)
num_station = len(stations)

# Process grid locations
stations["grid_x"] = m*stations['x_km']/x_edge
stations["grid_y"] = n*stations['y_km']/y_edge
stations["grid_z"] = np.abs(stations['depth_km'])
events["grid_x"] = m*events['x_km']/(x_edge)
events["grid_y"] = n*events['y_km']/(y_edge)
events["grid_z"] = events['z_km']

station_loc = stations[["grid_x", "grid_y", "grid_z"]].values

events.reset_index(inplace=True, drop=True)
events["index"] = events.index.values  # reindex starts from 0 continuously
event_loc = events[["grid_x", "grid_y", "grid_z"]].values
event_time = events["time"].values

event_index_map = {x: i for i, x in enumerate(events["event_index"])}
station_index_map = {x: i for i, x in enumerate(stations["station_id"])}
picks = picks[picks["event_index"] != -1]
picks["index"] = picks["event_index"].apply(lambda x: event_index_map[x])  # map index starts from 0 continuously
picks["phase_time"] = picks.apply(lambda x: (x["phase_time"] - event_time[x["index"]]).total_seconds(), axis=1)
picks["station_index"] = picks["station_id"].apply(lambda x: station_index_map[x])

# Load data
phase_dataset = PhaseDataset(picks,events,stations)
sampler = torch.utils.data.SequentialSampler(phase_dataset)
data_loader = DataLoader(phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None)
# Invert
eikonal_2d = Eikonal2D(m,n,h,tol,num_event,num_station,dx,dy)
optimize(args,config,data_loader,eikonal_2d)

# Plots, ideally save field
ff = torch.clone(eikonal_2d.f)
ff = torch.reshape(ff,(eikonal_2d.m,eikonal_2d.n))
ff = ff.detach().numpy()
# plots
plt.imshow(ff,vmin=1.0,vmax=2.0)
plt.colorbar() ; plt.savefig("Test_inverted")
