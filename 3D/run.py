import argparse
import json
from datetime import datetime, timedelta
import h5py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from pyproj import Proj
from torch import nn
from torch.utils.data import DataLoader

from data import PhaseDataset
from optimize import optimize
from model import Eikonal_3D


##################################################################################################################################
# Set paths to event, station, pick files
### --- ### 
save_file = 'result.h5' # name for output file
pcks = "./torch_syn_picks.csv"        # computed with syn_data
# pcks = "./data_benchmark/picks.csv" # computed with julia

sts = "./data_benchmark/stations.csv"
evnts = "./data_benchmark/events.csv"
cnfig = "./data_benchmark/config.json"

##################################################################################################################################
parser = argparse.ArgumentParser(description="ADTomo", add_help=True)
parser.add_argument("--config", default=cnfig, type=str, help="config file")
parser.add_argument("--stations", type=str, default=sts, help="station json")
parser.add_argument("--picks", type=str, default=pcks, help="picks csv")
parser.add_argument("--events", type=str, default=evnts, help="events csv")
parser.add_argument("-j", "--workers", default=0, type=int, metavar="N", help="number of data loading workers (default: 4)")
parser.add_argument("--opt", default="lbfgs", type=str, help="optimizer")
args = parser.parse_args()

# Read config file and set dimensions up
with open(args.config, "r") as fp:
    config = json.load(fp)
# set domain dimensions and solver args since already computed from preprocessing
m = config["m"]
n = config["n"]
l = config["l"]
h = config["h"]
dx = config["dx"]
dy = config["dy"]
dz = config["dz"]
tol = np.double(config["tol"])

# for real data
# smooth_hori = config["smooth_hori"]
# smooth_vert = config["smooth_vert"]
# lambda_p = config["lambda_p"]

##################################################################################################################################
# domain specifcations
proj = Proj(f"+proj=sterea +lon_0={ (config['minlongitude'] + config['maxlongitude']) / 2 } +lat_0={ (config['minlatitude'] + config['maxlatitude'] ) / 2 } +units=km")
x_min, y_min = proj(longitude=config["minlongitude"], latitude=config["minlatitude"]) # compute edges of bounding box in km
x_max, y_max = proj(longitude=config["maxlongitude"], latitude=config["maxlatitude"]) # compute edges of bounding box in km
z_edge = config["maxdepth"]


##################################################################################################################################
##################################################################################################################################
# Generate GIL7 
# vel_hx = np.array([0, 1, 3, 4, 5, 17, 25])
# vel_px = np.array([3.20, 4.50, 4.80, 5.51, 6.21, 6.89, 7.83])
# vel_sx = np.array([1.50, 2.40, 2.78, 3.18, 3.40, 3.98, 4.52])
# fvel0_p = np.ones((m, n, l)) ; fvel0_s = np.ones((m, n, l))
# ct = 0; nvp = vel_px[ct]; nvs = vel_sx[ct]
# for i in range(l):
#     if ct < len(vel_hx) and (i - dz) * h >= vel_hx[ct+1]:
#         print(i)
#         ct += 1
#         nvp = vel_px[ct]
#         nvs = vel_sx[ct]
#     fvel0_p[:, :, i] = nvp
#     fvel0_s[:, :, i] = nvs

# vel_p = fvel0_p #* 1.027
# vel_s = fvel0_s #* 1.07 #1.07
# ratio_s = vel_s / vel_p
# ratio_s = torch.tensor(ratio_s)
# vel_p = torch.tensor(vel_p)
# print(ratio_s);exit()

##################################################################################################################################
##################################################################################################################################
#------> Process stations
## Stations
stations = pd.read_csv(args.stations,index_col='station_id') 
stations["station_id"] = stations.index
stations[["x_km1", "y_km1"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)

stations.reset_index(inplace=True, drop=True)
stations["st_index"] = stations.index.values  # reindex starts from 0 continuously
stations["station_index"] = stations.index.values
stations.set_index("st_index", inplace=True)


##################################################################################################################################
#------> Process events
events = pd.read_csv(args.events, parse_dates=["time"])
events[["x_km1", "y_km1"]] = events.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)

num_event = len(events)
num_station = len(stations)

stations["grid_x"] = stations['x_km'] + dx
stations["grid_y"] = stations['y_km'] + dy
stations["grid_z"] = stations['z_km'] + dz
events["grid_x"] = events['x_km'] + dx
events["grid_y"] = events['y_km'] + dy
events["grid_z"] = events['z_km'] + dz


station_loc = stations[["grid_x", "grid_y", "grid_z"]].values

events.reset_index(inplace=True, drop=True)
events["index"] = events.index.values  # reindex starts from 0 continuously

event_loc = events[["grid_x", "grid_y", "grid_z"]].values
event_time = events["time"].values

event_index_map = {x: i for i, x in enumerate(events["event_index"])}
station_index_map = {x: i for i, x in enumerate(stations["station_id"])}

##################################################################################################################################
#------> Process picks
picks = pd.read_csv(args.picks, parse_dates=["phase_time"])
picks = picks[picks["event_index"] != -1]
picks["index"] = picks["event_index"].apply(lambda x: event_index_map[x])  # map index starts from 0 continuously
picks["phase_time"] = picks.apply(lambda x: (x["phase_time"] - event_time[x["index"]]).total_seconds(), axis=1)
picks["station_index"] = picks["station_id"].apply(lambda x: station_index_map[x])
print("Picks processed")

##################################################################################################################################
#------> Load dataset
phase_dataset = PhaseDataset(picks,events,stations)
sampler = torch.utils.data.SequentialSampler(phase_dataset)
data_loader = DataLoader(phase_dataset, batch_size=None, sampler=sampler, num_workers=args.workers, collate_fn=None)
print(data_loader)
print("Dataset loaded")

##################################################################################################################################
#------> f
# always prescribe an initial model / create one here
ftest = torch.ones(m,n,l,dtype=torch.double) * 6.0
##################################################################################################################################
#------> Invert
eikonal_3d = Eikonal_3D(m,n,l,h,tol,dx,dy,dz,num_event,num_station,f=ftest)
optimize(args,config,data_loader,eikonal_3d)
print(" *** Ended")

##################################################################################################################################
#------> Save 
ff = torch.clone(eikonal_3d.f)
ff = torch.reshape(ff,(eikonal_3d.m,eikonal_3d.n,eikonal_3d.l))
ff = ff.detach().numpy()

# hf = h5py.File('a_test_region_realdata.h5', 'w')
hf = h5py.File(save_file, 'w')
hf.create_dataset('data', data=ff)
hf.close()
