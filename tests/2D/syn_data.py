import torch
import numpy as np
import pandas as pd
import json
import torch.nn.functional as F
import torch.optim as optim
import eik2d_cpp

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from pyproj import Proj
from model import Eikonal2D

# Debug 
plotfield = True
plotgrid = True

#
np.random.seed(0)
num_station = 20
num_event = 1000

# Domain specifications
config = {
    "minlatitude": 30.0,
    "maxlatitude": 35.0,
    "minlongitude": 120.0,
    "maxlongitude": 125.0,
    "mindepth": 0.0,
    "maxdepth": 10.0,
    "degree2km": 111.19,
    "m": 40,
    "n": 40,
    "l": 2,
    "h": 1.0,
    "tol":1e-3,
}

# For 2D, don't need 'l' or 'depth'
m = config["m"]
n = config["n"]
h = config["h"]
tol = config["tol"]
maxdepth = config["maxdepth"]

# Construct field + optional plot
ftest = torch.ones(n,m, dtype=torch.double)
# ftest[12:20,20:24] = 2
# ftest[30:35,30:35] = 2
# ftest[10:25,30:35] = 2
ftest[5:15,5:15] = 2
ftest[25:35,25:35] = 2
ftest[5:15,25:35] = 2
ftest[25:35,5:15] = 2

if plotfield:
    plt.figure()
    plt.imshow(ftest,vmin=1.0, vmax=2.0)
    plt.colorbar() ; plt.savefig("Test_domain")
ftest = torch.flatten(ftest)

# Prepare grid
time0 = datetime.fromisoformat("2019-01-01T00:00:00")
depth0 = config["maxdepth"]
latitude0 = config["minlatitude"]
longitude0 = config["minlongitude"]
config["latitude0"] = latitude0
config["longitude0"] = longitude0
config["depth0"] = depth0
proj = Proj(f"+proj=sterea +lon_0={config['longitude0']} +lat_0={config['latitude0']} +units=km +x_0={0}")
with open(f"./config.json", "w") as f:
    json.dump(config, f)
dlon = np.abs(config['minlongitude'] - config['maxlongitude'])
dlat = np.abs(config['minlatitude'] - config['maxlatitude'])
x_edge, y_edge = proj(longitude=config["maxlongitude"], latitude=config["maxlatitude"])
dx = x_edge / m # dx in km
dy = y_edge / n # dy in km

# Stations
stations = []
for i in range(num_station):
    station_id = f"NC.{i:02d}"
    latitude = latitude0 + (np.random.rand() * (dlat-0.5))
    longitude = longitude0 + (np.random.rand() * (dlon-0.5))
    elevation_m = np.random.rand() * 1000 
    depth_km = -elevation_m / 1000
    stations.append([station_id, latitude, longitude, elevation_m, depth_km])
stations = pd.DataFrame(stations, columns=["station_id", "latitude", "longitude", "elevation_m", "depth_km"])
stations[["x_km", "y_km"]] = stations.apply(
    lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1
)
stations["z_km"] = stations["depth_km"]
stations.to_csv(f"./stations.csv", index=False)
stations_json = stations.copy()
stations_json.set_index("station_id", inplace=True)
stations_json = stations_json.to_dict(orient="index")
with open(f"./stations.json", "w") as f:
    json.dump(stations_json, f, indent=4)

# Calculate position of stations of grid
stations["grid_x"] = m*stations['x_km']/x_edge
stations["grid_y"] = n*stations['y_km']/y_edge
stations["grid_z"] = np.abs(stations['depth_km'])

# Events
event_index = 0
events = []
picks = []
for i in range(num_event):
    event_index += 1
    time = time0
    latitude = latitude0 + (np.random.rand() * (dlat-0.5)) 
    longitude = longitude0 + (np.random.rand() * (dlon-0.5)) 
    depth = np.random.rand() * (depth0 - 1) 
    events.append([time.strftime("%Y-%m-%dT%H:%M:%S.%f"), latitude, longitude, depth, event_index])
events = pd.DataFrame(events, columns=["time", "latitude", "longitude", "depth_km", "event_index"])
events[["x_km", "y_km"]] = events.apply(lambda x: pd.Series(proj(longitude=x.longitude, latitude=x.latitude)), axis=1)
events["z_km"] = events["depth_km"]
events.to_csv(f"./events.csv", index=False)

# Calculate position of events on grid
events["grid_x"] = m*events['x_km']/(x_edge)
events["grid_y"] = n*events['y_km']/(y_edge)
events["grid_z"] = events['z_km']


if plotgrid:
    plt.figure()
    plt.scatter(stations["longitude"], stations["latitude"], s=10, marker="^", label="stations")
    plt.scatter(events["longitude"], events["latitude"], s=10, marker=".", label="events")
    plt.axis("scaled")
    plt.legend()
    plt.savefig(f"./events_xy.png", dpi=300)

    plt.figure()
    plt.scatter(stations["grid_x"], stations["grid_y"], s=10, marker="^", label="stations")
    plt.scatter(events["grid_x"], events["grid_y"], s=10, marker=".", label="events")
    plt.axis("scaled")
    plt.legend()
    plt.savefig(f"./events_xy_grid.png", dpi=300)


### Forward computation
eikonal_2d = Eikonal2D(m,n,h,tol,num_event,num_station,dx,dy,f=ftest)

for i, station in stations.iterrows():
    print(i,' i= ',station["grid_x"],station["grid_y"],station["z_km"])

    grid_x = events["grid_x"]
    grid_y = events["grid_y"]

    txyz = eikonal_2d(int(station["grid_x"]),int(station["grid_y"]),grid_x,grid_y)

    event_index = 0
    for j, event in events.iterrows():
        event_index += 1
        x_km,y_km,z_km = event["grid_x"],event["grid_y"],event["grid_z"]
        print(' jj= ', j ,event["grid_x"],event["grid_y"],event["z_km"])
        arrival_time = time + timedelta(seconds=float(txyz[j]))
        pick = [station["station_id"], arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), event_index]
        picks.append(pick)

picks = pd.DataFrame(picks, columns=["station_id", "phase_time", "event_index"])
picks.to_csv(f"./picks.csv", index=False)

