import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import json

import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pandas as pd
from pyproj import Proj
from model import Eikonal_3D

np.random.seed(0)
num_station = 50
num_event = 100

config = {
    "minlatitude": 30.0,
    "maxlatitude": 30.5,
    "minlongitude": 130.0,
    "maxlongitude": 130.5,
    "mindepth": 0.0,
    "maxdepth": 20.0,
    "degree2km": 111.19,
    "h": 1.0,
    "tol":1e-3,
}


anomaly = False # User defined
# True puts 2 anomalies in the domain
# False gives a checkerboard

### Set parameters for domain
h = config["h"]
maxdepth = config["maxdepth"]
tol = np.double(config["tol"])
time = datetime.fromisoformat("2019-01-01T00:00:00")

##################################################################################################################################


# Prepare grid
time0 = datetime.fromisoformat("2019-01-01T00:00:00")
depth0 = config["maxdepth"]
latitude0 = config["minlatitude"]
longitude0 = config["minlongitude"]
config["latitude0"] = latitude0
config["longitude0"] = longitude0
config["depth0"] = depth0
# with open(f"./config.json", "w") as f:
#     json.dump(config, f)


##################################################################################################################################
# ---> Domain dimensions
proj = Proj(f"+proj=sterea +lon_0={ (config['minlongitude'] + config['maxlongitude']) / 2 } +lat_0={ (config['minlatitude'] + config['maxlatitude'] ) / 2 } +units=km")
x_min, y_min = proj(longitude=config["minlongitude"], latitude=config["minlatitude"]) # compute edges of bounding box in km
x_max, y_max = proj(longitude=config["maxlongitude"], latitude=config["maxlatitude"]) # compute edges of bounding box in km
z_edge = config["maxdepth"]

dx, dy = proj(longitude=config["minlongitude"],latitude=config["minlatitude"])
dx = np.ceil(np.abs(dx)) + 2
dy = np.ceil(np.abs(dy)) + 2
dz = 1

m = int(np.ceil(x_max + dx)) + 3
n = int(np.ceil(y_max + dy)) + 3
l = int(np.ceil(config['maxdepth']+dz)) + 1

print('dx,dy,dz')
print(dx,dy,dz)
print('m,n,l')
print(m,n,l) # 54,61,22 -- matches with julia values

##################################################################################################################################
# ---> Dump in file to read in run.py 
config['m'] = m
config['n'] = n
config['l'] = l
config['dx'] = dx
config['dy'] = dy
config['dz'] = dz
with open(f"./config.json", "w") as f:
    json.dump(config, f)

##################################################################################################################################
# ---> Prepare Domain 
ftest = torch.ones(m,n,l,dtype=torch.double) * 6.0


if anomaly:
    ftest[20-1:40-1, 20-1:40-1, 8-1:14-1] = 6.5
    ftest[5-1:25-1, 4-1:19-1, 2-1:7-1] = 5.5
else:
    lenn = 10
    for i in range(m):
        for j in range(n):
            for k in range(l):
                # Calculate even/odd index for checkerboard pattern
                even_sum = ( (i - i%lenn) // lenn) + ((j - j%lenn) // lenn) + ((k + k%lenn) // lenn)
                if even_sum % 2 == 0:  # Check for even sum (even = white square)
                    ftest[i, j, k] = 6.5
                    # vel_s[i, j, k] = 3.8
                else:
                    ftest[i, j, k] = 5.5
                    # vel_s[i, j, k] = 3.2

plt.figure()
plt.imshow(ftest[:,:,0])
plt.colorbar() ; plt.savefig("Test_domain_1")

plt.figure()
plt.imshow(ftest[:,:,5],vmin=5,vmax=7)
plt.colorbar() ; plt.savefig("Test_domain_5")

plt.figure()
plt.imshow(ftest[:,:,11],vmin=5,vmax=7)
plt.colorbar() ; plt.savefig("Test_domain_11")

##################################################################################################################################
# ---> Read stations + place on grid before computation
# Hardcoded since both checkerboard and anomaly test case use same stations
stations = pd.read_csv("./data_benchmark/stations.csv")
stations['grid_x'] = stations['x_km'] + dx 
stations['grid_y'] = stations['y_km'] + dy 
stations['grid_z'] = stations['z_km'] + dz 

events = pd.read_csv("./data_benchmark/events.csv")
events['grid_x'] = events['x_km'] + dx 
events['grid_y'] = events['y_km'] + dy 
events['grid_z'] = events['z_km'] + dz 

##################################################################################################################################
# ---> Compute only P picks
ftest = torch.flatten(ftest)
eikonal_3d = Eikonal_3D(m,n,l,h,tol,dx,dy,dz,num_event,num_station,f=ftest)

picks = []
phase = 'P'
for i, station in stations.iterrows():
    print(i,' i= ',station["grid_x"],station["grid_y"],station["grid_z"])

    grid_x = events["grid_x"]
    grid_y = events["grid_y"]
    grid_z = events["grid_z"]

    txyz = eikonal_3d(station["grid_x"],station["grid_y"],station["grid_z"],grid_x,grid_y,grid_z,'P')

    event_index = 0; phase = 'P'
    for j, event in events.iterrows():
        event_index += 1
        x_km,y_km,z_km = event["grid_x"],event["grid_y"],event["grid_z"]
        arrival_time = time + timedelta(seconds=float(txyz[j]))

        pick = [station["station_id"], arrival_time.strftime("%Y-%m-%dT%H:%M:%S.%f"), event_index, phase]
        picks.append(pick)


picks = pd.DataFrame(picks, columns=["station_id", "phase_time", "event_index", "phase_type"])
picks.to_csv(f"./torch_syn_picks.csv", index=False)


print("* Done")
