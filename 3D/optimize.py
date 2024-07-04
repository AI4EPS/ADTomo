import torch
import torch.optim as optim
import numpy as np
from model import Eikonal_3D

def optimize(args, config, data_loader, eikonal_3d):
    if (args.opt.lower() == "lbfgs") or (args.opt.lower() == "bfgs"):
        optimizer = optim.LBFGS(params=eikonal_3d.parameters(), max_iter=500, line_search_fn="strong_wolfe",tolerance_grad=1e-8)
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    
    def closure():

        optimizer.zero_grad()

        for meta in data_loader:

            loss = 0.0
            batch_picks = meta["picks"]
            stations = meta["stations"]
            events = meta["events"]

            # order goes like ... 
            # (1,P), (1,S), (2,P), (2,S), ...
            # Skip S for now

            for idx_sta, picks_station_phase in batch_picks.groupby(["station_index","phase_type"]):


                station_loc = stations.loc[idx_sta[0]]
                phase = idx_sta[1]

                if phase == 'P':

                    print(f"Station {station_loc['station_id']} at ({station_loc['latitude']}, {station_loc['longitude']})")
                    x1,y1,z1 = station_loc['grid_x'],station_loc['grid_y'],station_loc['grid_z']

                    # Only events that have a pick
                    grid_x = events.iloc[picks_station_phase["event_index"]-1]["grid_x"]
                    grid_y = events.iloc[picks_station_phase["event_index"]-1]["grid_y"]
                    grid_z = events.iloc[picks_station_phase["event_index"]-1]["grid_z"]

                    picks_time = torch.tensor(picks_station_phase["phase_time"].array)

                    txyz = eikonal_3d(x1,y1,z1,grid_x,grid_y,grid_z,phase)

                    loss += torch.sum((picks_time - txyz)**2)
                
                else:
                    continue
                    # Not doing S for now


        loss.backward(retain_graph=True)    
        return loss

    optimizer.step(closure)
