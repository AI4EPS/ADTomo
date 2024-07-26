import numpy as np
import torch
import torch.optim as optim

# from .model import Eikonal2D


def optimize(args, config, data_loader, eikonal_2d):
    if (args.opt.lower() == "lbfgs") or (args.opt.lower() == "bfgs"):
        optimizer = optim.LBFGS(params=eikonal_2d.parameters(), max_iter=1000, line_search_fn="strong_wolfe")
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")

    def closure():
        optimizer.zero_grad()
        for meta in data_loader:
            
            loss = 0.0
            batch_picks = meta["picks"]
            stations = meta["stations"]
            events = meta["events"]

            for idx_sta, picks_station in batch_picks.groupby("station_index"):
                station_loc = stations.loc[idx_sta]
                print(f"Station {station_loc['station_id']} at ({station_loc['latitude']}, {station_loc['longitude']})")
                x1,y1,z1 = station_loc['grid_x'],station_loc['grid_y'],station_loc['grid_z']

                # For that station, compute for all events
                grid_x = events["grid_x"]
                grid_y = events["grid_y"]
                picks_time = torch.tensor(picks_station["phase_time"].array)
                txyz = eikonal_2d(int(x1),int(y1),grid_x,grid_y) # Only at integers

                loss += torch.sum((picks_time - txyz)**2)

        loss.backward(retain_graph=True)    
        return loss

    optimizer.step(closure)
