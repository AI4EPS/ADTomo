# %%
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from adtomo import Eikonal3D

if __name__ == "__main__":

    # %%
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp:
        init_process_group(backend="gloo")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
        print(f"DDP rank {ddp_rank}, local rank {ddp_local_rank}, world size {ddp_world_size}")
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        print("Non-DDP run")

    # %%
    ######################################### Load Synthetic Data #########################################
    data_path = "data"
    events = pd.read_csv(f"{data_path}/events.csv", dtype={"event_id": str})
    stations = pd.read_csv(f"{data_path}/stations.csv", dtype={"station_id": str})
    picks = pd.read_csv(f"{data_path}/picks.csv", dtype={"event_id": str, "station_id": str})
    picks = picks.merge(events[["event_id", "event_time"]], on="event_id")

    assert (
        len(stations["station_id"].unique()) >= ddp_world_size
    ), f"Number of stations ({len(stations['station_id'].unique())}) must be larger than world size ({ddp_world_size})"

    #### make the time values relative to event time in seconds
    picks["phase_time_origin"] = picks["phase_time"].copy()
    picks["phase_time"] = (
        pd.to_datetime(picks["phase_time"]) - pd.to_datetime(picks["event_time"])
    ).dt.total_seconds()  # relative to event time (arrival time)
    picks.drop(columns=["event_time"], inplace=True)
    events["event_time_origin"] = events["event_time"].copy()
    events["event_time"] = np.zeros(len(events))  # relative to event time
    ####

    #### make the station and event index continuous (0, 1, 2, ...) for internal use
    # events = events.sort_values("event_index").set_index("event_index")
    # stations = stations.sort_values("station_index").set_index("station_index")
    events["idx_eve"] = np.arange(len(events))  # continuous index from 0 to num_event/num_station
    stations["idx_sta"] = np.arange(len(stations))
    picks = picks.merge(events[["event_id", "idx_eve"]], on="event_id")  ## idx_eve, and idx_sta are used internally
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
    ####

    with open(f"{data_path}/config.json", "r") as f:
        eikonal_config = json.load(f)
    num_event = len(events)
    num_station = len(stations)
    nx, ny, nz, h = eikonal_config["nx"], eikonal_config["ny"], eikonal_config["nz"], eikonal_config["h"]
    xgrid = torch.arange(0, nx, dtype=torch.float64) * h
    ygrid = torch.arange(0, ny, dtype=torch.float64) * h
    zgrid = torch.arange(0, nz, dtype=torch.float64) * h
    eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid, "zgrid": zgrid})
    vp = torch.ones((nx, ny, nz), dtype=torch.float64) * 6.0
    vs = vp / 1.73

    ## initial event location
    event_loc = events[["x_km", "y_km", "z_km"]].values

    eikonal3d = Eikonal3D(
        num_event,
        num_station,
        stations[["x_km", "y_km", "z_km"]].values,
        stations[["dt_s"]].values,
        event_loc,
        events[["event_time"]].values,
        vp,
        vs,
        # max_dvp=1.0,
        # max_dvs=0.5,
        # lambda_vp=1.0,
        # lambda_vs=1.0,
        config=eikonal_config,
    )
    preds, loss = eikonal3d(picks)

    ######################################### Optimize #########################################
    # %%
    vp = eikonal3d.vp0.detach().numpy() + eikonal3d.dvp.detach().numpy()
    vs = eikonal3d.vs0.detach().numpy() + eikonal3d.dvs.detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial3d_vp_vs_xy.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial3d_vp_vs_xz.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial3d_vp_vs_yz.png")

    raw_picks = picks.copy()
    picks = picks[picks["idx_sta"] % ddp_world_size == ddp_local_rank]
    print(f"Rank {ddp_rank} has {len(picks)} picks")
    if ddp:
        eikonal3d = DDP(eikonal3d)
    raw_eikonal3d = eikonal3d.module if ddp else eikonal3d

    if ddp_local_rank == 0:
        print("Initial loss:", loss.item())

    t0 = time.time()
    optimizer = optim.LBFGS(params=eikonal3d.parameters(), max_iter=1000, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        _, loss = eikonal3d(picks)
        loss.backward()

        if ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= ddp_world_size

        norm = torch.nn.utils.clip_grad_norm_(eikonal3d.parameters(), 1.0)

        return loss

    optimizer.step(closure)

    # # optimizer = optim.Adam(params=eikonal3d.parameters(), lr=0.1)
    # for _ in range(100):
    #     optimizer.zero_grad()
    #     preds, loss = eikonal3d(picks)
    #     loss.backward()
    #     optimizer.step()
    #     if ddp:
    #         dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    #         loss /= ddp_world_size
    #     norm = torch.nn.utils.clip_grad_norm_(eikonal3d.parameters(), 1.0)
    #     if master_process:
    #         print("Loss:", loss.item())

    if ddp_local_rank == 0:
        print(f"Inversion time: {time.time() - t0:.2f}s")

    # preds, loss = eikonal3d(picks)
    preds, loss = eikonal3d(raw_picks)
    if ddp_local_rank == 0:
        print("Final loss:", loss.item())

        vp = raw_eikonal3d.vp0.detach().numpy() + raw_eikonal3d.dvp.detach().numpy()
        vs = raw_eikonal3d.vs0.detach().numpy() + raw_eikonal3d.dvs.detach().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(vp[:, :, nz // 2], cmap="viridis")
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("Vp")
        im = ax[1].imshow(vs[:, :, nz // 2], cmap="viridis")
        fig.colorbar(im, ax=ax[1])
        # ax[1].set_title("Vs")
        if ddp:
            plt.savefig(f"{data_path}/inversed3d_vp_vs_xy_{ddp_local_rank}.png")
        else:
            plt.savefig(f"{data_path}/inversed3d_vp_vs_xy.png")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(vp[:, ny // 2, :], cmap="viridis")
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("Vp")
        im = ax[1].imshow(vs[:, ny // 2, :], cmap="viridis")
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("Vs")
        if ddp:
            plt.savefig(f"{data_path}/inversed3d_vp_vs_xz_{ddp_local_rank}.png")
        else:
            plt.savefig(f"{data_path}/inversed3d_vp_vs_xz.png")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(vp[nx // 2, :, :], cmap="viridis")
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("Vp")
        im = ax[1].imshow(vs[nx // 2, :, :], cmap="viridis")
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("Vs")
        if ddp:
            plt.savefig(f"{data_path}/inversed3d_vp_vs_yz_{ddp_local_rank}.png")
        else:
            plt.savefig(f"{data_path}/inversed3d_vp_vs_yz.png")
