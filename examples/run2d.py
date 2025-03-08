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

from adtomo.eikonal2d import Eikonal2D, eikonal2d_op, interp2d

# %%
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

    # # %%
    # ######################################## Create Synthetic Data #########################################
    np.random.seed(0)
    data_path = "data"
    result_path = "results"
    figure_path = "figures"
    if not os.path.exists(result_path):
        os.makedirs(result_path, exist_ok=True)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path, exist_ok=True)
    meta = np.load(f"{data_path}/SFCVM_5km.npz")
    assert meta["dx"] == meta["dy"]
    vp = meta["vp"][:, :, 0]
    vs = meta["vs"][:, :, 0]
    vp_mean = vp.mean()
    vs_mean = vs.mean()
    vp_min = vp.min()
    vs_min = vs.min()
    vp_max = vp.max()
    vs_max = vs.max()

    # # %%
    # nx, ny = vp.shape
    # h = float(meta["dx"])

    # eikonal_config = {"nx": nx, "ny": ny, "h": h}
    # with open(f"{result_path}/config.json", "w") as f:
    #     json.dump(eikonal_config, f)
    # xgrid = np.arange(0, nx) * h
    # ygrid = np.arange(0, ny) * h
    # eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid})
    # num_station = 20
    # num_event = 100
    # stations = []
    # for i in range(num_station):
    #     x = np.random.randint(0, nx) * h
    #     y = np.random.randint(0, ny) * h
    #     stations.append({"station_id": f"STA{i:02d}", "x_km": x, "y_km": y, "dt_s": 0.0})
    # stations = pd.DataFrame(stations)
    # stations["station_index"] = stations.index
    # stations.to_csv(f"{result_path}/stations.csv", index=False)
    # events = []
    # reference_time = pd.to_datetime("2021-01-01T00:00:00.000")
    # for i in range(num_event):
    #     x = np.random.uniform(xgrid[0], xgrid[-1])
    #     y = np.random.uniform(ygrid[0], ygrid[-1])
    #     t = i * 5
    #     # events.append({"event_id": i, "event_time": t, "x_km": x, "y_km": y})
    #     events.append({"event_id": i, "event_time": reference_time + pd.Timedelta(seconds=t), "x_km": x, "y_km": y})
    # events = pd.DataFrame(events)
    # events["event_index"] = events.index
    # events["event_time"] = events["event_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    # events.to_csv(f"{result_path}/events.csv", index=False)

    # # # %%
    # # vpvs_ratio = 1.73
    # # vp = torch.ones((nx, ny), dtype=torch.float64) * 6.0
    # # vs = vp / vpvs_ratio
    # # ### add anomaly
    # # vp[int(nx / 3) : int(2 * nx / 3), int(ny / 3) : int(2 * ny / 3)] *= 1.1
    # # vs[int(nx / 3) : int(2 * nx / 3), int(ny / 3) : int(2 * ny / 3)] *= 1.1

    # vp = torch.from_numpy(vp)
    # vs = torch.from_numpy(vs)

    # picks = []
    # for j, station in stations.iterrows():
    #     ix, iy = int(round(station["x_km"] / h)), int(round(station["y_km"] / h))
    #     tp2d = eikonal2d_op.forward(1.0 / vp, h, ix, iy).numpy()
    #     ts2d = eikonal2d_op.forward(1.0 / vs, h, ix, iy).numpy()
    #     for i, event in events.iterrows():
    #         if np.random.rand() < 0.5:
    #             tt = interp2d(tp2d, event["x_km"], event["y_km"], eikonal_config["xgrid"], eikonal_config["ygrid"], h)
    #             picks.append(
    #                 {
    #                     "event_id": event["event_id"],
    #                     "station_id": station["station_id"],
    #                     "phase_type": "P",
    #                     # "phase_time": event["event_time"] + tt,
    #                     "phase_time": pd.to_datetime(event["event_time"]) + pd.Timedelta(seconds=tt),
    #                     "travel_time": tt,
    #                 }
    #             )
    #         if np.random.rand() < 0.5:
    #             tt = interp2d(ts2d, event["x_km"], event["y_km"], eikonal_config["xgrid"], eikonal_config["ygrid"], h)
    #             picks.append(
    #                 {
    #                     "event_id": event["event_id"],
    #                     "station_id": station["station_id"],
    #                     "phase_type": "S",
    #                     # "phase_time": event["event_time"] + tt,
    #                     "phase_time": pd.to_datetime(event["event_time"]) + pd.Timedelta(seconds=tt),
    #                     "travel_time": tt,
    #                 }
    #             )
    # picks = pd.DataFrame(picks)
    # # use picks,  stations.index, events.index to set station_index and
    # picks["phase_time"] = picks["phase_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    # picks["event_index"] = picks["event_id"].map(events.set_index("event_id")["event_index"])
    # picks["station_index"] = picks["station_id"].map(stations.set_index("station_id")["station_index"])
    # picks.to_csv(f"{result_path}/picks.csv", index=False)
    # # %%
    # fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # im = ax[0].imshow(vp.T, cmap="bwr_r", vmin=vp_min, vmax=vp_max, origin="lower")
    # fig.colorbar(im, ax=ax[0])
    # ax[0].set_title("Vp")
    # im = ax[1].imshow(vs.T, cmap="bwr_r", vmin=vs_min, vmax=vs_max, origin="lower")
    # fig.colorbar(im, ax=ax[1])
    # ax[1].set_title("Vs")
    # plt.savefig(f"{figure_path}/true2d_vp_vs.png", bbox_inches="tight")

    # # %%
    # fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    # ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
    # ax[0, 0].plot(events["x_km"], events["y_km"], ".", label="Event")
    # ax[0, 0].set_xlabel("x (km)")
    # ax[0, 0].set_ylabel("y (km)")
    # ax[0, 0].set_aspect("equal")
    # ax[0, 0].legend()
    # ax[0, 0].set_title("Station and Event Locations")
    # plt.savefig(f"{figure_path}/station_event_2d.png", bbox_inches="tight")

    # # %%
    # fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(10, 10))
    # picks = picks.merge(stations, on="station_id")
    # mapping_color = lambda x: f"C{int(x)}"
    # ax[0, 0].scatter(pd.to_datetime(picks["phase_time"]), picks["x_km"], c=picks["event_index"].apply(mapping_color))
    # ax[0, 0].scatter(
    #     pd.to_datetime(events["event_time"]), events["x_km"], c=events["event_index"].apply(mapping_color), marker="x"
    # )
    # ax[0, 0].set_xlabel("Time (s)")
    # ax[0, 0].set_ylabel("x (km)")
    # ax[1, 0].scatter(pd.to_datetime(picks["phase_time"]), picks["y_km"], c=picks["event_index"].apply(mapping_color))
    # ax[1, 0].scatter(
    #     pd.to_datetime(events["event_time"]), events["y_km"], c=events["event_index"].apply(mapping_color), marker="x"
    # )
    # ax[1, 0].set_xlabel("Time (s)")
    # ax[1, 0].set_ylabel("y (km)")
    # plt.savefig(f"{figure_path}/picks_2d.png", bbox_inches="tight")

    # %%
    ######################################### Load Synthetic Data #########################################
    data_path = "results"
    events = pd.read_csv(f"{data_path}/events.csv")
    stations = pd.read_csv(f"{data_path}/stations.csv")
    picks = pd.read_csv(f"{data_path}/picks.csv")
    picks = picks.merge(events[["event_index", "event_time"]], on="event_index")

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

    events["idx_eve"] = np.arange(len(events))  # continuous index from 0 to num_event/num_station
    stations["idx_sta"] = np.arange(len(stations))
    picks = picks.merge(events[["event_id", "idx_eve"]], on="event_id")  ## idx_eve, and idx_sta are used internally
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")

    with open(f"{data_path}/config.json", "r") as f:
        eikonal_config = json.load(f)
    num_event = len(events)
    num_station = len(stations)
    nx, ny, h = eikonal_config["nx"], eikonal_config["ny"], eikonal_config["h"]
    # xgrid = torch.arange(-int(nx * 1.5), int(nx * 1.5), dtype=torch.float64) * h
    # ygrid = torch.arange(-int(ny * 1.5), int(ny * 1.5), dtype=torch.float64) * h
    scale_sub = 1.0
    h = h / scale_sub
    xgrid = torch.arange(-int(nx * 1.5), int(nx * 1.5), dtype=torch.float64) * h
    ygrid = torch.arange(-int(ny * 1.5), int(ny * 1.5), dtype=torch.float64) * h
    nx, ny = xgrid.shape[0], ygrid.shape[0]
    eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid, "nx": nx, "ny": ny, "h": h, "scale_sub": scale_sub})

    vp = torch.ones((nx, ny), dtype=torch.float64) * vp_mean
    vs = torch.ones((nx, ny), dtype=torch.float64) * vs_mean
    # vp = torch.ones((nx, ny), dtype=torch.float64) * 6.0
    # vs = vp / vpvs_ratio

    ## initial event location
    event_loc = events[["x_km", "y_km"]].values
    # event_loc = events[["x_km", "y_km"]].values + np.random.randn(num_event, 2) * 10
    # event_loc = events[["x_km", "y_km"]].values * 0.0 + stations[["x_km", "y_km"]].values.mean(axis=0)

    lambda_dvp = 1e-4
    lambda_dvs = 1e-4
    lambda_sp_ratio = 1e-4
    eikonal2d = Eikonal2D(
        num_event,
        num_station,
        stations[["x_km", "y_km"]].values,
        stations[["dt_s"]].values,
        event_loc,
        events[["event_time"]].values,
        vp,
        vs,
        # max_dvp=1.0,
        # max_dvs=0.5,
        lambda_dvp=lambda_dvp,
        lambda_dvs=lambda_dvs,
        lambda_sp_ratio=lambda_sp_ratio,
        config=eikonal_config,
    )
    preds, loss = eikonal2d(picks)

    ######################################### Optimize #########################################
    # %%
    vp = eikonal2d.vp0.detach().numpy() + eikonal2d.dvp.detach().numpy()
    vs = eikonal2d.vs0.detach().numpy() + eikonal2d.dvs.detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im = ax[0].imshow(vp.T, cmap="bwr_r", vmin=vp_min, vmax=vp_max, origin="lower")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs.T, cmap="bwr_r", vmin=vs_min, vmax=vs_max, origin="lower")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{figure_path}/initial2d_vp_vs.png", bbox_inches="tight")

    raw_picks = picks.copy()
    picks = picks[picks["idx_sta"] % ddp_world_size == ddp_local_rank]
    print(f"Rank {ddp_rank} has {len(picks)} picks")

    eikonal2d.dvp.requires_grad = True
    eikonal2d.dvs.requires_grad = True
    eikonal2d.event_loc.weight.requires_grad = False
    eikonal2d.event_time.weight.requires_grad = False
    if ddp_local_rank == 0:
        print(
            "Optimizing parameters:\n"
            + "\n".join(
                [f"{name}: {param.size()}" for name, param in eikonal2d.named_parameters() if param.requires_grad]
            ),
        )

    if ddp:
        eikonal2d = DDP(eikonal2d)
    raw_eikonal2d = eikonal2d.module if ddp else eikonal2d

    t0 = time.time()
    parameters = [param for param in eikonal2d.parameters() if param.requires_grad]
    optimizer = optim.LBFGS(params=parameters, max_iter=1000, line_search_fn="strong_wolfe")

    if ddp_local_rank == 0:
        print("Initial loss:", loss.item())

    def closure():
        optimizer.zero_grad()
        _, loss = eikonal2d(picks)
        loss.backward()

        if ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= ddp_world_size

        # norm = torch.nn.utils.clip_grad_norm_(eikonal2d.parameters(), 1.0)
        # if ddp_local_rank == 0:
        #     print(f"Gradient norm: {norm.item()}")

        return loss

    optimizer.step(closure)

    # # optimizer = optim.Adam(params=eikonal2d.parameters(), lr=0.1)
    # for _ in range(100):
    #     optimizer.zero_grad()
    #     preds, loss = eikonal2d(picks)
    #     loss.backward()
    #     optimizer.step()
    #     if ddp:
    #         dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    #         loss /= ddp_world_size
    #     norm = torch.nn.utils.clip_grad_norm_(eikonal2d.parameters(), 1.0)
    #     if master_process:
    #         print("Loss:", loss.item())

    if ddp_local_rank == 0:
        print(f"Inversion time: {time.time() - t0:.2f}s")

    preds, loss = eikonal2d(picks)

    if ddp_local_rank == 0:
        print("Final loss:", loss.item())

        vp = raw_eikonal2d.vp0.detach().numpy() + raw_eikonal2d.dvp.detach().numpy()
        vs = raw_eikonal2d.vs0.detach().numpy() + raw_eikonal2d.dvs.detach().numpy()
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        im = ax[0].imshow(vp.T, cmap="bwr_r", vmin=vp_min, vmax=vp_max, origin="lower")
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("Vp")
        im = ax[1].imshow(vs.T, cmap="bwr_r", vmin=vs_min, vmax=vs_max, origin="lower")
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("Vs")
        plt.savefig(f"{figure_path}/inverted2d_vp_vs.png", bbox_inches="tight")
        plt.savefig(
            f"{figure_path}/inverted2d_vp_vs_{lambda_dvp:.0e}_{lambda_dvs:.0e}_{lambda_sp_ratio:.0e}.png",
            bbox_inches="tight",
        )

        # %%
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
        # ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
        ax[0, 0].plot(events["x_km"], events["y_km"], ".", label="True Events")
        ax[0, 0].plot(event_loc[:, 0], event_loc[:, 1], "x", label="Initial Events")
        for i in range(len(event_loc)):
            ax[0, 0].plot(
                [events["x_km"].iloc[i], event_loc[i, 0]], [events["y_km"].iloc[i], event_loc[i, 1]], "k--", alpha=0.5
            )
        event_loc = raw_eikonal2d.event_loc.weight.detach().numpy()
        ax[0, 0].plot(event_loc[:, 0], event_loc[:, 1], "x", label="Inverted Events")
        for i in range(len(event_loc)):
            ax[0, 0].plot(
                [events["x_km"].iloc[i], event_loc[i, 0]], [events["y_km"].iloc[i], event_loc[i, 1]], "r--", alpha=0.5
            )
        ax[0, 0].set_xlabel("x (km)")
        ax[0, 0].set_ylabel("y (km)")
        ax[0, 0].legend()
        ax[0, 0].set_title("Station and Event Locations")
        plt.savefig(f"{figure_path}/inverted2d_station_event.png", bbox_inches="tight")
