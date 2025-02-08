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

from adtomo.eikonal3d import Eikonal3D, eikonal3d_op, interp3d

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
    ######################################## Create Synthetic Data #########################################
    np.random.seed(0)
    data_path = "data"
    result_path = "results"
    figure_path = "figures"
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    meta = np.load(f"{data_path}/SFCVM_5km.npz")
    assert meta["dx"] == meta["dy"] == meta["dz"]
    vp = meta["vp"]
    vs = meta["vs"]
    vp_mean = np.mean(vp, axis=(0, 1))
    vs_mean = np.mean(vs, axis=(0, 1))
    vp_min = np.min(vp, axis=(0, 1))
    vs_min = np.min(vs, axis=(0, 1))
    vp_max = np.max(vp, axis=(0, 1))
    vs_max = np.max(vs, axis=(0, 1))

    # %%
    nx, ny, nz = vp.shape
    h = float(meta["dx"])

    eikonal_config = {"nx": nx, "ny": ny, "nz": nz, "h": h}
    with open(f"{result_path}/config.json", "w") as f:
        json.dump(eikonal_config, f)
    xgrid = np.arange(0, nx) * h
    ygrid = np.arange(0, ny) * h
    zgrid = np.arange(0, nz) * h
    eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid, "zgrid": zgrid})
    num_station = 10
    num_event = 500
    stations = []
    for i in range(num_station):
        x = np.random.uniform(xgrid[0], xgrid[-1])
        y = np.random.uniform(ygrid[0], ygrid[-1])
        z = np.random.uniform(zgrid[0], zgrid[0] + 3 * h)
        stations.append({"station_id": f"STA{i:02d}", "x_km": x, "y_km": y, "z_km": z, "dt_s": 0.0})
    stations = pd.DataFrame(stations)
    stations["station_index"] = stations.index
    stations.to_csv(f"{result_path}/stations.csv", index=False)
    events = []
    reference_time = pd.to_datetime("2021-01-01T00:00:00.000")
    for i in range(num_event):
        x = np.random.uniform(xgrid[0], xgrid[-1])
        y = np.random.uniform(ygrid[0], ygrid[-1])
        z = np.random.uniform(zgrid[0], zgrid[-1])
        t = i * 5
        # events.append({"event_id": i, "event_time": t, "x_km": x, "y_km": y, "z_km": z})
        events.append(
            {"event_id": i, "event_time": reference_time + pd.Timedelta(seconds=t), "x_km": x, "y_km": y, "z_km": z}
        )
    events = pd.DataFrame(events)
    events["event_index"] = events.index
    events["event_time"] = events["event_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    events.to_csv(f"{result_path}/events.csv", index=False)

    # vpvs_ratio = 1.73
    # vp = torch.ones((nx, ny, nz), dtype=torch.float64) * 6.0
    # vs = vp / vpvs_ratio

    # ### add anomaly
    # vp[int(nx / 3) : int(2 * nx / 3), int(ny / 3) : int(2 * ny / 3), :] *= 1.1
    # vs[int(nx / 3) : int(2 * nx / 3), int(ny / 3) : int(2 * ny / 3), :] *= 1.1

    vp = torch.from_numpy(vp)
    vs = torch.from_numpy(vs)

    picks = []
    for j, station in stations.iterrows():
        ix, iy = int(round(station["x_km"] / h)), int(round(station["y_km"] / h))
        tp3d = eikonal3d_op.forward(1.0 / vp, h, station["x_km"] / h, station["y_km"] / h, station["z_km"] / h).numpy()
        ts3d = eikonal3d_op.forward(1.0 / vs, h, station["x_km"] / h, station["y_km"] / h, station["z_km"] / h).numpy()
        for i, event in events.iterrows():
            if np.random.rand() < 0.5:
                tt = interp3d(
                    tp3d,
                    event["x_km"],
                    event["y_km"],
                    event["z_km"],
                    eikonal_config["xgrid"],
                    eikonal_config["ygrid"],
                    eikonal_config["zgrid"],
                    h,
                )
                picks.append(
                    {
                        "event_id": event["event_id"],
                        "station_id": station["station_id"],
                        "phase_type": "P",
                        # "phase_time": event["event_time"] + tt,
                        "phase_time": pd.to_datetime(event["event_time"]) + pd.Timedelta(seconds=tt),
                        "travel_time": tt,
                    }
                )
            if np.random.rand() < 0.5:
                tt = interp3d(
                    ts3d,
                    event["x_km"],
                    event["y_km"],
                    event["z_km"],
                    eikonal_config["xgrid"],
                    eikonal_config["ygrid"],
                    eikonal_config["zgrid"],
                    h,
                )
                picks.append(
                    {
                        "event_id": event["event_id"],
                        "station_id": station["station_id"],
                        "phase_type": "S",
                        # "phase_time": event["event_time"] + tt,
                        "phase_time": pd.to_datetime(event["event_time"]) + pd.Timedelta(seconds=tt),
                        "travel_time": tt,
                    }
                )
    picks = pd.DataFrame(picks)
    # use picks,  stations.index, events.index to set station_index and
    picks["phase_time"] = picks["phase_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    picks["event_index"] = picks["event_id"].map(events.set_index("event_id")["event_index"])
    picks["station_index"] = picks["station_id"].map(stations.set_index("station_id")["station_index"])
    picks.to_csv(f"{result_path}/picks.csv", index=False)
    # %%
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, :, nz // 2], cmap="bwr_r", vmin=vp_min[nz // 2], vmax=vp_max[nz // 2])
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, :, nz // 2], cmap="bwr_r", vmin=vs_min[nz // 2], vmax=vs_max[nz // 2])
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{figure_path}/true3d_vp_vs_xy.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, ny // 2, :], cmap="bwr_r", vmin=vp_min.min(), vmax=vp_max.max())
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, ny // 2, :], cmap="bwr_r", vmin=vs_min.min(), vmax=vs_max.max())
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{figure_path}/true3d_vp_vs_xz.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[nx // 2, :, :], cmap="bwr_r", vmin=vp_min.min(), vmax=vp_max.max())
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[nx // 2, :, :], cmap="bwr_r", vmin=vs_min.min(), vmax=vs_max.max())
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{figure_path}/true3d_vp_vs_yz.png", bbox_inches="tight")

    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    ax[0, 0].scatter(stations["x_km"], stations["y_km"], c=stations["z_km"], marker="^", label="Station")
    ax[0, 0].scatter(events["x_km"], events["y_km"], c=events["z_km"], marker=".", label="Event")
    ax[0, 0].set_xlabel("x (km)")
    ax[0, 0].set_ylabel("y (km)")
    ax[0, 0].set_aspect("equal")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{figure_path}/station_event_3d.png", bbox_inches="tight")
    # %%
    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(10, 15))
    picks = picks.merge(stations, on="station_id")
    mapping_color = lambda x: f"C{int(x)}"
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    events["event_time"] = pd.to_datetime(events["event_time"])
    ax[0, 0].scatter(pd.to_datetime(picks["phase_time"]), picks["x_km"], c=picks["event_index"].apply(mapping_color))
    ax[0, 0].scatter(
        pd.to_datetime(events["event_time"]), events["x_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("x (km)")
    ax[1, 0].scatter(pd.to_datetime(picks["phase_time"]), picks["y_km"], c=picks["event_index"].apply(mapping_color))
    ax[1, 0].scatter(
        pd.to_datetime(events["event_time"]), events["y_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("y (km)")
    ax[2, 0].scatter(pd.to_datetime(picks["phase_time"]), picks["z_km"], c=picks["event_index"].apply(mapping_color))
    ax[2, 0].scatter(
        pd.to_datetime(events["event_time"]), events["z_km"], c=events["event_index"].apply(mapping_color), marker="x"
    )
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 0].set_ylabel("z (km)")
    plt.savefig(f"{figure_path}/picks_3d.png", bbox_inches="tight")
    # %%
    ######################################### Load Synthetic Data #########################################
    data_path = "results"
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

    vp = torch.ones((nx, ny, nz), dtype=torch.float64) * np.mean(vp_mean)
    vs = torch.ones((nx, ny, nz), dtype=torch.float64) * np.mean(vs_mean)

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
        lambda_vp=1.0,
        lambda_vs=1.0,
        config=eikonal_config,
    )
    preds, loss = eikonal3d(picks)

    ######################################### Optimize #########################################
    # %%
    vp = eikonal3d.vp0.detach().numpy() + eikonal3d.dvp.detach().numpy()
    vs = eikonal3d.vs0.detach().numpy() + eikonal3d.dvs.detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, :, nz // 2], cmap="bwr_r", vmin=vp_min[nz // 2], vmax=vp_max[nz // 2])
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, :, nz // 2], cmap="bwr_r", vmin=vs_min[nz // 2], vmax=vs_max[nz // 2])
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{figure_path}/initial3d_vp_vs_xy.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, ny // 2, :], cmap="bwr_r", vmin=vp_min.min(), vmax=vp_max.max())
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, ny // 2, :], cmap="bwr_r", vmin=vs_min.min(), vmax=vs_max.max())
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{figure_path}/initial3d_vp_vs_xz.png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[nx // 2, :, :], cmap="bwr_r", vmin=vp_min.min(), vmax=vp_max.max())
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[nx // 2, :, :], cmap="bwr_r", vmin=vs_min.min(), vmax=vs_max.max())
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{figure_path}/initial3d_vp_vs_yz.png", bbox_inches="tight")

    raw_picks = picks.copy()
    picks = picks[picks["idx_sta"] % ddp_world_size == ddp_local_rank]
    print(f"Rank {ddp_rank} has {len(picks)} picks")

    eikonal3d.dvp.requires_grad = True
    eikonal3d.dvs.requires_grad = True
    eikonal3d.event_loc.weight.requires_grad = False
    eikonal3d.event_time.weight.requires_grad = False
    if ddp_local_rank == 0:
        print(
            "Optimizing parameters:\n"
            + "\n".join(
                [f"{name}: {param.size()}" for name, param in eikonal3d.named_parameters() if param.requires_grad]
            ),
        )

    if ddp:
        eikonal3d = DDP(eikonal3d)
    raw_eikonal3d = eikonal3d.module if ddp else eikonal3d

    t0 = time.time()
    parameters = [param for param in eikonal3d.parameters() if param.requires_grad]
    optimizer = optim.LBFGS(params=parameters, max_iter=1000, line_search_fn="strong_wolfe")

    if ddp_local_rank == 0:
        print("Initial loss:", loss.item())

    def closure():
        optimizer.zero_grad()
        _, loss = eikonal3d(picks)
        loss.backward()

        if ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= ddp_world_size

        norm = torch.nn.utils.clip_grad_norm_(eikonal3d.parameters(), 1.0)
        # if ddp_local_rank == 0:
        # print(f"Gradient norm: {norm.item()}")

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
        im = ax[0].imshow(vp[:, :, nz // 2], cmap="bwr_r", vmin=vp_min[nz // 2], vmax=vp_max[nz // 2])
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("Vp")
        im = ax[1].imshow(vs[:, :, nz // 2], cmap="bwr_r", vmin=vs_min[nz // 2], vmax=vs_max[nz // 2])
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("Vs")
        plt.savefig(f"{figure_path}/inversed3d_vp_vs_xy.png", bbox_inches="tight")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(vp[:, ny // 2, :], cmap="bwr_r", vmin=vp_min.min(), vmax=vp_max.max())
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("Vp")
        im = ax[1].imshow(vs[:, ny // 2, :], cmap="bwr_r", vmin=vs_min.min(), vmax=vs_max.max())
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("Vs")
        plt.savefig(f"{figure_path}/inversed3d_vp_vs_xz.png", bbox_inches="tight")

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        im = ax[0].imshow(vp[nx // 2, :, :], cmap="bwr_r", vmin=vp_min.min(), vmax=vp_max.max())
        fig.colorbar(im, ax=ax[0])
        ax[0].set_title("Vp")
        im = ax[1].imshow(vs[nx // 2, :, :], cmap="bwr_r", vmin=vs_min.min(), vmax=vs_max.max())
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title("Vs")
        plt.savefig(f"{figure_path}/inversed3d_vp_vs_yz.png", bbox_inches="tight")

        # %%
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
        # ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
        ax[0, 0].plot(events["x_km"], events["y_km"], ".", label="True Events")
        ax[0, 0].plot(event_loc[:, 0], event_loc[:, 1], "x", label="Initial Events")
        for i in range(len(event_loc)):
            ax[0, 0].plot(
                [events["x_km"].iloc[i], event_loc[i, 0]], [events["y_km"].iloc[i], event_loc[i, 1]], "k--", alpha=0.5
            )
        event_loc = raw_eikonal3d.event_loc.weight.detach().numpy()
        ax[0, 0].plot(event_loc[:, 0], event_loc[:, 1], "x", label="Inverted Events")
        for i in range(len(event_loc)):
            ax[0, 0].plot(
                [events["x_km"].iloc[i], event_loc[i, 0]], [events["y_km"].iloc[i], event_loc[i, 1]], "r--", alpha=0.5
            )
        ax[0, 0].set_xlabel("x (km)")
        ax[0, 0].set_ylabel("y (km)")
        ax[0, 0].legend()
        ax[0, 0].set_title("Station and Event Locations")
        plt.savefig(f"{figure_path}/inverted3d_station_event_xy.png", bbox_inches="tight")

        # %%
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
        # ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
        ax[0, 0].plot(events["x_km"], events["z_km"], ".", label="True Events")
        ax[0, 0].plot(event_loc[:, 0], event_loc[:, 2], "x", label="Initial Events")
        for i in range(len(event_loc)):
            ax[0, 0].plot(
                [events["x_km"].iloc[i], event_loc[i, 0]], [events["z_km"].iloc[i], event_loc[i, 2]], "k--", alpha=0.5
            )
        event_loc = raw_eikonal3d.event_loc.weight.detach().numpy()
        ax[0, 0].plot(event_loc[:, 0], event_loc[:, 2], "x", label="Inverted Events")
        for i in range(len(event_loc)):
            ax[0, 0].plot(
                [events["x_km"].iloc[i], event_loc[i, 0]], [events["z_km"].iloc[i], event_loc[i, 2]], "r--", alpha=0.5
            )
        ax[0, 0].set_xlabel("x (km)")
        ax[0, 0].set_ylabel("z (km)")
        ax[0, 0].legend()
        ax[0, 0].set_title("Station and Event Locations")
        plt.savefig(f"{figure_path}/inverted3d_station_event_xz.png", bbox_inches="tight")

        # %%
        fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
        # ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
        ax[0, 0].plot(events["y_km"], events["z_km"], ".", label="True Events")
        ax[0, 0].plot(event_loc[:, 1], event_loc[:, 2], "x", label="Initial Events")
        for i in range(len(event_loc)):
            ax[0, 0].plot(
                [events["y_km"].iloc[i], event_loc[i, 1]], [events["z_km"].iloc[i], event_loc[i, 2]], "k--", alpha=0.5
            )
        event_loc = raw_eikonal3d.event_loc.weight.detach().numpy()
        ax[0, 0].plot(event_loc[:, 1], event_loc[:, 2], "x", label="Inverted Events")
        for i in range(len(event_loc)):
            ax[0, 0].plot(
                [events["y_km"].iloc[i], event_loc[i, 1]], [events["z_km"].iloc[i], event_loc[i, 2]], "r--", alpha=0.5
            )
        ax[0, 0].set_xlabel("y (km)")
        ax[0, 0].set_ylabel("z (km)")
        ax[0, 0].legend()
        ax[0, 0].set_title("Station and Event Locations")
        plt.savefig(f"{figure_path}/inverted3d_station_event_yz.png", bbox_inches="tight")
