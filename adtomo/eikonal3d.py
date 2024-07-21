# %%
# %%
import eikonal2d_op
import eikonal3d_op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def interp3d(time_table, x, y, z, xgrid, ygrid, zgrid, h):
    nx = len(xgrid)
    ny = len(ygrid)
    nz = len(zgrid)
    assert time_table.shape == (nx, ny, nz)

    ix0 = np.floor((x - xgrid[0]) / h).clip(0, nx - 2).astype(int)
    iy0 = np.floor((y - ygrid[0]) / h).clip(0, ny - 2).astype(int)
    iz0 = np.floor((z - zgrid[0]) / h).clip(0, nz - 2).astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1
    x = (np.clip(x, xgrid[0], xgrid[-1]) - xgrid[0]) / h
    y = (np.clip(y, ygrid[0], ygrid[-1]) - ygrid[0]) / h
    z = (np.clip(z, zgrid[0], zgrid[-1]) - zgrid[0]) / h

    Q000 = time_table[ix0, iy0, iz0]
    Q100 = time_table[ix1, iy0, iz0]
    Q010 = time_table[ix0, iy1, iz0]
    Q110 = time_table[ix1, iy1, iz0]
    Q001 = time_table[ix0, iy0, iz1]
    Q101 = time_table[ix1, iy0, iz1]
    Q011 = time_table[ix0, iy1, iz1]
    Q111 = time_table[ix1, iy1, iz1]

    t = (
        Q000 * (ix1 - x) * (iy1 - y) * (iz1 - z)
        + Q100 * (x - ix0) * (iy1 - y) * (iz1 - z)
        + Q010 * (ix1 - x) * (y - iy0) * (iz1 - z)
        + Q110 * (x - ix0) * (y - iy0) * (iz1 - z)
        + Q001 * (ix1 - x) * (iy1 - y) * (z - iz0)
        + Q101 * (x - ix0) * (iy1 - y) * (z - iz0)
        + Q011 * (ix1 - x) * (y - iy0) * (z - iz0)
        + Q111 * (x - ix0) * (y - iy0) * (z - iz0)
    )

    return t


def set_u0(v, x, y, z, xgrid, ygrid, zgrid, h):
    nx = len(xgrid)
    ny = len(ygrid)
    nz = len(zgrid)
    assert v.shape == (nx, ny, nz)

    ix0 = np.floor((x - xgrid[0]) / h).clip(0, nx - 2).astype(int)
    iy0 = np.floor((y - ygrid[0]) / h).clip(0, ny - 2).astype(int)
    iz0 = np.floor((z - zgrid[0]) / h).clip(0, nz - 2).astype(int)
    ix1 = ix0 + 1
    iy1 = iy0 + 1
    iz1 = iz0 + 1
    x = (np.clip(x, xgrid[0], xgrid[-1]) - xgrid[0]) / h
    y = (np.clip(y, ygrid[0], ygrid[-1]) - ygrid[0]) / h
    z = (np.clip(z, zgrid[0], zgrid[-1]) - zgrid[0]) / h

    u0 = np.ones([nx, ny, nz]) * 1000.0
    u0[ix0, iy0, iz0] = np.sqrt((x - ix0) ** 2 + (y - iy0) ** 2 + (z - iz0) ** 2) * h / v[ix0, iy0, iz0]
    u0[ix1, iy0, iz0] = np.sqrt((x - ix1) ** 2 + (y - iy0) ** 2 + (z - iz0) ** 2) * h / v[ix1, iy0, iz0]
    u0[ix0, iy1, iz0] = np.sqrt((x - ix0) ** 2 + (y - iy1) ** 2 + (z - iz0) ** 2) * h / v[ix0, iy1, iz0]
    u0[ix1, iy1, iz0] = np.sqrt((x - ix1) ** 2 + (y - iy1) ** 2 + (z - iz0) ** 2) * h / v[ix1, iy1, iz0]
    u0[ix0, iy0, iz1] = np.sqrt((x - ix0) ** 2 + (y - iy0) ** 2 + (z - iz1) ** 2) * h / v[ix0, iy0, iz1]
    u0[ix1, iy0, iz1] = np.sqrt((x - ix1) ** 2 + (y - iy0) ** 2 + (z - iz1) ** 2) * h / v[ix1, iy0, iz1]
    u0[ix0, iy1, iz1] = np.sqrt((x - ix0) ** 2 + (y - iy1) ** 2 + (z - iz1) ** 2) * h / v[ix0, iy1, iz1]
    u0[ix1, iy1, iz1] = np.sqrt((x - ix1) ** 2 + (y - iy1) ** 2 + (z - iz1) ** 2) * h / v[ix1, iy1, iz1]

    return u0


class Eikonal3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u0, f, h, x, y, z):
        u = eikonal3d_op.forward(u0, f, h, x, y, z)
        ctx.save_for_backward(u, u0, f)
        ctx.h = h
        ctx.x = x
        ctx.y = y
        ctx.z = z
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, u0, f = ctx.saved_tensors
        grad_u0, grad_f = eikonal3d_op.backward(grad_output, u, u0, f, ctx.h, ctx.x, ctx.y, ctx.z)
        return grad_u0, grad_f, None, None, None, None


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def clamp(input, min, max):
    return Clamp.apply(input, min, max)


class Eikonal3D(torch.nn.Module):
    def __init__(
        self,
        num_event,
        num_station,
        station_loc,
        station_dt,
        event_loc,
        event_time,
        vp,
        vs,
        config=None,
        dtype=torch.float64,
    ):
        super().__init__()
        self.dtype = dtype
        self.num_event = num_event
        self.event_loc = nn.Embedding(num_event, 3)
        self.event_time = nn.Embedding(num_event, 1)
        self.station_loc = nn.Embedding(num_station, 3)
        self.station_dt = nn.Embedding(num_station, 1)  # same statioin term for P and S
        self.station_loc.weight = torch.nn.Parameter(torch.tensor(station_loc, dtype=dtype), requires_grad=False)
        self.station_dt.weight = torch.nn.Parameter(torch.zeros(num_station, 1, dtype=dtype), requires_grad=False)

        self.event_loc.weight = torch.nn.Parameter(
            torch.tensor(event_loc, dtype=dtype).contiguous(), requires_grad=False
        )
        self.event_time.weight = torch.nn.Parameter(
            torch.tensor(event_time, dtype=dtype).contiguous(), requires_grad=False
        )
        self.vp = torch.nn.Parameter(vp, requires_grad=True)
        self.vs = torch.nn.Parameter(vs, requires_grad=True)

        # set config
        nx, ny, nz, h = config["nx"], config["ny"], config["nz"], config["h"]
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.h = h
        self.xgrid = torch.arange(0, nx, dtype=dtype) * h
        self.ygrid = torch.arange(0, ny, dtype=dtype) * h
        self.zgrid = torch.arange(0, nz, dtype=dtype) * h

    def interp(self, time_table, x, y, z):

        ix0 = torch.floor((x - self.xgrid[0]) / self.h).clamp(0, self.nx - 2).long()
        iy0 = torch.floor((y - self.ygrid[0]) / self.h).clamp(0, self.ny - 2).long()
        iz0 = torch.floor((z - self.zgrid[0]) / self.h).clamp(0, self.nz - 2).long()
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1
        # x = (torch.clamp(x, self.xgrid[0], self.xgrid[-1]) - self.xgrid[0]) / self.h
        # y = (torch.clamp(y, self.ygrid[0], self.ygrid[-1]) - self.ygrid[0]) / self.h
        # z = (torch.clamp(z, self.zgrid[0], self.zgrid[-1]) - self.zgrid[0]) / self.h
        x = (clamp(x, self.xgrid[0], self.xgrid[-1]) - self.xgrid[0]) / self.h
        y = (clamp(y, self.ygrid[0], self.ygrid[-1]) - self.ygrid[0]) / self.h
        z = (clamp(z, self.zgrid[0], self.zgrid[-1]) - self.zgrid[0]) / self.h

        Q000 = time_table[ix0, iy0, iz0]
        Q100 = time_table[ix1, iy0, iz0]
        Q010 = time_table[ix0, iy1, iz0]
        Q110 = time_table[ix1, iy1, iz0]
        Q001 = time_table[ix0, iy0, iz1]
        Q101 = time_table[ix1, iy0, iz1]
        Q011 = time_table[ix0, iy1, iz1]
        Q111 = time_table[ix1, iy1, iz1]

        t = (
            Q000 * (ix1 - x) * (iy1 - y) * (iz1 - z)
            + Q100 * (x - ix0) * (iy1 - y) * (iz1 - z)
            + Q010 * (ix1 - x) * (y - iy0) * (iz1 - z)
            + Q110 * (x - ix0) * (y - iy0) * (iz1 - z)
            + Q001 * (ix1 - x) * (iy1 - y) * (z - iz0)
            + Q101 * (x - ix0) * (iy1 - y) * (z - iz0)
            + Q011 * (ix1 - x) * (y - iy0) * (z - iz0)
            + Q111 * (x - ix0) * (y - iy0) * (z - iz0)
        )

        return t

    def set_u0(self, v, x, y, z):

        ix0 = torch.floor((x - self.xgrid[0]) / self.h).clamp(0, self.nx - 2).long()
        iy0 = torch.floor((y - self.ygrid[0]) / self.h).clamp(0, self.ny - 2).long()
        iz0 = torch.floor((z - self.zgrid[0]) / self.h).clamp(0, self.nz - 2).long()
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        iz1 = iz0 + 1
        # x = (torch.clamp(x, self.xgrid[0], self.xgrid[-1]) - self.xgrid[0]) / self.h
        # y = (torch.clamp(y, self.ygrid[0], self.ygrid[-1]) - self.ygrid[0]) / self.h
        # z = (torch.clamp(z, self.zgrid[0], self.zgrid[-1]) - self.zgrid[0]) / self.h
        x = (clamp(x, self.xgrid[0], self.xgrid[-1]) - self.xgrid[0]) / self.h
        y = (clamp(y, self.ygrid[0], self.ygrid[-1]) - self.ygrid[0]) / self.h
        z = (clamp(z, self.zgrid[0], self.zgrid[-1]) - self.zgrid[0]) / self.h

        u0 = torch.ones([self.nx, self.ny, self.nz], dtype=self.dtype) * 1000.0
        u0[ix0, iy0, iz0] = torch.sqrt((x - ix0) ** 2 + (y - iy0) ** 2 + (z - iz0) ** 2) * self.h / v[ix0, iy0, iz0]
        u0[ix1, iy0, iz0] = torch.sqrt((x - ix1) ** 2 + (y - iy0) ** 2 + (z - iz0) ** 2) * self.h / v[ix1, iy0, iz0]
        u0[ix0, iy1, iz0] = torch.sqrt((x - ix0) ** 2 + (y - iy1) ** 2 + (z - iz0) ** 2) * self.h / v[ix0, iy1, iz0]
        u0[ix1, iy1, iz0] = torch.sqrt((x - ix1) ** 2 + (y - iy1) ** 2 + (z - iz0) ** 2) * self.h / v[ix1, iy1, iz0]
        u0[ix0, iy0, iz1] = torch.sqrt((x - ix0) ** 2 + (y - iy0) ** 2 + (z - iz1) ** 2) * self.h / v[ix0, iy0, iz1]
        u0[ix1, iy0, iz1] = torch.sqrt((x - ix1) ** 2 + (y - iy0) ** 2 + (z - iz1) ** 2) * self.h / v[ix1, iy0, iz1]
        u0[ix0, iy1, iz1] = torch.sqrt((x - ix0) ** 2 + (y - iy1) ** 2 + (z - iz1) ** 2) * self.h / v[ix0, iy1, iz1]
        u0[ix1, iy1, iz1] = torch.sqrt((x - ix1) ** 2 + (y - iy1) ** 2 + (z - iz1) ** 2) * self.h / v[ix1, iy1, iz1]

        return u0

    def forward(self, picks):

        # %%
        loss = 0
        pred = []
        idx = []
        for (station_index_, phase_type_), picks_ in picks.groupby(["station_index", "phase_type"]):
            idx.append(picks_.index)
            station_loc = self.station_loc(torch.tensor(station_index_, dtype=torch.int64))

            event_index_ = torch.tensor(picks_["event_index"].values, dtype=torch.int64)
            event_loc = self.event_loc(event_index_)
            event_time = self.event_time(event_index_)

            if phase_type_ == "P":
                # up0 = self.set_u0(self.vp, station_loc[0], station_loc[1], station_loc[2])
                up0 = torch.ones((self.nx, self.ny, self.nz), dtype=self.dtype) * 1000.0
                tp3d = Eikonal3DFunction.apply(
                    up0,
                    1.0 / self.vp,
                    self.h,
                    station_loc[0] / self.h,
                    station_loc[1] / self.h,
                    station_loc[2] / self.h,
                )
                tt = self.interp(tp3d, event_loc[:, 0], event_loc[:, 1], event_loc[:, 2]).squeeze()  # travel time
                at = event_time.squeeze() + tt  # arrival time
                # pred.append(tt.detach().numpy())
                # loss += F.mse_loss(tt, torch.tensor(picks_["travel_time"].values, dtype=self.dtype).squeeze())
                pred.append(at.detach().numpy())
                loss += F.mse_loss(at, torch.tensor(picks_["phase_time"].values, dtype=self.dtype).squeeze())

            elif phase_type_ == "S":
                # us0 = self.set_u0(self.vs, station_loc[0], station_loc[1], station_loc[2])
                us0 = torch.zeros((self.nx, self.ny, self.nz), dtype=self.dtype) * 1000.0
                ts3d = Eikonal3DFunction.apply(
                    us0,
                    1.0 / self.vs,
                    self.h,
                    station_loc[0] / self.h,
                    station_loc[1] / self.h,
                    station_loc[2] / self.h,
                )
                tt = self.interp(ts3d, event_loc[:, 0], event_loc[:, 1], event_loc[:, 2]).squeeze()
                at = event_time.squeeze() + tt
                # pred.append(tt.detach().numpy())
                # loss += F.mse_loss(tt, torch.tensor(picks_["travel_time"].values, dtype=self.dtype).squeeze())
                pred.append(at.detach().numpy())
                loss += F.mse_loss(at, torch.tensor(picks_["phase_time"].values, dtype=self.dtype).squeeze())

        pred_df = pd.DataFrame(
            {
                "index": np.concatenate(idx),
                "pred_s": np.concatenate(pred),
            }
        )
        pred_df = pred_df.sort_values("index", ignore_index=True)
        return pred_df, loss


# %%
if __name__ == "__main__":

    # %%
    import json
    import os
    from datetime import datetime, timedelta

    ######################################## Create Synthetic Data #########################################
    np.random.seed(0)
    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    nx = 10
    ny = 10
    nz = 10
    h = 3.0
    eikonal_config = {"nx": nx, "ny": ny, "nz": nz, "h": h}
    with open(f"{data_path}/config.json", "w") as f:
        json.dump(eikonal_config, f)
    xgrid = np.arange(0, nx) * h
    ygrid = np.arange(0, ny) * h
    zgrid = np.arange(0, nz) * h
    eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid, "zgrid": zgrid})
    num_station = 10
    num_event = 50
    stations = []
    for i in range(num_station):
        x = np.random.uniform(xgrid[0], xgrid[-1])
        y = np.random.uniform(ygrid[0], ygrid[-1])
        z = np.random.uniform(zgrid[0], zgrid[0] + 3 * h)
        stations.append({"station_id": f"STA{i:02d}", "x_km": x, "y_km": y, "z_km": z, "dt_s": 0.0})
    stations = pd.DataFrame(stations)
    stations["station_index"] = stations.index
    stations.to_csv(f"{data_path}/stations.csv", index=False)
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
    events.to_csv(f"{data_path}/events.csv", index=False)
    vpvs_ratio = 1.73
    vp = torch.ones((nx, ny, nz), dtype=torch.float64) * 6.0
    vs = vp / vpvs_ratio

    ### add anomaly
    vp[int(nx / 3) : int(2 * nx / 3), int(ny / 3) : int(2 * ny / 3), :] *= 1.1
    vs[int(nx / 3) : int(2 * nx / 3), int(ny / 3) : int(2 * ny / 3), :] *= 1.1

    picks = []
    for j, station in stations.iterrows():
        ix, iy = int(round(station["x_km"] / h)), int(round(station["y_km"] / h))
        # up0 = set_u0(vp, station["x_km"], station["y_km"], station["z_km"], xgrid, ygrid, zgrid, h)
        # us0 = set_u0(vs, station["x_km"], station["y_km"], station["z_km"], xgrid, ygrid, zgrid, h)
        # up0 = torch.tensor(up0, dtype=torch.float64)
        # us0 = torch.tensor(us0, dtype=torch.float64)
        up0 = torch.ones((nx, ny, nz), dtype=torch.float64) * 1000.0
        us0 = torch.zeros((nx, ny, nz), dtype=torch.float64) * 1000.0
        tp3d = eikonal3d_op.forward(
            up0, 1.0 / vp, h, station["x_km"] / h, station["y_km"] / h, station["z_km"] / h
        ).numpy()
        ts3d = eikonal3d_op.forward(
            us0, 1.0 / vs, h, station["x_km"] / h, station["y_km"] / h, station["z_km"] / h
        ).numpy()
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
    picks.to_csv(f"{data_path}/picks.csv", index=False)
    # %%
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/true3d_vp_vs_xy.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/true3d_vp_vs_xz.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/true3d_vp_vs_yz.png")

    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    ax[0, 0].scatter(stations["x_km"], stations["y_km"], c=stations["z_km"], marker="^", label="Station")
    ax[0, 0].scatter(events["x_km"], events["y_km"], c=events["z_km"], marker=".", label="Event")
    ax[0, 0].set_xlabel("x (km)")
    ax[0, 0].set_ylabel("y (km)")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{data_path}/station_event_3d.png")
    # %%
    fig, ax = plt.subplots(3, 1, squeeze=False, figsize=(10, 15))
    picks = picks.merge(stations, on="station_id")
    mapping_color = lambda x: f"C{int(x)}"
    picks["phase_time"] = pd.to_datetime(picks["phase_time"])
    events["event_time"] = pd.to_datetime(events["event_time"])
    ax[0, 0].scatter(picks["phase_time"], picks["x_km"], c=picks["event_index"].apply(mapping_color))
    ax[0, 0].scatter(events["event_time"], events["x_km"], c=events["event_index"].apply(mapping_color), marker="x")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("x (km)")
    ax[1, 0].scatter(picks["phase_time"], picks["y_km"], c=picks["event_index"].apply(mapping_color))
    ax[1, 0].scatter(events["event_time"], events["y_km"], c=events["event_index"].apply(mapping_color), marker="x")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("y (km)")
    ax[2, 0].scatter(picks["phase_time"], picks["z_km"], c=picks["event_index"].apply(mapping_color))
    ax[2, 0].scatter(events["event_time"], events["z_km"], c=events["event_index"].apply(mapping_color), marker="x")
    ax[2, 0].set_xlabel("Time (s)")
    ax[2, 0].set_ylabel("z (km)")
    plt.savefig(f"{data_path}/picks_3d.png")
    # %%
    ######################################### Load Synthetic Data #########################################
    data_path = "data"
    events = pd.read_csv(f"{data_path}/events.csv")
    stations = pd.read_csv(f"{data_path}/stations.csv")
    picks = pd.read_csv(f"{data_path}/picks.csv")
    picks = picks.merge(events[["event_index", "event_time"]], on="event_index")

    #### make the time values relative to event time in seconds
    picks["phase_time_origin"] = picks["phase_time"].copy()
    picks["phase_time"] = (
        pd.to_datetime(picks["phase_time"]) - pd.to_datetime(picks["event_time"])
    ).dt.total_seconds()  # relative to event time (arrival time)
    picks.drop(columns=["event_time"], inplace=True)
    events["event_time_origin"] = events["event_time"].copy()
    events["event_time"] = np.zeros(len(events))  # relative to event time
    ####

    with open(f"{data_path}/config.json", "r") as f:
        eikonal_config = json.load(f)
    events = events.sort_values("event_index").set_index("event_index")
    stations = stations.sort_values("station_index").set_index("station_index")
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
        eikonal_config,
    )
    preds, loss = eikonal3d(picks)

    ######################################### Optimize #########################################
    # %%
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(eikonal3d.vp.detach().numpy()[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(eikonal3d.vs.detach().numpy()[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial3d_vp_vs_xy.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(eikonal3d.vp.detach().numpy()[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(eikonal3d.vs.detach().numpy()[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial3d_vp_vs_xz.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(eikonal3d.vp.detach().numpy()[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(eikonal3d.vs.detach().numpy()[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial3d_vp_vs_yz.png")

    optimizer = optim.LBFGS(params=eikonal3d.parameters(), max_iter=1000, line_search_fn="strong_wolfe")
    print("Initial loss:", loss.item())

    def closure():
        optimizer.zero_grad()
        _, loss = eikonal3d(picks)
        loss.backward()
        return loss

    optimizer.step(closure)

    preds, loss = eikonal3d(picks)
    print("Final loss:", loss.item())

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(eikonal3d.vp.detach().numpy()[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(eikonal3d.vs.detach().numpy()[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/inversed3d_vp_vs_xy.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(eikonal3d.vp.detach().numpy()[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(eikonal3d.vs.detach().numpy()[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/inversed3d_vp_vs_xz.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(eikonal3d.vp.detach().numpy()[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(eikonal3d.vs.detach().numpy()[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/inversed3d_vp_vs_yz.png")
