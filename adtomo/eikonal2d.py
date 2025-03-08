# %%
import eikonal2d_op
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def interp2d(time_table, r, z, rgrid, zgrid, h):
    nr = len(rgrid)
    nz = len(zgrid)
    assert time_table.shape == (nr, nz)

    ir0 = np.floor((r - rgrid[0]) / h).clip(0, nr - 2).astype(int)
    iz0 = np.floor((z - zgrid[0]) / h).clip(0, nz - 2).astype(int)
    ir1 = ir0 + 1
    iz1 = iz0 + 1
    r = (np.clip(r, rgrid[0], rgrid[-1]) - rgrid[0]) / h
    z = (np.clip(z, zgrid[0], zgrid[-1]) - zgrid[0]) / h

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    Q00 = time_table[ir0, iz0]
    Q01 = time_table[ir0, iz1]
    Q10 = time_table[ir1, iz0]
    Q11 = time_table[ir1, iz1]

    t = (
        Q00 * (ir1 - r) * (iz1 - z)
        + Q10 * (r - ir0) * (iz1 - z)
        + Q01 * (ir1 - r) * (z - iz0)
        + Q11 * (r - ir0) * (z - iz0)
    )

    return t


class Eikonal2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, h, x, y):
        u = eikonal2d_op.forward(f, h, x, y)
        ctx.save_for_backward(u, f)
        ctx.h = h
        ctx.x = x
        ctx.y = y
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, f = ctx.saved_tensors
        grad_f = eikonal2d_op.backward(grad_output, u, f, ctx.h, ctx.x, ctx.y)
        return grad_f, None, None, None


class Clamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def clamp(input, min, max):
    return Clamp.apply(input, min, max)


class Eikonal2D(torch.nn.Module):
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
        max_dvp=0.0,
        max_dvs=0.0,
        lambda_dvp=0.0,
        lambda_dvs=0.0,
        lambda_sp_ratio=0.0,
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
        self.vp0 = vp
        self.vs0 = vs
        self.dvp = torch.nn.Parameter(torch.zeros_like(vp), requires_grad=True)
        self.dvs = torch.nn.Parameter(torch.zeros_like(vs), requires_grad=True)
        self.max_dvp = max_dvp
        self.max_dvs = max_dvs
        self.lambda_dvp = lambda_dvp
        self.lambda_dvs = lambda_dvs
        self.lambda_sp_ratio = lambda_sp_ratio

        # self.smooth_kernel = torch.ones([1, 1, 5, 5], dtype=dtype) / (5 * 5)
        # self.smooth_kernel = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=dtype).view(1, 1, 3, 3)
        self.smooth_kernel = torch.tensor(
            [
                [-1, 1],
                [1, -1],
            ],
            dtype=dtype,
        ).view(1, 1, 2, 2)
        self.smooth_kernel = self.smooth_kernel / self.smooth_kernel.abs().sum()

        # set config
        nx, ny, h = config["nx"], config["ny"], config["h"]
        xgrid, ygrid = config["xgrid"], config["ygrid"]
        self.nx = nx
        self.ny = ny
        self.h = h
        self.xgrid = xgrid
        self.ygrid = ygrid

        # set subscale grid for interpolation
        # TODO: add align for scale_sub != 1
        self.scale_sub = config["scale_sub"]
        self.nx_sub = nx // self.scale_sub
        self.ny_sub = ny // self.scale_sub
        self.h_sub = h * self.scale_sub

    # def interp(self, time_table, x, y):

    #     ix0 = torch.floor((x - self.xgrid[0]) / self.h).clamp(0, self.nx - 2).long()
    #     iy0 = torch.floor((y - self.ygrid[0]) / self.h).clamp(0, self.ny - 2).long()
    #     ix1 = ix0 + 1
    #     iy1 = iy0 + 1
    #     # x = (torch.clamp(x, self.xgrid[0], self.xgrid[-1]) - self.xgrid[0]) / self.h
    #     # y = (torch.clamp(y, self.ygrid[0], self.ygrid[-1]) - self.ygrid[0]) / self.h
    #     x = (clamp(x, self.xgrid[0], self.xgrid[-1]) - self.xgrid[0]) / self.h
    #     y = (clamp(y, self.ygrid[0], self.ygrid[-1]) - self.ygrid[0]) / self.h

    #     ## https://en.wikipedia.org/wiki/Bilinear_interpolation

    #     Q00 = time_table[ix0, iy0]
    #     Q01 = time_table[ix0, iy1]
    #     Q10 = time_table[ix1, iy0]
    #     Q11 = time_table[ix1, iy1]

    #     t = (
    #         Q00 * (ix1 - x) * (iy1 - y)
    #         + Q10 * (x - ix0) * (iy1 - y)
    #         + Q01 * (ix1 - x) * (y - iy0)
    #         + Q11 * (x - ix0) * (y - iy0)
    #     )

    #     return t

    def interp(self, time_table, x, y):

        nx, ny = time_table.shape
        ix0 = torch.floor(x).clamp(0, nx - 2).long()
        iy0 = torch.floor(y).clamp(0, ny - 2).long()
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        x = clamp(x, 0, nx - 1)
        y = clamp(y, 0, ny - 1)

        ## https://en.wikipedia.org/wiki/Bilinear_interpolation

        Q00 = time_table[ix0, iy0]
        Q01 = time_table[ix0, iy1]
        Q10 = time_table[ix1, iy0]
        Q11 = time_table[ix1, iy1]

        t = (
            Q00 * (ix1 - x) * (iy1 - y)
            + Q10 * (x - ix0) * (iy1 - y)
            + Q01 * (ix1 - x) * (y - iy0)
            + Q11 * (x - ix0) * (y - iy0)
        )

        return t

    def forward(self, picks):

        loss = 0
        preds = []
        idx = []

        if self.max_dvp > 0 or self.max_dvs > 0:
            dvp = torch.tanh(self.dvp) * self.max_dvp
            dvs = torch.tanh(self.dvs) * self.max_dvs
        else:
            dvp = self.dvp
            dvs = self.dvs
        vp = self.vp0 + dvp
        vs = self.vs0 + dvs

        ## idx_sta an idx_eve are used internally to ensure continous index
        for (idx_sta_, phase_type_), picks_ in picks.groupby(["idx_sta", "phase_type"]):
            idx.append(picks_.index)
            station_loc = self.station_loc(torch.tensor(idx_sta_, dtype=torch.int64))

            idx_eve_ = torch.tensor(picks_["idx_eve"].values, dtype=torch.int64)
            event_loc = self.event_loc(idx_eve_)
            event_time = self.event_time(idx_eve_)
            obs = torch.tensor(picks_["phase_time"].values, dtype=self.dtype).squeeze()

            ## Option 1:
            # nx_sub = int(np.ceil((station_loc[0] - event_loc[:, 0]).abs().max().item() / self.h_sub)) * 2 + 1
            # ny_sub = int(np.ceil((station_loc[1] - event_loc[:, 1]).abs().max().item() / self.h_sub)) * 2 + 1
            # x0 = int(torch.round((station_loc[0] - self.xgrid[0]) / self.h).item())
            # x1 = x0 - nx_sub // 2
            # y0 = int(torch.round((station_loc[1] - self.ygrid[0]) / self.h).item())
            # y1 = y0 - ny_sub // 2
            # x1 = min(max(0, x1), self.nx - nx_sub)
            # y1 = min(max(0, y1), self.ny - ny_sub)

            ## Option 2:
            # event_loc = event_loc[
            #     ((event_loc[:, 0] - station_loc[0]).abs() < self.max_nx_sub * self.h_sub)
            #     & ((event_loc[:, 1] - station_loc[1]).abs() < self.max_ny_sub * self.h_sub)
            # ]
            selected = ((event_loc[:, 0] - station_loc[0]).abs() < self.nx_sub * self.h_sub) & (
                (event_loc[:, 1] - station_loc[1]).abs() < self.ny_sub * self.h_sub
            )
            event_loc = event_loc[selected]
            obs = obs[selected]

            nx_sub = int(
                np.ceil(
                    (
                        max(event_loc[:, 0].max().item(), station_loc[0].item())
                        - min(event_loc[:, 0].min().item(), station_loc[0].item())
                    )
                    / self.h
                )
            )
            ny_sub = int(
                np.ceil(
                    (
                        max(event_loc[:, 1].max().item(), station_loc[1].item())
                        - min(event_loc[:, 1].min().item(), station_loc[1].item())
                    )
                    / self.h
                )
            )
            x1 = int((min(event_loc[:, 0].min().item(), station_loc[0].item()) - self.xgrid[0].item()) // self.h)
            y1 = int((min(event_loc[:, 1].min().item(), station_loc[1].item()) - self.ygrid[0].item()) // self.h)

            # x1 -= 1
            # y1 -= 1
            # nx_sub += 3
            # ny_sub += 3
            nx_sub += 2
            ny_sub += 2

            # # ## DEBUG
            # nx_sub = int(27 * self.scale_sub) + 3
            # ny_sub = int(57 * self.scale_sub) + 3
            # x1 = int(torch.round((0 - self.xgrid[0]) / self.h).item())
            # y1 = int(torch.round((0 - self.ygrid[0]) / self.h).item())

            if phase_type_ == "P":
                vp_sub = vp[x1 : x1 + nx_sub, y1 : y1 + ny_sub]
                if self.h != self.h_sub:
                    vp_sub = (
                        F.interpolate(
                            vp_sub.unsqueeze(0).unsqueeze(0),
                            scale_factor=self.h / self.h_sub,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                tp2d = Eikonal2DFunction.apply(
                    1.0 / vp_sub,
                    self.h_sub,
                    (station_loc[0] - self.xgrid[0] - x1 * self.h) / self.h_sub,
                    (station_loc[1] - self.ygrid[0] - y1 * self.h) / self.h_sub,
                )

                tt = self.interp(
                    tp2d,
                    (event_loc[:, 0] - self.xgrid[0] - x1 * self.h) / self.h_sub,
                    (event_loc[:, 1] - self.ygrid[0] - y1 * self.h) / self.h_sub,
                )  # travel time
                pred = event_time.squeeze(-1) + tt  # arrival time
                # pred.append(tt.detach().numpy())
                # loss += F.mse_loss(tt, torch.tensor(picks_["travel_time"].values, dtype=self.dtype).squeeze())
                preds.append(pred.detach().numpy())
                loss += F.mse_loss(pred, obs)

            elif phase_type_ == "S":
                vs_sub = vs[x1 : x1 + nx_sub, y1 : y1 + ny_sub]
                if self.h != self.h_sub:
                    vs_sub = (
                        F.interpolate(
                            vs_sub.unsqueeze(0).unsqueeze(0),
                            scale_factor=self.h / self.h_sub,
                            mode="bilinear",
                            align_corners=False,
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                ts2d = Eikonal2DFunction.apply(
                    1.0 / vs_sub,
                    self.h_sub,
                    (station_loc[0] - self.xgrid[0] - x1 * self.h) / self.h_sub,
                    (station_loc[1] - self.ygrid[0] - y1 * self.h) / self.h_sub,
                )
                tt = self.interp(
                    ts2d,
                    (event_loc[:, 0] - self.xgrid[0] - x1 * self.h) / self.h_sub,
                    (event_loc[:, 1] - self.ygrid[0] - y1 * self.h) / self.h_sub,
                )
                # pred.append(tt.detach().numpy())
                # loss += F.mse_loss(tt, torch.tensor(picks_["travel_time"].values, dtype=self.dtype).squeeze())
                pred = event_time.squeeze(-1) + tt
                preds.append(pred.detach().numpy())
                loss += F.mse_loss(pred, obs)
        if self.lambda_dvp > 0:
            dvp_sub = dvp[x1 : x1 + nx_sub, y1 : y1 + ny_sub]
            dvp_sub = dvp_sub.unsqueeze(0).unsqueeze(0)
            if self.h != self.h_sub:
                dvp_sub = F.interpolate(dvp_sub, scale_factor=self.h / self.h_sub, mode="bilinear", align_corners=False)
            reg_dvp = F.conv2d(dvp_sub, self.smooth_kernel)
            loss += self.lambda_dvp * reg_dvp.abs().sum()

        if self.lambda_dvs > 0:
            dvs_sub = dvs[x1 : x1 + nx_sub, y1 : y1 + ny_sub]
            dvs_sub = dvs_sub.unsqueeze(0).unsqueeze(0)
            if self.h != self.h_sub:
                dvs_sub = F.interpolate(dvs_sub, scale_factor=self.h / self.h_sub, mode="bilinear", align_corners=False)
            reg_dvs = F.conv2d(dvs_sub, self.smooth_kernel)
            loss += self.lambda_dvs * reg_dvs.abs().sum()

        if self.lambda_sp_ratio > 0:
            vs_sub = vs[x1 : x1 + nx_sub, y1 : y1 + ny_sub]
            vp_sub = vp[x1 : x1 + nx_sub, y1 : y1 + ny_sub]
            vs_sub = vs_sub.unsqueeze(0).unsqueeze(0)
            vp_sub = vp_sub.unsqueeze(0).unsqueeze(0)
            if self.h != self.h_sub:
                vs_sub = F.interpolate(vs_sub, scale_factor=self.h_sub / self.h, mode="bilinear", align_corners=False)
                vp_sub = F.interpolate(vp_sub, scale_factor=self.h_sub / self.h, mode="bilinear", align_corners=False)
            sp_ratio_sub = vs_sub / vp_sub
            reg_sp_ratio = F.conv2d(sp_ratio_sub, self.smooth_kernel)
            loss += self.lambda_sp_ratio * reg_sp_ratio.abs().sum()

        pred_df = pd.DataFrame(
            {
                "index": np.concatenate(idx),
                "pred": np.concatenate(preds),
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
    h = 1.0
    eikonal_config = {"nx": nx, "ny": ny, "h": h}
    with open(f"{data_path}/config.json", "w") as f:
        json.dump(eikonal_config, f)
    xgrid = np.arange(0, nx) * h
    ygrid = np.arange(0, ny) * h
    eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid})
    num_station = 10
    num_event = 20
    stations = []
    for i in range(num_station):
        x = np.random.randint(0, nx) * h
        y = np.random.randint(0, ny) * h
        stations.append({"station_id": f"STA{i:02d}", "x_km": x, "y_km": y, "dt_s": 0.0})
    stations = pd.DataFrame(stations)
    stations["station_index"] = stations.index
    stations.to_csv(f"{data_path}/stations.csv", index=False)
    events = []
    reference_time = pd.to_datetime("2021-01-01T00:00:00.000")
    for i in range(num_event):
        x = np.random.uniform(xgrid[0], xgrid[-1])
        y = np.random.uniform(ygrid[0], ygrid[-1])
        t = i * 5
        # events.append({"event_id": i, "event_time": t, "x_km": x, "y_km": y})
        events.append({"event_id": i, "event_time": reference_time + pd.Timedelta(seconds=t), "x_km": x, "y_km": y})
    events = pd.DataFrame(events)
    events["event_index"] = events.index
    events["event_time"] = events["event_time"].apply(lambda x: x.isoformat(timespec="milliseconds"))
    events.to_csv(f"{data_path}/events.csv", index=False)
    vpvs_ratio = 1.73
    vp = torch.ones((nx, ny), dtype=torch.float64) * 6.0
    vs = vp / vpvs_ratio

    ### add anomaly
    vp[int(nx / 3) : int(2 * nx / 3), int(ny / 3) : int(2 * ny / 3)] *= 1.1
    vs[int(nx / 3) : int(2 * nx / 3), int(ny / 3) : int(2 * ny / 3)] *= 1.1

    picks = []
    for j, station in stations.iterrows():
        ix, iy = int(round(station["x_km"] / h)), int(round(station["y_km"] / h))
        tp2d = eikonal2d_op.forward(1.0 / vp, h, ix, iy).numpy()
        ts2d = eikonal2d_op.forward(1.0 / vs, h, ix, iy).numpy()
        for i, event in events.iterrows():
            if np.random.rand() < 0.5:
                tt = interp2d(tp2d, event["x_km"], event["y_km"], eikonal_config["xgrid"], eikonal_config["ygrid"], h)
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
                tt = interp2d(ts2d, event["x_km"], event["y_km"], eikonal_config["xgrid"], eikonal_config["ygrid"], h)
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
    im = ax[0].imshow(vp, cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs, cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/true2d_vp_vs.png")

    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
    ax[0, 0].plot(events["x_km"], events["y_km"], ".", label="Event")
    ax[0, 0].set_xlabel("x (km)")
    ax[0, 0].set_ylabel("y (km)")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{data_path}/station_event_2d.png")
    # %%
    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(10, 10))
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
    plt.savefig(f"{data_path}/picks_2d.png")

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
    events["idx_eve"] = np.arange(len(events))  # continuous index from 0 to num_event/num_station
    stations["idx_sta"] = np.arange(len(stations))
    picks = picks.merge(events[["event_id", "idx_eve"]], on="event_id")  ## idx_eve, and idx_sta are used internally
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
    num_event = len(events)
    num_station = len(stations)
    nx, ny, h = eikonal_config["nx"], eikonal_config["ny"], eikonal_config["h"]
    xgrid = torch.arange(0, nx, dtype=torch.float64) * h
    ygrid = torch.arange(0, ny, dtype=torch.float64) * h
    eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid})
    vp = torch.ones((nx, ny), dtype=torch.float64) * 6.0
    vs = vp / 1.73

    ## initial event location
    event_loc = events[["x_km", "y_km"]].values
    # event_loc = events[["x_km", "y_km"]].values + np.random.randn(num_event, 2) * 10
    # event_loc = events[["x_km", "y_km"]].values * 0.0 + stations[["x_km", "y_km"]].values.mean(axis=0)

    eikonal2d = Eikonal2D(
        num_event,
        num_station,
        stations[["x_km", "y_km"]].values,
        stations[["dt_s"]].values,
        # events[["x_km", "y_km"]].values,
        event_loc,
        events[["event_time"]].values,
        vp,
        vs,
        # max_dvp=0.0,
        # max_dvs=0.0,
        # lambda_vp=1.0,
        # lambda_vs=1.0,
        config=eikonal_config,
    )
    preds, loss = eikonal2d(picks)

    ######################################### Optimize #########################################
    # %%
    vp = eikonal2d.vp0.detach().numpy() + eikonal2d.dvp.detach().numpy()
    vs = eikonal2d.vs0.detach().numpy() + eikonal2d.dvs.detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp, cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs, cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial2d_vp_vs.png")

    eikonal2d.dvp.requires_grad = True
    eikonal2d.dvs.requires_grad = True
    eikonal2d.event_loc.weight.requires_grad = False
    eikonal2d.event_time.weight.requires_grad = False
    print(
        "Optimizing parameters:\n"
        + "\n".join([f"{name}: {param.size()}" for name, param in eikonal2d.named_parameters() if param.requires_grad]),
    )

    parameters = [param for param in eikonal2d.parameters() if param.requires_grad]
    optimizer = optim.LBFGS(params=parameters, max_iter=1000, line_search_fn="strong_wolfe")
    print("Initial loss:", loss.item())

    def closure():
        optimizer.zero_grad()
        _, loss = eikonal2d(picks)
        loss.backward()
        return loss

    optimizer.step(closure)

    preds, loss = eikonal2d(picks)
    print("Final loss:", loss.item())

    vp = eikonal2d.vp0.detach().numpy() + eikonal2d.dvp.detach().numpy()
    vs = eikonal2d.vs0.detach().numpy() + eikonal2d.dvs.detach().numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp, cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs, cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/inverted2d_vp_vs.png")

    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    # ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
    ax[0, 0].plot(events["x_km"], events["y_km"], ".", label="True Events")
    ax[0, 0].plot(event_loc[:, 0], event_loc[:, 1], "x", label="Initial Events")
    for i in range(len(event_loc)):
        ax[0, 0].plot(
            [events["x_km"].iloc[i], event_loc[i, 0]], [events["y_km"].iloc[i], event_loc[i, 1]], "k--", alpha=0.5
        )
    event_loc = eikonal2d.event_loc.weight.detach().numpy()
    ax[0, 0].plot(event_loc[:, 0], event_loc[:, 1], "x", label="Inverted Events")
    for i in range(len(event_loc)):
        ax[0, 0].plot(
            [events["x_km"].iloc[i], event_loc[i, 0]], [events["y_km"].iloc[i], event_loc[i, 1]], "r--", alpha=0.5
        )
    ax[0, 0].set_xlabel("x (km)")
    ax[0, 0].set_ylabel("y (km)")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{data_path}/inverted2d_station_event.png")
