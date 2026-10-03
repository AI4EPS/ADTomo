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


class Eikonal3DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, h, x, y, z):
        u = eikonal3d_op.forward(f, h, x, y, z)
        ctx.save_for_backward(u, f)
        ctx.h = h
        ctx.x = x
        ctx.y = y
        ctx.z = z
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, f = ctx.saved_tensors
        grad_f = eikonal3d_op.backward(grad_output, u, f, ctx.h, ctx.x, ctx.y, ctx.z)
        return grad_f, None, None, None, None


VPVS_RATIO_LOWER_BOUND = np.sqrt(4.0 / 3.0)  # Vp/Vs at zero bulk modulus (Poisson's ratio -1); no material is below it

## Expected variation of each inverted parameter; its penalties are divided by it. Vp/Vs varies about 4x less than
## velocity, i.e. 0.25 in log(Vp/Vs), which is about 0.75 in its parameter log(Vp/Vs - bound) near Vp/Vs = 1.73.
SCALE = {"vp": 1.0, "vs": 1.0, "vpvs_ratio": 0.75}


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
        vpvs_ratio=None,
        sigma_t=1.0,
        lambda_smooth=0.0,
        lambda_damp=0.0,
        lambda_monotonic=0.0,
        smooth_kernels=None,
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
        # set config
        nx, ny, nz, h = config["nx"], config["ny"], config["nz"], config["h"]
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.h = h
        self.xgrid = torch.arange(0, nx, dtype=dtype) * h
        self.ygrid = torch.arange(0, ny, dtype=dtype) * h
        self.zgrid = torch.arange(0, nz, dtype=dtype) * h

        ## inverted variables: (vp, vs), or (vs, vpvs_ratio) when vpvs_ratio is given (vp input is then ignored);
        ## each may be a scalar, 1D along depth (nz), or 3D (nx, ny, nz), and is expanded to 3D in velocity()
        ## Parameters are log velocity and log(Vp/Vs - bound): differences are relative changes, so every penalty is
        ## dimensionless, velocity stays positive and Vp/Vs stays above its physical bound.
        values = {"vs": vs, "vpvs_ratio": vpvs_ratio} if vpvs_ratio is not None else {"vp": vp, "vs": vs}
        self.offset = {"vp": 0.0, "vs": 0.0, "vpvs_ratio": VPVS_RATIO_LOWER_BOUND}
        self.init = {name: torch.log(torch.as_tensor(value, dtype=dtype) - self.offset[name]) for name, value in values.items()}
        self.params = nn.ParameterDict({name: nn.Parameter(value.clone()) for name, value in self.init.items()})

        ## loss = mean((residual / sigma_t)^2)
        ##      + sum over parameters m of [lambda_smooth * mean|kernels * m| + lambda_damp * mean|m - m_init|] / SCALE
        ##      + lambda_monotonic * mean relu(velocity decrease with depth), for 1D velocity
        ## where |.| and relu are smoothed within eps = 1e-3 of zero (smooth_l1_loss, softplus), so LBFGS sees a smooth loss
        self.sigma_t = sigma_t  # pick uncertainty (s)
        self.lambda_smooth = lambda_smooth
        self.lambda_damp = lambda_damp
        self.lambda_monotonic = lambda_monotonic
        ## smoothing kernels by parameter dimension; default: first difference per km along each axis (3D: x, y, then z)
        d = torch.tensor([-1.0, 1.0], dtype=dtype) / h
        self.smooth_kernels = {1: [d], 3: [d.view(2, 1, 1), d.view(1, 2, 1), d.view(1, 1, 2)]} | (smooth_kernels or {})

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

    def velocity(self):
        """Return the 3D (vp, vs) fields from the inversion parameters."""
        m = {name: (self.offset[name] + torch.exp(x)).expand(self.nx, self.ny, self.nz) for name, x in self.params.items()}
        vp = m["vpvs_ratio"] * m["vs"] if "vpvs_ratio" in m else m["vp"]
        return vp, m["vs"]

    def forward(self, picks):
        # %%
        loss = torch.tensor(0.0, dtype=self.dtype)
        pred = []
        idx = []
        residuals = []
        vp, vs = self.velocity()

        ## idx_sta an idx_eve are used internally to ensure continous index
        # for (station_index_, phase_type_), picks_ in picks.groupby(["station_index", "phase_type"]):
        for (idx_sta_, phase_type_), picks_ in picks.groupby(["idx_sta", "phase_type"]):
            idx.append(picks_.index)
            station_loc = self.station_loc(torch.tensor(idx_sta_, dtype=torch.int64))

            idx_eve_ = torch.tensor(picks_["idx_eve"].values, dtype=torch.int64)
            event_loc = self.event_loc(idx_eve_)
            event_time = self.event_time(idx_eve_)

            v = vp if phase_type_ == "P" else vs
            tt3d = Eikonal3DFunction.apply(
                1.0 / v,
                self.h,
                station_loc[0] / self.h,
                station_loc[1] / self.h,
                station_loc[2] / self.h,
            )
            tt = self.interp(tt3d, event_loc[:, 0], event_loc[:, 1], event_loc[:, 2])  # travel time
            at = event_time.squeeze(-1) + tt  # arrival time
            pred.append(at.detach().numpy())
            residuals.append(at - torch.tensor(picks_["phase_time"].values, dtype=self.dtype))

        ## data misfit: mean squared residual over all picks, in units of the pick uncertainty
        loss += (torch.cat(residuals) / self.sigma_t).pow(2).mean()

        ## regularization on each parameter at its own shape
        eps = 1e-3
        for name, x in self.params.items():
            if self.lambda_smooth > 0 and x.dim() > 0:
                conv = F.conv1d if x.dim() == 1 else F.conv3d
                r = torch.cat([conv(x[None, None], k[None, None]).flatten() for k in self.smooth_kernels[x.dim()]])
                loss += self.lambda_smooth * F.smooth_l1_loss(r, torch.zeros_like(r), beta=eps) / SCALE[name]
            if self.lambda_damp > 0:
                loss += self.lambda_damp * F.smooth_l1_loss(x, self.init[name], beta=eps) / SCALE[name]
            if self.lambda_monotonic > 0 and name in ("vp", "vs") and x.dim() == 1:
                d = x[:-1] - x[1:]  # positive where velocity decreases with depth
                loss += self.lambda_monotonic * F.softplus(d, beta=1 / eps).mean()

        pred_df = pd.DataFrame(
            {
                "index": np.concatenate(idx),
                "pred": np.concatenate(pred),
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
    events["idx_eve"] = np.arange(len(events))  # continuous index from 0 to num_event/num_station
    stations["idx_sta"] = np.arange(len(stations))
    picks = picks.merge(events[["event_id", "idx_eve"]], on="event_id")  ## idx_eve, and idx_sta are used internally
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
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
        # max_dvp=0.0,
        # max_dvs=0.0,
        # lambda_vp=1.0,
        # lambda_vs=1.0,
        config=eikonal_config,
    )
    preds, loss = eikonal3d(picks)

    ######################################### Optimize #########################################
    # %%
    vp, vs = [v.detach().numpy() for v in eikonal3d.velocity()]
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

    eikonal3d.params["vp"].requires_grad = True
    eikonal3d.params["vs"].requires_grad = True
    eikonal3d.event_loc.weight.requires_grad = False
    eikonal3d.event_time.weight.requires_grad = False
    print(
        "Optimizing parameters:\n"
        + "\n".join([f"{name}: {param.size()}" for name, param in eikonal3d.named_parameters() if param.requires_grad]),
    )

    parameters = [param for param in eikonal3d.parameters() if param.requires_grad]
    optimizer = optim.LBFGS(params=parameters, max_iter=1000, line_search_fn="strong_wolfe")
    print("Initial loss:", loss.item())

    def closure():
        optimizer.zero_grad()
        _, loss = eikonal3d(picks)
        loss.backward()
        return loss

    optimizer.step(closure)

    preds, loss = eikonal3d(picks)
    print("Final loss:", loss.item())

    vp, vs = [v.detach().numpy() for v in eikonal3d.velocity()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, :, nz // 2], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/inversed3d_vp_vs_xy.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[:, ny // 2, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/inversed3d_vp_vs_xz.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs[nx // 2, :, :], cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/inversed3d_vp_vs_yz.png")

    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    ax[0, 0].plot(events["x_km"], events["y_km"], ".", label="True Events")
    ax[0, 0].plot(event_loc[:, 0], event_loc[:, 1], "x", label="Initial Events")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{data_path}/inversed3d_station_event_xy.png")

    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    ax[0, 0].plot(events["x_km"], events["z_km"], ".", label="True Events")
    ax[0, 0].plot(event_loc[:, 0], event_loc[:, 2], "x", label="Initial Events")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{data_path}/inversed3d_station_event_xz.png")

    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    ax[0, 0].plot(events["y_km"], events["z_km"], ".", label="True Events")
    ax[0, 0].plot(event_loc[:, 1], event_loc[:, 2], "x", label="Initial Events")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{data_path}/inversed3d_station_event_yz.png")
