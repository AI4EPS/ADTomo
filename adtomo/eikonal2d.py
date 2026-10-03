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


VPVS_RATIO_LOWER_BOUND = np.sqrt(4.0 / 3.0)  # Vp/Vs at zero bulk modulus (Poisson's ratio -1); no material is below it

## Expected variation of each inverted parameter; its penalties are divided by it. Vp/Vs varies about 4x less than
## velocity, i.e. 0.25 in log(Vp/Vs), which is about 0.75 in its parameter log(Vp/Vs - bound) near Vp/Vs = 1.73.
SCALE = {"vp": 1.0, "vs": 1.0, "vpvs_ratio": 0.75}


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
        nx, ny, h = config["nx"], config["ny"], config["h"]
        self.nx, self.ny, self.h = nx, ny, h
        self.xgrid, self.ygrid = config["xgrid"], config["ygrid"]

        ## inverted variables: (vp, vs), or (vs, vpvs_ratio) when vpvs_ratio is given (vp input is then ignored);
        ## each may be a scalar, 1D along axis 1 (ny), or 2D (nx, ny), and is expanded to 2D in velocity()
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
        ## smoothing kernels by parameter dimension; default: first difference per km along each axis (2D: x, then z)
        d = torch.tensor([-1.0, 1.0], dtype=dtype) / h
        self.smooth_kernels = {1: [d], 2: [d.view(2, 1), d.view(1, 2)]} | (smooth_kernels or {})

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

    def velocity(self):
        """Return the 2D (vp, vs) fields from the inversion parameters."""
        m = {name: (self.offset[name] + torch.exp(x)).expand(self.nx, self.ny) for name, x in self.params.items()}
        vp = m["vpvs_ratio"] * m["vs"] if "vpvs_ratio" in m else m["vp"]
        return vp, m["vs"]

    def forward(self, picks):
        loss = torch.tensor(0.0, dtype=self.dtype)
        preds = []
        idx = []
        residuals = []

        vp, vs = self.velocity()

        ## idx_sta an idx_eve are used internally to ensure continous index
        for (idx_sta_, phase_type_), picks_ in picks.groupby(["idx_sta", "phase_type"]):
            station_loc = self.station_loc(torch.tensor(idx_sta_, dtype=torch.int64))

            idx_eve_ = torch.tensor(picks_["idx_eve"].values, dtype=torch.int64)
            event_loc = self.event_loc(idx_eve_)
            event_time = self.event_time(idx_eve_)
            obs = torch.tensor(picks_["phase_time"].values, dtype=self.dtype).squeeze()

            if not (
                (station_loc[0] > self.xgrid[0])
                and (station_loc[0] < self.xgrid[-1])
                and (station_loc[1] > self.ygrid[0])
                and (station_loc[1] < self.ygrid[-1])
            ):
                continue
            selected = (
                (event_loc[:, 0] > self.xgrid[0])
                & (event_loc[:, 0] < self.xgrid[-1])
                & (event_loc[:, 1] > self.ygrid[0])
                & (event_loc[:, 1] < self.ygrid[-1])
            )
            event_time = event_time[selected]
            event_loc = event_loc[selected]
            obs = obs[selected]
            if len(obs) == 0:
                continue

            ## Keep the original index
            idx.append(picks_.index.values[selected])
            v = vp if phase_type_ == "P" else vs
            tt2d = Eikonal2DFunction.apply(
                1.0 / v,
                self.h,
                (station_loc[0] - self.xgrid[0]) / self.h,
                (station_loc[1] - self.ygrid[0]) / self.h,
            )
            tt = self.interp(
                tt2d,
                (event_loc[:, 0] - self.xgrid[0]) / self.h,
                (event_loc[:, 1] - self.ygrid[0]) / self.h,
            )  # travel time
            pred = event_time.squeeze(-1) + tt  # arrival time
            preds.append(pred.detach().numpy())
            residuals.append(pred - obs)

        ## data misfit: mean squared residual over all picks, in units of the pick uncertainty
        loss += (torch.cat(residuals) / self.sigma_t).pow(2).mean()

        ## regularization on each parameter at its own shape
        eps = 1e-3
        for name, x in self.params.items():
            if self.lambda_smooth > 0 and x.dim() > 0:
                conv = F.conv1d if x.dim() == 1 else F.conv2d
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
    vp, vs = [v.detach().numpy() for v in eikonal2d.velocity()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(vp, cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(vs, cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial2d_vp_vs.png")

    eikonal2d.params["vp"].requires_grad = True
    eikonal2d.params["vs"].requires_grad = True
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

    vp, vs = [v.detach().numpy() for v in eikonal2d.velocity()]
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
