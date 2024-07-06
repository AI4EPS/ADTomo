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

    ir0 = np.floor((r - rgrid[0]) / h).clip(0, nr - 2).astype(np.int64)
    iz0 = np.floor((z - zgrid[0]) / h).clip(0, nz - 2).astype(np.int64)
    r = np.clip(r, rgrid[0], rgrid[-1])
    z = np.clip(z, zgrid[0], zgrid[-1])
    ir1 = ir0 + 1
    iz1 = iz0 + 1

    ## https://en.wikipedia.org/wiki/Bilinear_interpolation
    r0 = ir0 * h + rgrid[0]
    r1 = ir1 * h + rgrid[0]
    z0 = iz0 * h + zgrid[0]
    z1 = iz1 * h + zgrid[0]

    Q00 = time_table[ir0, iz0]
    Q01 = time_table[ir0, iz1]
    Q10 = time_table[ir1, iz0]
    Q11 = time_table[ir1, iz1]

    t = (
        1.0
        / (r1 - r0)
        / (z1 - z0)
        * (
            Q00 * (r1 - r) * (z1 - z)
            + Q10 * (r - r0) * (z1 - z)
            + Q01 * (r1 - r) * (z - z0)
            + Q11 * (r - r0) * (z - z0)
        )
    )

    return t


class Eikonal2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, h, ix, iy):
        u = eikonal2d_op.forward(f, h, ix, iy)
        ctx.save_for_backward(u, f)
        ctx.h = h
        ctx.ix = ix
        ctx.iy = iy
        return u

    @staticmethod
    def backward(ctx, grad_output):
        u, f = ctx.saved_tensors
        grad_f = eikonal2d_op.backward(grad_output, u, f, ctx.h, ctx.ix, ctx.iy)
        return grad_f, None, None, None


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
        nx, ny, h = config["nx"], config["ny"], config["h"]
        self.nx = nx
        self.ny = ny
        self.h = h
        self.xgrid = torch.arange(0, nx, dtype=dtype) * h
        self.ygrid = torch.arange(0, ny, dtype=dtype) * h

        # check

    # def calc_time(self, event_loc, station_loc, type):

    #     for event_xyz, station_xyz in zip(event_loc, station_loc):
    #         print(event_xyz, station_xyz)

    #         break
    #     return Eikonal2DFunction.apply(self.vp, 1.0, 0, 0)

    def interp(self, time_table, x, y):
        nx = len(self.xgrid)
        ny = len(self.ygrid)
        assert time_table.shape == (nx, ny)

        ir0 = torch.floor((x - self.xgrid[0]) / self.h).clamp(0, nx - 2).long()
        iz0 = torch.floor((y - self.ygrid[0]) / self.h).clamp(0, ny - 2).long()
        x = torch.clamp(x, self.xgrid[0], self.xgrid[-1])
        y = torch.clamp(y, self.ygrid[0], self.ygrid[-1])
        ir1 = ir0 + 1
        iz1 = iz0 + 1

        ## https://en.wikipedia.org/wiki/Bilinear_interpolation
        r0 = ir0 * self.h + self.xgrid[0]
        r1 = ir1 * self.h + self.xgrid[0]
        z0 = iz0 * self.h + self.ygrid[0]
        z1 = iz1 * self.h + self.ygrid[0]

        Q00 = time_table[ir0, iz0]
        Q01 = time_table[ir0, iz1]
        Q10 = time_table[ir1, iz0]
        Q11 = time_table[ir1, iz1]

        t = (
            1.0
            / (r1 - r0)
            / (z1 - z0)
            * (
                Q00 * (r1 - x) * (z1 - y)
                + Q10 * (x - r0) * (z1 - y)
                + Q01 * (r1 - x) * (y - z0)
                + Q11 * (x - r0) * (y - z0)
            )
        )

        return t

    def forward(self, picks):

        # %%
        loss = 0
        pred = []
        idx = []
        for (station_index_, phase_type_), picks_ in picks.groupby(["station_index", "phase_type"]):
            idx.append(picks_.index)
            station_loc = self.station_loc(torch.tensor(station_index_, dtype=torch.int64))
            ix = int(round(station_loc[0].item() / self.h))
            iy = int(round(station_loc[1].item() / self.h))

            event_index_ = torch.tensor(picks_["event_index"].values, dtype=torch.int64)
            event_loc = self.event_loc(event_index_)
            event_time = self.event_time(event_index_)

            if phase_type_ == "P":
                tp2d = Eikonal2DFunction.apply(1.0 / self.vp, self.h, ix, iy)
                tt = self.interp(tp2d, event_loc[:, 0], event_loc[:, 1]).squeeze()  # travel time
                at = event_time.squeeze() + tt  # arrival time
                # pred.append(tt.detach().numpy())
                # loss += F.mse_loss(tt, torch.tensor(picks_["travel_time"].values, dtype=self.dtype).squeeze())
                pred.append(at.detach().numpy())
                loss += F.mse_loss(at, torch.tensor(picks_["phase_time"].values, dtype=self.dtype).squeeze())

            elif phase_type_ == "S":
                ts2d = Eikonal2DFunction.apply(1.0 / self.vs, self.h, ix, iy)
                tt = self.interp(ts2d, event_loc[:, 0], event_loc[:, 1]).squeeze()
                # pred.append(tt.detach().numpy())
                # loss += F.mse_loss(tt, torch.tensor(picks_["travel_time"].values, dtype=self.dtype).squeeze())
                at = event_time.squeeze() + tt
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

    ######################################## Create Synthetic Data #########################################
    np.random.seed(0)
    data_path = "data"
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    nx = 10
    ny = 10
    h = 1
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
    for i in range(num_event):
        x = np.random.uniform(0, nx * h)
        y = np.random.uniform(0, ny * h)
        # t = np.random.rand()
        t = i * 5
        events.append({"event_id": i, "event_time": t, "x_km": x, "y_km": y})
    events = pd.DataFrame(events)
    events["event_index"] = events.index
    events.to_csv(f"{data_path}/events.csv", index=False)
    vpvs_ratio = 1.73
    vp = torch.ones((nx, ny), dtype=torch.float64) * 6.0
    vs = vp / vpvs_ratio

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
                        "phase_time": event["event_time"] + tt,
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
                        "phase_time": event["event_time"] + tt,
                        "travel_time": tt,
                    }
                )
    picks = pd.DataFrame(picks)
    # use picks,  stations.index, events.index to set station_index and
    picks["event_index"] = picks["event_id"].map(events.set_index("event_id")["event_index"])
    picks["station_index"] = picks["station_id"].map(stations.set_index("station_id")["station_index"])
    picks.to_csv(f"{data_path}/picks.csv", index=False)
    # %%
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(5, 5))
    ax[0, 0].plot(stations["x_km"], stations["y_km"], "^", label="Station")
    ax[0, 0].plot(events["x_km"], events["y_km"], ".", label="Event")
    ax[0, 0].set_xlabel("x (km)")
    ax[0, 0].set_ylabel("y (km)")
    ax[0, 0].legend()
    ax[0, 0].set_title("Station and Event Locations")
    plt.savefig(f"{data_path}/station_event.png")
    # %%
    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(10, 10))
    picks = picks.merge(stations, on="station_id")
    mapping_color = lambda x: f"C{int(x)}"
    ax[0, 0].scatter(picks["phase_time"], picks["x_km"], c=picks["event_index"].apply(mapping_color))
    ax[0, 0].scatter(events["event_time"], events["x_km"], c=events["event_index"].apply(mapping_color), marker="x")
    ax[0, 0].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("x (km)")
    ax[1, 0].scatter(picks["phase_time"], picks["y_km"], c=picks["event_index"].apply(mapping_color))
    ax[1, 0].scatter(events["event_time"], events["y_km"], c=events["event_index"].apply(mapping_color), marker="x")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("y (km)")
    plt.savefig(f"{data_path}/picks.png")
    # %%
    ######################################### Load Synthetic Data #########################################
    data_path = "data"
    events = pd.read_csv(f"{data_path}/events.csv")
    stations = pd.read_csv(f"{data_path}/stations.csv")
    picks = pd.read_csv(f"{data_path}/picks.csv")
    with open(f"{data_path}/config.json", "r") as f:
        eikonal_config = json.load(f)
    events = events.sort_values("event_index").set_index("event_index")
    stations = stations.sort_values("station_index").set_index("station_index")
    num_event = len(events)
    num_station = len(stations)
    nx, ny, h = eikonal_config["nx"], eikonal_config["ny"], eikonal_config["h"]
    xgrid = torch.arange(0, nx, dtype=torch.float64) * h
    ygrid = torch.arange(0, ny, dtype=torch.float64) * h
    eikonal_config.update({"xgrid": xgrid, "ygrid": ygrid})
    vp = torch.ones((nx, ny), dtype=torch.float64) * 6.0
    vs = vp / 1.73

    eikonal2d = Eikonal2D(
        num_event,
        num_station,
        stations[["x_km", "y_km"]].values,
        stations[["dt_s"]].values,
        events[["x_km", "y_km"]].values,
        events[["event_time"]].values,
        vp,
        vs,
        eikonal_config,
    )
    preds, loss = eikonal2d(picks)

    ######################################### Optimize #########################################
    # %%
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(eikonal2d.vp.detach().numpy(), cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(eikonal2d.vs.detach().numpy(), cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/initial_vp_vs.png")

    optimizer = optim.LBFGS(params=eikonal2d.parameters(), max_iter=1000, line_search_fn="strong_wolfe")
    print("Initial loss:", loss.item())

    def closure():
        optimizer.zero_grad()
        _, loss = eikonal2d(picks)
        loss.backward()
        return loss

    optimizer.step(closure)

    preds, loss = eikonal2d(picks)
    print("Final loss:", loss.item())

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    im = ax[0].imshow(eikonal2d.vp.detach().numpy(), cmap="viridis")
    fig.colorbar(im, ax=ax[0])
    ax[0].set_title("Vp")
    im = ax[1].imshow(eikonal2d.vs.detach().numpy(), cmap="viridis")
    fig.colorbar(im, ax=ax[1])
    ax[1].set_title("Vs")
    plt.savefig(f"{data_path}/inverted_vp_vs.png")
