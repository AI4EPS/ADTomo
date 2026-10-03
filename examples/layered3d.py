# %%
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from adtomo.eikonal3d import Eikonal3D, eikonal3d_op, interp3d

# %%
if __name__ == "__main__":

    # %%
    ######################################## Settings #########################################
    ## inverted variables, each with its own shape:
    ##   1d: Vs(z) + one Vp/Vs;  3d: Vs(x, y, z) + one Vp/Vs;  3d1d: Vs(x, y, z) + Vp/Vs(z);  3d3d: Vs(x, y, z) + Vp/Vs(x, y, z)
    model = os.environ.get("MODEL", "1d")
    noise = float(os.environ.get("NOISE", 0.02))  # Gaussian pick noise (s)
    ## smoothing weight: 0 without noise; with noise, the 2D example's values as a start
    lambda_smooth = float(os.environ.get("LAMBDA", 0.0 if noise == 0 else 10.0 if model == "1d" else 100.0))
    tag = f"{model}_{'noisy' if noise > 0 else 'clean'}"
    result_path = "results/layered3d"
    figure_path = "figures/layered3d"
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)

    # %%
    ######################################## Layered Model #########################################
    np.random.seed(0)
    nx, ny, nz, h = 16, 16, 16, 1.0  # grid nodes and spacing (km); z is depth, positive down
    xgrid = np.arange(nx) * h
    ygrid = np.arange(ny) * h
    zgrid = np.arange(nz) * h
    vpvs_ratio = 1.73
    layer_top = np.array([0.0, 4.0, 8.0, 12.0])  # km
    layer_vp = np.array([5.0, 5.8, 6.4, 7.0])  # km/s
    vp_true = np.tile(layer_vp[np.searchsorted(layer_top, zgrid, side="right") - 1], (nx, ny, 1))  # (nx, ny, nz)
    vs_true = vp_true / vpvs_ratio
    vpvs_ratio0 = 1.80  # homogeneous starting model: Vp 6.0 km/s, Vp/Vs 1.80
    vs0 = 6.0 / vpvs_ratio0

    # %%
    ######################################## Synthetic Data #########################################
    num_station = 16
    num_event = 50
    ## stations on the surface with one grid cell of topography; their own random generator keeps the events fixed
    station_xy = np.random.default_rng(3).uniform(xgrid[1], xgrid[-2], (num_station, 2))
    length = xgrid[-1] - xgrid[0]
    stations = pd.DataFrame(
        {
            "station_id": [f"STA{i:02d}" for i in range(num_station)],
            "x_km": station_xy[:, 0],
            "y_km": station_xy[:, 1],
            "z_km": 0.5 * h * (1.0 + np.sin(2 * np.pi * station_xy[:, 0] / length) * np.sin(2 * np.pi * station_xy[:, 1] / length)),
            "dt_s": 0.0,
        }
    )
    events = pd.DataFrame(
        {
            "event_id": np.arange(num_event),
            "x_km": np.random.uniform(xgrid[1], xgrid[-2], num_event),
            "y_km": np.random.uniform(ygrid[1], ygrid[-2], num_event),
            "z_km": np.random.uniform(3.0, zgrid[-2], num_event),
            "event_time": 0.0,  # travel time == arrival time
        }
    )

    tables = []
    for _, station in stations.iterrows():
        for phase_type, v in [("P", vp_true), ("S", vs_true)]:
            tt3d = eikonal3d_op.forward(
                torch.from_numpy(1.0 / v), h, station["x_km"] / h, station["y_km"] / h, station["z_km"] / h
            ).numpy()
            tt = interp3d(tt3d, events["x_km"].values, events["y_km"].values, events["z_km"].values, xgrid, ygrid, zgrid, h)
            tables.append(
                pd.DataFrame(
                    {"event_id": events["event_id"], "station_id": station["station_id"], "phase_type": phase_type, "phase_time": tt}
                )
            )
    picks = pd.concat(tables, ignore_index=True)
    picks["phase_time"] += np.random.default_rng(1).normal(0.0, noise, len(picks))

    events["idx_eve"] = np.arange(num_event)
    stations["idx_sta"] = np.arange(num_station)
    picks = picks.merge(events[["event_id", "idx_eve"]], on="event_id")
    picks = picks.merge(stations[["station_id", "idx_sta"]], on="station_id")
    print(f"{num_station} stations, {num_event} events, {len(picks)} picks; {model}, noise {noise} s, lambda {lambda_smooth}")

    # %%
    ######################################## Inversion #########################################
    config = {"nx": nx, "ny": ny, "nz": nz, "h": h}
    vs_shape, ratio_shape = {
        "1d": (nz, ()),
        "3d": ((nx, ny, nz), ()),
        "3d1d": ((nx, ny, nz), nz),
        "3d3d": ((nx, ny, nz), (nx, ny, nz)),
    }[model]
    sigma_t = max(noise, 1e-3)  # pick uncertainty; noise-free data get a nominal 1 ms
    eikonal3d = Eikonal3D(
        num_event,
        num_station,
        stations[["x_km", "y_km", "z_km"]].values,
        stations[["dt_s"]].values,
        events[["x_km", "y_km", "z_km"]].values,  # true locations, held fixed
        events[["event_time"]].values,
        None,  # vp is vpvs_ratio * vs
        np.full(vs_shape, vs0),
        vpvs_ratio=np.full(ratio_shape, vpvs_ratio0),
        sigma_t=sigma_t,
        lambda_smooth=lambda_smooth,
        config=config,
    )
    eikonal3d.event_loc.weight.requires_grad = False
    eikonal3d.event_time.weight.requires_grad = False

    parameters = [param for param in eikonal3d.parameters() if param.requires_grad]
    optimizer = optim.LBFGS(params=parameters, max_iter=5000, line_search_fn="strong_wolfe")
    history = {"loss": [], "misfit": []}  # misfit: mean (residual / sigma_t)^2, the data term of the loss

    def closure():
        optimizer.zero_grad()
        preds, loss = eikonal3d(picks)
        loss.backward()
        history["loss"].append(loss.item())
        history["misfit"].append(np.mean(((preds["pred"].values - picks["phase_time"].values) / sigma_t) ** 2))
        return loss

    t0 = time.time()
    optimizer.step(closure)
    print(f"Inversion: {time.time() - t0:.1f}s, {len(history['loss'])} evaluations, data misfit {history['misfit'][-1]:.3g}")

    # %%
    ######################################## Results #########################################
    vp_inv, vs_inv = [v.detach().numpy() for v in eikonal3d.velocity()]
    print(f"Vp/Vs: true {vpvs_ratio}, inverted {(vp_inv / vs_inv).min():.4f} to {(vp_inv / vs_inv).max():.4f}")
    print(f"Vs RMS error: {np.sqrt(np.mean((vs_inv - vs_true) ** 2)):.3f} km/s")
    np.savez(f"{result_path}/{tag}.npz", vp_true=vp_true, vs_true=vs_true, vp_inv=vp_inv, vs_inv=vs_inv)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    ax.semilogy(history["loss"], "k", label="Total loss")
    ax.semilogy(history["misfit"], label="Data misfit, mean (r/σ)²")
    if lambda_smooth > 0:
        ax.semilogy(np.array(history["loss"]) - np.array(history["misfit"]), label="Regularization")
    if noise > 0:
        ax.axhline(1.0, color="gray", linestyle="--", label="Noise level")
    ax.set_xlabel("Function evaluation")
    ax.set_title(f"{tag}, lambda = {lambda_smooth:g}")
    ax.legend()
    plt.savefig(f"{figure_path}/{tag}_loss.png", bbox_inches="tight", dpi=150)

    # %%
    ## one layout per case: rows of (variable, panel) pairs; a single Vp/Vs value is reported in the title
    ## panels: x-z section through the middle in y (true, inverted), map view at depth z_map, depth profile
    z_map = 6.0  # km, inside the second layer
    iy, iz = ny // 2, int(round(z_map / h))
    layouts = {
        "1d": [[("vs", "profile")]],
        "3d": [[("vs", "true"), ("vs", "inverted"), ("vs", "map"), ("vs", "profile")]],
        "3d1d": [[("vs", "true"), ("vs", "inverted"), ("vs", "map"), ("vs", "profile"), ("vpvs_ratio", "profile")]],
        "3d3d": [
            [("vs", "true"), ("vs", "inverted"), ("vs", "map"), ("vs", "profile")],
            [("vpvs_ratio", "true"), ("vpvs_ratio", "inverted"), ("vpvs_ratio", "map"), ("vpvs_ratio", "profile")],
        ],
    }
    inverted = {name: (eikonal3d.offset[name] + torch.exp(x)).detach().numpy() for name, x in eikonal3d.params.items()}
    truth = {"vs": vs_true, "vpvs_ratio": vp_true / vs_true}
    initial = {"vs": vs0, "vpvs_ratio": vpvs_ratio0}
    label = {"vs": "Vs (km/s)", "vpvs_ratio": "Vp/Vs"}
    extent_xz = [xgrid[0] - h / 2, xgrid[-1] + h / 2, zgrid[-1] + h / 2, zgrid[0] - h / 2]  # cells centered on nodes
    extent_xy = [xgrid[0] - h / 2, xgrid[-1] + h / 2, ygrid[-1] + h / 2, ygrid[0] - h / 2]

    layout = layouts[model]
    width = [1.3 if panel != "profile" else 0.8 for _, panel in max(layout, key=len)]
    fig, axes = plt.subplots(len(layout), len(width), figsize=(4 * sum(width), 4.5 * len(layout)), squeeze=False,
                             layout="constrained", gridspec_kw={"width_ratios": width})
    title = f"{tag}, lambda = {lambda_smooth:g}"
    if inverted["vpvs_ratio"].ndim == 0:
        title += f"; Vp/Vs {inverted['vpvs_ratio']:.4f} (true {vpvs_ratio}, initial {vpvs_ratio0})"
    fig.suptitle(title)
    for row, ax_row in zip(layout, axes):
        for (name, panel), ax in zip(row, ax_row):
            v = inverted[name]
            vmin = min(truth[name].min(), v.min(), initial[name])
            vmax = max(truth[name].max(), v.max(), initial[name])
            if panel in ("true", "inverted"):
                field = truth[name] if panel == "true" else v
                im = ax.imshow(field[:, iy, :].T, cmap="jet_r", vmin=vmin, vmax=vmax, origin="upper", extent=extent_xz)
                ax.plot(events["x_km"], events["z_km"], "k.", markersize=3)
                ax.plot(stations["x_km"], stations["z_km"], "k^", markersize=6)
                ax.set_xlabel("x (km)")
                ax.set_ylabel("Depth (km)")
                ax.set_title(f"{panel.capitalize()} {label[name]}, y = {ygrid[iy]:g} km")
                fig.colorbar(im, ax=ax, shrink=0.8)
            elif panel == "map":
                im = ax.imshow(v[:, :, iz].T, cmap="jet_r", vmin=vmin, vmax=vmax, origin="upper", extent=extent_xy)
                ax.plot(events["x_km"], events["y_km"], "k.", markersize=3)
                ax.plot(stations["x_km"], stations["y_km"], "k^", markersize=6)
                ax.set_xlabel("x (km)")
                ax.set_ylabel("y (km)")
                ax.set_title(f"Inverted {label[name]}, z = {zgrid[iz]:g} km")
                fig.colorbar(im, ax=ax, shrink=0.8)
            else:
                ax.plot(np.repeat(truth[name][0, 0], 2), np.stack([zgrid - h / 2, zgrid + h / 2], axis=1).ravel(), "k", label="True")
                ax.axvline(initial[name], color="gray", linestyle="--", label="Initial")
                if v.ndim == 1:
                    ax.plot(v, zgrid, "r.-", label="Inverted")
                else:
                    interior = v[1:-1, 1:-1].reshape(-1, nz)
                    ax.plot(interior.mean(axis=0), zgrid, "r.-", label="Inverted (mean over x, y)")
                    ax.fill_betweenx(zgrid, interior.min(axis=0), interior.max(axis=0), color="r", alpha=0.2, label="Range over x, y")
                ax.set_ylim(zgrid[-1] + h / 2, zgrid[0] - h / 2)
                ax.set_xlabel(label[name])
                ax.set_ylabel("Depth (km)")
                ax.set_title(f"{label[name]} profile")
                ax.legend(loc="lower left", fontsize=8)
        for ax in ax_row[len(row):]:
            ax.axis("off")
    plt.savefig(f"{figure_path}/{tag}.png", bbox_inches="tight", dpi=150)
