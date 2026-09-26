"""Generate the background and smooth checkerboard 3-D velocity models."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"


def plot_checkerboard(initial, true, amplitude, path):
    relative = (true["vp"] - initial["vp"]) / initial["vp"]
    depth_index = int(torch.argmin((initial["depth"] - 15.0).abs()))
    lat_index = int(relative.abs().amax(dim=(0, 2)).argmax())
    lon_index = int(relative.abs().amax(dim=(0, 1)).argmax())
    slices = (
        (relative[depth_index], [initial["lon"][0], initial["lon"][-1], initial["lat"][0], initial["lat"][-1]], "longitude (deg)", "latitude (deg)", "horizontal at 15 km"),
        (relative[:, lat_index, :], [initial["lon"][0], initial["lon"][-1], initial["depth"][-1], initial["depth"][0]], "longitude (deg)", "depth (km)", "longitude-depth"),
        (relative[:, :, lon_index], [initial["lat"][0], initial["lat"][-1], initial["depth"][-1], initial["depth"][0]], "latitude (deg)", "depth (km)", "latitude-depth"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for index, (axis, (field, extent, xlabel, ylabel, title)) in enumerate(zip(axes, slices)):
        image = axis.imshow(field, origin="upper", extent=extent, aspect="equal" if index == 0 else "auto", interpolation="bilinear", cmap="seismic", vmin=-amplitude, vmax=amplitude)
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
        figure.colorbar(image, ax=axis, label="Vp relative perturbation")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=0.05)
    parser.add_argument("--wavelength-lon", type=float, default=1.5, help="degrees longitude")
    parser.add_argument("--wavelength-lat", type=float, default=1.5, help="degrees latitude")
    parser.add_argument("--wavelength-depth", type=float, default=20.0, help="km depth")
    args = parser.parse_args()

    lon = torch.arange(-121.0, -117.5, 0.1, dtype=torch.float64)
    lat = torch.arange(33.5, 36.5, 0.1, dtype=torch.float64)
    depth = torch.arange(-2.0, 20.1, 2.0, dtype=torch.float64)
    depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
    vp = 5.5 + 0.03 * depth_grid.clamp_min(0.0)
    vs = vp / 1.73
    checker = (
        torch.cos(2.0 * torch.pi * (lon_grid - lon[0]) / args.wavelength_lon)
        * torch.cos(2.0 * torch.pi * (lat_grid - lat[0]) / args.wavelength_lat)
        * torch.cos(2.0 * torch.pi * (depth_grid - depth[0]) / args.wavelength_depth)
    )
    initial = {"lon": lon, "lat": lat, "depth": depth, "vp": vp, "vs": vs}
    true = {"lon": lon, "lat": lat, "depth": depth, "vp": vp * (1 + args.amplitude * checker), "vs": vs * (1 + args.amplitude * checker)}

    DATA.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    torch.save(initial, DATA / "model_initial.pt")
    torch.save(true, DATA / "model_true.pt")
    plot_checkerboard(initial, true, args.amplitude, FIGURES / "checkerboard.png")
    print(f"saved models to {DATA}; wavelengths={args.wavelength_lon}/{args.wavelength_lat}/{args.wavelength_depth} lon/lat/depth")


if __name__ == "__main__":
    main()
