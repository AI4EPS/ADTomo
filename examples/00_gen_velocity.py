"""Generate the initial and true 3-D velocity models."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


# Edit the experiment here.
AMPLITUDE = 0.15
WAVELENGTH_LON = 0.5       # degrees longitude
WAVELENGTH_LAT = 0.5       # degrees latitude
WAVELENGTH_DEPTH = 10.0     # km

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

lon_min = -120.75
lat_min = 33.75
depth_min = -2.0
depth_max = 16.0
lon = torch.linspace(lon_min, lon_min + 1.5 * WAVELENGTH_LON, 16, dtype=torch.float64)
lat = torch.linspace(lat_min, lat_min + 1.5 * WAVELENGTH_LAT, 16, dtype=torch.float64)
depth = torch.linspace(depth_min, depth_max, 10, dtype=torch.float64)
depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")

vp = 5.5 + 0.03 * depth_grid.clamp_min(0.0)
vs = vp / 1.73
checker = (
    torch.cos(2.0 * torch.pi * (lon_grid - lon[0]) / WAVELENGTH_LON)
    * torch.cos(2.0 * torch.pi * (lat_grid - lat[0]) / WAVELENGTH_LAT)
    * torch.cos(2.0 * torch.pi * (depth_grid - depth[0]) / WAVELENGTH_DEPTH)
)
initial = {"lon": lon, "lat": lat, "depth": depth, "vp": vp, "vs": vs}
true = {
    "lon": lon,
    "lat": lat,
    "depth": depth,
    "vp": vp * (1.0 + AMPLITUDE * checker),
    "vs": vs * (1.0 + AMPLITUDE * checker),
}

DATA.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
torch.save(initial, DATA / "model.pt")
torch.save(true, DATA / "model_true.pt")

relative = (true["vp"] - initial["vp"]) / initial["vp"]
depth_index = int(torch.argmin((depth - depth.mean()).abs()))
lat_index = int(relative.abs().amax(dim=(0, 2)).argmax())
lon_index = int(relative.abs().amax(dim=(0, 1)).argmax())
slices = (
    (relative[depth_index], [lon[0], lon[-1], lat[0], lat[-1]], "longitude (deg)", "latitude (deg)", f"horizontal at {depth[depth_index]:.1f} km"),
    (relative[:, lat_index, :], [lon[0], lon[-1], depth[-1], depth[0]], "longitude (deg)", "depth (km)", "longitude-depth"),
    (relative[:, :, lon_index], [lat[0], lat[-1], depth[-1], depth[0]], "latitude (deg)", "depth (km)", "latitude-depth"),
)
figure, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
for index, (axis, (field, extent, xlabel, ylabel, title)) in enumerate(zip(axes, slices)):
    image = axis.imshow(
        field,
        origin="upper",
        extent=extent,
        aspect="equal" if index == 0 else "auto",
        interpolation="bilinear",
        cmap="seismic",
        vmin=-AMPLITUDE,
        vmax=AMPLITUDE,
    )
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    figure.colorbar(image, ax=axis, label="Vp relative perturbation")
figure.savefig(FIGURES / "checkerboard.png", dpi=180)
plt.close(figure)

print(f"saved models to {DATA}; wavelengths={WAVELENGTH_LON}/{WAVELENGTH_LAT}/{WAVELENGTH_DEPTH} lon/lat/depth")
