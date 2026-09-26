"""Spherical velocity models and station-centered Cartesian forward grids."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


R_EARTH = 6371.0  # km


class VelocityModel(nn.Module):
    """Absolute Vp/Vs on a regular ``(depth, latitude, longitude)`` grid."""

    def __init__(self, lon, lat, depth, vp, vs, trainable=True):
        super().__init__()
        self.register_buffer("lon", torch.as_tensor(lon, dtype=torch.float64))
        self.register_buffer("lat", torch.as_tensor(lat, dtype=torch.float64))
        self.register_buffer("depth", torch.as_tensor(depth, dtype=torch.float64))
        self.vp = nn.Parameter(torch.as_tensor(vp, dtype=torch.float64).clone(), requires_grad=trainable)
        self.vs = nn.Parameter(torch.as_tensor(vs, dtype=torch.float64).clone(), requires_grad=trainable)


class VelocityModel1D(nn.Module):
    """Absolute Vp/Vs as functions of spherical depth only."""

    def __init__(self, depth, vp, vs, trainable=True):
        super().__init__()
        depth = torch.as_tensor(depth, dtype=torch.float64)
        assert depth.ndim == 1 and torch.all(depth[1:] > depth[:-1]), "depth must be a strictly increasing 1-D axis"
        self.register_buffer("depth", depth)
        self.vp = nn.Parameter(torch.as_tensor(vp, dtype=torch.float64).clone(), requires_grad=trainable)
        self.vs = nn.Parameter(torch.as_tensor(vs, dtype=torch.float64).clone(), requires_grad=trainable)

    @classmethod
    def from_3d(cls, model, trainable=True):
        """The horizontal mean of a 3-D model: ``V(d) = mean_{lat, lon} V(d, lat, lon)``."""
        return cls(model.depth, model.vp.detach().mean(dim=(1, 2)), model.vs.detach().mean(dim=(1, 2)), trainable)


def interpolate_1d(axis, values, query):
    """Piecewise-linear interpolation of ``values`` on a strictly increasing 1-D ``axis``."""
    upper = torch.searchsorted(axis, query.contiguous()).clamp(1, axis.numel() - 1)
    lower = upper - 1
    weight = (query - axis[lower]) / (axis[upper] - axis[lower])
    return (1.0 - weight) * values[lower] + weight * values[upper]


def spherical_to_ecef(lon, lat, depth):
    """Degrees east/north and km depth positive down to ECEF km."""
    lon = torch.deg2rad(lon)
    lat = torch.deg2rad(lat)
    radius = R_EARTH - depth
    return torch.stack(
        [
            radius * torch.cos(lat) * torch.cos(lon),
            radius * torch.cos(lat) * torch.sin(lon),
            radius * torch.sin(lat),
        ],
        dim=-1,
    )


def ecef_to_spherical(xyz):
    """ECEF km to longitude, latitude, depth in degrees/degrees/km."""
    radius = torch.linalg.vector_norm(xyz, dim=-1)
    lon = torch.rad2deg(torch.atan2(xyz[..., 1], xyz[..., 0]))
    lat = torch.rad2deg(torch.atan2(xyz[..., 2], torch.hypot(xyz[..., 0], xyz[..., 1])))
    return lon, lat, R_EARTH - radius


def local_basis(lon, lat):
    """Rows of the East, North, Down basis at longitude/latitude."""
    lon = torch.deg2rad(lon)
    lat = torch.deg2rad(lat)
    zero = torch.zeros_like(lon)
    east = torch.stack([-torch.sin(lon), torch.cos(lon), zero], dim=-1)
    north = torch.stack(
        [-torch.sin(lat) * torch.cos(lon), -torch.sin(lat) * torch.sin(lon), torch.cos(lat)], dim=-1
    )
    down = torch.stack(
        [-torch.cos(lat) * torch.cos(lon), -torch.cos(lat) * torch.sin(lon), -torch.sin(lat)], dim=-1
    )
    return torch.stack([east, north, down], dim=-2)


def ecef_to_local(xyz, origin, basis):
    """ECEF km to a local East/North/Down frame."""
    return (xyz - origin) @ basis.transpose(-1, -2)


def local_to_ecef(xyz, origin, basis):
    """Local East/North/Down km to ECEF."""
    return xyz @ basis + origin


class ForwardGrid:
    """A fixed East/North/Down box around one station for a 3-D model.

    Coordinates are ``(x, y, z)`` = (East, North, Down) km with the station at
    the origin; local fields use ``(z, y, x)`` tensor order. Every node samples
    the global model at its own spherical position. ``padding`` (default
    ``2 * spacing``) surrounds the build-time events; ``padding_above`` adds
    room above the shallowest point so relocated events can move upward.
    """

    def __init__(self, station_spherical, events_spherical, model, spacing, padding=None, padding_above=0.0):
        self.spacing = float(spacing)
        padding = 2.0 * self.spacing if padding is None else float(padding)
        station = torch.as_tensor(station_spherical, dtype=torch.float64).reshape(3)
        self.station_ecef = spherical_to_ecef(*station)
        self.basis = local_basis(station[0], station[1])

        points = torch.cat([torch.zeros(1, 3, dtype=torch.float64), self.to_local(events_spherical)])
        minimum, maximum = points.amin(dim=0).tolist(), points.amax(dim=0).tolist()
        n_west = math.ceil((padding - minimum[0]) / self.spacing)
        n_east = math.ceil((maximum[0] + padding) / self.spacing)
        n_south = math.ceil((padding - minimum[1]) / self.spacing)
        n_north = math.ceil((maximum[1] + padding) / self.spacing)
        n_up = math.ceil((float(padding_above) - minimum[2]) / self.spacing)
        n_down = math.ceil((maximum[2] + padding) / self.spacing)
        self.x = torch.arange(-n_west, n_east + 1, dtype=torch.float64) * self.spacing
        self.y = torch.arange(-n_south, n_north + 1, dtype=torch.float64) * self.spacing
        self.z = torch.arange(-n_up, n_down + 1, dtype=torch.float64) * self.spacing
        self.shape = (len(self.z), len(self.y), len(self.x))
        self.station_index = torch.tensor([n_west, n_south, n_up], dtype=torch.float64)

        z, y, x = torch.meshgrid(self.z, self.y, self.x, indexing="ij")
        lon, lat, depth = ecef_to_spherical(local_to_ecef(torch.stack([x, y, z], dim=-1), self.station_ecef, self.basis))
        self.sample_grid = torch.stack(
            [
                2.0 * (lon - model.lon[0]) / (model.lon[-1] - model.lon[0]) - 1.0,
                2.0 * (lat - model.lat[0]) / (model.lat[-1] - model.lat[0]) - 1.0,
                2.0 * (depth - model.depth[0]) / (model.depth[-1] - model.depth[0]) - 1.0,
            ],
            dim=-1,
        )[None]

    def to_local(self, events_spherical):
        """Event lon/lat/depth to East/North/Down km in this station's frame."""
        events = torch.as_tensor(events_spherical, dtype=torch.float64).reshape(-1, 3)
        return ecef_to_local(spherical_to_ecef(events[:, 0], events[:, 1], events[:, 2]), self.station_ecef, self.basis)

    def sample_model(self, model_field):
        """Differentiably sample a global ``(depth, latitude, longitude)`` field onto the box."""
        return F.grid_sample(model_field[None, None], self.sample_grid, mode="bilinear", padding_mode="border", align_corners=True)[0, 0]

    def sample_events(self, traveltime, events_spherical):
        """Trilinearly sample a ``(z, y, x)`` field at live (possibly trainable) event positions."""
        index = (self.to_local(events_spherical) - torch.stack([self.x[0], self.y[0], self.z[0]])) / self.spacing
        size = torch.tensor([len(self.x), len(self.y), len(self.z)], dtype=torch.float64)
        if torch.any(index < 0) or torch.any(index > size - 1):
            raise ValueError("current event location left the fixed 3-D forward grid")
        grid = (2.0 * index / (size - 1) - 1.0).view(1, -1, 1, 1, 3)
        return F.grid_sample(traveltime[None, None], grid, mode="bilinear", align_corners=True)[0, 0, :, 0, 0]


class ForwardGrid2D:
    """A fixed Cartesian vertical section around one station for a 1-D model.

    Coordinates are ``(x, y)`` = (horizontal distance, Down) km with the
    station at the origin; fields use ``(y, x)`` tensor order. The section is
    the azimuthal reduction of the East/North/Down frame: an event at local
    ``(E, N, D)`` sits at ``(sqrt(E^2 + N^2), D)``, and node ``(x, 0, y)``
    samples the model at its spherical depth ``R - sqrt(x^2 + (R - d_s - y)^2)``,
    so layers stay curved in the section. Nodes shallower than the model's
    first depth take its shallowest velocity. ``padding``/``padding_above`` as
    in :class:`ForwardGrid`.
    """

    def __init__(self, station_spherical, events_spherical, model, spacing, padding=None, padding_above=0.0):
        self.spacing = float(spacing)
        padding = 2.0 * self.spacing if padding is None else float(padding)
        station = torch.as_tensor(station_spherical, dtype=torch.float64).reshape(3)
        self.station_ecef = spherical_to_ecef(*station)
        self.basis = local_basis(station[0], station[1])

        events = self.to_section(events_spherical)
        n_left = max(1, math.ceil(padding / self.spacing))
        n_right = max(1, math.ceil((events[:, 0].max().item() + padding) / self.spacing))
        n_up = math.ceil((float(padding_above) - min(0.0, events[:, 1].min().item())) / self.spacing)
        n_down = max(1, math.ceil((max(0.0, events[:, 1].max().item()) + padding) / self.spacing))
        self.x = torch.arange(-n_left, n_right + 1, dtype=torch.float64) * self.spacing
        self.y = torch.arange(-n_up, n_down + 1, dtype=torch.float64) * self.spacing
        self.shape = (len(self.y), len(self.x))
        self.station_index = torch.tensor([n_left, n_up], dtype=torch.float64)

        y, x = torch.meshgrid(self.y, self.x, indexing="ij")
        local = torch.stack([x, torch.zeros_like(x), y], dim=-1)
        _, _, self.grid_depth = ecef_to_spherical(local_to_ecef(local, self.station_ecef, self.basis))

    def to_section(self, events_spherical):
        """Event lon/lat/depth to ``(sqrt(E^2 + N^2), D)`` in this station's section."""
        events = torch.as_tensor(events_spherical, dtype=torch.float64).reshape(-1, 3)
        local = ecef_to_local(spherical_to_ecef(events[:, 0], events[:, 1], events[:, 2]), self.station_ecef, self.basis)
        return torch.stack([torch.hypot(local[:, 0], local[:, 1]), local[:, 2]], dim=-1)

    def sample_model(self, model_field, model_depth):
        """Differentiably map a depth-only field onto the section at each node's spherical depth."""
        return interpolate_1d(model_depth, model_field, self.grid_depth.clamp_min(model_depth[0]))

    def sample_events(self, traveltime, events_spherical):
        """Bilinearly sample a ``(y, x)`` field at live (possibly trainable) event positions."""
        index = (self.to_section(events_spherical) - torch.stack([self.x[0], self.y[0]])) / self.spacing
        size = torch.tensor([len(self.x), len(self.y)], dtype=torch.float64)
        if torch.any(index < 0) or torch.any(index > size - 1):
            raise ValueError("current event location left the fixed 2-D forward grid")
        grid = (2.0 * index / (size - 1) - 1.0).view(1, -1, 1, 2)
        return F.grid_sample(traveltime[None, None], grid, mode="bilinear", align_corners=True)[0, 0, :, 0]
