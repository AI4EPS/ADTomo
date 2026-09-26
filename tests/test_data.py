"""The unified dataset design: 1-D averaging, shared observations, and the trainable interface."""

import pandas as pd
import torch

from adtomo import ForwardGrid, ForwardGrid2D, Tomography, Tomography2D, VelocityModel, VelocityModel1D, build_station_groups, set_trainable


def make_model_3d():
    lon = torch.arange(-120.6, -119.39, 0.1, dtype=torch.float64)
    lat = torch.arange(34.4, 35.61, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 45.1, 5.0, dtype=torch.float64)
    depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
    vp = 5.5 + 0.03 * depth_grid.clamp_min(0.0) + 0.2 * torch.sin(lon_grid) * torch.cos(lat_grid)
    return VelocityModel(lon, lat, depth, vp, vp / 1.73)


STATIONS = pd.DataFrame({"station_id": ["A", "B"], "longitude": [-120.0, -119.7], "latitude": [35.0, 35.2], "depth_km": [0.0, -0.5]})
EVENTS = pd.DataFrame({
    "event_id": ["E1", "E2"],
    "event_time": ["2026-01-01T00:00:00.000", "2026-01-01T00:01:00.000"],
    "longitude": [-119.9, -119.8],
    "latitude": [35.1, 35.0],
    "depth_km": [8.0, 12.0],
})
PICKS = pd.DataFrame({
    "event_id": ["E1", "E1", "E2", "E2", "E1"],
    "station_id": ["A", "A", "A", "B", "B"],
    "phase_type": ["P", "S", "P", "P", "S"],
    "phase_time": ["2026-01-01T00:00:02.500", "2026-01-01T00:00:04.250", "2026-01-01T00:01:03.000", "2026-01-01T00:01:05.500", "2026-01-01T00:00:07.000"],
})


def test_from_3d_is_the_horizontal_mean():
    model_3d = make_model_3d()
    model_1d = VelocityModel1D.from_3d(model_3d)
    assert torch.equal(model_1d.depth, model_3d.depth)
    assert torch.allclose(model_1d.vp, model_3d.vp.mean(dim=(1, 2)))
    assert torch.allclose(model_1d.vs, model_3d.vs.mean(dim=(1, 2)))
    assert model_1d.vp.requires_grad


def test_station_groups_share_observations_between_1d_and_3d():
    model_3d = make_model_3d()
    groups_3d = build_station_groups(STATIONS, EVENTS, PICKS, model_3d, "3d", spacing=5.0)
    groups_1d = build_station_groups(STATIONS, EVENTS, PICKS, VelocityModel1D.from_3d(model_3d), "1d", spacing=2.0)
    assert len(groups_3d) == len(groups_1d) == 2
    for (grid_3d, phases_3d), (grid_1d, phases_1d) in zip(groups_3d, groups_1d):
        assert isinstance(grid_3d, ForwardGrid) and isinstance(grid_1d, ForwardGrid2D)
        for (phase_3d, indices_3d, observed_3d), (phase_1d, indices_1d, observed_1d) in zip(phases_3d, phases_1d):
            assert phase_3d == phase_1d and torch.equal(indices_3d, indices_1d) and torch.equal(observed_3d, observed_1d)
    (_, phases_a), (_, phases_b) = groups_3d
    assert [phase for phase, _, _ in phases_a] == ["P", "S"]
    assert torch.equal(phases_a[0][1], torch.tensor([0, 1])) and torch.allclose(phases_a[0][2], torch.tensor([2.5, 3.0], dtype=torch.float64))
    assert torch.equal(phases_b[0][1], torch.tensor([1])) and torch.allclose(phases_b[0][2], torch.tensor([5.5], dtype=torch.float64))


def test_unknown_event_id_is_rejected():
    picks = PICKS.copy()
    picks.loc[0, "event_id"] = "missing"
    try:
        build_station_groups(STATIONS, EVENTS, picks, make_model_3d(), "3d", spacing=5.0)
    except ValueError as error:
        assert "unknown event_id" in str(error) and "missing" in str(error)
    else:
        raise AssertionError("an unknown event_id must not select the last catalog event")


def test_set_trainable_toggles_every_parameter_in_both_modes():
    model_3d = make_model_3d()
    event_loc = EVENTS[["longitude", "latitude", "depth_km"]].to_numpy()
    for tomography in (Tomography(model_3d, event_loc), Tomography2D(VelocityModel1D.from_3d(model_3d), event_loc)):
        for names in (["vp"], ["event_loc", "event_time"], ["vp", "vs", "event_loc", "event_time"]):
            parameters = set_trainable(tomography, names)
            assert tomography.model.vp.requires_grad == ("vp" in names)
            assert tomography.model.vs.requires_grad == ("vs" in names)
            assert tomography.event_loc.requires_grad == ("event_loc" in names)
            assert tomography.event_time_correction.requires_grad == ("event_time" in names)
            assert len(parameters) == len(names)
        try:
            set_trainable(tomography, ["vp", "dt0"])
        except ValueError:
            pass
        else:
            raise AssertionError("unknown trainable names must be rejected")
