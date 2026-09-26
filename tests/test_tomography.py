"""3-D tomography objective, summed-rank gradient consistency, and event-parameter gradients."""

import torch

from adtomo import ForwardGrid, Tomography, VelocityModel, predict_travel_times, smoothness


STATIONS = torch.tensor([[-120.0, 35.0, 0.0], [-119.7, 35.2, 0.0]], dtype=torch.float64)
EVENTS = torch.tensor([[-119.9, 35.1, 8.0], [-119.8, 35.0, 12.0]], dtype=torch.float64)


def make_model(trainable=True):
    lon = torch.arange(-120.6, -119.39, 0.1, dtype=torch.float64)
    lat = torch.arange(34.4, 35.61, 0.1, dtype=torch.float64)
    depth = torch.arange(-15.0, 45.1, 5.0, dtype=torch.float64)
    vp = torch.full((len(depth), len(lat), len(lon)), 6.0, dtype=torch.float64)
    return VelocityModel(lon, lat, depth, vp, vp / 1.73, trainable=trainable)


def make_groups(model):
    """Station i observes event i, with observations offset from the unperturbed model."""
    groups = []
    for i, station in enumerate(STATIONS):
        grid = ForwardGrid(station, EVENTS[i : i + 1], model, spacing=5.0)
        with torch.no_grad():
            observed_p = predict_travel_times(model, grid, "P", EVENTS[i : i + 1]) + 0.05
            observed_s = predict_travel_times(model, grid, "S", EVENTS[i : i + 1]) - 0.04
        groups.append((grid, [("P", torch.tensor([i]), observed_p), ("S", torch.tensor([i]), observed_s)]))
    return groups


def perturb(model):
    with torch.no_grad():
        model.vp[3, 4, 5] += 0.1
        model.vs[4, 5, 4] -= 0.05


def test_serial_objective_matches_data_and_regularization():
    model = make_model()
    groups = make_groups(model)
    tomography = Tomography(model, EVENTS, beta_vp=0.5, beta_vs=0.25, alpha_vp=0.125, alpha_vs=0.0625)
    perturb(model)

    loss = tomography(groups)
    residuals = [predict_travel_times(model, grid, phase, EVENTS[indices]) - observed for grid, phase_groups in groups for phase, indices, observed in phase_groups]
    data_sum = torch.cat(residuals).square().sum()
    dvp, dvs = model.vp - tomography.vp0, model.vs - tomography.vs0
    regularization = 0.5 * smoothness(dvp, model.lon, model.lat, model.depth) + 0.25 * smoothness(dvs, model.lon, model.lat, model.depth) + 0.125 * dvp.square().mean() + 0.0625 * dvs.square().mean()
    assert torch.allclose(loss, data_sum / 4 + regularization)
    assert torch.allclose(tomography.data_sum, data_sum)
    assert torch.allclose(tomography.data_loss, data_sum / 4)
    assert torch.allclose(tomography.regularization_loss, regularization)

    huber = Tomography(model, EVENTS, huber_delta=0.01)
    huber(groups)
    absolute = torch.cat(residuals).abs()
    assert torch.allclose(huber.data_sum, torch.where(absolute <= 0.01, absolute.square(), 0.01 * (2 * absolute - 0.01)).sum())


def test_summed_rank_gradients_match_serial():
    """Each rank scales its data sum by 1/total and the regularization by 1/world_size; summed gradients equal the serial ones."""
    reference_model = make_model()
    groups = make_groups(reference_model)
    reference = Tomography(reference_model, EVENTS, alpha_vp=0.25, alpha_vs=0.125)
    perturb(reference_model)
    reference(groups).backward()

    summed = None
    for local_groups in (groups[:1], groups[1:]):
        model = make_model()
        tomography = Tomography(model, EVENTS, alpha_vp=0.25, alpha_vs=0.125)
        perturb(model)
        tomography(local_groups, data_scale=1 / 4, regularization_scale=1 / 2).backward()
        gradients = (model.vp.grad, model.vs.grad, tomography.event_loc.grad, tomography.event_time_correction.grad)
        summed = gradients if summed is None else tuple(a + b for a, b in zip(summed, gradients))

    assert torch.allclose(summed[0], reference_model.vp.grad, atol=1e-11)
    assert torch.allclose(summed[1], reference_model.vs.grad, atol=1e-11)
    assert torch.allclose(summed[2], reference.event_loc.grad, atol=1e-11)
    assert torch.allclose(summed[3], reference.event_time_correction.grad, atol=1e-11)


def test_event_parameter_gradients_match_finite_differences():
    model = make_model(trainable=False)
    station, true_event = STATIONS[0], EVENTS[:1]
    with torch.no_grad():
        observed = predict_travel_times(model, ForwardGrid(station, true_event, model, spacing=5.0), "P", true_event)
    initial_event = true_event + torch.tensor([[0.02, -0.01, -1.0]], dtype=torch.float64)
    tomography = Tomography(model, initial_event)
    groups = [(ForwardGrid(station, initial_event, model, spacing=5.0), [("P", torch.tensor([0]), observed)])]

    tomography(groups).backward()
    assert torch.isfinite(tomography.event_loc.grad).all() and torch.isfinite(tomography.event_time_correction.grad).all()
    assert model.vp.grad is None

    step = 1e-5
    losses = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            tomography.event_loc[0, 0] = initial_event[0, 0] + sign * step
        losses.append(tomography(groups).item())
    finite_difference = (losses[0] - losses[1]) / (2.0 * step)
    assert abs(tomography.event_loc.grad[0, 0].item() - finite_difference) / (abs(finite_difference) + 1e-12) < 1e-3


def test_spherical_smoothness_is_cell_centered_and_volume_weighted():
    lon = torch.tensor([-121.0, -120.4, -119.7, -119.0], dtype=torch.float64)
    lat = torch.tensor([34.0, 34.6, 35.4], dtype=torch.float64)
    depth = torch.tensor([-2.0, 1.0, 6.0, 14.0], dtype=torch.float64)
    depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
    field = 0.02 * depth_grid + 0.3 * torch.deg2rad(lat_grid) + 0.2 * torch.deg2rad(lon_grid)

    radius = 6371.0 - depth
    dz = depth[1:] - depth[:-1]
    dphi = torch.deg2rad(lat[1:] - lat[:-1])
    dlambda = torch.deg2rad(lon[1:] - lon[:-1])
    cos_lat = torch.cos(torch.deg2rad(lat))
    radius_center = 0.5 * (radius[1:] + radius[:-1])
    phi_center = 0.5 * (torch.deg2rad(lat[1:]) + torch.deg2rad(lat[:-1]))
    volume = radius_center[:, None, None].square() * torch.cos(phi_center)[None, :, None]
    volume = volume * dz[:, None, None] * dphi[None, :, None] * dlambda[None, None, :]
    depth_nodes = (field[1:] - field[:-1]) / dz[:, None, None]
    grad_depth = 0.25 * (depth_nodes[:, :-1, :-1] + depth_nodes[:, 1:, :-1] + depth_nodes[:, :-1, 1:] + depth_nodes[:, 1:, 1:])
    lat_nodes = (field[:, 1:] - field[:, :-1]) / (radius[:, None, None] * dphi[None, :, None])
    grad_lat = 0.25 * (lat_nodes[:-1, :, :-1] + lat_nodes[1:, :, :-1] + lat_nodes[:-1, :, 1:] + lat_nodes[1:, :, 1:])
    lon_nodes = (field[:, :, 1:] - field[:, :, :-1]) / (radius[:, None, None] * cos_lat[None, :, None] * dlambda[None, None, :])
    grad_lon = 0.25 * (lon_nodes[:-1, :-1, :] + lon_nodes[1:, :-1, :] + lon_nodes[:-1, 1:, :] + lon_nodes[1:, 1:, :])
    expected = ((grad_depth.square() + grad_lat.square() + grad_lon.square()) * volume).sum() / volume.sum()
    assert torch.allclose(smoothness(field, lon, lat, depth), expected)


def test_spherical_smoothness_constant_gradient_and_autograd():
    lon = torch.linspace(-121.0, -119.0, 5, dtype=torch.float64)
    lat = torch.linspace(34.0, 36.0, 4, dtype=torch.float64)
    depth = torch.linspace(-2.0, 18.0, 6, dtype=torch.float64)
    constant = torch.ones((len(depth), len(lat), len(lon)), dtype=torch.float64)
    assert smoothness(constant, lon, lat, depth).item() == 0.0
    field = constant.clone().requires_grad_()
    value = smoothness(field, lon, lat, depth)
    assert torch.isfinite(value) and value.item() == 0.0
    value.backward()
    assert torch.isfinite(field.grad).all()

    varying = (torch.arange(field.numel(), dtype=torch.float64).reshape_as(field) / field.numel()).requires_grad_()
    positive = smoothness(varying, lon, lat, depth)
    assert torch.isfinite(positive) and positive.item() > 0.0
    positive.backward()
    assert torch.isfinite(varying.grad).all() and varying.grad.abs().sum() > 0.0


def test_regularization_weights_skip_unrequested_terms():
    smooth_model = make_model()
    smooth_groups = make_groups(smooth_model)
    smooth_only = Tomography(smooth_model, EVENTS, beta_vp=1.0)
    perturb(smooth_model)
    smooth_only(smooth_groups)
    assert smooth_only.smooth_vp.item() > 0.0
    assert smooth_only.damp_vp.item() == 0.0

    damp_model = make_model()
    damp_groups = make_groups(damp_model)
    damp_only = Tomography(damp_model, EVENTS, alpha_vp=1.0)
    perturb(damp_model)
    damp_only._smoothness = lambda field: (_ for _ in ()).throw(AssertionError("smoothing was evaluated"))
    damp_only(damp_groups)
    assert damp_only.damp_vp.item() > 0.0
    assert damp_only.smooth_vp.item() == 0.0

    unregularized_model = make_model()
    unregularized_groups = make_groups(unregularized_model)
    unregularized = Tomography(unregularized_model, EVENTS)
    perturb(unregularized_model)
    unregularized._smoothness = lambda field: (_ for _ in ()).throw(AssertionError("smoothing was evaluated"))
    loss = unregularized(unregularized_groups)
    assert unregularized.regularization_loss.item() == 0.0
    assert unregularized.smooth_vp.item() == 0.0 and unregularized.smooth_vs.item() == 0.0
    assert unregularized.damp_vp.item() == 0.0 and unregularized.damp_vs.item() == 0.0
    assert torch.allclose(loss, unregularized.data_loss)


def test_volume_average_is_resolution_stable():
    def field_on(depth, lat, lon):
        depth_grid, lat_grid, lon_grid = torch.meshgrid(depth, lat, lon, indexing="ij")
        return 0.02 * depth_grid + 0.3 * torch.deg2rad(lat_grid) + 0.2 * torch.deg2rad(lon_grid)

    coarse = (
        torch.linspace(-2.0, 18.0, 6, dtype=torch.float64),
        torch.linspace(34.0, 36.0, 5, dtype=torch.float64),
        torch.linspace(-121.0, -119.0, 5, dtype=torch.float64),
    )
    fine = (
        torch.linspace(-2.0, 18.0, 11, dtype=torch.float64),
        torch.linspace(34.0, 36.0, 9, dtype=torch.float64),
        torch.linspace(-121.0, -119.0, 9, dtype=torch.float64),
    )
    coarse_value = smoothness(field_on(*coarse), coarse[2], coarse[1], coarse[0])
    fine_value = smoothness(field_on(*fine), fine[2], fine[1], fine[0])
    assert torch.allclose(coarse_value, fine_value, rtol=2e-3, atol=1e-12)
