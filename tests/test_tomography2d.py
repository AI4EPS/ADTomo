"""Tomography2D checks: 1-D velocity with fixed events, then relocation with fixed velocity."""

import torch

from adtomo import ForwardGrid2D, Tomography2D, VelocityModel1D, predict_travel_times_2d, smoothness_1d


DEPTH = torch.arange(-5.0, 20.1, 1.0, dtype=torch.float64)
STATIONS = torch.tensor(
    [
        [-122.80, 38.80, 0.0],
        [-122.75, 38.83, 0.0],
        [-122.85, 38.78, 0.0],
        [-122.78, 38.86, 0.0],
        [-122.83, 38.84, 0.0],
    ],
    dtype=torch.float64,
)
EVENTS = torch.tensor([[-122.79, 38.81, 6.0], [-122.81, 38.82, 4.0]], dtype=torch.float64)
SPACING = 0.5


def test_smoothness_uses_physical_interval_lengths():
    depth = torch.tensor([0.0, 1.0, 4.0], dtype=torch.float64)
    field = torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64)
    dz = depth[1:] - depth[:-1]
    gradient = (field[1:] - field[:-1]) / dz
    expected = (gradient.square() * dz).sum() / dz.sum()
    assert torch.allclose(smoothness_1d(field, depth), expected)


def true_model():
    vp = 5.0 + 0.05 * DEPTH.clamp_min(0.0)
    return VelocityModel1D(DEPTH, vp, vp / 1.73, trainable=False)


def observe(model):
    observed = []
    for station in STATIONS:
        grid = ForwardGrid2D(station, EVENTS, model, spacing=SPACING)
        with torch.no_grad():
            observed.append({phase: predict_travel_times_2d(model, grid, phase, EVENTS) for phase in ("P", "S")})
    return observed


def make_groups(model, initial_loc, observed, time_shift=0.0):
    """Phase groups hold phase_time - t0_initial; shifting t0_initial by -time_shift adds time_shift."""
    indices = torch.arange(len(EVENTS))
    return [
        (
            ForwardGrid2D(station, initial_loc, model, spacing=SPACING, padding=4.0),
            [(phase, indices, times + time_shift) for phase, times in station_observed.items()],
        )
        for station, station_observed in zip(STATIONS, observed)
    ]


def optimize(tomography, groups, iterations, learning_rate, gamma=1.0):
    parameters = [parameter for parameter in tomography.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    initial_loss = tomography(groups).item()
    for _ in range(iterations):
        optimizer.zero_grad()
        loss = tomography(groups)
        loss.backward()
        optimizer.step()
        scheduler.step()
    return initial_loss, tomography(groups).item()


def test_velocity_only_recovers_sampled_depths():
    model = true_model()
    observed = observe(model)
    initial = VelocityModel1D(DEPTH, model.vp.detach() * 1.1, model.vs.detach() * 1.1, trainable=True)
    tomography = Tomography2D(initial, EVENTS)
    tomography.event_loc.requires_grad_(False)
    tomography.event_time_correction.requires_grad_(False)
    groups = make_groups(initial, EVENTS, observed)

    first, last = optimize(tomography, groups, 200, 0.02)

    assert last < first / 20
    assert torch.isfinite(initial.vp.grad).all() and torch.isfinite(initial.vs.grad).all()
    assert tomography.event_loc.grad is None and tomography.event_time_correction.grad is None
    sampled = (DEPTH >= 0.0) & (DEPTH <= 6.0)
    initial_error = (0.1 * model.vp[sampled]).abs().mean()
    final_error = (initial.vp.detach()[sampled] - model.vp[sampled]).abs().mean()
    assert final_error < initial_error / 2


def optimize_lbfgs(tomography, groups, parameters, rounds=5, max_iter=20):
    optimizer = torch.optim.LBFGS(
        parameters, max_iter=max_iter, line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14
    )

    def closure():
        optimizer.zero_grad()
        try:
            loss = tomography(groups)
        except ValueError:  # line-search probe left the forward grid: barrier
            return torch.full((), 1e6, dtype=torch.float64)
        loss.backward()
        return loss

    initial_loss = tomography(groups).item()
    for _ in range(rounds):
        optimizer.step(closure)
    return initial_loss, tomography(groups).item()


def test_location_only_recovers_perturbed_events():
    model = true_model()
    observed = observe(model)
    perturbation = torch.tensor([[0.015, -0.010, -1.5], [-0.010, 0.012, 1.0]], dtype=torch.float64)
    initial_loc = EVENTS + perturbation
    tomography = Tomography2D(model, initial_loc)
    groups = make_groups(model, initial_loc, observed, time_shift=0.4)

    first, last = optimize_lbfgs(tomography, groups, [tomography.event_loc, tomography.event_time_correction])

    assert last < first * 1e-6
    assert torch.isfinite(tomography.event_loc.grad).all()
    assert torch.isfinite(tomography.event_time_correction.grad).all()
    assert model.vp.grad is None
    recovered = tomography.event_loc.detach()
    initial_offset = torch.linalg.vector_norm(perturbation[:, :2], dim=-1)
    recovered_offset = torch.linalg.vector_norm(recovered[:, :2] - EVENTS[:, :2], dim=-1)
    assert torch.all(recovered_offset < initial_offset / 100)
    assert torch.all((tomography.event_loc.detach()[:, 2] - EVENTS[:, 2]).abs() < 1e-3)
    assert torch.all((tomography.event_time_correction.detach() - 0.4).abs() < 1e-3)


def test_beta_and_alpha_terms_use_zero_weight_short_circuit():
    truth = true_model()
    observed = observe(truth)
    model = VelocityModel1D(DEPTH, truth.vp.detach(), truth.vs.detach(), trainable=True)
    regularized = Tomography2D(model, EVENTS, beta_vp=0.5, beta_vs=0.25, alpha_vp=0.125, alpha_vs=0.0625)
    with torch.no_grad():
        model.vp *= 1.1
        model.vs *= 1.1
    groups = make_groups(model, EVENTS, observed)
    loss = regularized(groups)
    assert regularized.smooth_vp.item() > 0.0 and regularized.smooth_vs.item() > 0.0
    assert regularized.damp_vp.item() > 0.0 and regularized.damp_vs.item() > 0.0
    assert torch.isfinite(loss)

    unregularized = Tomography2D(model, EVENTS)
    unregularized(groups)
    assert unregularized.regularization_loss.item() == 0.0
    assert unregularized.smooth_vp.item() == 0.0 and unregularized.smooth_vs.item() == 0.0
    assert unregularized.damp_vp.item() == 0.0 and unregularized.damp_vs.item() == 0.0
