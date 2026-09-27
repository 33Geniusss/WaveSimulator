"""Matched-model tests: optimization correctness, not independent physics validation."""
from dataclasses import replace
import threading

import numpy as np
import pytest

from wave_lab.hybrid import refine_first_arrivals
from wave_lab.inversion import travel_times
from wave_lab.model import NoiseConfig, SimulationConfig, SimulationCancelled, add_measurement_noise, pick_arrival, simulate


@pytest.fixture(scope="module")
def acquisition():
    config = SimulationConfig(width=5, depth=2, x_min=-1, y_min=-1, spacing=.06,
                              interface_depth=.7, c1=1, c2=1.8, duration=4,
                              sponge_width=.4, max_frames=2)
    result = simulate(config)
    positions = [.5, 1, 1.8, 2.6, 3.6, 4.4]
    observed = [pick_arrival(result.times, result.signal(x)) for x in positions]
    return config, result, positions, observed


@pytest.fixture(scope="module")
def fitted(acquisition):
    config, result, positions, observed = acquisition
    return refine_first_arrivals(config, positions, observed, result.times, .0001)


def test_fd_refines_ray_start_and_exports_real_picks(acquisition, fitted):
    config, result, positions, observed = acquisition
    assert fitted.identifiable and fitted.termination == "target_rmse"
    assert fitted.c1 == pytest.approx(1, abs=.002)
    assert fitted.c2 == pytest.approx(1.8, abs=.002)
    assert fitted.rmse < fitted.fd_initial_rmse/100
    assert fitted.rmse < fitted.ray_stage["rmse"]
    assert 3 <= fitted.fd_evaluations <= 48
    predicted = simulate(replace(config, c1=fitted.c1, c2=fitted.c2), minimum_steps=len(result.times)-1)
    picks = [pick_arrival(result.times, np.interp(result.times, predicted.times, predicted.signal(x))) for x in positions]
    np.testing.assert_allclose(picks, fitted.fitted_times, atol=1e-12)
    assert not np.allclose(picks, travel_times(positions, config.interface_depth, fitted.c1, fitted.c2), atol=.001)
    assert fitted.to_dict()["evaluation_history"][0]["c1"] == pytest.approx(fitted.ray_stage["c1"])


def test_generating_velocities_are_not_optimizer_inputs(acquisition, fitted):
    config, result, positions, observed = acquisition
    changed = refine_first_arrivals(replace(config, c1=.5, c2=5), positions, observed, result.times, .0001)
    assert changed.c1 == fitted.c1 and changed.c2 == fitted.c2
    assert changed.evaluation_history == fitted.evaluation_history


def test_fixed_noisy_observations_are_not_replayed_into_candidates(acquisition):
    config, result, positions, _ = acquisition
    observed = [pick_arrival(result.times, add_measurement_noise(result.signal(x), i+1, NoiseConfig(True, .00001, 6730)))
                for i, x in enumerate(positions)]
    fit = refine_first_arrivals(config, positions, observed, result.times, .0001, max_evaluations=24)
    assert fit.observed_times == observed
    assert fit.rmse < fit.fd_initial_rmse
    assert fit.identifiable
    candidate = simulate(replace(config, c1=fit.c1, c2=fit.c2), minimum_steps=len(result.times)-1)
    expected = [pick_arrival(result.times, np.interp(result.times, candidate.times, candidate.signal(x))) for x in positions]
    np.testing.assert_allclose(expected, fit.fitted_times, atol=1e-12)


def test_budget_and_progress_include_diagnostic_solves(acquisition):
    config, result, positions, observed = acquisition
    progress = []
    fit = refine_first_arrivals(config, positions, observed, result.times, .0001, max_evaluations=8,
                               target_rmse=1e-10, progress=lambda *args: progress.append(args))
    assert fit.fd_evaluations <= len(fit.evaluation_history) <= 8
    assert [p[0] for p in progress] == list(range(1, len(progress)+1))
    assert fit.termination == "evaluation_budget"
    assert any("尚未达到" in warning for warning in fit.warnings)


@pytest.mark.parametrize("before", [True, False])
def test_cancellation(acquisition, before):
    config, result, positions, observed = acquisition
    event = threading.Event()
    if before:
        event.set()
    with pytest.raises(SimulationCancelled):
        refine_first_arrivals(config, positions, observed, result.times, .0001, cancel=event,
                              progress=lambda *args: event.set())


def test_unidentifiable_ray_data_skips_fd(acquisition):
    config, result, _, _ = acquisition
    fit = refine_first_arrivals(config, [.2, .4, .6], [.2, .4, .6], result.times, .0001)
    assert fit.c2 is None and fit.fd_evaluations == 0
    assert fit.termination == "ray_unidentifiable"


@pytest.mark.parametrize("change", [{"max_evaluations": 7}, {"target_rmse": float("nan")}, {"threshold": 0}])
def test_invalid_controls(acquisition, change):
    config, result, positions, observed = acquisition
    options = {"threshold": .0001, **change}
    with pytest.raises(ValueError):
        refine_first_arrivals(config, positions, observed, result.times, **options)


def test_acquisition_timestep_override(acquisition):
    config, result, _, _ = acquisition
    identical = simulate(config, minimum_steps=config.steps)
    np.testing.assert_array_equal(identical.plane, result.plane)
    finer = simulate(config, minimum_steps=2*config.steps)
    assert len(finer.times) == 2*config.steps+1
    assert np.isfinite(finer.plane).all()
    with pytest.raises(ValueError):
        simulate(config, minimum_steps=-1)
    with pytest.raises(ValueError, match="缓存"):
        simulate(replace(config, cache_mb=8), minimum_steps=100_000)
