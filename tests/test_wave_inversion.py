import json

import numpy as np
import pytest

from wave_lab.inversion import invert_first_arrivals, travel_times
from wave_lab.model import SimulationConfig, pick_arrival, ray_front, simulate


@pytest.mark.parametrize("c1,c2,depth", [(1., 2., 1.), (.7, 3., 2.), (3., 7., .3), (12., 30., 4.)])
def test_recovers_exact_ray_data_without_generating_speed_input(c1, c2, depth):
    # Independent Snell path construction, including physical head-wave validity.
    distances = np.array([.5, 1, 2, 3, 4.5, 6])*depth
    angle = np.arcsin(c1/c2)
    leg_x = depth*np.tan(angle)
    head = 2*depth/np.cos(angle)/c1+(distances-2*leg_x)/c2
    times = np.minimum(distances/c1, np.where(distances >= 2*leg_x, head, np.inf))
    fit = invert_first_arrivals(distances, times, depth)
    assert fit.identifiable
    assert fit.c1 == pytest.approx(c1, rel=1e-6)
    assert fit.c2 == pytest.approx(c2, rel=1e-6)
    assert fit.rmse < 1e-9
    json.dumps(fit.to_dict(), allow_nan=False)


def test_direct_only_data_does_not_invent_second_velocity():
    distances = np.arange(1, 7)*.2
    fit = invert_first_arrivals(distances, distances/1.3, 1.)
    assert fit.c1 == pytest.approx(1.3)
    assert fit.c2 is None and not fit.identifiable
    assert fit.condition_number is None
    assert any("无法确定 c2" in warning for warning in fit.warnings)


def test_known_depth_all_head_data_can_recover_both_speeds():
    distances = np.arange(5., 11.)
    times = distances/2+np.sqrt(3)
    fit = invert_first_arrivals(distances, times, 1.)
    assert fit.c1 == pytest.approx(1.)
    assert fit.c2 == pytest.approx(2.)
    assert fit.branches == ["头波"]*6
    assert any("截距" in warning for warning in fit.warnings)


def test_receiver_order_signs_and_changed_arrivals():
    distances = np.array([.5, 1, 2, 3, 4.5, 6])
    times = travel_times(distances, 1, 1, 2)
    order = [4, 0, 3, 1, 5, 2]
    fit = invert_first_arrivals(-distances[order], times[order], 1)
    assert fit.c1 == pytest.approx(1.)
    assert fit.c2 == pytest.approx(2.)
    np.testing.assert_allclose(fit.fitted_times, times[order], atol=1e-10)
    slower = invert_first_arrivals(distances, times*1.2, 1)
    assert slower.c1 == pytest.approx(1/1.2)
    assert slower.c2 == pytest.approx(2/1.2)


@pytest.mark.parametrize("offsets,arrivals,depth,match", [
    ([1]*5, [1]*5, 1, "不同"),
    ([1, 2, 3, 4, 5, 6], [1, 2, None, 4, 5, 6], 1, "未检出"),
    ([1, 2, 3, 4, 5, 6], [0, 2, 3, 4, 5, 6], 1, "t=0"),
    ([1, -1, 1, -1, 1, -1], [1]*6, 1, "不同"),
    ([1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], 0, "深度"),
    ([1, 2, 3, 4, 5, 6], [1, 2, float("nan"), 4, 5, 6], 1, "有限"),
])
def test_invalid_or_insufficient_inputs(offsets, arrivals, depth, match):
    with pytest.raises(ValueError, match=match):
        invert_first_arrivals(offsets, arrivals, depth)


def test_bounds_and_large_misfit_are_explicit():
    distances = [.5, 1, 2, 3, 4.5, 6]
    times = travel_times(distances, 1, 1, 2)
    limited = invert_first_arrivals(distances, times, 1, speed_min=.1, speed_max=1.8)
    assert limited.c1 <= 1.8 and (limited.c2 is None or limited.c2 <= 1.8+1e-10)
    assert any("搜索边界" in warning for warning in limited.warnings)
    bad = invert_first_arrivals(distances, [.3, 3, .2, 4, .1, 2], 1)
    assert any("误差较大" in warning for warning in bad.warnings)
    with pytest.raises(ValueError, match="上限"):
        invert_first_arrivals(distances, times, 1, speed_min=2, speed_max=1)


def test_fit_is_better_than_independent_dense_speed_grid():
    distances = np.array([.5, 1, 2, 3, 4.5, 6])
    times = travel_times(distances, 1, 1, 2)+np.array([.02, -.01, .03, -.01, .06, -.02])
    fit = invert_first_arrivals(distances, times, 1, speed_min=.5, speed_max=4)
    velocities = np.linspace(.5, 4, 500)
    c1, c2 = np.meshgrid(velocities, velocities)
    direct = distances/c1[..., None]
    head = distances/c2[..., None]+2*np.sqrt(np.maximum(0, 1/c1**2-1/c2**2))[..., None]
    residual = np.minimum(direct, head)-times
    scores = np.where(c2 > c1, np.mean(residual**2, axis=-1), np.inf)
    assert fit.rmse**2 <= float(scores.min())+1e-12


def test_pde_picks_invert_with_documented_ray_discrepancy():
    config = SimulationConfig(spacing=.05, max_frames=2)
    result = simulate(config)
    distances = [.5, 1, 2, 3, 4.5, 6]
    arrivals = [pick_arrival(result.times, result.signal(x)) for x in distances]
    fit = invert_first_arrivals(distances, arrivals, config.interface_depth,
                                sample_interval=float(result.times[1]))
    assert fit.identifiable
    assert abs(fit.c1-config.c1)/config.c1 < .15
    assert abs(fit.c2-config.c2)/config.c2 < .2
    # The actual measured times, not ray_front(config, x), enter the inverse.
    assert max(abs(np.array(arrivals)-np.array([ray_front(config, x) for x in distances]))) > .02
    assert fit.rmse > 1e-4


@pytest.mark.parametrize("count", [2, 3, 4, 8, 12, 25, 64])
def test_variable_receiver_counts_recover_both_speeds(count):
    distances = np.linspace(.5, 6, count)
    # Independent two-layer reference for c1=1, c2=2, h=1.
    times = np.minimum(distances, distances/2+np.sqrt(3))
    fit = invert_first_arrivals(distances, times, 1)
    assert fit.identifiable
    assert fit.c1 == pytest.approx(1, rel=1e-6)
    assert fit.c2 == pytest.approx(2, rel=1e-6)
    for data in (fit.offsets, fit.observed_times, fit.fitted_times, fit.residuals, fit.branches):
        assert len(data) == count
    if count == 2:
        assert any("只有 2 个" in warning for warning in fit.warnings)


@pytest.mark.parametrize("offsets,arrivals", [([], []), ([1, 2], [1]), ([[1, 2]], [[1, 2]])])
def test_variable_count_rejects_empty_mismatched_or_nonvector_data(offsets, arrivals):
    with pytest.raises(ValueError, match="等长的一维"):
        invert_first_arrivals(offsets, arrivals, 1)


@pytest.mark.parametrize("offsets,arrivals", [([1], [1]), ([0, 1], [0, 1]), ([0, 0], [0, 0])])
def test_too_few_informative_offsets_do_not_report_two_speeds(offsets, arrivals):
    with pytest.raises(ValueError, match="至少需要 2"):
        invert_first_arrivals(offsets, arrivals, 1)


def test_two_direct_receivers_still_cannot_determine_c2():
    fit = invert_first_arrivals([.5, 1], [.5, 1], 1)
    assert fit.c1 == pytest.approx(1)
    assert fit.c2 is None and not fit.identifiable


def test_two_head_receivers_and_duplicate_offsets():
    fit = invert_first_arrivals([5, 6], np.array([5, 6])/2+np.sqrt(3), 1)
    assert fit.c1 == pytest.approx(1) and fit.c2 == pytest.approx(2)
    repeated = invert_first_arrivals([.5, -.5, 6, 6], [.5, .5, 3+np.sqrt(3), 3+np.sqrt(3)], 1)
    assert repeated.c1 == pytest.approx(1) and repeated.c2 == pytest.approx(2)


def test_two_inconsistent_picks_do_not_force_a_head_branch():
    # At the coarse PDE mesh the early near-source threshold crossing can make
    # t/d increase with distance, contrary to faster-lower-layer first arrivals.
    fit = invert_first_arrivals([.5, 6], [.39484145291945266, 4.954116601619015], 1)
    assert fit.c2 is None and not fit.identifiable
    assert fit.rmse > 0
    assert any("无法确定 c2" in warning for warning in fit.warnings)
