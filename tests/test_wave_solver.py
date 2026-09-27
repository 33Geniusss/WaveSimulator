from dataclasses import replace
import json
import threading

import numpy as np
import pytest

from wave_lab.model import (Receiver, SimulationCancelled, SimulationConfig,
                            damping, export_results, load_configuration, pick_arrival,
                            pulse, ray_front, recommended_right_boundary, save_configuration, simulate)


def small_config(**kwargs):
    config = SimulationConfig(width=1, depth=1, x_min=-1, y_min=-1, spacing=0.05, c1=1, c2=1,
                              interface_depth=.5,
                              duration=0.63, sponge_width=0.3, boundary="dirichlet",
                              max_frames=10000, cache_mb=64)
    return replace(config, **kwargs)


def test_homogeneous_2d_exact_mode_second_order():
    errors = []
    for spacing in [0.1, 0.05, 0.025]:
        config = small_config(spacing=spacing)
        initial = lambda x, z: np.sin(np.pi * (x+1) / 2) * np.sin(np.pi * (z+1) / 2)
        result = simulate(config, initial=initial, source_enabled=False)
        xx, zz = np.meshgrid(result.x, result.z)
        exact = initial(xx, zz) * np.cos(np.pi / np.sqrt(2) * config.duration)
        errors.append(float(np.sqrt(np.mean((result.frames[-1] - exact)**2))))
    orders = np.log2(np.array(errors[:-1]) / errors[1:])
    assert np.all(orders > 1.85), (errors, orders)


def test_reflecting_boundary_exact_mode():
    config = small_config(spacing=0.025, boundary="reflecting")
    initial = lambda x, z: np.cos(np.pi * (x+1) / 2) * np.cos(np.pi * (z+1) / 2)
    result = simulate(config, initial=initial, source_enabled=False)
    xx, zz = np.meshgrid(result.x, result.z)
    exact = initial(xx, zz) * np.cos(np.pi / np.sqrt(2) * config.duration)
    assert np.max(np.abs(result.frames[-1] - exact)) < 0.001


def test_layered_lossless_discrete_energy():
    config = small_config(c2=2, duration=1.2)
    result = simulate(config, initial=lambda x, z: np.sin(np.pi*(x+1)/2)*np.sin(np.pi*(z+1)/2), source_enabled=False)
    assert len(result.frames) == len(result.times)
    u, v = result.frames[:-1].astype(float), result.frames[1:].astype(float)
    dt = result.times[1]
    dx, dz = result.x[1]-result.x[0], result.z[1]-result.z[0]
    c2 = np.where(result.z[:, None] < config.interface_depth, config.c1**2, config.c2**2)
    kinetic = np.sum(((v-u)/dt)**2 / c2, axis=(1, 2))
    potential_x = np.sum(np.diff(u, axis=2)*np.diff(v, axis=2)/dx**2, axis=(1, 2))
    potential_z = np.sum(np.diff(u, axis=1)*np.diff(v, axis=1)/dz**2, axis=(1, 2))
    energy = 0.5 * dx * dz * (kinetic + potential_x + potential_z)
    assert np.ptp(energy) / np.mean(energy) < 1e-5


def test_interface_normal_incidence_reflection_and_transmission():
    config = small_config(width=1, x_min=-1, depth=4.5, y_min=-1.5, spacing=0.025,
                          interface_depth=1.5, c2=2,
                          boundary="reflecting", duration=2.8, cache_mb=128)
    result = simulate(config, initial=lambda x, z: np.exp(-(z/0.15)**2), source_enabled=False)
    at2 = result.frames[:, np.argmin(abs(result.z-.5)), len(result.x)//2]
    at4 = result.frames[:, np.argmin(abs(result.z-2.5)), len(result.x)//2]
    incident = np.max(at2[(result.frame_times > 0.3) & (result.frame_times < 0.8)])
    reflected = np.max(at2[(result.frame_times > 2.2) & (result.frame_times < 2.8)])
    transmitted = np.max(at4[(result.frame_times > 1.7) & (result.frame_times < 2.3)])
    assert reflected / incident == pytest.approx(1/3, abs=0.05)
    assert transmitted / incident == pytest.approx(4/3, abs=0.08)


def test_sponge_reduces_late_reflection():
    # Horizontal plane pulse gives a clean right-wall echo, with no source ambiguity.
    config = small_config(width=4, x_min=-2, depth=2, y_min=-2, spacing=0.04,
                          interface_depth=1.5, duration=8,
                          sponge_width=1, sponge_strength=15, boundary="reflecting", max_frames=300)
    initial = lambda x, z: np.exp(-(x/0.2)**2) * np.ones_like(z)
    wall = simulate(config, initial=initial, source_enabled=False)
    sponge = simulate(replace(config, boundary="sponge"), initial=initial, source_enabled=False)
    mask = (wall.times > 5.0) & (wall.times < 8)
    wall_echo = np.max(np.abs(wall.signal(2)[mask]))
    sponge_echo = np.max(np.abs(sponge.signal(2)[mask]))
    assert sponge_echo < 0.4 * wall_echo, (sponge_echo, wall_echo)


def test_cached_plane_interpolation_and_memory_budget():
    result = simulate(small_config(boundary="reflecting", max_frames=8, cache_mb=8))
    j = 12
    x = (result.x[j] + result.x[j+1]) / 2
    np.testing.assert_allclose(result.signal(x), (result.plane[:, j] + result.plane[:, j+1]) / 2, atol=1e-7)
    assert len(result.frames) <= 8
    assert result.frame_times[0] == 0 and result.frame_times[-1] == result.config.duration
    assert result.frames.nbytes + result.plane.nbytes + result.times.nbytes + result.frame_times.nbytes <= 8 * 1024**2
    # Observation-plane data are not decimated with animation frames.
    assert len(result.times) > len(result.frames)


def test_compact_pulse_and_picker():
    assert pulse(-1, 3) == pulse(0, 3) == pulse(2/3, 3) == pulse(1, 3) == 0
    times = np.arange(6, dtype=float)
    assert pick_arrival(times, np.array([0, 0, .2, 0, -1, 0]), .1) == pytest.approx(1.5)
    assert pick_arrival(times, np.zeros(6)) is None
    assert pick_arrival(times, np.ones(6)*1e-9) is None
    # Earliest crossing, not the strongest (late) peak.
    assert pick_arrival(times, np.array([0, 0, .2, 0, -1, 0]), .1) < 2


def test_configuration_and_export_roundtrip(tmp_path):
    config = small_config()
    receivers = [Receiver(2, .8), Receiver(7, -.4)]
    path = tmp_path / "scene.json"
    save_configuration(path, config, receivers, .08)
    assert load_configuration(path) == (config, receivers, .08)
    result = simulate(config)
    export_results(tmp_path / "export", result, receivers, .08)
    signals = np.loadtxt(tmp_path / "export/signals.csv", skiprows=1, delimiter=",", encoding="utf-8-sig")
    np.testing.assert_allclose(signals[:, 1], result.signal(.8))
    archive = np.load(tmp_path / "export/wavefield.npz")
    np.testing.assert_array_equal(archive["frames"], result.frames)
    metadata = (tmp_path / "export/receivers.csv").read_text(encoding="utf-8-sig")
    assert "picked_arrival_s" in metadata and "2,0.8,0.0" in metadata
    assert "y_m" in metadata and "arrival_threshold_au" in metadata
    assert "y" in archive and "z" not in archive
    payload = json.loads(path.read_text())
    payload["receivers"][1]["id"] = 2
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_configuration(path)


def test_cancel_and_invalid_parameters():
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(SimulationCancelled):
        simulate(small_config(), cancel=cancel)
    for bad in [dict(c1=0), dict(source_depth=1.5), dict(cfl=1.1), dict(spacing=.0001), dict(duration=float("nan"))]:
        with pytest.raises(ValueError):
            replace(small_config(), **bad).validate()


def test_ray_reference_equal_depth_and_slow_lower_layer():
    config = SimulationConfig()
    h = config.interface_depth-config.source_depth
    expected = 6/config.c2 + 2*h/config.c1*np.sqrt(1-(config.c1/config.c2)**2)
    assert ray_front(config, config.source_x+6) == pytest.approx(expected)
    assert ray_front(replace(config, c2=.5), config.source_x+6) == 6


def test_picker_independent_of_future_echo_and_record_length():
    times = np.arange(8, dtype=float)
    signal = np.array([0, 0, .02, .04, 0, 0, 0, 0])
    picked = pick_arrival(times, signal, .01)
    assert picked == pytest.approx(1.5)
    signal[-1] = 1e6
    assert pick_arrival(times, signal, .01) == picked
    assert pick_arrival(times[:4], signal[:4], .01) == picked
    assert pick_arrival(times, np.zeros(8), .01) is None


def test_all_four_sponges_and_origin_in_undamped_interior():
    config = SimulationConfig()
    x = np.array([config.x_min, 0, config.width])
    y = np.array([config.y_min, 0, config.depth])
    gamma = damping(config, x, y)
    assert gamma[1, 1] == 0
    assert np.all(gamma[[0, 2], 1] > 0)  # top and bottom
    assert np.all(gamma[1, [0, 2]] > 0)  # left and right
    assert config.receiver_bounds[0] < 0 < config.receiver_bounds[1]
    for field in ("source_x", "source_depth"):
        with pytest.raises(ValueError, match="固定"):
            replace(config, **{field: .1}).validate()
    with pytest.raises(ValueError, match="缓冲"):
        replace(config, y_min=-.8).validate()


def test_origin_measurement_plane_and_negative_receiver_coordinates():
    config = small_config(boundary="sponge", duration=1, max_frames=10000)
    result = simulate(config)
    j = np.argmin(abs(result.y))
    assert result.y[j] == pytest.approx(0)
    np.testing.assert_allclose(result.plane, result.frames[:, j, :], atol=1e-7)
    # Symmetric homogeneous source/domain: left and right traces agree.
    np.testing.assert_allclose(result.signal(-.4), result.signal(.4), atol=1e-7)
    assert np.max(abs(result.signal(.4))) > .001
    # y=0 is an interior observation line: the wave also propagates upward.
    above = np.argmin(abs(result.y+.2))
    below = np.argmin(abs(result.y-.2))
    np.testing.assert_allclose(result.frames[:, above, :], result.frames[:, below, :], atol=1e-7)
    assert np.max(abs(result.frames[:, above, :])) > .001


def test_zero_plane_interpolation_when_zero_is_between_grid_rows():
    result = simulate(small_config(spacing=.07, boundary="sponge", max_frames=10000))
    assert np.min(abs(result.y)) > 1e-3
    index = int(np.searchsorted(result.y, 0)-1)
    weight = -result.y[index] / (result.y[index+1]-result.y[index])
    expected = (1-weight)*result.frames[:, index, :] + weight*result.frames[:, index+1, :]
    np.testing.assert_allclose(result.plane, expected, atol=1e-7)
    assert np.max(abs(result.signal(.4))) > .001


def test_top_sponge_suppresses_return_instead_of_reflecting_at_observation_line():
    config = small_config(width=4, x_min=-4, y_min=-2, depth=4, spacing=.05,
                          boundary="reflecting", sponge_width=.8, sponge_strength=4,
                          duration=5, max_frames=30)
    initial = lambda x, y: np.exp(-(y/.2)**2) * np.ones_like(x)
    wall = simulate(config, initial=initial, source_enabled=False)
    sponge = simulate(replace(config, boundary="sponge"), initial=initial, source_enabled=False)
    late = wall.times > 1.0
    wall_peak = float(np.max(abs(wall.signal(0)[late])))
    sponge_peak = float(np.max(abs(sponge.signal(0)[late])))
    # Zero-velocity initial pulse splits into two travelling waves of amplitude 1/2.
    assert wall_peak == pytest.approx(.5, abs=.04)
    assert sponge_peak < .2 * wall_peak, (wall_peak, sponge_peak)


def test_expanded_boundaries_preserve_default_first_picks():
    config = SimulationConfig(spacing=.05, max_frames=2)
    original = simulate(config)
    expanded = simulate(replace(config, x_min=-3, y_min=-3, width=9, depth=5))
    for position in [.5, 1, 2, 3, 4.5, 6]:
        a = pick_arrival(original.times, original.signal(position))
        b = pick_arrival(expanded.times, expanded.signal(position))
        assert a is not None and b is not None
        assert abs(a-b) < .005
    # This is actual propagation, not a picker returning the geometric prediction.
    assert pick_arrival(original.times, original.signal(6)) < 6 / config.c1


def test_outgoing_edge_releases_normal_pulse_even_without_sponge_damping():
    config = small_config(width=8, x_min=-8, y_min=-2, depth=4, spacing=.05,
                          boundary="reflecting", sponge_strength=0, duration=5, max_frames=2)
    initial = lambda x, y: np.exp(-(y/.2)**2) * np.ones_like(x)
    wall = simulate(config, initial=initial, source_enabled=False)
    outgoing = simulate(replace(config, boundary="sponge"), initial=initial, source_enabled=False)
    echo = (wall.times > 3.5) & (wall.times < 4.5)
    assert np.max(abs(outgoing.signal(0)[echo])) < .05 * np.max(abs(wall.signal(0)[echo]))


def test_v2_configuration_rejects_moving_origin_and_nonzero_receiver_y(tmp_path):
    path = tmp_path / "scene.json"
    save_configuration(path, SimulationConfig(), [Receiver(1, -.5), Receiver(2, 0)], 1e-4)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert all(r["y"] == 0 for r in payload["receivers"])
    bad = json.loads(path.read_text())
    bad["receivers"][0]["y"] = .1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="y 坐标"):
        load_configuration(path)
    payload["schema_version"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="v2"):
        load_configuration(path)


def test_crossover_and_automatic_right_boundary_cover_useful_region():
    config = SimulationConfig(auto_right=True)
    dc = config.crossover_distance
    assert dc == pytest.approx(2*np.sqrt(3))
    assert ray_front(config, .9*dc) == pytest.approx(.9*dc/config.c1)
    assert ray_front(config, 1.1*dc) < 1.1*dc/config.c1
    receivers = [Receiver(1, .5), Receiver(2, 6)]
    right = recommended_right_boundary(config, receivers)
    planned = replace(config, width=right)
    assert right == pytest.approx(7.15)
    assert planned.receiver_bounds[1] > 1.5*dc
    assert right-config.sponge_width >= 6+config.c1/config.frequency
    deeper = replace(config, interface_depth=2)
    assert recommended_right_boundary(deeper, receivers) > right
    low_contrast = replace(config, c2=1.01)
    assert recommended_right_boundary(low_contrast, receivers) > right
    for speed in [1, .5]:
        assert replace(config, c2=speed).crossover_distance is None
        assert recommended_right_boundary(replace(config, c2=speed), receivers) is None


def test_auto_boundary_settings_roundtrip_and_custom_factor(tmp_path):
    config = SimulationConfig(auto_right=True, crossover_factor=2)
    receivers = [Receiver(1, 1), Receiver(2, 5)]
    right = recommended_right_boundary(config, receivers)
    config = replace(config, width=right)
    path = tmp_path / "auto.json"
    save_configuration(path, config, receivers)
    loaded, points, threshold = load_configuration(path)
    assert loaded == config and points == receivers
    assert loaded.receiver_bounds[1] > 2*loaded.crossover_distance
    with pytest.raises(ValueError, match="覆盖倍数"):
        replace(config, crossover_factor=1).validate()
