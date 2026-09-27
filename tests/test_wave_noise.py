import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from wave_lab.model import (NoiseConfig, Receiver, SimulationConfig, SimulationResult,
                            add_measurement_noise, export_results, load_configuration,
                            pick_arrival, save_configuration)


def test_noise_statistics_and_independent_reproducible_channels():
    clean = np.zeros(100000)
    config = NoiseConfig(True, .002, 6730)
    first = add_measurement_noise(clean, 1, config)
    second = add_measurement_noise(clean, 2, config)
    assert abs(first.mean()) < .002 * .015
    assert first.std() == pytest.approx(.002, rel=.015)
    assert abs(np.corrcoef(first, second)[0, 1]) < .015
    np.testing.assert_array_equal(first, add_measurement_noise(clean, 1, config))
    assert not np.array_equal(first, add_measurement_noise(clean, 1, replace(config, seed=6731)))
    np.testing.assert_array_equal(clean, np.zeros(100000))


def test_disabled_and_zero_noise_restore_original_without_mutation():
    clean = np.linspace(-.02, .02, 200, dtype=np.float32)
    original = clean.copy()
    for noise in (NoiseConfig(), NoiseConfig(True, 0)):
        observed = add_measurement_noise(clean, 1, noise)
        np.testing.assert_array_equal(observed, original)
        observed[:] = 1
        np.testing.assert_array_equal(clean, original)


def test_noise_can_cause_false_detection_without_a_wave():
    times = np.linspace(0, 1, 1000)
    clean = np.zeros_like(times)
    observed = add_measurement_noise(clean, 1, NoiseConfig(True, .01))
    assert pick_arrival(times, clean, .0001) is None
    assert pick_arrival(times, observed, .0001) is not None
    assert pick_arrival(times, observed, 1) is None


@pytest.mark.parametrize("noise", [NoiseConfig(True, -1), NoiseConfig(True, float("nan")),
                                   NoiseConfig(True, .01, -1), NoiseConfig(True, .01, 1.5),
                                   NoiseConfig("yes"), NoiseConfig(True, .01, True)])
def test_reject_invalid_noise_settings(noise):
    with pytest.raises(ValueError):
        noise.validate()


def test_configuration_roundtrip_and_older_v2_defaults(tmp_path):
    path = tmp_path / "noise.json"
    noise = NoiseConfig(True, .0003, 456)
    save_configuration(path, SimulationConfig(), [Receiver(1, 1)], noise=noise)
    assert load_configuration(path, include_noise=True)[3] == noise
    assert len(load_configuration(path)) == 3
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["noise"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_configuration(path, include_noise=True)[3] == NoiseConfig()
    payload["noise"] = {"std_au": -1}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_configuration(path)


def test_export_matches_observed_picks_and_preserves_clean_field(tmp_path):
    times = np.linspace(0, 1, 101)
    plane = np.column_stack((np.zeros_like(times), np.sin(times)*.001))
    result = SimulationResult(SimulationConfig(), np.array([0., 2.]), np.array([0., 1.]),
                              times, np.zeros((2, 2, 2)), times[[0, -1]], plane)
    receivers = [Receiver(3, 1), Receiver(1, .5)]
    noise = NoiseConfig(True, .0002, 134)
    original = plane.copy()
    export_results(tmp_path, result, receivers, .0001, noise)
    observed = np.loadtxt(tmp_path / "signals.csv", delimiter=",", skiprows=1, encoding="utf-8-sig")
    clean = np.loadtxt(tmp_path / "signals_clean.csv", delimiter=",", skiprows=1, encoding="utf-8-sig")
    with (tmp_path / "receivers.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for index, receiver in enumerate(receivers):
        expected = add_measurement_noise(result.signal(receiver.x), receiver.id, noise)
        np.testing.assert_array_equal(observed[:, index+1], expected)
        np.testing.assert_array_equal(clean[:, index+1], result.signal(receiver.x))
        assert float(rows[index]["picked_arrival_s"]) == pick_arrival(times, expected, .0001)
    with np.load(tmp_path / "wavefield.npz") as data:
        np.testing.assert_array_equal(data["observation_plane"], original)
    np.testing.assert_array_equal(result.plane, original)
    assert load_configuration(tmp_path / "configuration.json", include_noise=True)[3] == noise
