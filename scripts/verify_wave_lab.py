"""Reproduce numerical evidence and render the default desktop application."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from wave_lab.app import WaveLabWindow
from wave_lab.model import Receiver, SimulationConfig, simulate, pick_arrival, ray_front, recommended_right_boundary, save_configuration


def main():
    destination = ROOT / "output" / "wave_demo_auto"
    destination.mkdir(parents=True, exist_ok=True)
    config = SimulationConfig(auto_right=True)
    default_receivers = [Receiver(i+1, d) for i, d in enumerate([.5, 1, 2, 3, 4.5, 6])]
    config = replace(config, width=recommended_right_boundary(config, default_receivers))
    scenes = ROOT / "scenes"
    scenes.mkdir(exist_ok=True)
    for name, scene_config, offsets in [
        ("two_layer_wide", config, [.5, 1, 2, 3, 4.5, 6]),
        ("two_layer_short", config, [.5, 1, 1.5, 2, 2.5, 3]),
        ("homogeneous", replace(config, c2=config.c1), [.5, 1, 2, 3, 4.5, 6]),
    ]:
        receivers = [Receiver(i+1, d) for i, d in enumerate(offsets)]
        right = recommended_right_boundary(scene_config, receivers)
        if right is not None:
            scene_config = replace(scene_config, width=right)
        save_configuration(scenes / f"{name}.json", scene_config, receivers)
    result = simulate(config)
    report = {
        "model": "u_tt + 2 gamma u_t = c(y)^2 Laplacian(u) + f; constant density",
        "coordinate_system": "source=(0,0), receivers y=0, positive y down; four-sided sponge plus Mur outgoing edges; no surface at y=0",
        "arrival_threshold_au": 1e-4,
        "auto_right_boundary": {"crossover_m": config.crossover_distance,
                                "coverage_factor": config.crossover_factor,
                                "right_boundary_m": config.width,
                                "receiver_right_limit_m": config.receiver_bounds[1]},
        "default_run": {
            "grid_nx_nz": config.shape,
            "steps": config.steps,
            "dt_s": float(result.times[1]),
            "cached_frames": len(result.frames),
            "cache_mib": config.estimated_cache_bytes / 1024**2,
            "solver_elapsed_seconds": result.elapsed_seconds,
            "finite": bool(np.isfinite(result.frames).all() and np.isfinite(result.plane).all()),
        },
    }
    errors = []
    for spacing in [.1, .05, .025]:
        test_config = replace(config, width=1, depth=1, x_min=-1, y_min=-1, spacing=spacing, c1=1, c2=1,
                              interface_depth=.5,
                              duration=.63, boundary="dirichlet", sponge_width=.3)
        initial = lambda x, z: np.sin(np.pi*(x+1)/2)*np.sin(np.pi*(z+1)/2)
        test_result = simulate(test_config, initial=initial, source_enabled=False)
        x, z = np.meshgrid(test_result.x, test_result.z)
        exact = initial(x, z)*np.cos(np.pi/np.sqrt(2)*test_config.duration)
        errors.append(float(np.sqrt(np.mean((test_result.frames[-1]-exact)**2))))
    report["homogeneous_exact_mode"] = {"spacings_m": [.1, .05, .025], "rms_errors": errors, "observed_orders": list(np.log2(np.array(errors[:-1])/errors[1:]))}
    # Independent boundary-contamination diagnostic at a fixed physical mesh/source.
    check_config = replace(config, spacing=.05, max_frames=10)
    base = simulate(check_config)
    expanded = simulate(replace(check_config, x_min=check_config.x_min-1, y_min=check_config.y_min-1, width=check_config.width+1, depth=check_config.depth+1))
    differences = []
    for position in [.5, 1, 2, 3, 4.5, 6]:
        a = pick_arrival(base.times, base.signal(position))
        b = pick_arrival(expanded.times, expanded.signal(position))
        differences.append({"receiver_x": position, "base_pick_s": a, "expanded_pick_s": b,
                            "absolute_change_s": None if a is None or b is None else abs(a-b)})
    report["boundary_enlargement_check"] = {"spacing_m": .05, "extra_buffer_m": 1, "results": differences,
        "scope": "Single scene/threshold diagnostic, not a universal accuracy or boundary-error guarantee."}
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    window = WaveLabWindow()
    window._populate_config(config)
    window.refresh_scene()
    window.show()
    window.accept_result(result)
    window.signal_mode.setCurrentIndex(1)
    window.tabs.setCurrentIndex(1)
    window.slider.setValue(int(np.argmin(abs(result.frame_times-4.0))))
    app.processEvents()
    window.export_to(destination)
    report["default_receivers"] = [
        {"id": r.id, "x_m": r.x, "picked_s": pick_arrival(result.times, result.signal(r.x)),
         "ray_front_s": ray_front(config, r.x), "peak_au": float(np.max(abs(result.signal(r.x))))}
        for r in window.receivers
    ]
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    (destination / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
