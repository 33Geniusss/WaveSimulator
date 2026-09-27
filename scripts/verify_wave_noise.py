"""Render the noise controls and export a reproducible noisy example."""
from pathlib import Path
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PySide6 import QtCore, QtWidgets

from wave_lab.app import WaveLabWindow
from wave_lab.model import pick_arrival, simulate


def main():
    destination = ROOT / "output/wave_demo_noise"
    destination.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = WaveLabWindow()
    window.show()
    result = simulate(window.config)
    window.accept_result(result)
    window.signal_mode.setCurrentIndex(1)
    window.tabs.setCurrentIndex(1)
    window.slider.setValue(int(np.argmin(abs(result.frame_times-4))))
    app.processEvents()
    window.grab().save(str(destination / "noise_off.png"))
    # Deliberately visible noise, above the default detection threshold:
    # demonstrates false triggering rather than silently hiding it.
    window.noise_std.setValue(.0002)
    window.noise_enabled.setChecked(True)
    app.processEvents()
    window.export_to(destination)
    report = {"noise_std_au": window.noise_std.value(), "seed": window.noise_seed.value(),
              "threshold_au": window.threshold.value(),
              "scope": "Illustrative strong-noise case, not an error-rate study.",
              "receivers": []}
    for receiver in window.receivers:
        clean = result.signal(receiver.x)
        observed = window.receiver_signal(receiver)
        report["receivers"].append({"id": receiver.id, "x_m": receiver.x,
                                   "clean_pick_s": pick_arrival(result.times, clean, window.threshold.value()),
                                   "noisy_pick_s": pick_arrival(result.times, observed, window.threshold.value()),
                                   "measured_noise_std_au": float(np.std(observed-clean))})
    (destination / "verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
