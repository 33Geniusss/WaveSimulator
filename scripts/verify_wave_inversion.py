"""Reproduce GUI inversion evidence using real simulated receiver records."""
from pathlib import Path
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6 import QtCore, QtWidgets

from wave_lab.app import WaveLabWindow
import numpy as np

from wave_lab.model import Receiver, simulate


def main():
    destination = ROOT / "output/wave_demo_inversion"
    destination.mkdir(parents=True, exist_ok=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = WaveLabWindow()
    window.inverse_mode.setCurrentIndex(1)  # This evidence script exercises ray-only inversion.
    window.show()
    result = simulate(window.config)
    window.accept_result(result)
    window.run_inversion()
    app.processEvents()
    window.export_to(destination)
    evidence = {"scope": "PDE waveforms -> threshold picks -> known-depth ray inversion; no statistical accuracy guarantee.",
                "simulation_reference_only": {"c1": result.config.c1, "c2": result.config.c2},
                "clean": window.inverse_result.to_dict()}
    window.noise_enabled.setChecked(True)
    window.run_inversion()
    evidence["default_noise"] = window.inverse_result.to_dict() if window.inverse_result else window.inverse_summary.text()
    app.processEvents()
    window.grab().save(str(destination / "noise_fit.png"))
    window.noise_std.setValue(.001)
    window.run_inversion()
    evidence["strong_noise_diagnostic"] = window.inverse_summary.text()
    app.processEvents()
    window.grab().save(str(destination / "noise_rejection.png"))
    window.noise_enabled.setChecked(False)
    for receiver, position in zip(window.receivers, [.2, .4, .6, .8, 1, 1.2]):
        window.move_receiver(receiver.id, position)
    window.run_inversion()
    evidence["short_array"] = window.inverse_result.to_dict()
    app.processEvents()
    window.grab().save(str(destination / "short_array.png"))
    evidence["variable_receiver_counts"] = []
    for count in [2, 4, 9, 25]:
        window.receivers = [Receiver(i+1, float(x)) for i, x in enumerate(np.linspace(.5, 6, count))]
        window.next_receiver_id = count+1
        window.refresh_handles()
        window.refresh_table()
        window.rebuild_waveforms()
        window.run_inversion()
        evidence["variable_receiver_counts"].append({"count": count, "fit": window.inverse_result.to_dict()})
    app.processEvents()
    window.grab().save(str(destination / "variable_receivers.png"))
    (destination / "verification.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    print(json.dumps({"clean_c1": evidence["clean"]["c1"], "clean_c2": evidence["clean"]["c2"],
                      "clean_rmse_s": evidence["clean"]["rmse"],
                      "short_array_c2": evidence["short_array"]["c2"]}))


if __name__ == "__main__":
    main()
