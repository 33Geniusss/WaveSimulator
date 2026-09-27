"""Run the default GUI hybrid fit through Qt's real event loop and export evidence."""
from pathlib import Path
import json
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6 import QtCore, QtWidgets
from wave_lab.app import WaveLabWindow
from wave_lab.model import simulate


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = WaveLabWindow()
    window.show()
    destination = ROOT / "output/wave_demo_hybrid"
    destination.mkdir(parents=True, exist_ok=True)
    evidence = {"scope": "Same-model synthetic records; known source/grid/boundaries. No independent accuracy or speedup claim.",
                "platform": app.platformName(), "style": app.style().objectName(), "gui_heartbeats": 0}
    stage = ["clean"]
    errors = []
    timer = QtCore.QTimer()

    def heartbeat():
        evidence["gui_heartbeats"] += 1
    timer.timeout.connect(heartbeat)
    timer.start(20)

    def finish_stage():
        try:
            fit = window.inverse_result
            assert fit is not None and fit.method == "ray_initialized_fd_first_arrival_least_squares", window.inverse_summary.text()
            evidence[stage[0]] = fit.to_dict()
            print(json.dumps({"stage": stage[0], "c1": fit.c1, "c2": fit.c2, "rmse": fit.rmse,
                              "fd_calls": fit.fd_evaluations, "seconds": fit.elapsed_seconds}), flush=True)
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            app.processEvents()
            window.export_to(destination if stage[0] == "clean" else destination / "noise")
            if stage[0] == "clean":
                stage[0] = "noise"
                window.noise_enabled.setChecked(True)
                QtCore.QTimer.singleShot(0, start_fit)
            else:
                (destination / "verification.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
                window.close()
        except Exception as error:
            errors.append(error)
            window.close()

    def start_fit():
        window.run_inversion()
        if not window._inverse_running:
            errors.append(RuntimeError(window.inverse_summary.text()))
            window.close()
            return
        window.inverse_worker.finished.connect(finish_stage)
        window.inverse_worker.progress.connect(lambda n, total, score, c1, c2:
                                               print(f"{stage[0]} candidate {n}/{total}, best_rmse={score}", flush=True) if n % 8 == 0 else None)

    window.accept_result(simulate(window.config))
    QtCore.QTimer.singleShot(0, start_fit)
    app.exec()
    timer.stop()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    if errors:
        raise errors[0]


if __name__ == "__main__":
    main()
