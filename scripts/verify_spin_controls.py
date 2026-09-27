"""Check native Qt spinbox hit targets and capture the three editor pages.

Run normally for the desktop platform, or set QT_QPA_PLATFORM=offscreen for CI.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6 import QtCore, QtTest, QtWidgets
from wave_lab.app import WaveLabWindow


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = WaveLabWindow()
    window.show()
    app.processEvents()
    destination = ROOT / "output/wave_spin_fix"
    destination.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, spin, row=None):
        parent = spin.parentWidget()
        while parent:
            if isinstance(parent, QtWidgets.QScrollArea):
                parent.ensureWidgetVisible(spin)
            parent = parent.parentWidget()
        app.processEvents()
        for up in (True, False):
            option = QtWidgets.QStyleOptionSpinBox()
            spin.initStyleOption(option)
            sub = QtWidgets.QStyle.SubControl.SC_SpinBoxUp if up else QtWidgets.QStyle.SubControl.SC_SpinBoxDown
            rect = spin.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_SpinBox, option, sub, spin)
            assert not rect.intersects(spin.lineEdit().geometry()), name
            target = spin.childAt(rect.center()) or spin
            before = spin.value()
            QtTest.QTest.mouseClick(target, QtCore.Qt.MouseButton.LeftButton, pos=target.mapFrom(spin, rect.center()))
            app.processEvents()
            expected = before+(spin.singleStep() if up else -spin.singleStep())
            assert abs(spin.value()-expected) < 1e-9, name
            if row is not None:
                for point in (rect.adjusted(2, 2, -2, -2).topLeft(), rect.adjusted(2, 2, -2, -2).bottomRight()):
                    mapped = spin.mapTo(window.receiver_table.viewport(), point)
                    assert window.receiver_table.indexAt(mapped).row() == row
            checks.append({"control": name, "direction": "up" if up else "down",
                           "before": before, "after": spin.value(), "button_rect": rect.getRect()})

    check("coverage_factor", window.coverage_factor)
    window.auto_right.setChecked(False)
    for name, spin in window.fields.items():
        check(name, spin)
    window.auto_right.setChecked(True)
    window.tabs.widget(0).verticalScrollBar().setValue(0)
    window.noise_enabled.setChecked(True)
    for name in ("threshold", "noise_std", "noise_seed"):
        check(name, getattr(window, name))
    app.processEvents()
    window.grab().save(str(destination / "parameters.png"))
    # processEvents alone does not drain DeferredDelete without app.exec().
    # Parameter edits replaced the table's old cell editors during this audit.
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    window.tabs.setCurrentIndex(1)
    for row in range(len(window.receivers)):
        window.receiver_table.scrollToItem(window.receiver_table.item(row, 0))
        check(f"receiver_{row+1}", window.receiver_table.cellWidget(row, 1), row)
    window.grab().save(str(destination / "receivers.png"))
    window.tabs.setCurrentIndex(2)
    for name in ("inverse_min", "inverse_max", "inverse_budget", "inverse_tolerance"):
        check(name, getattr(window, name))
    window.grab().save(str(destination / "inversion.png"))
    report = {"platform": app.platformName(), "style": app.style().objectName(),
              "device_pixel_ratio": window.devicePixelRatioF(), "checks": checks}
    (destination / "verification.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "checks"} | {"passed_clicks": len(checks)}))
    window.close()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    main()
