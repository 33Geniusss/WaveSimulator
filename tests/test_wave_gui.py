import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
import time

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtTest, QtWidgets

from wave_lab.app import WaveLabWindow
from wave_lab.model import NoiseConfig, Receiver, SimulationConfig, pick_arrival, recommended_right_boundary, save_configuration, simulate


@pytest.fixture(scope="module")
def application():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def window(application):
    widget = WaveLabWindow()
    widget.inverse_mode.setCurrentIndex(1)  # Legacy geometric-mode regression tests.
    config = SimulationConfig(width=4, depth=2, x_min=-1, y_min=-1, spacing=.05,
                              interface_depth=1, duration=2.4,
                              sponge_width=.4, max_frames=60)
    widget._populate_config(config)
    widget.receivers = [Receiver(1, 1.2), Receiver(2, 2.4)]
    widget.refresh_scene()
    widget.refresh_table()
    widget.show()
    application.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    application.processEvents()


def wait_for(application, predicate, timeout=15):
    start = time.monotonic()
    while not predicate():
        application.processEvents()
        time.sleep(.005)  # Release the GIL so Python solver workers can run.
        assert time.monotonic() - start < timeout, "GUI operation timed out"
    application.processEvents()


def test_worker_playback_and_parameter_invalidation(application, window):
    beats = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: beats.append(time.monotonic()))
    timer.start(5)
    QtTest.QTest.mouseClick(window.run_button, QtCore.Qt.MouseButton.LeftButton)
    assert not window.config_page.isEnabled()
    wait_for(application, lambda: window.result is not None and not window.worker.isRunning())
    timer.stop()
    assert beats, "GUI event loop must remain live while worker computes"
    assert window.play_button.isEnabled() and window.config_page.isEnabled()
    QtTest.QTest.mouseClick(window.play_button, QtCore.Qt.MouseButton.LeftButton)
    wait_for(application, lambda: window.current_frame >= 2)
    QtTest.QTest.mouseClick(window.play_button, QtCore.Qt.MouseButton.LeftButton)
    assert not window.timer.isActive()
    window.slider.setValue(20)
    assert window.current_frame == 20
    assert window.cursors[0].value() == window.result.frame_times[20]
    QtTest.QTest.mouseClick(window.reset_button, QtCore.Qt.MouseButton.LeftButton)
    assert window.current_frame == 0
    window.fields["c2"].setValue(2.5)
    assert window.result is None and not window.play_button.isEnabled()
    assert not window.export_button.isEnabled()


def test_actual_mouse_drag_changes_signal_without_recompute(application, window):
    result = simulate(window.config)
    window.accept_result(result)
    before = result.signal(window.receivers[0].x).copy()
    handle = window.handles[1]
    viewport = window.field_widget.viewport()
    start = window.field_widget.mapFromScene(handle.mapToScene(QtCore.QPointF(0, 0)))
    target_scene = window.field_plot.vb.mapViewToScene(QtCore.QPointF(1.7, .65))
    end = window.field_widget.mapFromScene(target_scene)
    QtTest.QTest.mouseMove(viewport, start)
    QtTest.QTest.mousePress(viewport, QtCore.Qt.MouseButton.LeftButton, pos=start)
    for fraction in [.25, .5, .75, 1]:
        point = start + (end-start)*fraction
        event = QtGui.QMouseEvent(QtCore.QEvent.Type.MouseMove, QtCore.QPointF(point),
                                 QtCore.QPointF(viewport.mapToGlobal(point)),
                                 QtCore.Qt.MouseButton.NoButton, QtCore.Qt.MouseButton.LeftButton,
                                 QtCore.Qt.KeyboardModifier.NoModifier)
        QtWidgets.QApplication.sendEvent(viewport, event)
        application.processEvents()
    QtTest.QTest.mouseRelease(viewport, QtCore.Qt.MouseButton.LeftButton, pos=end)
    application.processEvents()
    assert window.receivers[0].x == pytest.approx(1.7, abs=.03)
    assert handle.pos().y() == window.config.source_depth
    assert window.result is result
    assert not np.allclose(before, result.signal(window.receivers[0].x))
    assert window.receiver_table.currentRow() == 0
    handle.setPos(-100, 10)
    assert handle.pos().x() == window.config.receiver_bounds[0]
    assert handle.pos().y() == window.config.source_depth


def test_receiver_controls_modes_export_and_load(application, window, tmp_path):
    window.accept_result(simulate(window.config))
    original = window.result
    window.tabs.setCurrentIndex(1)
    QtTest.QTest.mouseClick(window.add_button, QtCore.Qt.MouseButton.LeftButton)
    assert len(window.receivers) == 3
    QtTest.QTest.mouseClick(window.remove_button, QtCore.Qt.MouseButton.LeftButton)
    assert len(window.receivers) == 2
    window.signal_mode.setCurrentIndex(1)
    window.threshold.setValue(.0002)
    window.ray_checkbox.setChecked(True)
    assert window.result is original
    window.receiver_table.cellWidget(0, 1).setValue(1.6)
    assert window.receivers[0].x == pytest.approx(1.6)
    assert window.handles[1].pos().x() == pytest.approx(1.6)
    window.export_to(tmp_path / "run")
    assert (tmp_path / "run/screenshot.png").stat().st_size > 10000
    assert (tmp_path / "run/signals.csv").exists()
    window.save_to(tmp_path / "saved.json")
    window.move_receiver(1, 2)
    window.load_from(tmp_path / "saved.json")
    assert window.result is None
    assert window.receivers[0].x == pytest.approx(1.6)
    assert window.threshold.value() == .0002


def test_worker_cancellation(application, window):
    window.fields["spacing"].setValue(.01)
    window.fields["duration"].setValue(10)
    window.start_simulation()
    QtTest.QTest.mouseClick(window.cancel_button, QtCore.Qt.MouseButton.LeftButton)
    wait_for(application, lambda: not window.worker.isRunning())
    assert window.result is None
    assert window.config_page.isEnabled()
    assert window.run_button.isEnabled()
    assert not window.export_button.isEnabled()


def test_no_detection_and_not_yet_arrived_are_distinct(application, window):
    result = simulate(window.config)
    window.accept_result(result)
    assert "待到" in window.receiver_table.item(0, 2).text()
    window.slider.setValue(len(result.frames)-1)
    assert "已到" in window.receiver_table.item(0, 2).text()
    result.plane[:] = 0
    window.rebuild_waveforms()
    assert window.receiver_table.item(0, 2).text() == "未检出"


def test_gui_fixed_origin_zero_depth_and_negative_x(application, window, tmp_path):
    assert "source_x" not in window.fields and "source_depth" not in window.fields
    assert window.config.source_x == window.config.source_depth == 0
    assert window.boundary.count() == 1 and window.boundary.currentData() == "sponge"
    assert len(window.sponge_lines) == 4
    window.move_receiver(1, -.5)
    assert window.receivers[0].x == -.5
    assert window.handles[1].pos().y() == 0
    window.handles[1].setPos(0, 100)
    assert window.receivers[0].x == 0
    assert window.handles[1].pos().y() == 0
    window.save_to(tmp_path / "origin.json")
    window.load_from(tmp_path / "origin.json")
    assert window.receivers[0].x == 0
    assert all(handle.pos().y() == 0 for handle in window.handles.values())


def test_automatic_crossover_tracks_medium_and_preserves_receivers(application, window, tmp_path):
    positions = [r.x for r in window.receivers]
    window.auto_right.setChecked(True)
    before = window.config.width
    assert window.fields["width"].isReadOnly()
    assert before == recommended_right_boundary(window.config, window.receivers)
    assert window.config.receiver_bounds[1] > 1.5*window.config.crossover_distance
    assert "尚无测点" in window.coverage_label.text()
    window.fields["c2"].setValue(1.1)
    assert window.config.width > before
    assert [r.x for r in window.receivers] == positions
    assert window.crossover_line.value() == window.config.crossover_distance
    assert "时长可能不足" in window.coverage_label.text()
    window.save_to(tmp_path / "auto.json")
    window.auto_right.setChecked(False)
    assert not window.fields["width"].isReadOnly()
    window.load_from(tmp_path / "auto.json")
    assert window.auto_right.isChecked()
    assert window.fields["width"].isReadOnly()
    window.coverage_factor.setValue(2)
    assert window.config.receiver_bounds[1] > 2*window.config.crossover_distance


def test_auto_boundary_missing_head_branch_and_oversized_domain(application, window):
    window.auto_right.setChecked(True)
    window.fields["c2"].setValue(1)
    assert not window.fields["width"].isReadOnly()
    assert not window.crossover_line.isVisible()
    assert "不存在" in window.coverage_label.text()
    window.fields["depth"].setValue(20)
    window.fields["interface_depth"].setValue(10)
    window.fields["c2"].setValue(1.01)
    assert "超过" in window.coverage_label.text()
    assert not window.run_button.isEnabled()
    assert window._auto_right_error


def test_noise_toggle_seed_threshold_and_receiver_changes(application, window):
    result = simulate(window.config)
    window.accept_result(result)
    receiver = window.receivers[0]
    clean = result.signal(receiver.x).copy()
    field = result.plane.copy()
    assert not window.noise_enabled.isChecked()
    assert not window.noise_std.isEnabled()
    QtTest.QTest.mouseClick(window.noise_enabled, QtCore.Qt.MouseButton.LeftButton,
                          pos=QtCore.QPoint(8, window.noise_enabled.height()//2))
    assert window.noise_std.isEnabled()
    window.noise_std.setValue(.001)
    observed = window.receiver_signal(receiver)
    assert not np.array_equal(observed, clean)
    curves = [item for item in window.signal_plot.listDataItems() if item.name() == "R1"]
    np.testing.assert_array_equal(curves[0].getData()[1], observed)
    arrival = pick_arrival(result.times, observed, window.threshold.value())
    assert window.receiver_table.item(0, 2).text().startswith(f"{arrival:.3f}")
    window.threshold.setValue(.0002)
    window.signal_mode.setCurrentIndex(1)
    window.slider.setValue(10)
    window.add_receiver()
    np.testing.assert_array_equal(window.receiver_signal(receiver), observed)
    window.move_receiver(receiver.id, 1.6)
    np.testing.assert_allclose(window.receiver_signal(receiver)-result.signal(receiver.x), observed-clean, atol=1e-17)
    window.move_receiver(receiver.id, 1.2)
    QtTest.QTest.mouseClick(window.noise_resample, QtCore.Qt.MouseButton.LeftButton)
    assert window.noise_seed.value() == 6731
    assert not np.array_equal(window.receiver_signal(receiver), observed)
    window.noise_seed.setValue(6730)
    np.testing.assert_array_equal(window.receiver_signal(receiver), observed)
    window.noise_enabled.setChecked(False)
    np.testing.assert_array_equal(window.receiver_signal(receiver), clean)
    window.noise_enabled.setChecked(True)
    np.testing.assert_array_equal(window.receiver_signal(receiver), observed)
    assert window.result is result
    np.testing.assert_array_equal(result.plane, field)


def test_noise_gui_save_load_and_export_match_display(application, window, tmp_path):
    window.accept_result(simulate(window.config))
    window.noise_enabled.setChecked(True)
    window.noise_std.setValue(.0005)
    window.noise_seed.setValue(42)
    expected = window.receiver_signal(window.receivers[0])
    window.export_to(tmp_path / "run")
    actual = np.loadtxt(tmp_path / "run/signals.csv", delimiter=",", skiprows=1, encoding="utf-8-sig")
    np.testing.assert_array_equal(actual[:, 1], expected)
    window.save_to(tmp_path / "scene.json")
    window.noise_enabled.setChecked(False)
    window.noise_seed.setValue(11)
    window.load_from(tmp_path / "scene.json")
    assert window.read_noise() == NoiseConfig(True, .0005, 42)
    window.accept_result(simulate(window.config))
    np.testing.assert_array_equal(window.receiver_signal(window.receivers[0]), expected)


def prepare_inverse_window(window):
    config = SimulationConfig(spacing=.05, max_frames=30)
    window._populate_config(config)
    window.receivers = [Receiver(i+1, x) for i, x in enumerate([.5, 1, 2, 3, 4.5, 6])]
    window.refresh_scene()
    window.refresh_table()
    window.accept_result(simulate(config))


def prepare_hybrid_window(window):
    config = SimulationConfig(width=5, depth=2, x_min=-1, y_min=-1, spacing=.06,
                              interface_depth=.7, c1=1, c2=1.8, duration=4,
                              sponge_width=.4, max_frames=20)
    window._populate_config(config)
    window.receivers = [Receiver(i+1, x) for i, x in enumerate([.5, 1, 1.8, 2.6, 3.6, 4.4])]
    window.refresh_scene()
    window.refresh_table()
    window.accept_result(simulate(config))
    window.inverse_mode.setCurrentIndex(0)


def test_hybrid_background_result_and_export(application, window, tmp_path):
    import json
    prepare_hybrid_window(window)
    original = window.result
    ticks = []
    timer = QtCore.QTimer()
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start(10)
    window.run_inversion()
    assert window._inverse_running and window.inverse_cancel.isEnabled()
    assert not window.config_page.isEnabled() and not window.tabs.widget(1).isEnabled()
    assert not window.threshold.isEnabled() and not window.export_button.isEnabled()
    with pytest.raises(ValueError, match="反演"):
        window.export_to(tmp_path / "busy")
    wait_for(application, lambda: not window._inverse_running)
    timer.stop()
    fit = window.inverse_result
    assert ticks and fit is not None and fit.identifiable
    assert fit.c2 == pytest.approx(1.8, abs=.002)
    assert window.result is original and window.config_page.isEnabled()
    assert window.inverse_button.isEnabled() and not window.inverse_cancel.isEnabled()
    assert window.inverse_table.rowCount() == 6
    assert "几何初值" in window.inverse_summary.text() and "有限差分求解" in window.inverse_summary.text()
    window.export_to(tmp_path / "hybrid")
    payload = json.loads((tmp_path / "hybrid/inversion.json").read_text(encoding="utf-8"))
    assert payload["method"] == "ray_initialized_fd_first_arrival_least_squares"
    assert payload["ray_stage"]["c1"] == fit.ray_stage["c1"]
    assert payload["fd_evaluations"] > 0 and len(payload["evaluation_history"]) >= payload["fd_evaluations"]
    window.inverse_budget.setValue(30)
    assert window.inverse_result is None


def test_hybrid_cancel_keeps_observation_and_unblocks_controls(application, window):
    prepare_hybrid_window(window)
    original = window.result
    window.run_inversion()
    QtTest.QTest.mouseClick(window.inverse_cancel, QtCore.Qt.MouseButton.LeftButton)
    wait_for(application, lambda: not window._inverse_running)
    assert window.inverse_result is None and window.result is original
    assert "取消" in window.inverse_summary.text()
    assert window.config_page.isEnabled() and window.threshold.isEnabled()
    assert window.inverse_button.isEnabled() and not window.noise_std.isEnabled()


def test_hybrid_rejects_changed_input_during_background_job(application, window):
    prepare_hybrid_window(window)
    window.run_inversion()
    # A programmatic edit bypasses disabled controls: still cancel/discard.
    window.threshold.setValue(.0002)
    wait_for(application, lambda: not window._inverse_running)
    assert window.inverse_result is None
    assert window.inverse_button.isEnabled()


def test_hybrid_settings_roundtrip_and_legacy_defaults(application, window, tmp_path):
    import json
    path = tmp_path / "scene.json"
    window.inverse_mode.setCurrentIndex(0)
    window.inverse_budget.setValue(32)
    window.inverse_tolerance.setValue(.0002)
    window.save_to(path)
    window.inverse_mode.setCurrentIndex(1)
    window.load_from(path)
    assert window.inverse_mode.currentData() == "hybrid" and window.inverse_budget.value() == 32
    assert window.inverse_tolerance.value() == .0002
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inversion"] = {"speed_min": .1, "speed_max": 100}
    path.write_text(json.dumps(payload), encoding="utf-8")
    window.load_from(path)
    assert window.inverse_mode.currentData() == "hybrid" and window.inverse_budget.value() == 48
    payload["inversion"]["max_evaluations"] = 201
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="预算"):
        window.load_from(path)


def test_six_receiver_inversion_ui_export_and_invalidation(application, window, tmp_path):
    assert not window.inverse_button.isEnabled()
    prepare_inverse_window(window)
    result = window.result
    window.tabs.setCurrentIndex(2)
    application.processEvents()
    QtTest.QTest.mouseClick(window.inverse_button, QtCore.Qt.MouseButton.LeftButton)
    fit = window.inverse_result
    assert fit is not None and fit.identifiable
    assert window.inverse_table.rowCount() == 6
    assert window.result_tabs.currentIndex() == 1
    assert "c1 估计" in window.inverse_summary.text()
    original_config = result.config
    result.config = replace(original_config, c1=.7, c2=4)
    window.run_inversion()
    assert window.inverse_result.c1 == fit.c1 and window.inverse_result.c2 == fit.c2
    result.config = original_config
    fit = window.inverse_result
    window.slider.setValue(5)
    window.signal_mode.setCurrentIndex(1)
    window.ray_checkbox.setChecked(True)
    assert window.inverse_result is fit  # display-only changes preserve the fit
    window.export_to(tmp_path / "run")
    import json
    payload = json.loads((tmp_path / "run/inversion.json").read_text(encoding="utf-8"))
    assert payload["c1"] == fit.c1 and payload["c2"] == fit.c2
    assert payload["receiver_ids"] == [1, 2, 3, 4, 5, 6]
    observed = np.loadtxt(tmp_path / "run/signals.csv", delimiter=",", skiprows=1, encoding="utf-8-sig")
    for i in range(6):
        assert payload["observed_times"][i] == pick_arrival(observed[:, 0], observed[:, i+1], window.threshold.value())
    window.threshold.setValue(.0002)
    assert window.result is result and window.inverse_result is None
    assert window.inverse_table.rowCount() == 0
    window.export_to(tmp_path / "run")
    assert not (tmp_path / "run/inversion.json").exists()
    window.run_inversion()
    assert window.inverse_result is not None
    window.move_receiver(1, .7)
    assert window.inverse_result is None
    window.run_inversion()
    window.noise_enabled.setChecked(True)
    assert window.inverse_result is None
    window.noise_enabled.setChecked(False)
    window.run_inversion()
    window.inverse_max.setValue(50)
    assert window.inverse_result is None
    window.add_receiver()
    assert window.inverse_button.isEnabled()
    assert "7 个" in window.inverse_button.text()
    window.run_inversion()
    assert window.inverse_result is not None
    assert window.inverse_table.rowCount() == 7
    window.inverse_min.setValue(.2)
    window.save_to(tmp_path / "settings.json")
    window.inverse_min.setValue(.1)
    window.inverse_max.setValue(100)
    window.load_from(tmp_path / "settings.json")
    assert window.inverse_min.value() == .2 and window.inverse_max.value() == 50


def test_inversion_reports_missing_and_noise_triggered_picks(application, window):
    prepare_inverse_window(window)
    window.threshold.setValue(1)
    window.run_inversion()
    assert window.inverse_result is None
    assert "未检出" in window.inverse_summary.text()
    window.threshold.setValue(.0001)
    window.noise_std.setValue(.001)
    window.noise_enabled.setChecked(True)
    window.run_inversion()
    assert window.inverse_result is None
    assert "t=0" in window.inverse_summary.text()


def test_variable_count_ui_export_config_and_add_past_old_limit(application, window, tmp_path):
    import json
    prepare_inverse_window(window)
    original = window.result
    for count in [2, 4, 9, 24]:
        window.receivers = [Receiver(i+1, float(x)) for i, x in enumerate(np.linspace(1, 6, count))]
        window.next_receiver_id = count+1
        window.refresh_handles()
        window.refresh_table()
        window.rebuild_waveforms()
        assert window.inverse_result is None and window.inverse_button.isEnabled()
        window.run_inversion()
        assert window.inverse_result is not None and window.inverse_result.identifiable
        assert window.inverse_table.rowCount() == count
        assert len(window.inverse_result.observed_times) == count
        assert window.result is original
    window.tabs.setCurrentIndex(1)
    application.processEvents()
    QtTest.QTest.mouseClick(window.add_button, QtCore.Qt.MouseButton.LeftButton)
    assert len(window.receivers) == 25 and window.inverse_result is None
    window.run_inversion()
    assert window.inverse_table.rowCount() == 25
    window.export_to(tmp_path / "many")
    payload = json.loads((tmp_path / "many/inversion.json").read_text(encoding="utf-8"))
    assert len(payload["receiver_ids"]) == len(payload["observed_times"]) == 25
    observed = np.loadtxt(tmp_path / "many/signals.csv", delimiter=",", skiprows=1, encoding="utf-8-sig")
    assert observed.shape[1] == 26
    window.load_from(tmp_path / "many/configuration.json")
    assert len(window.receivers) == 25
    assert window.result is None and window.inverse_result is None


def test_single_receiver_shows_insufficient_information(application, window):
    window.receivers = [Receiver(1, 1.2)]
    window.refresh_handles()
    window.refresh_table()
    window.accept_result(simulate(window.config))
    assert window.inverse_button.isEnabled()
    window.run_inversion()
    assert window.inverse_result is None
    assert "至少需要 2" in window.inverse_summary.text()


def spin_button_rect(spin, up):
    option = QtWidgets.QStyleOptionSpinBox()
    spin.initStyleOption(option)
    subcontrol = QtWidgets.QStyle.SubControl.SC_SpinBoxUp if up else QtWidgets.QStyle.SubControl.SC_SpinBoxDown
    return spin.style().subControlRect(QtWidgets.QStyle.ComplexControl.CC_SpinBox, option, subcontrol, spin)


def click_spin_target(spin, point):
    # Route to the actual child under the pointer. Sending directly to the
    # spinbox bypasses the Windows 11 bug: its line edit covered the up arrow.
    target = spin.childAt(point) or spin
    QtTest.QTest.mouseClick(target, QtCore.Qt.MouseButton.LeftButton, pos=target.mapFrom(spin, point))


@pytest.mark.parametrize("style_name", ["Fusion", "windows11"])
def test_all_numeric_arrows_have_unobstructed_click_regions(application, window, style_name):
    available = {key.lower() for key in QtWidgets.QStyleFactory.keys()}
    if style_name.lower() not in available:
        pytest.skip(f"Qt style {style_name} not available")
    previous = application.style().objectName()
    application.setStyle(style_name)
    try:
        window.auto_right.setChecked(True)
        application.processEvents()

        def check_editor(spin):
            parent = spin.parentWidget()
            while parent is not None:
                if isinstance(parent, QtWidgets.QScrollArea):
                    parent.ensureWidgetVisible(spin)
                parent = parent.parentWidget()
            application.processEvents()
            for up in (True, False):
                rect = spin_button_rect(spin, up)
                assert rect.width() >= 24
                assert not rect.intersects(spin.lineEdit().geometry())
                inner = rect.adjusted(3, 3, -3, -3)
                # Center and corners, not just the tiny arrow glyph itself.
                for point in (inner.center(), inner.topLeft(), inner.topRight(), inner.bottomLeft(), inner.bottomRight()):
                    before = spin.value()
                    click_spin_target(spin, point)
                    application.processEvents()
                    expected = before+(spin.singleStep() if up else -spin.singleStep())
                    expected = max(spin.minimum(), min(spin.maximum(), expected))
                    assert spin.value() == pytest.approx(expected, abs=1e-10)
        # Coverage needs auto mode; right boundary needs manual mode.
        check_editor(window.coverage_factor)
        window.auto_right.setChecked(False)
        for spin in window.fields.values():
            check_editor(spin)
        window.noise_enabled.setChecked(True)
        for spin in (window.threshold, window.noise_std, window.noise_seed):
            check_editor(spin)
        window.tabs.setCurrentIndex(2)
        window.inverse_mode.setCurrentIndex(0)
        for spin in (window.inverse_min, window.inverse_max, window.inverse_budget, window.inverse_tolerance):
            check_editor(spin)
        window.tabs.setCurrentIndex(1)
        check_editor(window.receiver_table.cellWidget(0, 1))
        window.add_receiver()
        check_editor(window.receiver_table.cellWidget(len(window.receivers)-1, 1))
        for row in range(len(window.receivers)):
            spin = window.receiver_table.cellWidget(row, 1)
            for up in (True, False):
                point = spin.mapTo(window.receiver_table.viewport(), spin_button_rect(spin, up).center())
                assert window.receiver_table.indexAt(point).row() == row
    finally:
        application.setStyle(previous)


def test_numeric_arrow_repeat_and_keyboard_editing(application, window):
    spin = window.noise_seed
    window.noise_enabled.setChecked(True)
    point = spin_button_rect(spin, True).center()
    before = spin.value()
    for _ in range(12):
        click_spin_target(spin, point)
    assert spin.value() == before+12
    # Keep native auto-repeat and normal keyboard entry.
    QtTest.QTest.mousePress(spin, QtCore.Qt.MouseButton.LeftButton, pos=point)
    QtTest.QTest.qWait(800)
    QtTest.QTest.mouseRelease(spin, QtCore.Qt.MouseButton.LeftButton, pos=point)
    assert spin.value() > before+13
    spin.lineEdit().selectAll()
    QtTest.QTest.keyClicks(spin.lineEdit(), "12345")
    QtTest.QTest.keyClick(spin.lineEdit(), QtCore.Qt.Key.Key_Return)
    assert spin.value() == 12345


def test_numeric_arrows_respect_readonly_disabled_and_limits(application, window):
    window.auto_right.setChecked(True)
    spin = window.fields["width"]
    before = spin.value()
    assert spin.isReadOnly()
    click_spin_target(spin, spin_button_rect(spin, True).center())
    assert spin.value() == before
    spin = window.noise_seed
    before = spin.value()
    assert not spin.isEnabled()
    click_spin_target(spin, spin_button_rect(spin, True).center())
    assert spin.value() == before
    window.noise_enabled.setChecked(True)
    for value, up in ((spin.minimum(), False), (spin.maximum(), True)):
        spin.setValue(value)
        click_spin_target(spin, spin_button_rect(spin, up).center())
        assert spin.value() == value
