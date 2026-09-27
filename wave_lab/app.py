from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import threading
import os
import json

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from .inversion import invert_first_arrivals, travel_times
from .hybrid import refine_first_arrivals

from .model import (NoiseConfig, Receiver, SimulationCancelled, SimulationConfig, SimulationResult,
                    add_measurement_noise,
                    export_results, load_configuration, pick_arrival, ray_front,
                    recommended_right_boundary, save_configuration, simulate)


COLORS = ["#38bdf8", "#fb923c", "#a78bfa", "#4ade80", "#f472b6", "#facc15", "#2dd4bf", "#e879f9"]


def configure_font():
    # Qt's offscreen platform on Windows does not enumerate system fonts.
    # Explicit registration also makes automated render QA representative.
    if "Microsoft YaHei UI" not in QtGui.QFontDatabase.families():
        font_directory = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        for filename in ("msyh.ttc", "msyhbd.ttc"):
            if (font_directory / filename).exists():
                QtGui.QFontDatabase.addApplicationFont(str(font_directory / filename))
    app = QtWidgets.QApplication.instance()
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))


class ReceiverHandle(pg.TargetItem):
    selected = QtCore.Signal(int)

    def __init__(self, receiver: Receiver, depth: float, bounds: tuple[float, float]):
        self.receiver_id = receiver.id
        self.fixed_depth = depth
        self.bounds = bounds
        color = COLORS[(receiver.id - 1) % len(COLORS)]
        super().__init__((receiver.x, depth), size=17, symbol="o", pen=pg.mkPen(color, width=2),
                         brush=pg.mkBrush(color + "60"), hoverPen="w", movable=True,
                         label=f"R{receiver.id}", labelOpts={"color": color, "offset": (0, 16)})
        self.setZValue(20)

    def setPos(self, *args):
        position = pg.Point(*args)
        super().setPos(float(np.clip(position.x(), *self.bounds)), self.fixed_depth)

    def mouseDragEvent(self, event):
        if event.isStart():
            self.selected.emit(self.receiver_id)
        super().mouseDragEvent(event)

    def mouseClickEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.selected.emit(self.receiver_id)
            event.accept()
        else:
            event.ignore()


class SolverWorker(QtCore.QThread):
    progress = QtCore.Signal(int, float, object)
    completed = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.cancel_event = threading.Event()

    def run(self):
        try:
            result = simulate(self.config, self.cancel_event, self.progress.emit)
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.completed.emit(result)
        except SimulationCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class InverseWorker(QtCore.QThread):
    progress = QtCore.Signal(int, int, object, float, float)
    completed = QtCore.Signal(object)
    cancelled = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, config, positions, arrivals, times, threshold, options, parent=None):
        super().__init__(parent)
        self.arguments = (config, list(positions), list(arrivals), times.copy(), threshold)
        self.options = dict(options)
        self.cancel_event = threading.Event()

    def run(self):
        try:
            fit = refine_first_arrivals(*self.arguments, **self.options,
                                       cancel=self.cancel_event, progress=self.progress.emit)
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.completed.emit(fit)
        except SimulationCancelled:
            self.cancelled.emit()
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class WaveLabWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        configure_font()
        self.setWindowTitle("Wave Lab 2D · 多接收器波速反演")
        self.resize(1440, 960)
        self.config = SimulationConfig(auto_right=True)
        self.receivers = [Receiver(i + 1, self.config.source_x + d) for i, d in enumerate([0.5, 1, 2, 3, 4.5, 6])]
        self.next_receiver_id = 7
        self.result: SimulationResult | None = None
        self.inverse_result = None
        self.inverse_signature = None
        self.inverse_worker = None
        self._inverse_running = False
        self._inverse_job_signature = None
        self.worker: SolverWorker | None = None
        self.handles: dict[int, ReceiverHandle] = {}
        self.cursors = []
        self.current_frame = 0
        self._loading = False
        self._auto_right_error = ""
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(30)
        self.timer.timeout.connect(self.advance)
        self.elapsed = QtCore.QElapsedTimer()
        self.play_time = 0.0
        self.display_limit = 1.0
        self._build_ui()
        self._populate_config(self.config)
        self.refresh_scene()
        self.refresh_table()
        self.rebuild_waveforms()
        self.set_result_controls(False)

    def _build_ui(self):
        pg.setConfigOptions(antialias=True, background="#101b2b", foreground="#cbd5e1")
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        title = QtWidgets.QLabel("WAVE LAB 2D     S = (0, 0) · 接收器 y = 0 · 四侧吸收")
        title.setStyleSheet("font-size: 21px; font-weight: 600; padding: 8px;")
        layout.addWidget(title)
        body = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        layout.addWidget(body, 1)
        sidebar = QtWidgets.QWidget()
        side = QtWidgets.QVBoxLayout(sidebar)
        side.setContentsMargins(2, 2, 6, 2)
        tabs = QtWidgets.QTabWidget()
        side.addWidget(tabs, 1)
        config_page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(config_page)
        form.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.fields = {}
        self.auto_right = QtWidgets.QCheckBox("按交叉距离自动设置右边界")
        self.auto_right.toggled.connect(self.parameters_changed)
        form.addRow(self.auto_right)
        self.coverage_factor = QtWidgets.QDoubleSpinBox()
        self.coverage_factor.setRange(1.1, 3)
        self.coverage_factor.setDecimals(2)
        self.coverage_factor.setSingleStep(.1)
        self.coverage_factor.setValue(1.5)
        self.coverage_factor.setSuffix(" × dc")
        self.coverage_factor.setKeyboardTracking(False)
        self.coverage_factor.setToolTip("默认有效观测范围至少覆盖 1.5 倍交叉距离，另留缓冲与吸收层；这是可调工程余量，不保证弱头波一定被检出。")
        self.coverage_factor.valueChanged.connect(self.parameters_changed)
        form.addRow("头波区覆盖范围", self.coverage_factor)
        definitions = [
            ("x_min", "左边界 x / m", -100, -0.1, 0.5, 2),
            ("width", "右边界 x / m", 0.5, 100, 0.5, 2),
            ("y_min", "上边界 y / m", -100, -0.1, 0.5, 2),
            ("depth", "下边界 y / m", 0.5, 100, 0.5, 2),
            ("spacing", "网格间距上限 / m", 0.005, 0.5, 0.005, 3),
            ("c1", "上层波速 / m/s", 0.1, 100, 0.1, 2),
            ("c2", "下层波速 / m/s", 0.1, 100, 0.1, 2),
            ("interface_depth", "界面深度 / m", 0.01, 99, 0.05, 2),
            ("frequency", "脉冲频率 / Hz", 0.1, 100, 0.5, 2),
            ("duration", "模拟时长 / s", 0.1, 100, 0.5, 2),
            ("sponge_width", "吸收层宽度 / m", 0.01, 20, 0.1, 2),
            ("sponge_strength", "吸收阻尼 / 1/s", 0, 100, 1, 1),
        ]
        for key, label, minimum, maximum, step, decimals in definitions:
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(minimum, maximum)
            box.setSingleStep(step)
            box.setDecimals(decimals)
            box.setKeyboardTracking(False)
            box.valueChanged.connect(self.parameters_changed)
            self.fields[key] = box
            form.addRow(label, box)
        self.boundary = QtWidgets.QComboBox()
        self.boundary.addItem("四侧吸收（无地表反射）", "sponge")
        self.boundary.currentIndexChanged.connect(self.parameters_changed)
        form.addRow("边界条件", self.boundary)
        self.resolution_label = QtWidgets.QLabel()
        self.resolution_label.setWordWrap(True)
        form.addRow(self.resolution_label)
        hint = QtWidgets.QLabel("源固定 (0, 0)，所有接收器 y=0。\ny 向下为正；y<0 是上层介质缓冲。\n观测线不反射，四周吸收返回波。")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #94a3b8; padding-top: 8px;")
        form.addRow(hint)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(config_page)
        tabs.addTab(scroll, "场景参数")
        self.config_page = config_page

        receiver_page = QtWidgets.QWidget()
        receiver_layout = QtWidgets.QVBoxLayout(receiver_page)
        info = QtWidgets.QLabel("拖动圆点或修改 x；所有接收器 y=0。\n首波使用固定绝对幅值阈值，不依赖后续峰值。")
        info.setWordWrap(True)
        receiver_layout.addWidget(info)
        self.receiver_table = QtWidgets.QTableWidget(0, 3)
        self.receiver_table.setHorizontalHeaderLabels(["接收器", "x / m", "首波 / s"])
        self.receiver_table.verticalHeader().hide()
        self.receiver_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.receiver_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.receiver_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        receiver_layout.addWidget(self.receiver_table, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.add_button = QtWidgets.QPushButton("+ 添加")
        self.remove_button = QtWidgets.QPushButton("− 删除选中")
        self.add_button.clicked.connect(self.add_receiver)
        self.remove_button.clicked.connect(self.remove_receiver)
        buttons.addWidget(self.add_button)
        buttons.addWidget(self.remove_button)
        receiver_layout.addLayout(buttons)
        tabs.addTab(receiver_page, "接收器")
        self.tabs = tabs

        inverse_page = QtWidgets.QWidget()
        inverse_layout = QtWidgets.QVBoxLayout(inverse_page)
        inverse_hint = QtWidgets.QLabel("使用当前所有接收器的首波到时。\n界面深度已知，源时刻为 t=0。\n拟合假设：水平两层、c2 > c1。")
        inverse_hint.setWordWrap(True)
        inverse_layout.addWidget(inverse_hint)
        self.inverse_depth = QtWidgets.QLabel()
        inverse_layout.addWidget(self.inverse_depth)
        self.inverse_mode = QtWidgets.QComboBox()
        self.inverse_mode.addItem("几何初值 + 有限差分到时拟合", "hybrid")
        self.inverse_mode.addItem("仅几何射线（快速对照）", "ray")
        self.inverse_mode.currentIndexChanged.connect(self.refresh_inverse_state)
        inverse_layout.addWidget(self.inverse_mode)
        bounds_form = QtWidgets.QFormLayout()
        self.inverse_min = QtWidgets.QDoubleSpinBox()
        self.inverse_max = QtWidgets.QDoubleSpinBox()
        for editor, value in ((self.inverse_min, .1), (self.inverse_max, 100.)):
            editor.setRange(.01, 1000)
            editor.setDecimals(2)
            editor.setValue(value)
            editor.setKeyboardTracking(False)
            editor.setSuffix(" m/s")
            editor.valueChanged.connect(self.refresh_inverse_state)
        bounds_form.addRow("波速搜索下限", self.inverse_min)
        bounds_form.addRow("波速搜索上限", self.inverse_max)
        self.inverse_budget = QtWidgets.QSpinBox()
        self.inverse_budget.setRange(8, 200)
        self.inverse_budget.setValue(48)
        self.inverse_budget.setKeyboardTracking(False)
        self.inverse_budget.valueChanged.connect(self.refresh_inverse_state)
        bounds_form.addRow("有限差分搜索预算", self.inverse_budget)
        self.inverse_tolerance = QtWidgets.QDoubleSpinBox()
        self.inverse_tolerance.setRange(1e-8, 1)
        self.inverse_tolerance.setDecimals(8)
        self.inverse_tolerance.setValue(.0001)
        self.inverse_tolerance.setSingleStep(.0001)
        self.inverse_tolerance.setKeyboardTracking(False)
        self.inverse_tolerance.valueChanged.connect(self.refresh_inverse_state)
        bounds_form.addRow("目标到时 RMSE / s", self.inverse_tolerance)
        inverse_layout.addLayout(bounds_form)
        self.inverse_button = QtWidgets.QPushButton("用当前接收器反演")
        self.inverse_button.clicked.connect(self.run_inversion)
        inverse_layout.addWidget(self.inverse_button)
        self.inverse_cancel = QtWidgets.QPushButton("取消反演")
        self.inverse_cancel.setEnabled(False)
        self.inverse_cancel.clicked.connect(self.cancel_inversion)
        inverse_layout.addWidget(self.inverse_cancel)
        self.inverse_progress = QtWidgets.QProgressBar()
        self.inverse_progress.setRange(0, 48)
        self.inverse_progress.setValue(0)
        inverse_layout.addWidget(self.inverse_progress)
        self.inverse_summary = QtWidgets.QLabel()
        self.inverse_summary.setWordWrap(True)
        self.inverse_summary.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        inverse_layout.addWidget(self.inverse_summary)
        inverse_layout.addStretch()
        inverse_note = QtWidgets.QLabel("几何初值后用有限差分精修到时。\n源、深度、网格及边界保持已知；候选波形不叠加随机噪声。\n生成波速仅供结束后的误差对照。")
        inverse_note.setWordWrap(True)
        inverse_note.setStyleSheet("color: #94a3b8;")
        inverse_layout.addWidget(inverse_note)
        inverse_scroll = QtWidgets.QScrollArea()
        inverse_scroll.setWidgetResizable(True)
        inverse_scroll.setWidget(inverse_page)
        tabs.addTab(inverse_scroll, "波速反演")

        self.run_button = QtWidgets.QPushButton("计算波场")
        self.run_button.setMinimumHeight(42)
        self.run_button.clicked.connect(self.start_simulation)
        side.addWidget(self.run_button)
        self.cancel_button = QtWidgets.QPushButton("取消计算")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_simulation)
        side.addWidget(self.cancel_button)
        self.progress = QtWidgets.QProgressBar()
        side.addWidget(self.progress)
        file_row = QtWidgets.QHBoxLayout()
        self.save_button = QtWidgets.QPushButton("保存配置")
        self.load_button = QtWidgets.QPushButton("载入配置")
        self.save_button.clicked.connect(self.save_dialog)
        self.load_button.clicked.connect(self.load_dialog)
        file_row.addWidget(self.save_button)
        file_row.addWidget(self.load_button)
        side.addLayout(file_row)
        export_row = QtWidgets.QHBoxLayout()
        self.export_button = QtWidgets.QPushButton("导出数据")
        self.screenshot_button = QtWidgets.QPushButton("保存截图")
        self.export_button.clicked.connect(self.export_dialog)
        self.screenshot_button.clicked.connect(self.screenshot_dialog)
        export_row.addWidget(self.export_button)
        export_row.addWidget(self.screenshot_button)
        side.addLayout(export_row)
        body.addWidget(sidebar)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(4, 2, 2, 2)
        self.coverage_label = QtWidgets.QLabel()
        self.coverage_label.setWordWrap(True)
        self.coverage_label.setStyleSheet("color: #fbbf24; padding: 5px;")
        right_layout.addWidget(self.coverage_label)
        vertical = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        self.result_tabs = QtWidgets.QTabWidget()
        self.result_tabs.addTab(vertical, "波场与波形")
        right_layout.addWidget(self.result_tabs, 1)
        inverse_view = QtWidgets.QWidget()
        inverse_view_layout = QtWidgets.QVBoxLayout(inverse_view)
        self.inverse_plot = pg.PlotWidget()
        self.inverse_plot.setLabel("bottom", "源距 |x|", units="m")
        self.inverse_plot.setLabel("left", "首波到时", units="s")
        self.inverse_plot.showGrid(x=True, y=True, alpha=.2)
        inverse_view_layout.addWidget(self.inverse_plot, 1)
        self.inverse_table = QtWidgets.QTableWidget(0, 6)
        self.inverse_table.setHorizontalHeaderLabels(["接收器", "源距 / m", "拾取 / s", "拟合 / s", "拟合−拾取 / s", "拟合分支"])
        self.inverse_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.inverse_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.inverse_table.setFixedHeight(245)
        inverse_view_layout.addWidget(self.inverse_table)
        note = QtWidgets.QLabel("圆点：观测到时；黄色叉号：有限差分预测（连线仅引导视线）；虚线：几何初值预测。仅几何模式显示射线曲线。\n分支仅为射线参考。残差小不保证波速正确；同模型合成数据结果不等于独立验证。")
        note.setWordWrap(True)
        inverse_view_layout.addWidget(note)
        self.result_tabs.addTab(inverse_view, "走时拟合")
        self.field_widget = pg.GraphicsLayoutWidget()
        self.field_plot = self.field_widget.addPlot(title="二维压力波场 · 拖动彩色圆点改变测点")
        self.field_plot.setLabel("bottom", "水平位置 x", units="m")
        self.field_plot.setLabel("left", "y（向下为正）", units="m")
        self.field_plot.invertY(True)
        self.field_plot.setAspectLocked(True)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.field_plot.addItem(self.image)
        cmap = pg.ColorMap([0, 0.5, 1], ["#2563eb", "#101b2b", "#f43f5e"])
        self.colorbar = pg.ColorBarItem(values=(-1, 1), colorMap=cmap, label="压力 / a.u.", interactive=False, width=15)
        self.colorbar.setImageItem(self.image, insert_in=self.field_plot)
        self.interface_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#94a3b8", width=1.5, style=QtCore.Qt.PenStyle.DashLine))
        self.field_plot.addItem(self.interface_line)
        self.plane_line = pg.InfiniteLine(angle=0, pen=pg.mkPen("#64748b", style=QtCore.Qt.PenStyle.DotLine))
        self.field_plot.addItem(self.plane_line)
        self.crossover_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#fbbf24", width=1.5, style=QtCore.Qt.PenStyle.DashLine), label="dc={value:.2f} m", labelOpts={"position": .9, "color": "#fbbf24"})
        self.field_plot.addItem(self.crossover_line)
        self.source_item = pg.ScatterPlotItem(size=17, symbol="star", pen="#fff7ed", brush="#fbbf24")
        self.source_item.setZValue(25)
        self.field_plot.addItem(self.source_item)
        self.source_label = pg.TextItem("S (0, 0)", color="#fbbf24", anchor=(0.5, 1.3))
        self.field_plot.addItem(self.source_label)
        self.layer_label = pg.TextItem(color="#cbd5e1", anchor=(1, 0))
        self.field_plot.addItem(self.layer_label)
        self.sponge_lines = []
        vertical.addWidget(self.field_widget)
        self.signal_widget = pg.GraphicsLayoutWidget()
        vertical.addWidget(self.signal_widget)
        vertical.setSizes([520, 300])
        noise_controls = QtWidgets.QHBoxLayout()
        self.noise_enabled = QtWidgets.QCheckBox("随机波形噪声")
        self.noise_enabled.setToolTip("在各接收器记录上叠加独立高斯噪声；不改变上方二维波场。关闭即恢复原始记录。")
        self.noise_std = QtWidgets.QDoubleSpinBox()
        self.noise_std.setDecimals(8)
        self.noise_std.setRange(0, 1)
        self.noise_std.setValue(NoiseConfig().std_au)
        self.noise_std.setSingleStep(.00001)
        self.noise_std.setKeyboardTracking(False)
        self.noise_std.setToolTip("每个采样点的噪声标准差（绝对幅值），不随信号强弱缩放。")
        self.noise_seed = QtWidgets.QSpinBox()
        self.noise_seed.setRange(0, 2147483647)
        self.noise_seed.setValue(NoiseConfig().seed)
        self.noise_seed.setKeyboardTracking(False)
        self.noise_seed.setToolTip("相同种子和接收器编号复现相同噪声。修改阈值、回放或移动测点不会重新抽样。")
        self.noise_resample = QtWidgets.QPushButton("换一组噪声")
        self.noise_resample.clicked.connect(lambda: self.noise_seed.setValue((self.noise_seed.value()+1) % 2147483648))
        for widget in (self.noise_enabled, QtWidgets.QLabel("标准差 / a.u."), self.noise_std,
                       QtWidgets.QLabel("种子"), self.noise_seed, self.noise_resample):
            noise_controls.addWidget(widget)
        noise_controls.addStretch()
        right_layout.addLayout(noise_controls)
        self.noise_label = QtWidgets.QLabel()
        self.noise_label.setWordWrap(True)
        self.noise_label.setStyleSheet("color: #94a3b8;")
        right_layout.addWidget(self.noise_label)
        self.noise_enabled.toggled.connect(self.rebuild_waveforms)
        self.noise_std.valueChanged.connect(self.rebuild_waveforms)
        self.noise_seed.valueChanged.connect(self.rebuild_waveforms)
        wave_controls = QtWidgets.QHBoxLayout()
        self.signal_mode = QtWidgets.QComboBox()
        self.signal_mode.addItems(["波形叠加", "分道排列（各道归一化）"])
        self.signal_mode.currentIndexChanged.connect(self.rebuild_waveforms)
        wave_controls.addWidget(self.signal_mode)
        wave_controls.addWidget(QtWidgets.QLabel("首波阈值 / a.u."))
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setDecimals(8)
        self.threshold.setRange(1e-8, 1)
        self.threshold.setValue(1e-4)
        self.threshold.setSingleStep(1e-5)
        self.threshold.setKeyboardTracking(False)
        self.threshold.setToolTip("首次 |压力| 达到此绝对幅值的时刻。不会用后面的强波重新设置门槛；也不会用射线参考帮助寻找首波。")
        self.threshold.valueChanged.connect(self.rebuild_waveforms)
        wave_controls.addWidget(self.threshold)
        self.ray_checkbox = QtWidgets.QCheckBox("几何前沿参考")
        self.ray_checkbox.setToolTip("理想射线到时；未包含有限脉冲、阈值及离散误差。")
        self.ray_checkbox.toggled.connect(self.rebuild_waveforms)
        wave_controls.addWidget(self.ray_checkbox)
        wave_controls.addWidget(QtWidgets.QLabel("波场增益"))
        self.gain = QtWidgets.QComboBox()
        for gain in (1, 2, 4, 8, 16):
            self.gain.addItem(f"{gain}×", gain)
        self.gain.setCurrentIndex(2)
        self.gain.currentIndexChanged.connect(self.apply_gain)
        self.gain.setToolTip("只调整颜色范围；不改变波场或接收器数据。大振幅可能颜色饱和。")
        wave_controls.addWidget(self.gain)
        wave_controls.addStretch()
        right_layout.addLayout(wave_controls)
        playback = QtWidgets.QHBoxLayout()
        self.play_button = QtWidgets.QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_playback)
        self.reset_button = QtWidgets.QPushButton("回到起点")
        self.reset_button.clicked.connect(self.reset_playback)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.slider.sliderPressed.connect(self.pause_playback)
        self.speed = QtWidgets.QComboBox()
        for value in [0.25, 0.5, 1, 2, 4]:
            self.speed.addItem(f"{value:g}×", value)
        self.speed.setCurrentIndex(2)
        self.time_label = QtWidgets.QLabel("t = 0.000 s")
        self.time_label.setMinimumWidth(150)
        for widget in (self.play_button, self.reset_button, self.slider, self.speed, self.time_label):
            playback.addWidget(widget, 1 if widget is self.slider else 0)
        right_layout.addLayout(playback)
        body.addWidget(right)
        body.setSizes([330, 1110])
        self.status = QtWidgets.QLabel("准备就绪：默认场景可直接计算。")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #94a3b8; padding: 6px;")
        layout.addWidget(self.status)
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #142033; color: #e2e8f0; }
            QLineEdit, QComboBox, QTableWidget { background: #0f1929; padding: 4px; }
            /* Define the entire spinbox geometry together: Windows 11's native
               side-by-side arrows otherwise overlap the styled QLineEdit. */
            QAbstractSpinBox {
                background: #0f1929; border: 1px solid #3b526e;
                border-radius: 3px; padding: 4px 4px 4px 6px; min-height: 24px;
            }
            QAbstractSpinBox QLineEdit { padding: 0; border: none; background: transparent; }
            QAbstractSpinBox::up-button {
                subcontrol-origin: border; subcontrol-position: top right;
                width: 26px; background: #263b55;
                border-left: 1px solid #3b526e;
            }
            QAbstractSpinBox::down-button {
                subcontrol-origin: border; subcontrol-position: bottom right;
                width: 26px; background: #263b55;
                border-left: 1px solid #3b526e; border-top: 1px solid #3b526e;
            }
            QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover { background: #365779; }
            QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed { background: #0284c7; }
            QAbstractSpinBox::up-arrow { image: url("__ASSETS__/spin-up.svg"); width: 12px; height: 8px; }
            QAbstractSpinBox::down-arrow { image: url("__ASSETS__/spin-down.svg"); width: 12px; height: 8px; }
            QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {
                image: url("__ASSETS__/spin-up-disabled.svg");
            }
            QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {
                image: url("__ASSETS__/spin-down-disabled.svg");
            }
            QAbstractSpinBox:disabled, QAbstractSpinBox:read-only { color: #64748b; }
            QPushButton { background: #263b55; border: 1px solid #3b526e; border-radius: 4px; padding: 6px; }
            QPushButton:hover { background: #365779; } QPushButton:disabled { color: #64748b; }
            QTabBar::tab { padding: 9px 16px; } QTabBar::tab:selected { background: #294462; }
            QProgressBar { border: 1px solid #3b526e; text-align: center; }
            QProgressBar::chunk { background: #0284c7; }
            QHeaderView::section { background: #263b55; padding: 5px; }
        """.replace("__ASSETS__", Path(__file__).with_name("assets").as_posix()))

    def _populate_config(self, config):
        self._loading = True
        self.config = config
        self.auto_right.setChecked(config.auto_right)
        self.coverage_factor.setValue(config.crossover_factor)
        for key, field in self.fields.items():
            field.setValue(getattr(config, key))
        self.boundary.setCurrentIndex(self.boundary.findData(config.boundary))
        self._loading = False
        self.apply_auto_right()
        self.update_resolution()

    def read_config(self):
        return replace(self.config, **{key: field.value() for key, field in self.fields.items()}, boundary=self.boundary.currentData(), auto_right=self.auto_right.isChecked(), crossover_factor=self.coverage_factor.value())

    def apply_auto_right(self):
        self._auto_right_error = ""
        config = self.config
        dc = config.crossover_distance
        active = config.auto_right and dc is not None
        self.fields["width"].setReadOnly(active)
        self.coverage_factor.setEnabled(config.auto_right)
        if active:
            target = recommended_right_boundary(config, self.receivers)
            if target > self.fields["width"].maximum():
                self._auto_right_error = f"自动右边界需 {target:.2f} m，超过界面支持的 100 m。请调整速度差/深度，或关闭自动模式手动研究。"
            else:
                blocker = QtCore.QSignalBlocker(self.fields["width"])
                self.fields["width"].setValue(target)
                del blocker
                self.config = replace(config, width=target)
        self.update_coverage()

    def update_coverage(self):
        if not hasattr(self, "coverage_label"):
            return
        config = self.config
        dc = config.crossover_distance
        self.crossover_line.setVisible(dc is not None)
        if dc is None:
            self.coverage_label.setText("c2 ≤ c1：不存在下层更快的头波领先区；不自动扩展，可手动设置右边界。")
            return
        self.crossover_line.setPos(dc)
        far_count = sum(abs(r.x) > dc for r in self.receivers)
        near_count = sum(abs(r.x) < dc for r in self.receivers)
        right_limit = config.receiver_bounds[1]
        text = f"交叉距离 dc = {dc:.3f} m · 测点右限 {right_limit:.3f} m · 头波领先区测点 {far_count}/{len(self.receivers)}"
        if right_limit <= dc:
            text += "\n右侧有效观测区未越过 dc，请扩大右边界。"
        if far_count == 0:
            text += "\n尚无测点位于头波领先区；将部分接收器移到黄色虚线右侧。"
        elif near_count == 0:
            text += "\n建议同时保留 dc 以内的测点，用于比较直达波。"
        target = max(config.crossover_factor*dc, max([abs(r.x) for r in self.receivers], default=0))
        suggested_time = ray_front(config, target) + 2/config.frequency
        if config.duration < suggested_time:
            text += f"\n时长可能不足：覆盖目标位置及完整源脉冲建议 ≥ {suggested_time:.2f} s。"
        if self._auto_right_error:
            text += "\n" + self._auto_right_error
        self.coverage_label.setText(text)

    def update_resolution(self):
        try:
            if self._auto_right_error:
                raise ValueError(self._auto_right_error)
            self.config.validate()
            nx, nz = self.config.shape
            ppw = min(self.config.c1, self.config.c2) / (self.config.frequency * self.config.spacing)
            warning = "\n频率相对网格偏高，建议加密网格。" if ppw < 12 else ""
            self.resolution_label.setText(f"网格 {nx} × {nz} · {self.config.steps} 时间步\n波场 {self.config.frame_count} 帧 · 缓存约 {self.config.estimated_cache_bytes / 1024**2:.1f} MiB\n中心波长约 {ppw:.1f} 个网格{warning}")
            self.run_button.setEnabled(not (self.worker and self.worker.isRunning()))
        except (ValueError, OverflowError) as error:
            self.resolution_label.setText(str(error))
            self.run_button.setEnabled(False)

    def parameters_changed(self, *_):
        if self._loading:
            return
        self.config = self.read_config()
        self.apply_auto_right()
        self.invalidate()
        self.update_resolution()
        try:
            self.config.validate()
        except (ValueError, OverflowError):
            return
        lo, hi = self.config.receiver_bounds
        for receiver in self.receivers:
            receiver.x = float(np.clip(receiver.x, lo, hi))
        self.refresh_scene()
        self.refresh_table()

    def invalidate(self):
        self.pause_playback()
        self.result = None
        self.image.clear()
        self.current_frame = 0
        self.progress.setValue(0)
        self.set_result_controls(False)
        self.rebuild_waveforms()
        self.status.setText("场景参数已修改：请重新计算波场。")
        self.time_label.setText("未计算")

    def set_result_controls(self, enabled):
        for widget in (self.play_button, self.reset_button, self.slider, self.export_button):
            widget.setEnabled(enabled)

    def refresh_scene(self):
        config = self.config
        nx, nz = config.shape
        dx, dz = config.x_span / (nx - 1), config.y_span / (nz - 1)
        self.image.setImage(np.zeros((nz, nx), dtype=np.float32), autoLevels=False, levels=(-1, 1))
        self.image.setRect(QtCore.QRectF(config.x_min - dx / 2, config.y_min - dz / 2, config.x_span + dx, config.y_span + dz))
        self.colorbar.setLevels((-1, 1))
        self.field_plot.setRange(xRange=(config.x_min, config.width), yRange=(config.y_min, config.depth), padding=0.015)
        self.interface_line.setPos(config.interface_depth)
        self.plane_line.setPos(config.source_depth)
        self.source_item.setData([config.source_x], [config.source_depth])
        self.source_label.setPos(config.source_x, config.source_depth)
        self.layer_label.setText(f"c1 = {config.c1:g} m/s\nc2 = {config.c2:g} m/s")
        self.layer_label.setPos(config.width - 0.1, config.interface_depth + 0.08)
        for line in self.sponge_lines:
            self.field_plot.removeItem(line)
        self.sponge_lines = []
        if config.boundary == "sponge":
            for position, angle in [(config.x_min + config.sponge_width, 90), (config.width - config.sponge_width, 90), (config.y_min + config.sponge_width, 0), (config.depth - config.sponge_width, 0)]:
                line = pg.InfiniteLine(position, angle=angle, pen=pg.mkPen("#475569", style=QtCore.Qt.PenStyle.DashLine))
                self.field_plot.addItem(line)
                self.sponge_lines.append(line)
        self.refresh_handles()

    def refresh_handles(self):
        for handle in self.handles.values():
            self.field_plot.removeItem(handle)
        self.handles = {}
        for receiver in self.receivers:
            handle = ReceiverHandle(receiver, self.config.source_depth, self.config.receiver_bounds)
            handle.sigPositionChanged.connect(self.handle_moved)
            handle.selected.connect(self.select_receiver)
            self.field_plot.addItem(handle)
            self.handles[receiver.id] = handle

    def refresh_table(self):
        self.receiver_table.setRowCount(len(self.receivers))
        lo, hi = self.config.receiver_bounds
        for row, receiver in enumerate(self.receivers):
            item = QtWidgets.QTableWidgetItem(f"R{receiver.id}")
            item.setForeground(QtGui.QColor(COLORS[(receiver.id - 1) % len(COLORS)]))
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.receiver_table.setItem(row, 0, item)
            position = QtWidgets.QDoubleSpinBox()
            position.setDecimals(3)
            position.setRange(lo, hi)
            position.setSingleStep(0.05)
            position.setValue(receiver.x)
            position.setKeyboardTracking(False)
            position.valueChanged.connect(lambda x, rid=receiver.id: self.move_receiver(rid, x))
            self.receiver_table.setCellWidget(row, 1, position)
            # The full upper/lower click regions must fit inside the cell,
            # including Windows font metrics and high-DPI scaling.
            self.receiver_table.setRowHeight(row, max(self.receiver_table.verticalHeader().defaultSectionSize(), position.sizeHint().height()+2))
            arrival = QtWidgets.QTableWidgetItem("待计算")
            arrival.setFlags(arrival.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
            self.receiver_table.setItem(row, 2, arrival)
        self.update_arrival_table()
        self.update_coverage()

    def select_receiver(self, receiver_id):
        self.tabs.setCurrentIndex(1)
        row = next(i for i, receiver in enumerate(self.receivers) if receiver.id == receiver_id)
        self.receiver_table.selectRow(row)

    def handle_moved(self, handle):
        self.move_receiver(handle.receiver_id, handle.pos().x())

    def move_receiver(self, receiver_id, x):
        x = float(np.clip(x, *self.config.receiver_bounds))
        for row, receiver in enumerate(self.receivers):
            if receiver.id == receiver_id:
                receiver.x = x
                handle = self.handles.get(receiver_id)
                if handle is not None and abs(handle.pos().x() - x) > 1e-10:
                    blocker = QtCore.QSignalBlocker(handle)
                    handle.setPos(x, self.config.source_depth)
                    del blocker
                spin = self.receiver_table.cellWidget(row, 1)
                if spin is not None:
                    blocker = QtCore.QSignalBlocker(spin)
                    spin.setValue(x)
                    del blocker
                break
        self.rebuild_waveforms()

    def add_receiver(self):
        lo, hi = self.config.receiver_bounds
        positions = [lo] + sorted(r.x for r in self.receivers) + [hi]
        a, b = max(zip(positions[:-1], positions[1:]), key=lambda pair: pair[1] - pair[0])
        self.receivers.append(Receiver(self.next_receiver_id, (a + b) / 2))
        self.next_receiver_id += 1
        self.refresh_handles()
        self.refresh_table()
        self.rebuild_waveforms()
        self.select_receiver(self.receivers[-1].id)

    def remove_receiver(self):
        row = self.receiver_table.currentRow()
        if len(self.receivers) <= 1:
            self.status.setText("至少保留一个接收器。")
            return
        if row < 0:
            self.status.setText("请先选择要删除的接收器。")
            return
        self.receivers.pop(row)
        self.refresh_handles()
        self.refresh_table()
        self.rebuild_waveforms()

    def read_noise(self):
        return NoiseConfig(self.noise_enabled.isChecked(), self.noise_std.value(), self.noise_seed.value())

    def current_inverse_signature(self):
        return (id(self.result), tuple((r.id, r.x) for r in self.receivers), self.read_noise(),
                self.threshold.value(), self.inverse_min.value(), self.inverse_max.value(),
                self.inverse_mode.currentData(), self.inverse_budget.value(), self.inverse_tolerance.value())

    def refresh_inverse_state(self, *_):
        if not hasattr(self, "inverse_plot") or not hasattr(self, "threshold"):
            return
        if self._inverse_running:
            if self._inverse_job_signature != self.current_inverse_signature():
                self.cancel_inversion()
            return
        hybrid = self.inverse_mode.currentData() == "hybrid"
        self.inverse_budget.setEnabled(hybrid)
        self.inverse_tolerance.setEnabled(hybrid)
        self.inverse_depth.setText(f"已知界面深度 h = {self.config.interface_depth:g} m")
        valid_bounds = self.inverse_min.value() < self.inverse_max.value()
        self.inverse_button.setText(f"用当前 {len(self.receivers)} 个接收器反演")
        ready = self.result is not None and bool(self.receivers) and valid_bounds
        self.inverse_button.setEnabled(ready)
        if self.inverse_result is not None and self.inverse_signature == self.current_inverse_signature():
            return
        self.inverse_result = None
        self.inverse_signature = None
        self.inverse_plot.clear()
        self.inverse_plot.setTitle("等待波速反演")
        self.inverse_table.setRowCount(0)
        text = "请先计算波场，再点击反演。" if self.result is None else "尚无当前数据的反演结果；点击上方按钮拟合。"
        if len(self.receivers) < 2:
            text = "当前只有 1 个接收器，无法同时确定两个波速；请增加不同源距的测点。"
        if not valid_bounds:
            text = "反演波速搜索下限必须小于上限。"
        self.inverse_summary.setText(text)

    def run_inversion(self):
        if self._inverse_running:
            return
        self.refresh_inverse_state()
        if self.result is None:
            return
        arrivals = [pick_arrival(self.result.times, self.receiver_signal(r), self.threshold.value()) for r in self.receivers]
        try:
            fit = invert_first_arrivals([r.x for r in self.receivers], arrivals,
                                       self.result.config.interface_depth,
                                       speed_min=self.inverse_min.value(), speed_max=self.inverse_max.value(),
                                       sample_interval=float(self.result.times[1]-self.result.times[0]))
        except ValueError as error:
            self.inverse_summary.setText(f"无法反演：{error}")
            return
        if self.inverse_mode.currentData() == "hybrid" and fit.identifiable:
            self.pause_playback()
            self.inverse_result = None
            self.inverse_signature = None
            self._inverse_job_signature = self.current_inverse_signature()
            self.inverse_worker = InverseWorker(self.result.config, [r.x for r in self.receivers], arrivals,
                                                self.result.times, self.threshold.value(),
                                                {"speed_min": self.inverse_min.value(), "speed_max": self.inverse_max.value(),
                                                 "max_evaluations": self.inverse_budget.value(), "target_rmse": self.inverse_tolerance.value()}, self)
            self.inverse_worker.progress.connect(self.on_inverse_progress)
            self.inverse_worker.completed.connect(self.accept_inverse_result)
            self.inverse_worker.cancelled.connect(lambda: self.inverse_summary.setText("反演已取消；原波场和观测记录保留。"))
            self.inverse_worker.failed.connect(lambda message: self.inverse_summary.setText(f"有限差分反演失败：{message}"))
            self.inverse_worker.finished.connect(self.inverse_finished)
            self.set_inverse_busy(True)
            self.inverse_plot.clear()
            self.inverse_table.setRowCount(0)
            self.inverse_plot.setTitle("有限差分到时拟合进行中")
            self.inverse_progress.setRange(0, self.inverse_budget.value())
            self.inverse_progress.setFormat("已评估 %v / %m 个候选")
            self.inverse_progress.setValue(0)
            self.inverse_summary.setText(f"几何初值：c1={fit.c1:.5g}，c2={fit.c2:.5g} m/s\n正在用有限差分精修，可能需要几十秒或更久。")
            self.tabs.setCurrentIndex(2)
            self.result_tabs.setCurrentIndex(1)
            self.inverse_worker.start()
            return
        if self.inverse_mode.currentData() == "hybrid":
            fit.warnings.append("几何初值信息不足，未启动有限差分精修。")
        self.display_inverse_result(fit)

    def set_inverse_busy(self, busy):
        self._inverse_running = busy
        for widget in (self.config_page, self.tabs.widget(1), self.load_button, self.save_button,
                       self.inverse_mode, self.inverse_min, self.inverse_max,
                       self.inverse_budget, self.inverse_tolerance, self.noise_enabled,
                       self.noise_std, self.noise_seed, self.noise_resample, self.threshold):
            widget.setEnabled(not busy)
        for handle in self.handles.values():
            handle.setEnabled(not busy)
        self.inverse_cancel.setEnabled(busy)
        self.inverse_button.setEnabled(not busy and self.result is not None)
        self.export_button.setEnabled(not busy and self.result is not None)
        self.run_button.setEnabled(not busy)
        if not busy:
            noise_on = self.noise_enabled.isChecked()
            for widget in (self.noise_std, self.noise_seed, self.noise_resample):
                widget.setEnabled(noise_on)
            hybrid = self.inverse_mode.currentData() == "hybrid"
            self.inverse_budget.setEnabled(hybrid)
            self.inverse_tolerance.setEnabled(hybrid)
            self.update_resolution()

    def on_inverse_progress(self, count, budget, rmse, c1, c2):
        self.inverse_progress.setValue(count)
        score = "暂无完整候选到时" if rmse is None else f"最佳到时 RMSE={rmse:.6g} s"
        self.inverse_summary.setText(f"有限差分搜索 {count}/{budget}\n候选 c1={c1:.5g}，c2={c2:.5g}\n{score}\n可取消；观测噪声保持固定。")

    def cancel_inversion(self):
        if self.inverse_worker is not None and self._inverse_running:
            self.inverse_worker.cancel_event.set()
            self.inverse_cancel.setEnabled(False)
            self.inverse_summary.setText("正在取消有限差分反演……")

    def accept_inverse_result(self, fit):
        if self.inverse_worker.cancel_event.is_set():
            self.inverse_summary.setText("反演已取消；原波场和观测记录保留。")
            return
        if self._inverse_job_signature != self.current_inverse_signature():
            self.inverse_summary.setText("输入已改变，已丢弃旧输入的反演结果。")
            return
        self.display_inverse_result(fit)

    def inverse_finished(self):
        self.set_inverse_busy(False)
        self._inverse_job_signature = None

    def display_inverse_result(self, fit):
        self.inverse_result = fit
        self.inverse_signature = self.current_inverse_signature()
        lines = [f"c1 估计：{fit.c1:.5g} m/s",
                 "c2：无法确定" if fit.c2 is None else f"c2 估计：{fit.c2:.5g} m/s",
                 f"走时 RMSE：{fit.rmse:.5g} s", f"最大残差：{fit.max_residual:.5g} s"]
        if fit.crossover_distance is not None:
            lines.append(f"拟合交叉距离：{fit.crossover_distance:.4g} m")
        if fit.condition_number is not None:
            lines.append(f"局部灵敏度条件数：{fit.condition_number:.3g}")
        if fit.ray_stage is not None:
            ray_c2 = "无法确定" if fit.ray_stage['c2'] is None else f"{fit.ray_stage['c2']:.5g}"
            lines.extend(["", f"几何初值：{fit.ray_stage['c1']:.5g} / {ray_c2} m/s",
                          f"有限差分求解：{fit.fd_evaluations} 次 · {fit.elapsed_seconds:.1f} s"])
            if fit.fd_initial_rmse is not None:
                lines.append(f"精修前 FD RMSE：{fit.fd_initial_rmse:.5g} s")
            lines.append("达到目标 RMSE" if fit.termination == "target_rmse" else "精修已停止（见下方诊断）")
        # Generating speeds are used ONLY after the inverse solve, for evaluation.
        truth = self.result.config
        lines.extend(["", "模拟设置（仅供结果对照）：", f"c1={truth.c1:g}，c2={truth.c2:g} m/s",
                      f"c1 相对误差：{100*abs(fit.c1-truth.c1)/truth.c1:.2f}%"])
        if fit.c2 is not None:
            lines.append(f"c2 相对误差：{100*abs(fit.c2-truth.c2)/truth.c2:.2f}%")
        lines.extend(["", *fit.warnings])
        if self.noise_enabled.isChecked() and self.noise_std.value() > 0:
            lines.append("当前使用带噪记录，请比较不同门槛和种子。")
        self.inverse_summary.setText("\n".join(lines))
        plot = self.inverse_plot
        plot.clear()
        is_fd = fit.method == "ray_initialized_fd_first_arrival_least_squares"
        plot.setTitle(f"{len(self.receivers)} 道首波：" + ("几何初值 → 有限差分精修" if is_fd else "几何射线拟合"))
        xx = np.linspace(0, 1.08*max(fit.offsets), 400)
        if is_fd:
            order = np.argsort(fit.offsets)
            plot.plot(np.asarray(fit.offsets)[order], np.asarray(fit.fitted_times)[order],
                      pen=pg.mkPen("#fbbf24", width=2), symbol="x", symbolSize=10, symbolPen="#fbbf24")
            yy = travel_times(xx, fit.depth, fit.ray_stage["c1"], fit.ray_stage["c2"])
            plot.plot(xx, yy, pen=pg.mkPen("#94a3b8", style=QtCore.Qt.PenStyle.DashLine))
        else:
            yy = travel_times(xx, fit.depth, fit.c1, fit.c2 or fit.c1)
            plot.plot(xx, yy, pen=pg.mkPen("#fbbf24", width=2))
        self.inverse_table.setRowCount(len(self.receivers))
        for i, receiver in enumerate(self.receivers):
            color = COLORS[(receiver.id-1) % len(COLORS)]
            plot.plot([fit.offsets[i]], [fit.observed_times[i]], pen=None, symbol="o", symbolSize=10, symbolBrush=color)
            plot.plot([fit.offsets[i]]*2, [fit.observed_times[i], fit.fitted_times[i]], pen=pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine))
            label = pg.TextItem(f"R{receiver.id}", color=color, anchor=(0, 1.2))
            label.setPos(fit.offsets[i], fit.observed_times[i])
            plot.addItem(label)
            values = [f"R{receiver.id}", f"{fit.offsets[i]:.3f}", f"{fit.observed_times[i]:.5f}",
                      f"{fit.fitted_times[i]:.5f}", f"{fit.residuals[i]:+.5f}", fit.branches[i]]
            for column, value in enumerate(values):
                self.inverse_table.setItem(i, column, QtWidgets.QTableWidgetItem(value))
        plot.enableAutoRange()
        self.tabs.setCurrentIndex(2)
        self.result_tabs.setCurrentIndex(1)

    def receiver_signal(self, receiver):
        return add_measurement_noise(self.result.signal(receiver.x), receiver.id, self.read_noise())

    def update_arrival_table(self):
        now = self.result.frame_times[self.current_frame] if self.result is not None else 0
        for row, receiver in enumerate(self.receivers):
            item = self.receiver_table.item(row, 2)
            if item is None:
                continue
            if self.result is None:
                item.setText("待计算")
                continue
            arrival = pick_arrival(self.result.times, self.receiver_signal(receiver), self.threshold.value())
            item.setText("未检出" if arrival is None else f"{arrival:.3f}" + (" 待到" if arrival > now else " 已到"))

    def rebuild_waveforms(self, *_):
        if not hasattr(self, "signal_widget"):
            return
        self.refresh_inverse_state()
        noise = self.read_noise()
        for control in (self.noise_std, self.noise_seed, self.noise_resample):
            control.setEnabled(noise.enabled and not self._inverse_running)
        self.noise_label.setText("检测和导出使用带噪波形；门槛过低可能被噪声提前触发。" if noise.enabled and noise.std_au > 0
                                else "噪声已关闭或强度为零：检测和导出使用原始波形。")
        self.update_coverage()
        self.signal_widget.clear()
        self.cursors = []
        self.signal_plot = self.signal_widget.addPlot()
        plot = self.signal_plot
        plot.setLabel("bottom", "时间", units="s")
        plot.showGrid(x=True, y=True, alpha=0.15)
        plot.setXRange(0, self.config.duration, padding=0)
        if self.result is None:
            plot.setTitle("接收器压力记录 · 计算完成后可回放、移动测点")
            plot.setLabel("left", "压力 / a.u.")
            return
        result = self.result
        separated = self.signal_mode.currentIndex() == 1
        plot.setTitle("完整记录 · 实线游标为当前时刻 · 圆点为阈值首波" + (" · 虚线为射线参考" if self.ray_checkbox.isChecked() else ""))
        plot.setLabel("left", "接收器（各道归一化）" if separated else "压力 / a.u.")
        legend = plot.addLegend(offset=(10, 10), colCount=min(6, len(self.receivers))) if not separated else None
        ticks = []
        for i, receiver in enumerate(self.receivers):
            color = COLORS[(receiver.id - 1) % len(COLORS)]
            signal = self.receiver_signal(receiver)
            scale = float(np.max(np.abs(signal))) or 1.0
            values = signal / scale * 0.38 + i if separated else signal
            curve = plot.plot(result.times, values, pen=pg.mkPen(color, width=1.5), name=f"R{receiver.id}")
            curve.setClipToView(True)
            ticks.append((i, f"R{receiver.id}"))
            arrival = pick_arrival(result.times, signal, self.threshold.value())
            if arrival is not None:
                y = float(np.interp(arrival, result.times, values))
                plot.plot([arrival], [y], pen=None, symbol="o", symbolSize=6, symbolBrush=color, symbolPen=color)
            if self.ray_checkbox.isChecked():
                front = ray_front(result.config, receiver.x)
                if separated:
                    plot.plot([front, front], [i - 0.4, i + 0.4], pen=pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine))
                else:
                    plot.addItem(pg.InfiniteLine(front, pen=pg.mkPen(color, style=QtCore.Qt.PenStyle.DashLine)))
        if separated:
            plot.getAxis("left").setTicks([ticks])
            plot.setYRange(-0.6, len(self.receivers) - 0.4, padding=0)
        else:
            plot.enableAutoRange(axis="y")
        cursor = pg.InfiniteLine(result.frame_times[self.current_frame], pen=pg.mkPen("#ffffff", width=2))
        cursor.setZValue(30)
        plot.addItem(cursor)
        self.cursors.append(cursor)
        self.update_arrival_table()

    def start_simulation(self):
        if self._inverse_running:
            return
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            self.config = self.read_config()
            self.apply_auto_right()
            if self._auto_right_error:
                raise ValueError(self._auto_right_error)
            self.config.validate()
        except (ValueError, OverflowError) as error:
            self.show_error(str(error))
            return
        self.invalidate()
        self.refresh_scene()
        self.config_page.setEnabled(False)
        self.load_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.worker = SolverWorker(self.config, self)
        self.worker.progress.connect(self.on_progress)
        self.worker.completed.connect(self.accept_result)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.failed.connect(self.show_error)
        self.worker.finished.connect(self.worker_finished)
        self.status.setText("正在计算真实二维波场；上图显示计算预览，完成后可同步回放。")
        self.worker.start()

    def on_progress(self, percent, time, frame):
        self.progress.setValue(percent)
        self.display_limit = max(float(np.max(np.abs(frame))), 1e-6)
        self.apply_gain()
        self.image.setImage(frame, autoLevels=False)
        self.time_label.setText(f"预览 t = {time:.3f} s")

    def accept_result(self, result):
        self.result = result
        self.current_frame = 0
        self.slider.setRange(0, len(result.frames) - 1)
        self.slider.setValue(0)
        self.set_result_controls(True)
        self.progress.setValue(100)
        # Fixed symmetric scale throughout playback; avoid frame-by-frame flicker.
        self.display_limit = max(float(np.max(np.abs(result.frames))), 1e-6)
        self.apply_gain()
        self.rebuild_waveforms()
        self.show_frame(0)
        self.status.setText(f"计算完成：{result.config.steps} 步，{len(result.frames)} 帧，用时 {result.elapsed_seconds:.2f} s。拖动接收器可立即重采样；播放速度不改变物理时间步。")

    def apply_gain(self, *_):
        limit = self.display_limit / self.gain.currentData()
        self.colorbar.setLevels((-limit, limit))

    def on_cancelled(self):
        self.refresh_scene()
        self.time_label.setText("已取消")
        self.status.setText("计算已取消，未保留不完整结果。可以修改参数后重新计算。")

    def worker_finished(self):
        self.config_page.setEnabled(True)
        self.load_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.update_resolution()

    def cancel_simulation(self):
        if self.worker is not None:
            self.worker.cancel_event.set()
            self.cancel_button.setEnabled(False)
            self.status.setText("正在取消计算……")

    def show_frame(self, index):
        if self.result is None:
            return
        index = int(np.clip(index, 0, len(self.result.frames) - 1))
        self.current_frame = index
        self.image.setImage(self.result.frames[index], autoLevels=False)
        time = float(self.result.frame_times[index])
        self.time_label.setText(f"t = {time:.3f} / {self.config.duration:g} s")
        for cursor in self.cursors:
            cursor.setPos(time)
        self.update_arrival_table()
        if not self.timer.isActive():
            self.play_time = time

    def toggle_playback(self):
        if self.result is None:
            return
        if self.timer.isActive():
            self.pause_playback()
            return
        if self.current_frame == len(self.result.frames) - 1:
            self.slider.setValue(0)
        self.play_time = float(self.result.frame_times[self.current_frame])
        self.elapsed.start()
        self.timer.start()
        self.play_button.setText("暂停")

    def pause_playback(self):
        self.timer.stop()
        if hasattr(self, "play_button"):
            self.play_button.setText("播放")

    def reset_playback(self):
        self.pause_playback()
        self.slider.setValue(0)
        self.show_frame(0)

    def advance(self):
        if self.result is None:
            self.pause_playback()
            return
        self.play_time += self.elapsed.restart() / 1000 * self.speed.currentData()
        index = min(len(self.result.frames) - 1, int(np.searchsorted(self.result.frame_times, self.play_time, side="right") - 1))
        self.slider.setValue(max(0, index))
        if self.play_time >= self.config.duration:
            self.pause_playback()

    def save_to(self, path):
        config = self.read_config()
        if self.inverse_min.value() >= self.inverse_max.value():
            raise ValueError("反演波速搜索下限必须小于上限。")
        if self._auto_right_error:
            raise ValueError(self._auto_right_error)
        config.validate()
        save_configuration(path, config, self.receivers, self.threshold.value(), self.read_noise())
        self.save_inverse_settings(path)

    def save_inverse_settings(self, path):
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["inversion"] = {"speed_min": self.inverse_min.value(), "speed_max": self.inverse_max.value(),
                                "mode": self.inverse_mode.currentData(),
                                "max_evaluations": self.inverse_budget.value(), "target_rmse": self.inverse_tolerance.value()}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")

    def load_from(self, path):
        if self._inverse_running or (self.worker is not None and self.worker.isRunning()):
            raise ValueError("请先等待或取消当前计算。")
        config, receivers, threshold, noise = load_configuration(path, include_noise=True)
        settings = json.loads(Path(path).read_text(encoding="utf-8")).get("inversion", {})
        lower, upper = settings.get("speed_min", .1), settings.get("speed_max", 100.)
        mode = settings.get("mode", "hybrid")
        budget, tolerance = settings.get("max_evaluations", 48), settings.get("target_rmse", .0001)
        if mode not in ("hybrid", "ray") or type(budget) is not int or not 8 <= budget <= 200:
            raise ValueError("反演模式或有限差分预算无效（预算须为 8–200 次）。")
        if type(tolerance) not in (int, float) or not np.isfinite(tolerance) or not 1e-8 <= tolerance <= 1 or abs(round(tolerance, 8)-tolerance) > 1e-12:
            raise ValueError("目标到时 RMSE 须在 0.00000001–1 秒内，最多八位小数。")
        for value, editor in ((lower, self.inverse_min), (upper, self.inverse_max)):
            if type(value) not in (int, float) or not np.isfinite(value) or not editor.minimum() <= value <= editor.maximum() or abs(round(value, 2)-value) > 1e-10:
                raise ValueError("反演搜索范围须为 0.01–1000 m/s 内的数值，最多两位小数。")
        if lower >= upper:
            raise ValueError("反演波速搜索下限必须小于上限。")
        if abs(round(noise.std_au, self.noise_std.decimals()) - noise.std_au) > 1e-12:
            raise ValueError("噪声标准差超过界面支持的 8 位小数精度。")
        if config.boundary != "sponge":
            raise ValueError("交互软件固定使用四侧吸收；反射/固定端边界只供数值验证使用。")
        # Reject values the editor cannot represent; never silently round a loaded scene.
        for name, editor in self.fields.items():
            value = getattr(config, name)
            if not editor.minimum() <= value <= editor.maximum() or abs(round(value, editor.decimals()) - value) > 1e-10:
                raise ValueError(f"配置参数 {name} 超出界面范围或小数精度；请调整 JSON 数值。")
        self.invalidate()
        self.receivers = receivers
        self._populate_config(config)
        self.next_receiver_id = max(r.id for r in receivers) + 1
        self.threshold.setValue(threshold)
        self.noise_std.setValue(noise.std_au)
        self.noise_seed.setValue(noise.seed)
        self.noise_enabled.setChecked(noise.enabled)
        self.inverse_min.setValue(lower)
        self.inverse_max.setValue(upper)
        self.inverse_mode.setCurrentIndex(self.inverse_mode.findData(mode))
        self.inverse_budget.setValue(budget)
        self.inverse_tolerance.setValue(tolerance)
        self.refresh_scene()
        self.refresh_table()
        self.rebuild_waveforms()
        self.status.setText(f"已载入配置：{Path(path).name}。点击计算波场开始。")

    def save_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存场景配置", "wave_scene.json", "JSON (*.json)")
        if path:
            try:
                self.save_to(path)
                self.status.setText(f"配置已保存：{path}")
            except Exception as error:
                self.show_error(str(error))

    def load_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "载入场景配置", "", "JSON (*.json)")
        if path:
            try:
                self.load_from(path)
            except Exception as error:
                self.show_error(str(error))

    def export_to(self, directory):
        if self._inverse_running:
            raise ValueError("请先等待或取消当前反演，再导出结果。")
        if self.result is None:
            raise ValueError("请先完成一次模拟。")
        if self.inverse_min.value() >= self.inverse_max.value():
            raise ValueError("反演波速搜索下限必须小于上限。")
        export_results(directory, self.result, self.receivers, self.threshold.value(), self.read_noise())
        self.save_inverse_settings(Path(directory) / "configuration.json")
        self.refresh_inverse_state()
        inverse_path = Path(directory) / "inversion.json"
        if self.inverse_result is not None:
            payload = self.inverse_result.to_dict()
            payload.update(receiver_ids=[r.id for r in self.receivers],
                           arrival_threshold_au=self.threshold.value(),
                           simulation_reference_only={"c1": self.result.config.c1, "c2": self.result.config.c2})
            inverse_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        elif inverse_path.exists():
            inverse_path.unlink()  # Do not leave an older fit beside new observations.
        self.grab().save(str(Path(directory) / "screenshot.png"))

    def export_dialog(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择导出父目录（自动新建结果子目录）")
        if directory:
            try:
                from datetime import datetime
                destination = Path(directory) / datetime.now().strftime("wave_run_%Y%m%d_%H%M%S_%f")
                self.export_to(destination)
                self.status.setText(f"已导出配置、接收器、全采样波形、波场缓存及截图；有效反演结果另存 inversion.json：{destination}")
            except Exception as error:
                self.show_error(str(error))

    def screenshot_dialog(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存软件截图", "wave_lab.png", "PNG (*.png)")
        if path and not self.grab().save(path):
            self.show_error("无法写入截图。")

    def show_error(self, message):
        self.status.setText(f"错误：{message}")
        QtWidgets.QMessageBox.warning(self, "Wave Lab 2D", message)

    def closeEvent(self, event):
        self.pause_playback()
        if self.inverse_worker is not None and self.inverse_worker.isRunning():
            self.inverse_worker.cancel_event.set()
            if not self.inverse_worker.wait(5000):
                event.ignore()
                return
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel_event.set()
            if not self.worker.wait(5000):
                event.ignore()
                return
        event.accept()


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setFont(QtGui.QFont("Microsoft YaHei UI", 10))
    window = WaveLabWindow()
    window.show()
    code = app.exec()
    window.deleteLater()
    QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
