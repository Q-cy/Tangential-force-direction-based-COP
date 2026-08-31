"""滑移频带功率占比、速度谱和旁路相对基线频谱窗口。"""

from __future__ import annotations

import threading
from collections import deque

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from ..config import SpectrumConfig


class SpectrumWindow(QtWidgets.QMainWindow):
    """显示速度谱、相对基线谱/瀑布和唯一实时判定值。"""

    def __init__(self, config: SpectrumConfig | None = None,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = (config or SpectrumConfig()).validate()
        self.setWindowTitle("CoP Friction Spectrum")
        self.resize(self.config.window_width, self.config.window_height)
        self._lock = threading.Lock()
        self._latest_snapshot = None
        self._history: deque = deque()
        self._progress = (0, self.config.required_samples)

        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)
        self.status_label = QtWidgets.QLabel(
            f"等待速度 STFT：0/{self.config.required_samples} 个位置点"
        )
        layout.addWidget(self.status_label)
        self.friction_label = QtWidgets.QLabel("频谱摩擦：WAITING（等待完整速度窗）")
        self.friction_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.friction_label)

        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.setTitle("CoP velocity STFT", size="12pt", bold=True)
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Velocity amplitude")
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spectrum_plot.setXRange(
            self.config.analysis_min_frequency_hz,
            self.config.analysis_max_frequency_hz,
        )
        self.spectrum_plot.addLegend()
        self._curve_x = self.spectrum_plot.plot(pen=pg.mkPen("r", width=2), name="velocity X")
        self._curve_y = self.spectrum_plot.plot(pen=pg.mkPen("b", width=2), name="velocity Y")
        self._curve_combined = self.spectrum_plot.plot(pen=pg.mkPen("g", width=2), name="velocity combined")
        layout.addWidget(self.spectrum_plot, stretch=2)

        self.relative_plot = pg.PlotWidget()
        self.relative_plot.setTitle(
            "Power relative to frozen contact baseline",
            size="12pt",
            bold=True,
        )
        self.relative_plot.setLabel("bottom", "Frequency", units="Hz")
        self.relative_plot.setLabel("left", "Relative power", units="dB")
        self.relative_plot.showGrid(x=True, y=True, alpha=0.3)
        self.relative_plot.setXRange(
            self.config.analysis_min_frequency_hz,
            self.config.analysis_max_frequency_hz,
        )
        self._relative_curve = self.relative_plot.plot(
            pen=pg.mkPen("m", width=2), name="relative dB"
        )
        self._relative_zero = pg.InfiniteLine(
            pos=0.0,
            angle=0,
            pen=pg.mkPen((160, 160, 160), style=QtCore.Qt.DashLine),
        )
        self.relative_plot.addItem(self._relative_zero)
        layout.addWidget(self.relative_plot, stretch=1)

        self.waterfall_plot = pg.PlotWidget()
        self.waterfall_plot.setTitle("Relative-power STFT history", size="12pt", bold=True)
        self.waterfall_plot.setLabel("bottom", "Time", units="s")
        self.waterfall_plot.setLabel("left", "Frequency", units="Hz")
        self.waterfall_plot.showGrid(x=True, y=True, alpha=0.2)
        self.waterfall_plot.setYRange(
            self.config.analysis_min_frequency_hz,
            self.config.analysis_max_frequency_hz,
        )
        self._waterfall_image = pg.ImageItem()
        self.waterfall_plot.addItem(self._waterfall_image)
        layout.addWidget(self.waterfall_plot, stretch=2)

        self.setCentralWidget(central)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(max(20, int(self.config.update_interval_s * 1000.0)))
        self.show()

    def submit(self, snapshot) -> None:
        """从采集线程提交不可变快照，不直接操作 Qt 图元。"""
        with self._lock:
            self._latest_snapshot = snapshot
            self._progress = None
            self._history.append(snapshot)
            cutoff = float(snapshot.spectrum_time_s) - self.config.history_duration_s
            while self._history and float(self._history[0].spectrum_time_s) < cutoff:
                self._history.popleft()

    def submit_progress(self, ready_samples: int, required_samples: int) -> None:
        """提交连续速度窗积累进度。"""
        try:
            ready = max(0, int(ready_samples))
            required = max(1, int(required_samples))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("频谱窗口进度必须是整数") from exc
        ready = min(ready, required)
        with self._lock:
            if ready < required:
                self._latest_snapshot = None
                self._history.clear()
                self._progress = (ready, required)
            else:
                self._progress = None

    def _format_friction(self, snapshot) -> str:
        """格式化唯一比值、阈值和时间滞回配置。"""
        if snapshot is None:
            return "频谱摩擦：WAITING（等待完整速度窗）"
        state = "静摩擦 STICK" if snapshot.state_name == "STICK" else "滑动摩擦 SLIP"
        low, high = self.config.slip_band_hz
        return (
            f"频谱摩擦：{state} | slip_band_power_ratio={snapshot.slip_band_power_ratio:.3f} "
            f"({low:g}-{high:g}Hz / {self.config.analysis_min_frequency_hz:g}-"
            f"{self.config.analysis_max_frequency_hz:g}Hz) | threshold={snapshot.threshold:.3f} | "
            f"连续窗 enter={self.config.enter_windows}, exit={self.config.exit_windows}"
        )

    def _clear_plots(self) -> None:
        self._curve_x.setData([], [])
        self._curve_y.setData([], [])
        self._curve_combined.setData([], [])
        self._relative_curve.setData([], [])
        self._waterfall_image.clear()

    def refresh(self) -> None:
        """在 Qt 主线程刷新同一快照的曲线、状态和相对dB瀑布。"""
        with self._lock:
            latest = self._latest_snapshot
            history = tuple(self._history)
            progress = self._progress
        if progress is not None:
            ready, required = progress
            self._clear_plots()
            self.friction_label.setText("频谱摩擦：WAITING（等待完整速度窗）")
            self.status_label.setText(f"等待速度 STFT：{ready}/{required} 个位置点")
            return
        if latest is None:
            return
        frequencies = np.asarray(latest.frequency_hz, dtype=np.float64)
        amplitude_x = np.asarray(latest.velocity_amplitude_x, dtype=np.float32)
        amplitude_y = np.asarray(latest.velocity_amplitude_y, dtype=np.float32)
        amplitude_combined = np.asarray(latest.velocity_amplitude_combined, dtype=np.float32)
        relative_power_db = np.asarray(latest.relative_power_db, dtype=np.float32)
        self._curve_x.setData(frequencies, amplitude_x)
        self._curve_y.setData(frequencies, amplitude_y)
        self._curve_combined.setData(frequencies, amplitude_combined)
        self._relative_curve.setData(frequencies, relative_power_db)
        finite = amplitude_combined[np.isfinite(amplitude_combined)]
        maximum = float(np.max(finite)) if finite.size else 0.0
        self.spectrum_plot.setYRange(0.0, max(maximum * 1.1, 1e-6), padding=0)
        finite_relative = relative_power_db[np.isfinite(relative_power_db)]
        if finite_relative.size:
            self.relative_plot.setYRange(
                min(float(np.min(finite_relative)) - 3.0, -3.0),
                max(float(np.max(finite_relative)) + 3.0, 3.0),
                padding=0,
            )
        self.friction_label.setText(self._format_friction(latest))
        baseline_text = "相对基线已冻结" if latest.baseline_established else "相对基线收集中"
        self.status_label.setText(
            f"t={latest.spectrum_time_s:.3f}s | {baseline_text} | snapshots={len(history)} | "
            f"{self.config.window_duration_s:.3g}s velocity STFT @ {self.config.sample_rate_hz:.1f}Hz"
        )
        if not history:
            return
        image = np.stack([
            np.asarray(item.relative_power_db, dtype=np.float32)
            for item in history
        ], axis=0).T
        finite_history = image[np.isfinite(image)]
        if finite_history.size:
            low = float(np.min(finite_history))
            high = float(np.percentile(finite_history, self.config.color_percentile))
        else:
            low, high = 0.0, 1.0
        high = max(high, low + 1e-6)
        self._waterfall_image.setImage(
            np.nan_to_num(image, nan=low), autoLevels=False, levels=(low, high)
        )
        times = np.asarray([float(item.spectrum_time_s) for item in history])
        start_t, end_t = float(times[0]), float(times[-1])
        width = max(end_t - start_t, self.config.update_interval_s)
        self._waterfall_image.setRect(QtCore.QRectF(
            start_t,
            self.config.analysis_min_frequency_hz,
            width,
            self.config.analysis_max_frequency_hz - self.config.analysis_min_frequency_hz,
        ))
        self.waterfall_plot.setXRange(start_t, start_t + width, padding=0)

    def closeEvent(self, event) -> None:
        """关闭显示定时器，但不停止采集或删除分析器历史。"""
        self.timer.stop()
        event.accept()


__all__ = ["SpectrumWindow"]
