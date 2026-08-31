"""CoP速度STFT、旁路相对基线与滑移频带功率占比的内部实现。"""

from __future__ import annotations

import os
import tempfile
from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import numpy as np

from ..config import SpectrumConfig


def _analysis_frequency_axis(config: SpectrumConfig) -> np.ndarray:
    """返回配置分析频带内的 rFFT 频率轴。"""
    full = np.fft.rfftfreq(config.window_samples, d=1.0 / config.sample_rate_hz)
    selected = ((full >= config.analysis_min_frequency_hz - 1e-12)
                & (full <= config.analysis_max_frequency_hz + 1e-12))
    return full[selected]


class SpectralFrictionState(IntEnum):
    """频谱摩擦判定的内部状态。"""

    WAITING = 0
    STICK = 1
    SLIP = 2


@dataclass(frozen=True, slots=True)
class SpectrumSnapshot:
    """一份供状态、GUI 和 NPZ 共用的不可变频谱快照。"""

    frequency_hz: np.ndarray
    spectrum_time_s: float
    velocity_amplitude_x: np.ndarray
    velocity_amplitude_y: np.ndarray
    velocity_amplitude_combined: np.ndarray
    baseline_power: np.ndarray
    relative_power_db: np.ndarray
    baseline_established: bool
    slip_band_power_ratio: float
    friction_state: SpectralFrictionState
    threshold: float
    revision: int

    def __post_init__(self) -> None:
        arrays = {
            "frequency_hz": np.asarray(self.frequency_hz, dtype=np.float64).copy(),
            "velocity_amplitude_x": np.asarray(self.velocity_amplitude_x, dtype=np.float32).copy(),
            "velocity_amplitude_y": np.asarray(self.velocity_amplitude_y, dtype=np.float32).copy(),
            "velocity_amplitude_combined": np.asarray(self.velocity_amplitude_combined, dtype=np.float32).copy(),
            "baseline_power": np.asarray(self.baseline_power, dtype=np.float32).copy(),
            "relative_power_db": np.asarray(self.relative_power_db, dtype=np.float32).copy(),
        }
        size = arrays["frequency_hz"].size
        for name in (
            "velocity_amplitude_x", "velocity_amplitude_y",
            "velocity_amplitude_combined", "baseline_power", "relative_power_db",
        ):
            if arrays[name].size != size:
                raise ValueError(f"SpectrumSnapshot.{name} 长度不匹配")
        for array in arrays.values():
            array.setflags(write=False)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "spectrum_time_s", float(self.spectrum_time_s))
        object.__setattr__(self, "baseline_established", bool(self.baseline_established))
        object.__setattr__(self, "slip_band_power_ratio", float(self.slip_band_power_ratio))
        object.__setattr__(self, "friction_state", SpectralFrictionState(self.friction_state))
        object.__setattr__(self, "threshold", float(self.threshold))
        object.__setattr__(self, "revision", int(self.revision))

    @property
    def time_s(self) -> float:
        """返回相对第一帧压力时间的频谱时间。"""
        return self.spectrum_time_s

    @property
    def state(self) -> SpectralFrictionState:
        """返回内部摩擦状态。"""
        return self.friction_state

    @property
    def state_name(self) -> str:
        """返回 WAITING、STICK 或 SLIP。"""
        return self.friction_state.name

    @property
    def amplitude_x(self) -> np.ndarray:
        """返回 X 方向速度单边幅值谱。"""
        return self.velocity_amplitude_x

    @property
    def amplitude_y(self) -> np.ndarray:
        """返回 Y 方向速度单边幅值谱。"""
        return self.velocity_amplitude_y

    @property
    def amplitude_combined(self) -> np.ndarray:
        """返回 X/Y 合成速度幅值谱。"""
        return self.velocity_amplitude_combined


class CopSpectrumAnalyzer:
    """重采样CoP，以功率占比判定状态，并旁路记录冻结相对基线。"""

    def __init__(self, config: SpectrumConfig | None = None) -> None:
        self.config = (config or SpectrumConfig()).validate()
        self.sample_rate_hz = float(self.config.sample_rate_hz)
        self.sample_period_s = 1.0 / self.sample_rate_hz
        self.window_samples = int(self.config.window_samples)
        self._required_samples = int(self.config.required_samples)
        self._frequency_hz = _analysis_frequency_axis(self.config)
        low, high = self.config.slip_band_hz
        self._slip_band_mask = ((self._frequency_hz >= low - 1e-12)
                                & (self._frequency_hz <= high + 1e-12))
        if not self._frequency_hz.size or not np.any(self._slip_band_mask):
            raise ValueError("分析频带或滑移频带没有对应 FFT 频点")
        self._hann = np.hanning(self.window_samples + 1)[:-1]
        self._hann_sum = float(np.sum(self._hann))
        if self._hann_sum <= 0.0:
            raise ValueError("SpectrumConfig 产生了无效 Hann 窗")
        size = max(self._required_samples, 4)
        self._resampled_times: deque[float] = deque(maxlen=size)
        self._resampled_x: deque[float] = deque(maxlen=size)
        self._resampled_y: deque[float] = deque(maxlen=size)
        self._last_input: tuple[float, float, float] | None = None
        self._last_grid_t: float | None = None
        self._time_origin_t: float | None = None
        self._last_snapshot_t: float | None = None
        self._friction_state = SpectralFrictionState.WAITING
        self._enter_count = 0
        self._exit_count = 0
        self._baseline_power: np.ndarray | None = None
        self._baseline_power_samples: list[np.ndarray] = []
        self._baseline_start_t: float | None = None
        self._revision = 0
        self._snapshots: list[SpectrumSnapshot] = []
        self._recent_snapshots: deque[SpectrumSnapshot] = deque()

    @property
    def frequencies_hz(self) -> np.ndarray:
        """返回完整分析频带的频率轴副本。"""
        return self._frequency_hz.copy()

    @property
    def ready_samples(self) -> int:
        """返回当前连续段已积累的 CoP 位置点数。"""
        return len(self._resampled_x)

    @property
    def required_samples(self) -> int:
        """返回形成完整速度窗所需的 CoP 位置点数。"""
        return self._required_samples

    @property
    def snapshots(self) -> tuple[SpectrumSnapshot, ...]:
        """返回完整会话快照历史。"""
        return tuple(self._snapshots)

    @property
    def recent_snapshots(self) -> tuple[SpectrumSnapshot, ...]:
        """返回 GUI 历史时长内的快照。"""
        return tuple(self._recent_snapshots)

    def get_snapshots(self) -> tuple[SpectrumSnapshot, ...]:
        """返回完整会话快照历史。"""
        return self.snapshots

    def get_recent_snapshots(self) -> tuple[SpectrumSnapshot, ...]:
        """返回最近快照历史。"""
        return self.recent_snapshots

    @property
    def current_snapshot(self) -> SpectrumSnapshot | None:
        """返回最近快照；尚未积满完整窗时返回 ``None``。"""
        return self._snapshots[-1] if self._snapshots else None

    @property
    def friction_state(self) -> SpectralFrictionState:
        """返回当前内部频谱摩擦状态。"""
        return self._friction_state

    @property
    def baseline_power(self) -> np.ndarray | None:
        """返回已冻结的逐频点基线副本；尚未建立时返回``None``。"""
        return None if self._baseline_power is None else self._baseline_power.copy()

    def reset(self, *, reset_time_origin: bool = False) -> None:
        """重置连续窗和状态，不删除已经生成的会话历史。"""
        self._clear_segment(reset_baseline=True)
        if reset_time_origin:
            self._time_origin_t = None

    def process(self, rx_t: float, cop_x: float, cop_y: float, state: int) -> SpectrumSnapshot | None:
        """提交一帧未滤波绝对 CoP，按更新间隔最多返回一个快照。"""
        try:
            timestamp, x_value, y_value, state_value = float(rx_t), float(cop_x), float(cop_y), int(state)
        except (TypeError, ValueError, OverflowError):
            self._clear_segment(reset_baseline=True)
            return None
        if self._time_origin_t is None and np.isfinite(timestamp):
            self._time_origin_t = timestamp
        if (state_value != self.config.required_cop_state or not np.isfinite(timestamp)
                or not np.isfinite(x_value) or not np.isfinite(y_value)):
            self._clear_segment(reset_baseline=True)
            return None
        if self._last_input is None:
            self._start_segment(timestamp, x_value, y_value)
            return None
        previous_t, previous_x, previous_y = self._last_input
        gap = timestamp - previous_t
        if gap <= 0.0:
            self._clear_segment(reset_baseline=True)
            return None
        if gap > self.config.max_gap_s:
            # 通信gap不跨段插值；已冻结基线保留，未完成基线重新收集。
            self._clear_segment(reset_baseline=False)
            if gap > 0.0:
                self._start_segment(timestamp, x_value, y_value)
            return None
        self._append_interpolated(previous_t, previous_x, previous_y, timestamp, x_value, y_value)
        self._last_input = (timestamp, x_value, y_value)
        return self._maybe_make_snapshot(self._last_grid_t)

    def _clear_segment(self, *, reset_baseline: bool) -> None:
        self._resampled_times.clear()
        self._resampled_x.clear()
        self._resampled_y.clear()
        self._last_input = None
        self._last_grid_t = None
        self._last_snapshot_t = None
        self._friction_state = SpectralFrictionState.WAITING
        self._enter_count = 0
        self._exit_count = 0
        if reset_baseline:
            self._baseline_power = None
            self._baseline_power_samples.clear()
            self._baseline_start_t = None
        elif self._baseline_power is None:
            self._baseline_power_samples.clear()
            self._baseline_start_t = None

    def _start_segment(self, timestamp: float, x_value: float, y_value: float) -> None:
        self._resampled_times.append(timestamp)
        self._resampled_x.append(x_value)
        self._resampled_y.append(y_value)
        self._last_input = (timestamp, x_value, y_value)
        self._last_grid_t = timestamp

    def _append_interpolated(self, previous_t: float, previous_x: float, previous_y: float,
                             timestamp: float, x_value: float, y_value: float) -> None:
        if self._last_grid_t is None:
            self._last_grid_t = previous_t
        next_grid_t = self._last_grid_t + self.sample_period_s
        while next_grid_t <= timestamp + 1e-12:
            alpha = (next_grid_t - previous_t) / (timestamp - previous_t)
            self._resampled_times.append(next_grid_t)
            self._resampled_x.append(previous_x + alpha * (x_value - previous_x))
            self._resampled_y.append(previous_y + alpha * (y_value - previous_y))
            self._last_grid_t = next_grid_t
            next_grid_t += self.sample_period_s

    def _velocity_spectrum(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values_x = np.asarray(self._resampled_x, dtype=np.float64)[-self._required_samples:]
        values_y = np.asarray(self._resampled_y, dtype=np.float64)[-self._required_samples:]
        velocity_x = np.diff(values_x) / self.sample_period_s
        velocity_y = np.diff(values_y) / self.sample_period_s
        velocity_x -= np.mean(velocity_x)
        velocity_y -= np.mean(velocity_y)
        spectrum_x = np.abs(np.fft.rfft(velocity_x * self._hann)) / self._hann_sum
        spectrum_y = np.abs(np.fft.rfft(velocity_y * self._hann)) / self._hann_sum
        if spectrum_x.size > 2:
            spectrum_x[1:-1] *= 2.0
            spectrum_y[1:-1] *= 2.0
        full = np.fft.rfftfreq(self.window_samples, d=self.sample_period_s)
        selected = ((full >= self.config.analysis_min_frequency_hz - 1e-12)
                    & (full <= self.config.analysis_max_frequency_hz + 1e-12))
        amplitude_x, amplitude_y = spectrum_x[selected], spectrum_y[selected]
        return amplitude_x, amplitude_y, np.hypot(amplitude_x, amplitude_y)

    def _power_ratio(self, amplitude_x: np.ndarray, amplitude_y: np.ndarray) -> float:
        """返回滑移频带功率除以完整分析频带功率。"""
        power = np.asarray(amplitude_x, dtype=np.float64) ** 2 + np.asarray(amplitude_y, dtype=np.float64) ** 2
        finite = np.isfinite(power)
        total = float(np.sum(power[finite]))
        return 0.0 if total <= np.finfo(float).eps else float(np.sum(power[finite & self._slip_band_mask]) / total)

    def _update_baseline(
        self,
        grid_timestamp: float,
        amplitude_x: np.ndarray,
        amplitude_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """旁路收集并冻结基线，返回基线、相对dB及建立状态。

        该方法不读取或修改摩擦状态、ratio和滞回计数。
        """
        power = (
            np.asarray(amplitude_x, dtype=np.float64) ** 2
            + np.asarray(amplitude_y, dtype=np.float64) ** 2
        )
        if self._baseline_power is None:
            if self._baseline_start_t is None:
                self._baseline_start_t = grid_timestamp
            self._baseline_power_samples.append(power.copy())
            if (
                grid_timestamp - self._baseline_start_t + 1e-12
                >= self.config.baseline_duration_s
            ):
                self._baseline_power = np.nanmedian(
                    np.stack(self._baseline_power_samples, axis=0), axis=0
                )
                self._baseline_power_samples.clear()
        if self._baseline_power is None:
            nan_values = np.full(power.shape, np.nan, dtype=np.float64)
            return nan_values, nan_values.copy(), False
        baseline = self._baseline_power.copy()
        floor = self.config.baseline_power_floor
        relative_db = 10.0 * np.log10(
            np.maximum(
                (power + floor) / (baseline + floor),
                np.finfo(float).tiny,
            )
        )
        return baseline, relative_db, True

    def _update_state(self, ratio: float) -> None:
        if not np.isfinite(ratio):
            return
        threshold = self.config.slip_band_power_ratio_threshold
        if self._friction_state == SpectralFrictionState.STICK:
            self._enter_count = self._enter_count + 1 if ratio >= threshold else 0
            if self._enter_count >= self.config.enter_windows:
                self._friction_state = SpectralFrictionState.SLIP
                self._enter_count = 0
                self._exit_count = 0
        elif self._friction_state == SpectralFrictionState.SLIP:
            self._exit_count = self._exit_count + 1 if ratio < threshold else 0
            if self._exit_count >= self.config.exit_windows:
                self._friction_state = SpectralFrictionState.STICK
                self._enter_count = 0
                self._exit_count = 0

    def _make_snapshot(self, grid_timestamp: float) -> SpectrumSnapshot:
        amplitude_x, amplitude_y, amplitude_combined = self._velocity_spectrum()
        ratio = self._power_ratio(amplitude_x, amplitude_y)
        baseline_power, relative_power_db, baseline_established = (
            self._update_baseline(grid_timestamp, amplitude_x, amplitude_y)
        )
        if self._friction_state == SpectralFrictionState.WAITING:
            # 首窗必须先输出STICK，同时把本窗ratio计入之后的连续窗证据。
            self._friction_state = SpectralFrictionState.STICK
            self._enter_count = 1 if ratio >= self.config.slip_band_power_ratio_threshold else 0
            self._exit_count = 0
        else:
            self._update_state(ratio)
        time_s = 0.0 if self._time_origin_t is None else grid_timestamp - self._time_origin_t
        self._revision += 1
        snapshot = SpectrumSnapshot(
            self._frequency_hz,
            time_s,
            amplitude_x,
            amplitude_y,
            amplitude_combined,
            baseline_power,
            relative_power_db,
            baseline_established,
            ratio,
            self._friction_state,
            self.config.slip_band_power_ratio_threshold,
            self._revision,
        )
        self._last_snapshot_t = grid_timestamp
        self._snapshots.append(snapshot)
        self._recent_snapshots.append(snapshot)
        cutoff = time_s - self.config.history_duration_s
        while self._recent_snapshots and self._recent_snapshots[0].spectrum_time_s < cutoff:
            self._recent_snapshots.popleft()
        return snapshot

    def _maybe_make_snapshot(self, grid_timestamp: float | None) -> SpectrumSnapshot | None:
        if grid_timestamp is None or len(self._resampled_x) < self._required_samples:
            return None
        if self._last_snapshot_t is not None and grid_timestamp - self._last_snapshot_t < self.config.update_interval_s - 1e-12:
            return None
        return self._make_snapshot(grid_timestamp)

    def save_npz(self, path: str | os.PathLike[str], csv_file_name: str = "") -> bool:
        """原子保存新会话的精简频谱 schema；没有快照时不创建文件。"""
        if not self._snapshots:
            return False
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frequency_hz = np.asarray(self._snapshots[0].frequency_hz, dtype=np.float64)
        fields = {
            name: np.stack([getattr(s, name) for s in self._snapshots]).astype(np.float32)
            for name in (
                "velocity_amplitude_x", "velocity_amplitude_y",
                "velocity_amplitude_combined", "baseline_power", "relative_power_db",
            )
        }
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(mode="w+b", dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp", delete=False) as temporary:
                temporary_name = temporary.name
                np.savez_compressed(
                    temporary, frequency_hz=frequency_hz,
                    spectrum_time_s=np.asarray([s.spectrum_time_s for s in self._snapshots], dtype=np.float64),
                    **fields,
                    slip_band_power_ratio=np.asarray([s.slip_band_power_ratio for s in self._snapshots], dtype=np.float64),
                    friction_state=np.asarray([int(s.friction_state) for s in self._snapshots], dtype=np.int8),
                    baseline_established=np.asarray(
                        [s.baseline_established for s in self._snapshots], dtype=bool
                    ),
                    threshold=np.float64(self.config.slip_band_power_ratio_threshold),
                    sample_rate_hz=np.float64(self.config.sample_rate_hz),
                    window_duration_s=np.float64(self.config.window_duration_s),
                    update_interval_s=np.float64(self.config.update_interval_s),
                    analysis_frequency_hz=np.asarray([self.config.analysis_min_frequency_hz, self.config.analysis_max_frequency_hz]),
                    slip_band_hz=np.asarray(self.config.slip_band_hz),
                    enter_windows=np.int64(self.config.enter_windows),
                    exit_windows=np.int64(self.config.exit_windows),
                    baseline_duration_s=np.float64(self.config.baseline_duration_s),
                    baseline_power_floor=np.float64(self.config.baseline_power_floor),
                    max_gap_s=np.float64(self.config.max_gap_s),
                    required_cop_state=np.int64(self.config.required_cop_state),
                    window_name=np.asarray("periodic_hann_velocity_stft"),
                    csv_file_name=np.asarray(csv_file_name),
                )
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, target)
            temporary_name = None
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
        return True


__all__ = ["CopSpectrumAnalyzer", "SpectrumSnapshot", "SpectralFrictionState"]
