"""
预滑动检测模块（CoP 频域分析法 — 12x7 PZT 适配版）

判据（忠实于原方案）：
  每 60ms 取一帧 -> get_cop() -> 中心化 -> ρ = hypot(x_c, y_c)
  -> 滑动窗口 N=32 -> 去 DC -> FFT -> 取 bin 1..4 最大幅值 = feature
  -> is_pre_slip = (feature >= 3 * baseline)

边沿检测：False -> True 跳变累计 slip_count（一次预滑动事件 = 1 次）。
"""
import numpy as np
from collections import deque
from typing import Optional


def _median_filter(signal: list, window: int) -> list:
    """滑动窗口中值滤波：边界处用较短窗口，返回等长列表"""
    if window <= 1 or len(signal) == 0:
        return list(signal)
    out = []
    n = len(signal)
    for i in range(n):
        start = max(0, i - window // 2)
        end = min(n, start + window)
        start = max(0, end - window)
        out.append(float(np.median(signal[start:end])))
    return out


class PreSlipDetector:
    """预滑动检测器：复用 PZTSensorAngle.get_cop() 拿原始 CoP，与 origin-locking 并行独立运行"""

    def __init__(self, sensor_rows: int = 12, sensor_cols: int = 7,
                 N: int = 32, fs_hz: float = 1 / 0.06, low_bin_count: int = 4,
                 median_window: int = 3,
                 auto_baseline_warmup: int = 10):
        self.x_center = (sensor_cols - 1) / 2.0
        self.y_center = (sensor_rows - 1) / 2.0
        self.N = N
        self.fs = fs_hz
        self.low_bins = list(range(1, low_bin_count + 1))
        self.median_window = median_window
        self.auto_baseline_warmup = auto_baseline_warmup
        self.rho_buf = deque(maxlen=N)
        self.x_buf = deque(maxlen=N)
        self.y_buf = deque(maxlen=N)
        self.baseline_amp = 0.0
        self.last_push_t = 0.0
        self.slip_count = 0
        self._prev_is_pre = False
        self._cop_sensor = None
        self._baseline_samples = []
        self._baseline_locked = False

    def bind(self, cop_sensor) -> None:
        """绑定 PZTSensorAngle 实例（必须在 update 前调用）"""
        self._cop_sensor = cop_sensor

    def set_baseline(self, amp: float) -> None:
        """静态基线标定（启动后空载 10s 调用一次）"""
        self.baseline_amp = float(amp)

    def reset(self) -> None:
        """清空窗口与计数（手动重置用）"""
        self.rho_buf.clear()
        self.x_buf.clear()
        self.y_buf.clear()
        self.slip_count = 0
        self._prev_is_pre = False
        self.last_push_t = 0.0
        self._baseline_samples.clear()
        self._baseline_locked = False

    def update(self, adc_data: np.ndarray, now_t: float) -> Optional[dict]:
        """60ms 门控：每 60ms 取一帧做 FFT 判定
        返回 None（未到 60ms）或 {feature, is_pre_slip, drift_xy, rho,
                                slip_count, is_new_event}
        """
        if self._cop_sensor is None:
            raise RuntimeError("PreSlipDetector.bind(cop_sensor) 未调用")

        period = 1.0 / self.fs
        if now_t - self.last_push_t < period - 1e-3:
            return None
        self.last_push_t = now_t

        cop_x, cop_y = self._cop_sensor.get_cop(adc_data)
        x_c = cop_x - self.x_center
        y_c = cop_y - self.y_center
        rho = float(np.hypot(x_c, y_c))

        self.rho_buf.append(rho)
        self.x_buf.append(x_c)
        self.y_buf.append(y_c)
        if len(self.rho_buf) < self.N:
            return None

        rho_smooth = _median_filter(list(self.rho_buf), self.median_window)
        rho_arr = np.array(rho_smooth, dtype=np.float32)
        rho_arr = rho_arr - float(np.mean(rho_arr))
        spec = np.abs(np.fft.rfft(rho_arr))
        feature = float(np.max(spec[self.low_bins]))

        drift_xy = (float(np.mean(self.x_buf)), float(np.mean(self.y_buf)))

        # 自动基线：前 N 帧 feature 中位数作为 baseline（替代手动 set_baseline）
        if not self._baseline_locked:
            self._baseline_samples.append(feature)
            if len(self._baseline_samples) >= self.auto_baseline_warmup:
                self.baseline_amp = float(np.median(self._baseline_samples))
                self._baseline_locked = True
            return {
                "feature": feature, "is_pre_slip": False,
                "drift_xy": drift_xy, "rho": rho,
                "slip_count": self.slip_count, "is_new_event": False,
            }

        is_pre = feature > 15  # 固定绝对阈值

        is_new_event = is_pre and not self._prev_is_pre
        if is_new_event:
            self.slip_count += 1
        self._prev_is_pre = is_pre

        return {
            "feature": feature,
            "is_pre_slip": is_pre,
            "drift_xy": drift_xy,
            "rho": rho,
            "slip_count": self.slip_count,
            "is_new_event": is_new_event,
        }