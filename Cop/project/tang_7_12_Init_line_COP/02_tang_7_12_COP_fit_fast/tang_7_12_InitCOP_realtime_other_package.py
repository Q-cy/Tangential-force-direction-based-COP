"""六维力/标定力等标量角度工具（PZT/CoP 由 main 直接调用 package_note）。"""

from collections import deque

import numpy as np


def compute_vector_angle(x: float, y: float) -> float:
    angle = float(np.degrees(np.arctan2(y, x + 1e-8)))
    if angle < 0:
        angle += 360
    return angle


def compute_6Dforce_angle(fx: float, fy: float) -> float:
    return compute_vector_angle(fx, fy)


def angle_difference(a1: float, a2: float) -> float:
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)


def compute_adc_variance(adc_data) -> float:
    """单帧 84 通道 ADC 的空间方差（不修改任何状态）。"""
    arr = np.asarray(adc_data, dtype=np.float32)
    mean = float(arr.mean())
    return float(((arr - mean) ** 2).mean())


class AdcVarianceTracker:
    """维护最近 N 帧 ADC 空间方差的滑动窗口，返回当前 std。"""

    def __init__(self, window: int = 30):
        self._history = deque(maxlen=window)

    def update(self, adc_data) -> tuple[float, float]:
        var = compute_adc_variance(adc_data)
        self._history.append(var)
        if len(self._history) >= 2:
            std = float(np.std(np.fromiter(self._history, dtype=np.float32)))
        else:
            std = 0.0
        return var, std

    def reset(self) -> None:
        self._history.clear()

