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
