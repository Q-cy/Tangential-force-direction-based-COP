"""压力阵列的全局滑移检测。

本模块只负责滑移状态和运动方向估计，不负责 CoP、标定、CSV 或 GUI。
算法对应 ``eskin_gripper_sdk`` 的 ``TangentialSensor``：在归一化压力斑块
上做有限半径的零填充平移搜索，以余弦相关及相对零平移的 improvement
确认真实斑块运动；CoP 短窗位移和相对 anchor 的大位移作为运动证据，并以
连续帧滞回避免状态抖动。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
import math

import numpy as np

from ..config import SlipConfig


class TangentialMotionState(IntEnum):
    """全局切向运动状态。"""

    NO_CONTACT = 0
    STICK = 1
    SLIP = 2


@dataclass(frozen=True)
class SlipResult:
    """一帧滑移检测结果。

    Attributes:
        motion_state: ``NO_CONTACT``、``STICK`` 或 ``SLIP``。
        is_slipping: 当前是否处于滑移状态。
        motion_distance: 短窗首尾 CoP 位移，单位为 cell。
        confidence: 压力斑块平移确认的余弦相关置信度；未确认时为 0。
        direction_x/direction_y: 当前用于切向方向的运动向量。
        angle_vector_magnitude: 该方向向量的模长，单位为 cell。
        reanchored: 本帧是否刚刚退出滑移并完成重新锚定。
        patch_row_shift/patch_col_shift: 最优斑块平移，单位为 cell。
        patch_correlation: 最优平移的余弦相关值。
        patch_improvement: 最优平移相对零平移的相关提升。
    """

    motion_state: TangentialMotionState = TangentialMotionState.NO_CONTACT
    is_slipping: bool = False
    motion_distance: float = 0.0
    confidence: float = 0.0
    direction_x: float = 0.0
    direction_y: float = 0.0
    angle_vector_magnitude: float = 0.0
    reanchored: bool = False
    patch_row_shift: int = 0
    patch_col_shift: int = 0
    patch_correlation: float = 0.0
    patch_improvement: float = 0.0

    @property
    def slip_motion_distance(self) -> float:
        """返回短窗滑移位移别名。"""
        return self.motion_distance

    @property
    def slip_confidence(self) -> float:
        """返回滑移置信度别名。"""
        return self.confidence


@dataclass(frozen=True)
class _MotionSample:
    cop_x: float
    cop_y: float
    normalized: np.ndarray


class SlipDetector:
    """逐帧检测 12×7 压力阵列的全局滑移。

    Args:
        config: 滑移参数；为 ``None`` 时使用 ``SlipConfig`` 默认值。
        rows: 压力阵列行数，默认 12。
        cols: 压力阵列列数，默认 7。

    ``update`` 的 ``cop_x/cop_y`` 必须使用当前项目的全局 CoP 坐标约定：
    x 为列、y 为行。角度坐标转换由上层复用
    ``PRSensorAngle._compute_cop_angle``，本类不自行规定 0 度方向。
    每个 ``SlipDetector`` 实例都有自己的历史、anchor 和滞回计数，双传感器
    不得共享同一个实例。
    """

    def __init__(self, config: SlipConfig | None = None,
                 rows: int = 12, cols: int = 7) -> None:
        self.config = (config or SlipConfig()).validate()
        if rows <= 0 or cols <= 0:
            raise ValueError("SlipDetector.rows/cols 必须大于 0")
        self.rows = int(rows)
        self.cols = int(cols)
        self._history: deque[_MotionSample] = deque(
            maxlen=self.config.window_frames
        )
        self._state = TangentialMotionState.NO_CONTACT
        self._anchor_x: float | None = None
        self._anchor_y: float | None = None
        self._enter_counter = 0
        self._exit_counter = 0
        self._direction_x = 0.0
        self._direction_y = 0.0
        self._last = SlipResult()

    @property
    def motion_state(self) -> TangentialMotionState:
        """返回当前运动状态。"""
        return self._state

    @property
    def is_slipping(self) -> bool:
        """返回当前是否处于 ``SLIP``。"""
        return self._state is TangentialMotionState.SLIP

    @property
    def anchor(self) -> tuple[float | None, float | None]:
        """返回 detector 内部维护的滑移/静摩擦 anchor。"""
        return self._anchor_x, self._anchor_y

    def reset(self) -> SlipResult:
        """清除无接触时的全部滑移历史并返回 ``NO_CONTACT`` 结果。"""
        self._history.clear()
        self._state = TangentialMotionState.NO_CONTACT
        self._anchor_x = self._anchor_y = None
        self._enter_counter = self._exit_counter = 0
        self._direction_x = self._direction_y = 0.0
        self._last = SlipResult()
        return self._last

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        """按整帧压力和归一化，保持斑块形状而消除总压变化。"""
        total = float(np.sum(matrix))
        if total <= 0.0:
            return np.zeros_like(matrix, dtype=np.float64)
        return np.asarray(matrix, dtype=np.float64) / total

    def _patch_translation(self, reference: np.ndarray,
                           current: np.ndarray) -> tuple[int, int, float, float]:
        """复刻 C++ 的零填充平移和余弦相关搜索。"""
        zero = 0.0
        best = 0.0
        best_row = best_col = 0
        radius = self.config.patch_search_radius

        def correlation(row_shift: int, col_shift: int) -> float:
            shifted = np.zeros_like(reference)
            source_r0 = max(0, -row_shift)
            source_r1 = min(self.rows, self.rows - row_shift)
            source_c0 = max(0, -col_shift)
            source_c1 = min(self.cols, self.cols - col_shift)
            dest_r0 = max(0, row_shift)
            dest_r1 = dest_r0 + max(0, source_r1 - source_r0)
            dest_c0 = max(0, col_shift)
            dest_c1 = dest_c0 + max(0, source_c1 - source_c0)
            if source_r1 > source_r0 and source_c1 > source_c0:
                shifted[dest_r0:dest_r1, dest_c0:dest_c1] = reference[
                    source_r0:source_r1, source_c0:source_c1
                ]
            denominator = math.sqrt(
                float(np.sum(shifted * shifted))
                * float(np.sum(current * current))
            )
            return float(np.sum(shifted * current) / denominator) if denominator > 1e-12 else 0.0

        zero = correlation(0, 0)
        best = zero
        for row_shift in range(-radius, radius + 1):
            for col_shift in range(-radius, radius + 1):
                if row_shift == 0 and col_shift == 0:
                    continue
                score = correlation(row_shift, col_shift)
                if score > best + 1e-6:
                    best = score
                    best_row, best_col = row_shift, col_shift
        return best_row, best_col, max(0.0, min(1.0, best)), best - zero

    def update(self, pressure: np.ndarray, cop_x: float, cop_y: float,
               contact: bool, ready: bool = True) -> SlipResult:
        """用一帧压力矩阵和全局 CoP 更新状态。

        Args:
            pressure: 形状为 ``(rows, cols)`` 或可展平为该形状的压力矩阵。
            cop_x/cop_y: 当前全局 CoP，x 为列、y 为行。
            contact: 上游 CoP 状态机是否确认接触。
            ready: 当前帧是否允许推进短窗状态；接触存在但尚未完成
                CoP 精修时保持 ``STICK`` 且不推进历史。无接触或无效 CoP
                才会完整 reset。

        Returns:
            SlipResult: 不可变的当前状态、距离、置信度和方向结果。

        Raises:
            ValueError: 压力形状错误或配置/输入不合法。
        """
        matrix = np.asarray(pressure, dtype=np.float64)
        expected_shape = (self.rows, self.cols)
        if matrix.size != self.rows * self.cols:
            raise ValueError(f"压力矩阵元素数必须为{self.rows * self.cols}")
        matrix = matrix.reshape(expected_shape)
        valid_cop = np.isfinite(cop_x) and np.isfinite(cop_y)
        if not contact or not valid_cop or float(np.sum(matrix)) <= 0.0:
            return self.reset()

        normalized = self._normalize(matrix)
        current = _MotionSample(float(cop_x), float(cop_y), normalized)
        if not ready:
            # 接触已经存在但 CoP 精修尚未完成时，严格保持 STICK 的接触
            # 语义，不推进短窗历史；参考 C++ 只在精修完成后调用
            # updateSlipState。
            self._state = TangentialMotionState.STICK
            self._anchor_x, self._anchor_y = current.cop_x, current.cop_y
            self._history.clear()
            self._enter_counter = self._exit_counter = 0
            self._direction_x = self._direction_y = 0.0
            self._last = SlipResult(TangentialMotionState.STICK, False)
            return self._last
        if self._state is TangentialMotionState.NO_CONTACT:
            self._state = TangentialMotionState.STICK
            self._anchor_x, self._anchor_y = current.cop_x, current.cop_y
            self._history.clear()
        self._history.append(current)

        if not self.config.enabled or len(self._history) < self.config.window_frames:
            self._last = SlipResult(self._state, self.is_slipping)
            return self._last

        first = self._history[0]
        last = self._history[-1]
        motion_x = last.cop_x - first.cop_x
        motion_y = last.cop_y - first.cop_y
        motion_distance = math.hypot(motion_x, motion_y)
        row_shift, col_shift, correlation, improvement = self._patch_translation(
            first.normalized, last.normalized
        )
        patch_confirms = (
            (row_shift != 0 or col_shift != 0)
            and correlation >= self.config.patch_min_correlation
            and improvement >= self.config.patch_min_improvement
        )
        confidence = correlation if patch_confirms else 0.0

        reanchored = False
        if self._state is not TangentialMotionState.SLIP:
            distance_from_anchor = math.hypot(
                current.cop_x - float(self._anchor_x),
                current.cop_y - float(self._anchor_y),
            )
            large_translation = distance_from_anchor >= self.config.reanchor_distance
            evidence = motion_distance >= self.config.enter_distance and (
                patch_confirms or large_translation
            )
            self._enter_counter = self._enter_counter + 1 if evidence else 0
            if self._enter_counter >= self.config.enter_frames:
                self._state = TangentialMotionState.SLIP
                self._enter_counter = self._exit_counter = 0
                self._direction_x, self._direction_y = motion_x, motion_y
                self._anchor_x, self._anchor_y = current.cop_x, current.cop_y
        else:
            if motion_distance > 1e-6:
                alpha = self.config.direction_smoothing
                self._direction_x = (1.0 - alpha) * self._direction_x + alpha * motion_x
                self._direction_y = (1.0 - alpha) * self._direction_y + alpha * motion_y
            self._anchor_x, self._anchor_y = current.cop_x, current.cop_y
            self._exit_counter = (
                self._exit_counter + 1
                if motion_distance <= self.config.exit_distance else 0
            )
            if self._exit_counter >= self.config.exit_frames:
                self._state = TangentialMotionState.STICK
                self._enter_counter = self._exit_counter = 0
                self._direction_x = self._direction_y = 0.0
                motion_distance = 0.0
                confidence = 0.0
                self._history.clear()
                self._history.append(current)
                reanchored = True

        magnitude = math.hypot(self._direction_x, self._direction_y) if self.is_slipping else 0.0
        self._last = SlipResult(
            motion_state=self._state,
            is_slipping=self.is_slipping,
            motion_distance=float(motion_distance),
            confidence=float(confidence),
            direction_x=float(self._direction_x if self.is_slipping else 0.0),
            direction_y=float(self._direction_y if self.is_slipping else 0.0),
            angle_vector_magnitude=float(magnitude),
            reanchored=reanchored,
            patch_row_shift=int(row_shift),
            patch_col_shift=int(col_shift),
            patch_correlation=float(correlation),
            patch_improvement=float(improvement),
        )
        return self._last

    def process(self, pressure: np.ndarray, cop_x: float, cop_y: float,
                contact: bool, ready: bool = True) -> SlipResult:
        """``update`` 的语义别名，便于在逐帧处理管线中调用。"""
        return self.update(pressure, cop_x, cop_y, contact, ready)


__all__ = ["TangentialMotionState", "SlipResult", "SlipDetector"]
