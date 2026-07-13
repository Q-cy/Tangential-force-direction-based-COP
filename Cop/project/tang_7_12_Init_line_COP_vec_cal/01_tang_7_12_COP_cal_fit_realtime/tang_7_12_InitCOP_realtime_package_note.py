import numpy as np
import threading
from collections import deque


class PZTSensorAngle:
    """压阻传感器：12×7（默认）PZT 阵列的 CoP 角度估计"""

    def __init__(self, rows: int = 12, cols: int = 7,
                 k: float = 5, collect_frames: int = 20,
                 stability_frames: int = 5):
        self.rows = rows
        self.cols = cols
        self.k = k
        self.collect_frames = collect_frames
        self.stability_frames = stability_frames

        self._buf = deque(maxlen=collect_frames)
        self._thresh = None
        self._lock = threading.Lock()

        self._origin_x = None
        self._origin_y = None
        self._contact_init = False
        self._low_counter = 0

    # ---------- 公共 API ----------

    def get_all(self, adc_data) -> tuple[float, float, float]:
        """
        输入 rows*cols 个 ADC 值，一次性输出 (angle, dx, dy) 三件套。

        :param adc_data: list/np.array，长度为 rows*cols 的 ADC 原始数据
        :return: (angle, dx, dy)
            · angle: PZT 角度（0~360°）
            · dx:    CoP X 方向位移（列方向，cells）
            · dy:    CoP Y 方向位移（行方向，cells）
        :raises ValueError: ADC 数据长度不等于 rows*cols 时抛出
        """
        expected = self.rows * self.cols
        if len(adc_data) != expected:
            raise ValueError(f"ADC数据长度必须为{expected}")

        dx, dy = self._compute_cop(adc_data)
        angle = self._compute_cop_angle(dx, dy)
        return angle, dx, dy

    def get_angle(self, adc_data) -> float:
        """便捷接口：只输出 PZT 角度（0~360°）。等价于 get_all(...)[0]。"""
        angle, _, _ = self.get_all(adc_data)
        return angle

    def reset_origin(self) -> None:
        """清掉首次接触 origin 与低压计数；阈值（若已确定）保留。"""
        self._origin_x = None
        self._origin_y = None
        self._contact_init = False
        self._low_counter = 0

    # ---------- 内部算法 ----------

    def _update_dynamic_threshold(self, total_pressure: float) -> None:
        with self._lock:
            if self._thresh is None:
                self._buf.append(total_pressure)
                if len(self._buf) >= self.collect_frames:
                    self._thresh = self.k * float(np.mean(self._buf))

    def _compute_cop(self, raw_frame) -> tuple[float, float]:
        rows, cols = self.rows, self.cols
        frame_flat = np.asarray(raw_frame, dtype=np.float32).flatten()
        frame2d = frame_flat.reshape(rows, cols)

        total_pressure = float(np.sum(frame2d))

        self._update_dynamic_threshold(total_pressure)

        # thresh 未确定不进入低压分支
        if self._thresh is not None:
            if total_pressure < self._thresh:
                self._low_counter += 1
            else:
                self._low_counter = 0

            if self._low_counter >= self.stability_frames:
                self.reset_origin()
                return 0.0, 0.0

        if total_pressure == 0:
            return 0.0, 0.0

        x_grid = np.tile(np.arange(cols), (rows, 1))
        y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols)
        cop_x = np.sum(frame2d * x_grid) / total_pressure
        cop_y = np.sum(frame2d * y_grid) / total_pressure

        delta_x = 0.0
        delta_y = 0.0
        if not self._contact_init:
            self._origin_x = cop_x
            self._origin_y = cop_y
            self._contact_init = True
        else:
            delta_x = cop_x - self._origin_x
            delta_y = cop_y - self._origin_y
        return delta_x, delta_y

    @staticmethod
    def _compute_angle(x: float, y: float) -> float:
        epsilon = 1e-8
        angle = np.degrees(np.arctan2(y, x + epsilon))
        if angle < 0:
            angle += 360
        return angle

    @staticmethod
    def _compute_cop_angle(px: float, py: float) -> float:
        return PZTSensorAngle._compute_angle(-px, -py)
