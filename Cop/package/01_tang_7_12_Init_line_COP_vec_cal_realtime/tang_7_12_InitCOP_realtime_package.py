import numpy as np
import threading
from collections import deque


class PZTSensorAngle:
    def __init__(self, rows: int = 12, cols: int = 7,
                 threshold_factor: float = 5, collect_frames: int = 20,
                 stability_frames: int = 5,
                 reset_at_frame: int = 0,
                 refine_cnt: int = 10,
                 refine_distance: float = 0.1):
        self.rows = rows
        self.cols = cols
        self.threshold_factor = threshold_factor
        self.collect_frames = collect_frames
        self.stability_frames = stability_frames
        self._reset_at_frame = reset_at_frame
        self._frame_count = 0

        self._refine_cnt = refine_cnt
        self._refine_distance = refine_distance
        self._refine_enabled = (refine_cnt > 0) and (refine_distance > 0)
        self._refine_cand_x = None
        self._refine_cand_y = None
        self._refine_curr = 0
        self._refined = False

        self._pressure_history = deque(maxlen=collect_frames)
        self._thresh = None
        self._lock = threading.Lock()

        self._origin_x = None
        self._origin_y = None
        self._contact_init = False
        self._low_counter = 0

    # ---------- 公共 API ----------

    def get_all(self, adc_data) -> tuple[float, float, float]:
        expected = self.rows * self.cols
        if len(adc_data) != expected:
            raise ValueError(f"ADC数据长度必须为{expected}")

        dx, dy = self._compute_delta_cop(adc_data)
        angle = self._compute_cop_angle(dx, dy)
        return angle, dx, dy

    def get_angle(self, adc_data) -> float:
        angle, _, _ = self.get_all(adc_data)
        return angle

    def reset_origin(self) -> None:
        self._origin_x = None
        self._origin_y = None
        self._contact_init = False
        self._low_counter = 0
        self._refine_cand_x = None
        self._refine_cand_y = None
        self._refine_curr = 0
        self._refined = False

    # ---------- 内部算法 ----------

    def _compute_cop(self, frame2d: np.ndarray, total_pressure: float) -> tuple[float, float]:
        x_grid = np.tile(np.arange(self.cols), (self.rows, 1))
        y_grid = np.repeat(np.arange(self.rows), self.cols).reshape(self.rows, self.cols)
        cop_x = float(np.sum(frame2d * x_grid) / total_pressure)
        cop_y = float(np.sum(frame2d * y_grid) / total_pressure)
        return cop_x, cop_y

    def _update_dynamic_threshold(self, total_pressure: float) -> None:
        with self._lock:
            if self.collect_frames <= 0:
                if self._thresh is None:
                    self._thresh = 0
                return
            if self._thresh is None:
                self._pressure_history.append(total_pressure)
                if len(self._pressure_history) >= self.collect_frames:
                    self._thresh = self.threshold_factor * float(np.mean(self._pressure_history))

    def _compute_delta_cop(self, raw_frame) -> tuple[float, float]:
        self._frame_count += 1
        if self._reset_at_frame > 0 and self._frame_count == self._reset_at_frame:
            self.reset_origin()

        rows, cols = self.rows, self.cols
        frame_flat = np.asarray(raw_frame, dtype=np.float32).flatten()
        frame2d = frame_flat.reshape(rows, cols)

        total_pressure = float(np.sum(frame2d))

        self._update_dynamic_threshold(total_pressure)

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

        cop_x, cop_y = self._compute_cop(frame2d, total_pressure)

        delta_x = 0.0
        delta_y = 0.0
        if not self._contact_init:
            if self._thresh is not None and total_pressure > self._thresh:
                self._origin_x = cop_x
                self._origin_y = cop_y
                self._contact_init = True
                if self._refine_enabled:
                    self._refine_cand_x = cop_x
                    self._refine_cand_y = cop_y
                    self._refine_curr = 1
            return 0.0, 0.0

        delta_x = cop_x - self._origin_x
        delta_y = cop_y - self._origin_y

        if self._refine_enabled and not self._refined:
            if self._refine_cand_x is None:
                self._refine_cand_x = cop_x
                self._refine_cand_y = cop_y
                self._refine_curr = 1
            else:
                dist = float(np.hypot(cop_x - self._refine_cand_x,
                                       cop_y - self._refine_cand_y))
                if dist <= self._refine_distance:
                    self._refine_curr += 1
                else:
                    self._refine_cand_x = cop_x
                    self._refine_cand_y = cop_y
                    self._refine_curr = 1

            if self._refine_curr >= self._refine_cnt:
                self._origin_x = self._refine_cand_x
                self._origin_y = self._refine_cand_y
                self._refined = True

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
        return PZTSensorAngle._compute_angle(px, -py)
