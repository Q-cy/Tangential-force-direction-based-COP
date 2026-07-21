"""压阻传感器 CoP 角度估计（PZTSensorAngle 类，支持多 sensor 实例）"""

import numpy as np
import threading
from collections import deque


class PZTSensorAngle:
    """压阻传感器：12×7（默认）PZT 阵列的 CoP 角度估计"""

    def __init__(self, rows: int = 12, cols: int = 7,
                 threshold_factor: float = 5, collect_frames: int = 20,
                 stability_frames: int = 5,
                 reset_at_frame: int = 0,
                 refine_cnt: int = 10,
                 refine_distance: float = 0.1):
        """
        构造一个 PZTSensorAngle 角度估计实例。

        :param rows: 传感器阵列行数（默认 12）。
        :param cols: 传感器阵列列数（默认 7），输入 ADC 序列长度 = rows * cols。
        :param threshold_factor: 动态低压阈值倍数：thresh = threshold_factor × mean(前 collect_frames 帧总压力)。
                                值越大，越难被判定为"低压"。
        :param collect_frames: 学习动态阈值用的样本窗口大小（默认 20 帧）。
                              0 = 不学阈值, _thresh 直接 = 0（首帧非零压力锁 origin）。
        :param stability_frames: 阈值确定后，连续多少帧低压自动 reset_origin（默认 5）。
        :param reset_at_frame: 在第 N 帧自动调一次 reset_origin() 后重新锁新 origin；
                                0 = 禁用自动 reset（默认）。
        :param refine_cnt: origin 锁定后，连续多少帧 CoP 稳定触发"二次精修"
                                        （调用 reset_origin() 让下帧自然重新锁）。默认 10。
                                        0 = 禁用精修。
        :param refine_distance: 判定"稳定"的 CoP 与候选点欧氏距离阈值（cells，默认 0.1）。
                                      0 = 禁用精修。
        """
        self.rows = rows
        self.cols = cols
        self.threshold_factor = threshold_factor
        self.collect_frames = collect_frames
        self.stability_frames = stability_frames
        self._reset_at_frame = reset_at_frame
        self._frame_count = 0

        # 二次精修（post-refine）参数与状态
        # _refine_enabled 派生标志在构造时算一次, 运行时不再改
        self._refine_cnt = refine_cnt
        self._refine_distance = refine_distance
        self._refine_enabled = (refine_cnt > 0) and (refine_distance > 0)
        # 候选点/稳定计数/已精修标志 —— 跟随 origin 生命周期
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
        """
        输入 rows*cols 个 ADC 值，一次性输出 (angle, dx, dy)。

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

        dx, dy = self._compute_delta_cop(adc_data)
        angle = self._compute_cop_angle_ccw90(dx, dy)
        return angle, dx, dy

    def get_angle(self, adc_data) -> float:
        """便捷接口：只输出 PZT 角度（0~360°）。等价于 get_all(...)[0]。"""
        angle, _, _ = self.get_all(adc_data)
        return angle

    def reset_origin(self) -> None:
        """清掉首次接触 origin 与低压计数，同时清掉二次精修状态（候选点、稳定计数、已精修标志）；
        阈值（若已确定）保留。
        """
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
        """计算 2D 帧的 CoP (X, Y) — 压力加权中心, 几何意义.

        输入:  (rows × cols) 帧 + 总压力 total_pressure
        输出:  (cop_x, cop_y) 浮动位置, 已是 cell 单位
        """
        x_grid = np.tile(np.arange(self.cols), (self.rows, 1))
        y_grid = np.repeat(np.arange(self.rows), self.cols).reshape(self.rows, self.cols)
        cop_x = float(np.sum(frame2d * x_grid) / total_pressure)
        cop_y = float(np.sum(frame2d * y_grid) / total_pressure)
        return cop_x, cop_y

    def _update_dynamic_threshold(self, total_pressure: float) -> None:
        with self._lock:
            # collect_frames=0: 跳过历史累积, _thresh 直接 = 0
            # (低压判定 `total_pressure < 0` 永不触发, 几乎所有帧都视为"非低")
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

        # CoP 加权均值
        cop_x, cop_y = self._compute_cop(frame2d, total_pressure)

        delta_x = 0.0
        delta_y = 0.0
        if not self._contact_init:
            # 等待阈值已定 + 当前压力 > 阈值 才锁 origin(防闲置帧"先入为主"污染基准)
            if self._thresh is not None and total_pressure > self._thresh:
                # 真首次接触: 同时初始化 origin 与二次精修候选
                self._origin_x = cop_x
                self._origin_y = cop_y
                self._contact_init = True
                if self._refine_enabled:
                    # origin 锁定时同步把候选点设到这一帧的 CoP, 稳定计数从 1 开始
                    self._refine_cand_x = cop_x
                    self._refine_cand_y = cop_y
                    self._refine_curr = 1
            return 0.0, 0.0

        delta_x = cop_x - self._origin_x
        delta_y = cop_y - self._origin_y

        # 二次精修: 寻找连续 N 帧 CoP 几乎不动的"稳定点"
        if self._refine_enabled and not self._refined:
            if self._refine_cand_x is None:
                # 防御性: 首帧(origin 刚锁时已 init)理论上不会到这里, 写在以防 manual 重置
                self._refine_cand_x = cop_x
                self._refine_cand_y = cop_y
                self._refine_curr = 1
            else:
                # 欧氏距离
                dist = float(np.hypot(cop_x - self._refine_cand_x,
                                       cop_y - self._refine_cand_y))
                if dist <= self._refine_distance:
                    # 离候选点近 → 累计稳定计数
                    self._refine_curr += 1
                else:
                    # 离候选点远 → 重置计数
                    self._refine_cand_x = cop_x
                    self._refine_cand_y = cop_y
                    self._refine_curr = 1

            if self._refine_curr >= self._refine_cnt:
                # 候选点作为新 origin
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

    @staticmethod
    def _compute_cop_angle_ccw90(px: float, py: float) -> float:
        return PZTSensorAngle._compute_angle(-py, px)
    