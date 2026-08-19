"""面向用户的最小压力传感器 API。"""

import sys
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .processing.calibration import FitCalibrationModel
from .processing.cop import PRSensorAngle
from .sensors.pressure import PRESSURE_SENSOR_PORT, PressureSensor


def compute_vector_angle(x: float, y: float) -> float:
    """返回二维向量的 0~360° 方向角。"""
    angle = float(np.degrees(np.arctan2(y, x + 1e-8)))
    return angle + 360.0 if angle < 0 else angle


def angle_difference(a: float, b: float) -> float:
    """返回两个 0~360 度方向角之间的最小绝对环绕误差。"""
    difference = abs(float(a) - float(b)) % 360.0
    return min(difference, 360.0 - difference)


@dataclass
class TangentialSample:
    """一个合法压力帧及其全部最小 API 计算结果。"""

    raw: np.ndarray
    matrix: np.ndarray
    gradient: np.ndarray
    minimum: float
    maximum: float
    total: float
    mean: float
    cop_x: float
    cop_y: float
    angle: float
    dx: float
    dy: float
    state: int
    calibrated_fx: float
    calibrated_fy: float
    calibrated_fz: float
    calibrated_angle: float
    request_seq: int = -1
    tx_t: float = float("nan")
    rx_t: float = float("nan")
    latency_s: float = float("nan")
    origin_x: float | None = None
    origin_y: float | None = None
    contact: bool = False
    display_contact: bool = False
    refined: bool = False
    region_mask: np.ndarray | None = None
    regions: list[dict] = field(default_factory=list)
    centroid: tuple[float, float] | None = None
    rel_ms: int = 0

    @property
    def raw_2d(self) -> np.ndarray:
        """兼容直观命名：12×7 ADC 矩阵。"""
        return self.matrix

    @property
    def adc_sum(self) -> float:
        return self.total

    @property
    def min(self) -> float:
        return self.minimum

    @property
    def max(self) -> float:
        return self.maximum

    @property
    def sum(self) -> float:
        return self.total

    @property
    def copX(self) -> float:
        return self.cop_x

    @property
    def copY(self) -> float:
        return self.cop_y


class TangentialFrameProcessor:
    """复用 PRSensorAngle 和运行时标定模块处理一个84通道压力帧。"""

    def __init__(self, cop_sensor=None, calibration=None, cal_dim="3D",
                 region_mode="full", median_window=5):
        if region_mode not in ("full", "region", "both"):
            raise ValueError("region_mode 必须是 full、region 或 both")
        if median_window <= 0:
            raise ValueError("median_window 必须大于0")
        self.cop_sensor = cop_sensor or PRSensorAngle()
        self.calibration = calibration
        self.cal_dim = cal_dim
        self.region_mode = region_mode
        self._dx_values = deque(maxlen=median_window)
        self._dy_values = deque(maxlen=median_window)

    def _predict(self, dx, dy, total):
        if self.calibration is None:
            return (float("nan"),) * 3
        if isinstance(self.calibration, FitCalibrationModel):
            return self.calibration.predict(dx, dy, total, self.cal_dim)
        values = list(self.calibration.predict([dx, dy, total]))
        values.extend([float("nan")] * (3 - len(values)))
        return tuple(float(value) for value in values[:3])

    def process(self, raw, frame=None) -> TangentialSample:
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        expected = self.cop_sensor.rows * self.cop_sensor.cols
        if values.size != expected:
            raise ValueError(f"压力帧通道数必须为{expected}，实际为{values.size}")
        matrix = values.reshape(self.cop_sensor.rows, self.cop_sensor.cols)
        self.cop_sensor.dynamic_threshold(matrix)

        use_full = self.region_mode in ("full", "both")
        use_region = self.region_mode in ("region", "both")
        if use_full:
            angle, dx, dy, cop_x, cop_y = self.cop_sensor.get_all(values)
            origin_x, origin_y = self.cop_sensor.get_origin()
            state = self.cop_sensor.get_state()
            gradient = np.asarray(
                self.cop_sensor.get_gradient(values), dtype=np.float64
            )
            centroid = self.cop_sensor._compute_centroid(matrix)
        else:
            angle = dx = dy = 0.0
            cop_x = cop_y = float("nan")
            origin_x = origin_y = None
            state = 0
            gradient = np.zeros(
                (self.cop_sensor.rows, self.cop_sensor.cols, 2),
                dtype=np.float64,
            )
            centroid = None

        if use_region:
            regions = self.cop_sensor._compute_region_delta_cop(matrix)
            region_mask = np.zeros(matrix.shape, dtype=np.int32)
            for region in regions:
                for row, col in region["coords"]:
                    region_mask[row, col] = region["id"]
        else:
            regions = []
            region_mask = np.zeros(matrix.shape, dtype=np.int32)

        self._dx_values.append(dx)
        self._dy_values.append(dy)
        filtered_dx = float(np.median(self._dx_values))
        filtered_dy = float(np.median(self._dy_values))
        total = float(np.sum(values))
        cal_fx, cal_fy, cal_fz = self._predict(filtered_dx, filtered_dy, total)
        cal_angle = (
            compute_vector_angle(cal_fx, cal_fy)
            if np.isfinite(cal_fx) and np.isfinite(cal_fy)
            else float("nan")
        )
        contact = state > 0
        display_contact = contact
        if use_region and not use_full:
            display_contact = any(
                region.get("contact_init", False) for region in regions
            )

        metadata = frame or {}
        sample = TangentialSample(
            raw=values.copy(),
            matrix=matrix.copy(),
            gradient=gradient,
            minimum=float(np.min(values)),
            maximum=float(np.max(values)),
            total=total,
            mean=float(np.mean(values)),
            cop_x=float(cop_x),
            cop_y=float(cop_y),
            angle=float(angle),
            dx=filtered_dx,
            dy=filtered_dy,
            state=int(state),
            calibrated_fx=cal_fx,
            calibrated_fy=cal_fy,
            calibrated_fz=cal_fz,
            calibrated_angle=cal_angle,
            request_seq=int(metadata.get("request_seq", -1)),
            tx_t=float(metadata.get("tx_t", float("nan"))),
            rx_t=float(metadata.get("rx_t", float("nan"))),
            latency_s=float(metadata.get("latency_s", float("nan"))),
            origin_x=origin_x,
            origin_y=origin_y,
            contact=bool(contact),
            display_contact=bool(display_contact),
            refined=state == 2,
            region_mask=region_mask,
            regions=regions,
            centroid=centroid,
        )
        return sample


class TangentialSensorAPI:
    """最小压力采集 API；负责设备生命周期并返回 ``TangentialSample``。"""

    def __init__(self, sensor=None, processor=None, sensor_factory=None,
                 model_path=None, pressure_port=PRESSURE_SENSOR_PORT):
        if sensor is None:
            if sensor_factory is None:
                sensor_factory = PressureSensor
            sensor = sensor_factory(port=pressure_port)
        if processor is None:
            calibration = (
                FitCalibrationModel.from_default()
                if model_path is None
                else FitCalibrationModel.from_path(model_path)
            )
            processor = TangentialFrameProcessor(
                calibration=calibration
            )
        self.sensor = sensor
        self.processor = processor
        self._closed = False

    def read(self, timeout_s=0.1) -> TangentialSample | None:
        if self._closed:
            raise RuntimeError("TangentialSensorAPI 已关闭")
        frame = self.sensor.read_frame(timeout_s=timeout_s)
        if frame is None:
            return None
        raw = self.sensor.decode(frame["raw"])
        return self.processor.process(raw, frame)

    def close(self):
        if self._closed:
            return
        self._closed = True
        close = getattr(self.sensor, "close", None)
        if close is not None:
            close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def format_terminal_sample(sample: TangentialSample) -> str:
    """生成行列数量和字段宽度固定的终端文本。"""
    rows = [" ".join(f"{value:7.0f}" for value in row) for row in sample.matrix]
    rows.extend([
        f"min={sample.minimum:12.3f} max={sample.maximum:12.3f} "
        f"sum={sample.total:14.3f} mean={sample.mean:12.3f}",
        f"copX={sample.cop_x:11.4f} copY={sample.cop_y:11.4f} "
        f"angle={sample.angle:10.3f}",
        f"Fx_cal={sample.calibrated_fx:10.4f} "
        f"Fy_cal={sample.calibrated_fy:10.4f} "
        f"Fz_cal={sample.calibrated_fz:10.4f}",
    ])
    return "\n".join(rows)


class FixedTerminalRenderer:
    """每帧只执行一次 write/flush 的固定布局终端渲染器。"""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self._first_frame = True

    def render(self, sample: TangentialSample) -> str:
        text = format_terminal_sample(sample)
        prefix = "\x1b[2J\x1b[H" if self._first_frame else "\x1b[H"
        self._first_frame = False
        self.stream.write(prefix + text + "\n")
        self.stream.flush()
        return text
