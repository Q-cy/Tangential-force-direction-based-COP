from typing import Any, TextIO
import numpy as np
from ..config import PressureConfig, ProcessingConfig

def compute_vector_angle(x: float, y: float) -> float: ...
def angle_difference(a: float, b: float) -> float: ...

class TangentialSample:
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
    request_seq: int
    tx_t: float
    rx_t: float
    latency_s: float
    rel_ms: int

class TangentialFrameProcessor:
    def __init__(self, cop_sensor: Any = ..., calibration: Any = ...,
                 cal_dim: str | None = ..., region_mode: str | None = ...,
                 median_window: int | None = ...,
                 processing_config: ProcessingConfig | None = ...) -> None: ...
    def process(self, raw: Any, frame: Any = ...) -> TangentialSample: ...

class TangentialSensorAPI:
    def __init__(self, sensor: Any = ..., processor: Any = ...,
                 sensor_factory: Any = ..., model_path: Any = ...,
                 pressure_port: str | None = ...,
                 config: PressureConfig | None = ...,
                 processing_config: ProcessingConfig | None = ...) -> None: ...
    def read(self, timeout_s: float = ...) -> TangentialSample | None: ...
    def close(self) -> None: ...
    def __enter__(self) -> TangentialSensorAPI: ...
    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...

def format_terminal_sample(sample: TangentialSample) -> str: ...

class FixedTerminalRenderer:
    def __init__(self, stream: TextIO | None = ...) -> None: ...
    def render(self, sample: TangentialSample) -> str: ...
