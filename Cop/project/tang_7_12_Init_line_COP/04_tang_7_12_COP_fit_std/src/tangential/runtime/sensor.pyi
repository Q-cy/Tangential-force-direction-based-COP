from typing import Any, TextIO
import numpy as np
from ..config import PressureConfig, ProcessingConfig
from ..processing.slip import SlipDetector, SlipResult, TangentialMotionState

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
    motion_state: TangentialMotionState
    is_slipping: bool
    slip_motion_distance: float
    slip_confidence: float
    angle_vector_magnitude: float

class TangentialFrameProcessor:
    def __init__(self, cop_sensor: Any = ..., calibration: Any = ...,
                 cal_dim: str | None = ..., region_mode: str | None = ...,
                 median_window: int | None = ...,
                 processing_config: ProcessingConfig | None = ...,
                 slip_detector: SlipDetector | None = ...) -> None: ...
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
