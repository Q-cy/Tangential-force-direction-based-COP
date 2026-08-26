from typing import Any, TextIO
import numpy as np
from ..config import PressureConfig, ProcessingConfig
from ..processing.slip import SlipDetector, TangentialMotionState

def compute_vector_angle(x: float, y: float) -> float: ...
def angle_difference(a: float, b: float) -> float: ...

class TangentialFrame:
    base_data: np.ndarray
    adc_sum: float
    cop_x: float
    cop_y: float
    angle: float
    dx: float
    dy: float
    motion_state: TangentialMotionState

class TangentialFrameProcessor:
    def __init__(self, cop_sensor: Any = ..., calibration: Any = ...,
                 cal_dim: str | None = ..., region_mode: str | None = ...,
                 median_window: int | None = ...,
                 processing_config: ProcessingConfig | None = ...,
                 slip_detector: SlipDetector | None = ...) -> None: ...
    def process_frame(self, raw_data: Any, frame: Any = ...) -> TangentialFrame: ...

class TangentialSensorAPI:
    def __init__(self, sensor: Any = ..., processor: Any = ...,
                 sensor_factory: Any = ..., model_path: Any = ...,
                 pressure_port: str | None = ...,
                 config: PressureConfig | None = ...,
                 processing_config: ProcessingConfig | None = ...) -> None: ...
    def read(self, timeout_s: float = ...) -> TangentialFrame | None: ...
    def close(self) -> None: ...
    def __enter__(self) -> TangentialSensorAPI: ...
    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None: ...

def format_terminal_sample(sample: TangentialFrame) -> str: ...

class FixedTerminalRenderer:
    def __init__(self, stream: TextIO | None = ...) -> None: ...
    def render(self, sample: TangentialFrame) -> str: ...
