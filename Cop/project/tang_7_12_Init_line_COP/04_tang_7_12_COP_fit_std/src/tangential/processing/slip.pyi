from enum import IntEnum
from typing import Any
import numpy as np
from ..config import SlipConfig

class TangentialMotionState(IntEnum):
    NO_CONTACT: int
    STICK: int
    SLIP: int

class SlipResult:
    motion_state: TangentialMotionState
    is_slipping: bool
    motion_distance: float
    confidence: float
    direction_x: float
    direction_y: float
    angle_vector_magnitude: float
    reanchored: bool
    patch_row_shift: int
    patch_col_shift: int
    patch_correlation: float
    patch_improvement: float
    @property
    def slip_motion_distance(self) -> float: ...
    @property
    def slip_confidence(self) -> float: ...

class SlipDetector:
    def __init__(self, config: SlipConfig | None = ..., rows: int = ..., cols: int = ...) -> None: ...
    @property
    def motion_state(self) -> TangentialMotionState: ...
    @property
    def is_slipping(self) -> bool: ...
    @property
    def anchor(self) -> tuple[float | None, float | None]: ...
    def reset(self) -> SlipResult: ...
    def update(self, pressure: Any, cop_x: float, cop_y: float,
               contact: bool, ready: bool = ...) -> SlipResult: ...
    def process(self, pressure: Any, cop_x: float, cop_y: float,
                contact: bool, ready: bool = ...) -> SlipResult: ...
