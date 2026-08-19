"""Tangential Sensor SDK 公共入口。

普通用户只需要从本模块导入，不需要了解内部文件结构。
"""

from .api import (
    FixedTerminalRenderer,
    TangentialFrameProcessor,
    TangentialSample,
    TangentialSensorAPI,
    compute_vector_angle,
    format_terminal_sample,
)
from .config import FullApplicationConfig
from .processing.calibration import FitCalibrationModel
from .processing.cop import PRSensorAngle
from .sensors.pressure import PressureSensor

TangentialSensor = TangentialSensorAPI

__all__ = [
    "TangentialSensor",
    "TangentialSensorAPI",
    "TangentialSample",
    "TangentialFrameProcessor",
    "FixedTerminalRenderer",
    "FitCalibrationModel",
    "FullApplicationConfig",
    "PRSensorAngle",
    "PressureSensor",
    "compute_vector_angle",
    "format_terminal_sample",
]

__version__ = "0.1.0"
