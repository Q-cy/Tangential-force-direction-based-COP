"""Tangential SDK 的可读公开 API 门面。

具体采集和单帧处理实现位于 ``tangential.runtime.sensor``。本模块只重新
导出稳定公共符号，方便阅读源码、生成文档和后续替换运行时实现。
"""

from .runtime.sensor import (
    FixedTerminalRenderer,
    TangentialFrameProcessor,
    TangentialFrame,
    TangentialSensorAPI,
    angle_difference,
    compute_vector_angle,
    format_terminal_sample,
)
from .processing.slip import SlipDetector, SlipResult, TangentialMotionState

# 控制 from tangential import * 会导入哪些
__all__ = [
    "FixedTerminalRenderer",
    "TangentialFrame",
    "TangentialFrameProcessor",
    "TangentialSensorAPI",
    "angle_difference",
    "compute_vector_angle",
    "format_terminal_sample",
    "TangentialMotionState",
    "SlipResult",
    "SlipDetector",
]
