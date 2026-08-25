"""Tangential 运行时实现。

这里承载压力采集 API、完整会话和同步适配器；用户代码应优先从
``tangential`` 顶层导入公开对象。
"""

from .sensor import (
    FixedTerminalRenderer,
    TangentialFrameProcessor,
    TangentialSample,
    TangentialSensorAPI,
    angle_difference,
    compute_vector_angle,
    format_terminal_sample,
)
__all__ = [
    "FixedTerminalRenderer",
    "TangentialFrameProcessor",
    "TangentialSample",
    "TangentialSensorAPI",
    "angle_difference",
    "compute_vector_angle",
    "format_terminal_sample",
]
