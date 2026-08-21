"""压力阵列几何处理与六维力标定模型。

本子包公开 ``PRSensorAngle`` 和 ``FitCalibrationModel``：前者实现阈值、
CoP、接触状态、梯度和区域处理，后者读取内置或外部模型并执行标定预测。
它们只处理已解码的数据，不直接访问串口、创建 GUI 或构造 108 列 CSV。
"""

from .calibration import FitCalibrationModel
from .cop import PRSensorAngle
from .slip import SlipDetector, SlipResult, TangentialMotionState

__all__ = [
    "FitCalibrationModel", "PRSensorAngle", "TangentialMotionState",
    "SlipResult", "SlipDetector",
]
