"""Tangential Sensor SDK 的稳定公共 Python API 入口。

普通用户应优先从 ``tangential`` 导入本模块显式导出的传感器、单帧处理、
标定、训练和离线绘图符号；``__all__`` 是稳定导出边界。内部的
``sensors``、``acquisition``、``processing``、``storage`` 和 ``gui`` 子包
分别负责协议采集、缓存同步、算法、CSV 持久化和可选实时显示，主要供
SDK 内部组合使用。导入本入口不应启动采集、创建 Qt 窗口或加载
PyQtGraph/Matplotlib；完整 GUI 应由 ``tangential app`` 按需启动。
"""

from .api import (
    FixedTerminalRenderer,
    TangentialFrameProcessor,
    TangentialSample,
    TangentialSensorAPI,
    angle_difference,
    compute_vector_angle,
    format_terminal_sample,
)
from .application import run_application, run_dual_application
from .config import (
    CalibrationConfig,
    CopConfig,
    ForceConfig,
    FullApplicationConfig,
    GuiConfig,
    OutputConfig,
    PlotConfig,
    PressureConfig,
    ProcessingConfig,
    SyncConfig,
    TrainingConfig,
)
from .processing.calibration import FitCalibrationModel
from .processing.cop import PRSensorAngle
from .sensors.pressure import PressureSensor
from .tools.training import TrainingResult, train_model
from .tools.plotting import PlotResult, plot_csv, plot_full_analysis

# 高级采集别名：推荐用户使用，负责压力传感器生命周期和逐帧结果读取。
TangentialSensor = TangentialSensorAPI

__all__ = [
    "TangentialSensor",          # 高级采集 API 别名：读取完整 TangentialSample。
    "TangentialSensorAPI",       # 底层压力采集 API：管理传感器并产生逐帧样本。
    "TangentialSample",           # 单帧结果：保存 ADC、统计值、CoP、角度、梯度和标定值。
    "TangentialFrameProcessor",   # 单帧处理器：调用既有 CoP、梯度和标定算法。
    "FixedTerminalRenderer",      # 终端渲染器：按固定布局显示一帧 12×7 数据和指标。
    "FitCalibrationModel",        # 标定模型 API：加载内置/外部模型并预测 Fx/Fy/Fz。
    "FullApplicationConfig",      # 完整应用配置：端口、保存目录、模型和采集参数。
    "PressureConfig",              # 压力设备和轮询配置。
    "ForceConfig",                 # 六维力设备和校零配置。
    "CopConfig",                   # CoP、阈值、区域和精修配置。
    "ProcessingConfig",           # 单帧处理和标定维度配置。
    "CalibrationConfig",          # 外部/内置模型路径配置。
    "SyncConfig",                  # 匹配窗口、主循环和缓存配置。
    "OutputConfig",                # CSV 输出目录配置。
    "GuiConfig",                   # GUI 显示配置。
    "PRSensorAngle",              # CoP 处理器：阈值、接触状态、区域、梯度和角度计算。
    "PressureSensor",              # 底层压力驱动：负责 PZT 串口协议、帧解析和时序统计。
    "compute_vector_angle",        # 角度工具：计算二维向量方向角（单位为度）。
    "angle_difference",            # 角度工具：计算两个方向角的最小环绕差值。
    "format_terminal_sample",      # 终端工具：把样本格式化为固定布局文本。
    "TrainingConfig",              # 训练配置：定义数据筛选、拟合类型和输出选项。
    "TrainingResult",              # 训练结果：返回模型路径、评估指标和写回信息。
    "train_model",                 # 训练入口：读取 CSV、拟合模型并生成 fit_coefs.bin。
    "PlotConfig",                  # 绘图配置：定义 CSV、列、行范围和输出参数。
    "PlotResult",                  # 绘图结果：返回生成的图像、分析和误差文件信息。
    "plot_csv",                    # 绘图入口：按真实 CSV 表头绘制指定列和区间。
    "plot_full_analysis",          # 绘图入口：生成完整 108 列采集数据分析图。
    "run_application",             # 完整应用入口：按配置启动采集和 GUI。
    "run_dual_application",        # 双完整应用入口：共用一个 Qt 应用、独立两路会话。
]

# SDK 发行版本号，供运行时检查兼容性和 CLI --version 使用。
__version__ = "0.3.0"
