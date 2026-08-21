"""完整应用的公开入口。

Qt、PyQtGraph 和完整会话只在调用 ``run_application`` 时惰性加载；普通
``import tangential`` 和最小压力 API 不会加载这些可选依赖。
"""

from __future__ import annotations

from .config import FullApplicationConfig


def run_application(config: FullApplicationConfig | None = None) -> int:
    """启动完整采集与实时 GUI 应用。

    Args:
        config: 完整应用配置；省略时使用环境变量和内置默认值。

    Returns:
        int: Qt 应用正常退出时返回 ``0``。

    Raises:
        Exception: 设备、配置、Qt 或采集会话错误向调用方传播。
    """
    from .runtime.session import FullApplicationRunner, acquisition_loop

    runner = FullApplicationRunner(
        acquisition_loop,
        config=config or FullApplicationConfig(),
    )
    runner.run()
    return 0


def run_dual_application(
    config_a: FullApplicationConfig,
    config_b: FullApplicationConfig,
) -> int:
    """启动两个完整、相互隔离的实时采集 GUI。

    Args:
        config_a: Sensor A 的完整配置，包含独立压力/力端口、输出目录、
            模型和窗口标题。
        config_b: Sensor B 的完整配置，字段语义同 ``config_a``；两个配置
            必须使用不同物理压力串口、不同输出目录，启用力通道时还必须
            使用不同物理力串口。

    Returns:
        int: 两路 Qt 应用正常退出时返回 ``0``。

    Raises:
        ValueError: 两路设备或输出目录冲突。
        Exception: Qt、设备、采集、CSV 或分析图错误向调用方传播。

    Side Effects:
        只创建一个 ``QApplication``，显示两个 ``RealTimePlot`` 窗口，
        启动两个 ``acquisition_loop`` 后台线程；退出时分别关闭会话并
        保存两路结束分析图。
    """
    from .runtime.session import DualApplicationRunner

    runner = DualApplicationRunner(config_a, config_b)
    runner.run()
    return 0


__all__ = ["run_application", "run_dual_application"]
