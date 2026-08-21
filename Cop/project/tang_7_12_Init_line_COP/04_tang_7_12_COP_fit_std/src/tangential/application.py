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


__all__ = ["run_application"]
