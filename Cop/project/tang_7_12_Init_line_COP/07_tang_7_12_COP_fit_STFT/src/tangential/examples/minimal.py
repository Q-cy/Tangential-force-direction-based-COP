"""最小压力采集示例。

该示例是 ``tangential example`` 使用的唯一逐帧循环：不启动 Qt、六维力、
CSV 或完整应用会话，只读取压力帧并打印单帧结果。
"""

from __future__ import annotations

from ..api import FixedTerminalRenderer
from ..config import ArrayConfig, PressureConfig, ProcessingConfig
from ..runtime.sensor import TangentialSensorAPI


def run(
    config: PressureConfig | None = None,
    *,
    model_path: str | None = None,
    processing_config: ProcessingConfig | None = None,
    array_config: ArrayConfig | None = None,
    timeout_s: float = 0.1,
) -> int:
    """启动最小压力采集循环。

    Args:
        config: 压力端口、轮询频率和串口超时配置。
        model_path: 外部模型路径；省略时使用内置模型。
        processing_config: CoP、梯度和标定处理配置。
        array_config: 整个采集、处理和终端显示链共用的阵列布局。
        timeout_s: 单帧读取等待时间，单位为秒。

    Returns:
        int: 循环被调用方正常中断时返回 ``0``。
    """
    active_processing = processing_config or ProcessingConfig()
    active_array = array_config or ArrayConfig()
    with TangentialSensorAPI(
        config=config or PressureConfig(),
        model_path=model_path,
        processing_config=active_processing,
        array_config=active_array,
    ) as sensor:
        renderer = FixedTerminalRenderer(
            array_config=active_array,
        )
        while True:
            frame = sensor.read(timeout_s=timeout_s)
            if frame is not None:
                renderer.render(frame)
    return 0


def main() -> int:
    """命令行运行最小示例，并把 Ctrl+C 视为正常退出。"""
    try:
        return run()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
