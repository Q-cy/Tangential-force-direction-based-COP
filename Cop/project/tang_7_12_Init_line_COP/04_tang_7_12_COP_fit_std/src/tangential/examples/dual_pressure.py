"""同时采集两个互不共享状态的压力传感器示例。

每个 ``TangentialSensorAPI`` 都拥有独立串口、采集进程、IPC队列、CoP
状态机和标定处理器。示例使用两个读取线程并行消费两路队列，避免一个设备
的读取超时阻塞另一个设备。两个配置不得指向同一个物理串口。
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TextIO

from ..config import PressureConfig
from ..runtime.sensor import TangentialSensorAPI


def _canonical_port(port: str) -> str:
    """解析串口真实路径，用于识别指向同一设备的不同符号链接。

    Args:
        port: 用户传入的串口路径，例如 ``/dev/serial/by-id/...``。

    Returns:
        str: 规范化后的绝对真实路径；路径尚不存在时也返回规范化结果。
    """
    return os.path.realpath(os.path.abspath(port))


def _format_summary(label: str, port: str, sample: Any) -> str:
    """把一路压力样本格式化为单行摘要。

    Args:
        label: 传感器显示名称。
        port: 该传感器配置的串口路径。
        sample: ``TangentialSample`` 或具有相同摘要字段的对象；``None``
            表示本轮读取超时。

    Returns:
        str: 包含端口、序号、ADC总和、CoP和角度的单行文本。
    """
    if sample is None:
        return f"{label}({port}): timeout"
    return (
        f"{label}({port}): seq={sample.request_seq} sum={sample.total:.0f} "
        f"cop=({sample.cop_x:.3f},{sample.cop_y:.3f}) "
        f"angle={sample.angle:.2f}°"
    )


def run(
    sensor_a: PressureConfig,
    sensor_b: PressureConfig,
    *,
    model_path: str | None = None,
    timeout_s: float = 0.1,
    stream: TextIO | None = None,
    sensor_factory=TangentialSensorAPI,
    max_iterations: int | None = None,
) -> int:
    """并行采集两个压力传感器，直到中断或达到测试迭代数。

    Args:
        sensor_a: 第一只压力传感器的完整设备和轮询配置。
        sensor_b: 第二只压力传感器的完整设备和轮询配置。
        model_path: 两路处理器使用的外部模型；``None`` 使用内置模型。
        timeout_s: 每一路单次 ``read`` 的最长等待时间，单位为秒。
        stream: 摘要输出流；``None`` 使用标准输出。
        sensor_factory: 传感器API工厂，生产环境使用默认值，测试可注入。
        max_iterations: 最大读取轮数；``None`` 表示持续运行。

    Returns:
        int: 正常完成时返回 ``0``。

    Raises:
        ValueError: 两个配置指向同一物理串口、超时不为正数，或迭代数非法。
        Exception: 任一传感器连接、采集、解码或处理失败时向调用方传播；
            退出上下文时两路资源都会关闭。

    Side Effects:
        启动两个独立压力采集进程和两个消费线程，并向输出流写摘要。
    """
    sensor_a.validate()
    sensor_b.validate()
    if _canonical_port(sensor_a.port) == _canonical_port(sensor_b.port):
        raise ValueError("两个压力传感器不能使用同一个物理串口")
    if timeout_s <= 0:
        raise ValueError("timeout_s 必须大于 0")
    if max_iterations is not None and max_iterations < 0:
        raise ValueError("max_iterations 不能为负数")

    output = stream or sys.stdout
    completed = 0
    with (
        sensor_factory(config=sensor_a, model_path=model_path) as api_a,
        sensor_factory(config=sensor_b, model_path=model_path) as api_b,
        ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-pressure") as pool,
    ):
        while max_iterations is None or completed < max_iterations:
            future_a = pool.submit(api_a.read, timeout_s=timeout_s)
            future_b = pool.submit(api_b.read, timeout_s=timeout_s)
            sample_a = future_a.result()
            sample_b = future_b.result()
            output.write(
                _format_summary("A", sensor_a.port, sample_a)
                + " | "
                + _format_summary("B", sensor_b.port, sample_b)
                + "\n"
            )
            output.flush()
            completed += 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """创建双压力示例的命令行解析器。"""
    parser = argparse.ArgumentParser(description="同时采集两个独立压力传感器")
    parser.add_argument("--port-a", required=True, help="第一只压力传感器串口")
    parser.add_argument("--port-b", required=True, help="第二只压力传感器串口")
    parser.add_argument("--model", help="可选外部 fit_coefs.bin")
    parser.add_argument("--timeout", type=float, default=0.1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析两个端口并运行示例；Ctrl+C视为正常停止。"""
    args = _build_parser().parse_args(argv)
    try:
        return run(
            PressureConfig(port=args.port_a),
            PressureConfig(port=args.port_b),
            model_path=args.model,
            timeout_s=args.timeout,
        )
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
