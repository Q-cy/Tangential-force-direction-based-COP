"""双路完整实时采集示例。

该示例与 ``examples/full.py`` 使用同一套完整会话：每一路都显示压力表、
梯度、CoP、角度、标定和实时曲线，保存完整 108 列 CSV，并在退出时生成
结束分析图。两路只共享一个 ``QApplication``，设备、会话、窗口和输出目录
均保持独立。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..application import run_dual_application
from ..config import (
    CalibrationConfig,
    ForceConfig,
    FullApplicationConfig,
    GuiConfig,
    OutputConfig,
    PressureConfig,
    default_save_dir,
)


def build_config(
    *,
    pressure_port: str,
    force_port: str | None,
    save_dir: str | os.PathLike[str],
    model_path: str | None,
    window_title: str,
) -> FullApplicationConfig:
    """创建一路双传感器完整应用配置。

    Args:
        pressure_port: 该路压力阵列串口。
        force_port: 可选六维力串口；为 ``None`` 时显式禁用力通道。
        save_dir: 该路独立 CSV 和结束分析图目录。
        model_path: 可选外部模型路径；为 ``None`` 时使用内置模型。
        window_title: 该路 GUI 窗口标题。

    Returns:
        FullApplicationConfig: 可直接传给 ``run_dual_application`` 的配置。

    Side Effects:
        只创建并校验配置对象，不打开串口、不创建窗口。
    """
    force = ForceConfig(
        enabled=force_port is not None,
        port=force_port or "/dev/ttyUSB1",
    )
    return FullApplicationConfig(
        pressure=PressureConfig(port=pressure_port),
        force=force,
        calibration=CalibrationConfig(model_path=model_path),
        output=OutputConfig(save_dir=str(save_dir)),
        gui=GuiConfig(window_title=window_title),
    )


def run(
    config_a: FullApplicationConfig,
    config_b: FullApplicationConfig,
    *,
    runner=run_dual_application,
) -> int:
    """运行两路完整 GUI 采集，并在异常时联动关闭两路。

    Args:
        config_a: Sensor A 的完整配置。
        config_b: Sensor B 的完整配置。
        runner: 双路应用入口；生产环境使用 ``run_dual_application``，测试
            可注入记录调用的替代函数。

    Returns:
        int: 双路 Qt 应用正常退出时返回 ``0``。

    Raises:
        ValueError: 两路压力串口、启用的力串口或输出目录冲突。
        Exception: 设备、Qt、采集、CSV 或分析图错误向上传播。

    Side Effects:
        创建一个 Qt 应用和两个完整会话；每路会创建独立采集资源、完整
        CSV 以及退出后的分析 PNG。
    """
    return runner(config_a, config_b)


def build_configs_from_args(args: argparse.Namespace) -> tuple[
    FullApplicationConfig, FullApplicationConfig
]:
    """把双路命令行参数转换成两份完整配置。

    Args:
        args: ``_build_parser`` 或统一 CLI 生成的参数对象。

    Returns:
        tuple: ``(config_a, config_b)``，两路输出目录默认分别为
            ``<base>/sensor_a`` 和 ``<base>/sensor_b``。

    Raises:
        ValueError: 两个显式力端口相同，或参数导致配置非法。
    """
    base = Path(args.save_dir or default_save_dir())
    save_a = Path(args.save_dir_a or base / "sensor_a")
    save_b = Path(args.save_dir_b or base / "sensor_b")
    model_a = args.model_a or args.model
    model_b = args.model_b or args.model
    return (
        build_config(
            pressure_port=args.port_a,
            force_port=args.force_port_a,
            save_dir=save_a,
            model_path=model_a,
            window_title="Sensor A",
        ),
        build_config(
            pressure_port=args.port_b,
            force_port=args.force_port_b,
            save_dir=save_b,
            model_path=model_b,
            window_title="Sensor B",
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    """创建双路完整 GUI 示例的命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="同时运行两个独立的完整压力采集 GUI（CSV 为 108 列）"
    )
    parser.add_argument("--port-a", required=True, help="Sensor A 压力串口")
    parser.add_argument("--port-b", required=True, help="Sensor B 压力串口")
    parser.add_argument("--force-port-a", help="可选 Sensor A 六维力串口")
    parser.add_argument("--force-port-b", help="可选 Sensor B 六维力串口")
    parser.add_argument("--save-dir", help="两路输出目录的父目录")
    parser.add_argument("--save-dir-a", help="Sensor A CSV/分析图目录")
    parser.add_argument("--save-dir-b", help="Sensor B CSV/分析图目录")
    parser.add_argument("--model", help="两路共用的外部 fit_coefs.bin")
    parser.add_argument("--model-a", help="Sensor A 外部模型，覆盖 --model")
    parser.add_argument("--model-b", help="Sensor B 外部模型，覆盖 --model")
    return parser


def run_from_namespace(args: argparse.Namespace) -> int:
    """执行统一 CLI 已解析的双路完整 GUI 参数。

    Args:
        args: 包含本模块命令行参数的 ``argparse.Namespace``。

    Returns:
        int: 正常退出码 ``0``。

    Side Effects:
        打开两个完整采集窗口并写入两路独立输出目录。
    """
    config_a, config_b = build_configs_from_args(args)
    return run(config_a, config_b)


def main(argv: list[str] | None = None) -> int:
    """解析双路完整 GUI 参数并运行；Ctrl+C 视为正常停止。

    Args:
        argv: 命令行参数；为 ``None`` 时读取 ``sys.argv``。

    Returns:
        int: 正常退出码 ``0``。
    """
    args = _build_parser().parse_args(argv)
    try:
        return run_from_namespace(args)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
