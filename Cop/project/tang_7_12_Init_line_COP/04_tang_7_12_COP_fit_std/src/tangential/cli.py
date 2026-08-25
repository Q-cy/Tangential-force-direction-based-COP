"""Tangential SDK 的唯一命令行入口。

本模块只负责参数解析、子命令分派和顶层错误码；具体采集、绘图和训练
逻辑由对应的公共模块实现。
"""

from __future__ import annotations

import argparse                 # arg = argument（参数） + parse（解析）
import json
import sys
from pathlib import Path

from . import __version__

VERSION = __version__


def _build_parser() -> argparse.ArgumentParser:
    """创建完整的 ``tangential`` 命令行参数解析器。

    Returns:
        argparse.ArgumentParser（解析器实例对象）: 包含 ``example``、``app``、``dual``、
            ``plot`` 和 ``fit`` 五个必选子命令的解析器。

    Side Effects:
        仅在内存中构造解析器和参数定义，不读取硬件或文件。
    """
    parser = argparse.ArgumentParser(                                       # 实例化一个顶层命令解析器对象
        prog="tangential",                                                  # prog = program 
        description="12×7 PZT 压力阵列与六维力采集、分析和标定工具",
    )
    parser.add_argument("--version", action="version", version=VERSION)     # 顶层解析器实例.add_argument
    commands = parser.add_subparsers(dest="command", required=True)         # 子命令，dest = destination，执行 tangential app → args.command = "app"

    example = commands.add_parser("example", help="运行最小压力采集示例")
    example.add_argument("--pressure-port")                                 # 选项，tangential example --pressure-port /dev/ttyUSB0
    example.add_argument("--model")
    example.add_argument("--timeout", type=float)
    example.set_defaults(handler=_handle_example)                           # 任务处理函数

    app = commands.add_parser("app", help="运行完整采集和实时 GUI")
    app.add_argument("--pressure-port")                                     # argparse 自动规则：命令行选项名字里的横杠 -，自动转换为下划线 _，作为 Namespace(args) 对象的属性名。
    app.add_argument("--force-port")
    app.add_argument("--save-dir")
    app.add_argument("--model")
    app.add_argument("--max-time-diff-ms", type=float)
    app.set_defaults(handler=_handle_app)

    dual = commands.add_parser(
        "dual", help="运行两个独立的完整压力采集 GUI（各自保存 CSV 和分析图）"
    )
    dual.add_argument("--port-a", required=True)
    dual.add_argument("--port-b", required=True)
    dual.add_argument("--force-port-a")
    dual.add_argument("--force-port-b")
    dual.add_argument("--save-dir")
    dual.add_argument("--save-dir-a")
    dual.add_argument("--save-dir-b")
    dual.add_argument("--model")
    dual.add_argument("--model-a")
    dual.add_argument("--model-b")
    dual.set_defaults(handler=_handle_dual)

    plot = commands.add_parser("plot", help="离线绘制 CSV")
    plot.add_argument("--dir")
    plot.add_argument("--files")
    plot.add_argument("--columns")
    plot.add_argument("--rows")
    plot.add_argument("--xcol")
    plot.add_argument("--title")
    plot.add_argument("--save")
    plot.add_argument("--error-ref")
    plot.add_argument("--mode", choices=("plot", "full_analysis"))
    plot.add_argument("--list", action="store_true")
    plot.set_defaults(handler=_handle_plot)

    fit = commands.add_parser("fit", help="训练并保存拟合模型")
    fit.add_argument("--xy-csv", required=True)
    fit.add_argument("--z-csv", required=True)
    fit.add_argument("--output-model")
    fit.add_argument("--output-plot")
    fit.add_argument("--dim", type=int, choices=(1, 2, 3))
    fit.add_argument("--poly-order", type=int, choices=(1, 2, 3))
    fit.add_argument("--fx-type")
    fit.add_argument("--fy-type")
    fit.add_argument("--fz-type")
    valid = fit.add_mutually_exclusive_group()
    valid.add_argument("--valid-only", dest="valid_only", action="store_true")
    valid.add_argument("--no-valid-only", dest="valid_only", action="store_false")
    fit.set_defaults(valid_only=None)
    split = fit.add_mutually_exclusive_group()
    split.add_argument("--split-sign", dest="split_sign", action="store_true")
    split.add_argument("--no-split-sign", dest="split_sign", action="store_false")
    fit.set_defaults(split_sign=None)
    one = fit.add_mutually_exclusive_group()
    one.add_argument("--one-on-one", dest="one_on_one", action="store_true")
    one.add_argument("--no-one-on-one", dest="one_on_one", action="store_false")
    fit.set_defaults(one_on_one=None)
    fit.add_argument("--write-back")
    fit.add_argument("--force", action="store_true")
    fit.set_defaults(handler=_handle_fit)
    return parser


def _handle_example(args: argparse.Namespace) -> int:
    """执行最小压力采集示例并持续刷新终端显示。

    Args:
        args (argparse.Namespace): ``_build_parser`` 生成的参数对象，使用
            ``pressure_port``、``model`` 和 ``timeout`` 字段。

    Returns:
        int: 正常循环不会返回；若循环被外部方式结束，约定成功码为 0。

    Raises:
        Exception: 传感器连接、模型加载、读取或渲染错误向上抛出，由
            ``main`` 转换为错误码 1。

    Side Effects:
        打开压力传感器并持续向标准输出写入 12×7 ADC 和计算结果；离开
        ``with`` 块时关闭传感器。
    """
    from .config import PressureConfig
    from .examples.minimal import run

    config = PressureConfig()
    if args.pressure_port is not None:                           # args.pressure_port 源于 example.add_argument("--pressure-port") 
        config.port = args.pressure_port
    return run(
        config,
        model_path=args.model,
        timeout_s=0.1 if args.timeout is None else args.timeout,
    )


def _handle_app(args: argparse.Namespace) -> int:
    """根据命令行参数启动完整采集和实时 GUI 应用。

    Args:
        args (argparse.Namespace): 包含压力/六维力端口、保存目录、模型路径
            和毫秒同步窗口的解析参数。

    Returns:
        int: 完整应用正常退出时的进程成功码 ``0``。

    Raises:
        Exception: 配置、Qt、设备、采集或文件错误向上抛出，由 ``main``
            转换为错误码 1。

    Side Effects:
        创建 Qt 应用、打开传感器、运行采集会话并写入 CSV/实时图像。
    """
    from .config import (
        CalibrationConfig,
        FullApplicationConfig,
        ForceConfig,
        OutputConfig,
        PressureConfig,
        SyncConfig,
    )
    from .examples.full import main

    pressure = PressureConfig()
    force = ForceConfig()
    output = OutputConfig()
    calibration = CalibrationConfig()
    sync = SyncConfig()
    if args.pressure_port is not None:
        pressure.port = args.pressure_port
    if args.force_port is not None:
        force.port = args.force_port
    if args.save_dir is not None:
        output.save_dir = args.save_dir
    if args.model is not None:
        calibration.model_path = args.model
    if args.max_time_diff_ms is not None:
        sync.max_time_diff_s = args.max_time_diff_ms / 1000.0
    return main(FullApplicationConfig(
        pressure=pressure,
        force=force,
        output=output,
        calibration=calibration,
        sync=sync,
    ))


def _handle_dual(args: argparse.Namespace) -> int:
    """启动两个独立的完整 GUI 会话。

    Args:
        args: ``dual`` 子命令解析结果，端口、模型和输出目录参数由双路
            示例统一转换为两份 ``FullApplicationConfig``。

    Returns:
        int: 双路应用正常退出时返回 ``0``。

    Raises:
        Exception: 参数、设备、Qt 或采集错误向 ``main`` 传播并转换为错误码。
    """
    from .examples.dual_sensor import run_from_namespace

    return run_from_namespace(args)


def _parse_columns(value: str | None):
    """把逗号分隔的列名参数解析为去空白后的列表。

    Args:
        value (str | None): CLI 的 ``--columns`` 原始值；空值或只含空白项
            时视为空配置。

    Returns:
        list[str] | None: 非空列名列表；没有有效列名时返回 ``None``。
    """
    if not value:
        return None
    columns = [item.strip() for item in value.split(",") if item.strip()]
    return columns or None


def _handle_plot(args: argparse.Namespace) -> int:
    """执行文件列表、普通 CSV 绘图或完整分析绘图子命令。

    Args:
        args (argparse.Namespace): 包含目录、文件、列、行范围、输出路径、
            模式和 ``--list`` 等字段的解析参数。

    Returns:
        int: 成功完成列表或绘图后返回 ``0``。

    Raises:
        Exception: CSV 解析、参数校验、绘图依赖或输出文件错误向上抛出，
            由 ``main`` 转换为错误码 1。

    Side Effects:
        ``--list`` 时向标准输出打印 JSON；绘图模式可能写入 PNG 和误差 CSV。
    """
    from .tools import plotting

    if args.list:
        result = [
            {
                "path": str(info.path),
                "modified_at": info.modified_at.isoformat(),
                "size_bytes": info.size_bytes,
                "row_count": info.row_count,
            }
            for info in plotting.list_files(args.dir or plotting.PlotConfig().directory)
        ]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    config = plotting.PlotConfig()
    if args.files is not None:
        config.files = args.files
    if args.dir is not None:
        config.directory = args.dir
    parsed_columns = _parse_columns(args.columns)
    if parsed_columns is not None:
        config.columns = tuple(parsed_columns)
    if args.rows is not None:
        config.rows = args.rows
    if args.xcol is not None:
        config.x_column = args.xcol
    if args.title is not None:
        config.title = args.title
    if args.save is not None:
        config.save_path = args.save
    if args.error_ref is not None:
        config.error_ref = args.error_ref
    if args.mode is not None:
        config.mode = args.mode
    result = (
        plotting.plot_full_analysis(config)
        if config.mode == "full_analysis"
        else plotting.plot_csv(config)
    )
    print(f"已保存图像: {result.save_path}")
    if result.error_path is not None:
        print(f"已保存误差: {result.error_path}")
    return 0


def _handle_fit(args: argparse.Namespace) -> int:
    """执行离线拟合并输出模型、评估图和可选 CSV 写回结果。

    Args:
        args (argparse.Namespace): 包含 XY/Z 输入、模型输出、拟合维度、拟合
            类型、筛选和写回保护选项的解析参数。

    Returns:
        int: 训练及输出成功完成时返回 ``0``。

    Raises:
        Exception: 输入数据、拟合、模型写出或受保护写回失败时向上抛出，
            由 ``main`` 转换为错误码 1。

    Side Effects:
        读取训练 CSV，写入模型和评估图；只有指定 ``--write-back`` 且满足
        ``--force`` 保护条件时才修改 CSV。
    """
    from .config import TrainingConfig
    from .tools.training import train_model

    config = TrainingConfig(xy_csv=args.xy_csv, z_csv=args.z_csv)
    overrides = {
        "output_model": args.output_model,
        "output_plot": args.output_plot,
        "dim": args.dim,
        "poly_order": args.poly_order,
        "fx": args.fx_type,
        "fy": args.fy_type,
        "fz": args.fz_type,
        "valid_only": args.valid_only,
        "split_sign": args.split_sign,
        "one_on_one": args.one_on_one,
        "write_back": args.write_back,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    config.force = args.force
    result = train_model(config)
    print(f"模型已保存: {result.model_path}")
    if result.plot_path is not None:
        print(f"报告已保存: {result.plot_path}")
    if result.written_path is not None:
        print(f"CSV已写回: {result.written_path}")
    return 0


def main(argv=None) -> int:
    """解析命令行参数、执行子命令并统一转换进程退出码。

    Args:
        argv (Sequence[str] | None): 待解析的参数序列；``None`` 时由
            ``argparse`` 从 ``sys.argv`` 读取。

    Returns:
        int: 成功返回 ``0``；未捕获的运行时异常打印到标准错误并返回 ``1``；
            argparse 的参数错误仍按其约定终止并使用 ``2``。

    Side Effects:
        可能打开硬件、创建 Qt、读取/写入 CSV、模型或图片，具体由子命令决定。
        ``KeyboardInterrupt`` 被视为正常停止并返回 ``0``。
    """
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
        return int(args.handler(args))
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
