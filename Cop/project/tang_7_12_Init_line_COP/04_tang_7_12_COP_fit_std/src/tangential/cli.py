"""Tangential SDK 的唯一命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


VERSION = "0.2.0"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tangential",
        description="12×7 PZT 压力阵列与六维力采集、分析和标定工具",
    )
    parser.add_argument("--version", action="version", version=VERSION)
    commands = parser.add_subparsers(dest="command", required=True)

    example = commands.add_parser("example", help="运行最小压力采集示例")
    example.add_argument("--pressure-port", default="/dev/ttyUSB0")
    example.add_argument("--model")
    example.add_argument("--timeout", type=float, default=0.1)
    example.set_defaults(handler=_handle_example)

    app = commands.add_parser("app", help="运行完整采集和实时 GUI")
    app.add_argument("--pressure-port", default="/dev/ttyUSB0")
    app.add_argument("--force-port", default="/dev/ttyUSB1")
    app.add_argument("--save-dir", default="./data")
    app.add_argument("--model")
    app.add_argument("--max-time-diff-ms", type=float, default=15.0)
    app.set_defaults(handler=_handle_app)

    plot = commands.add_parser("plot", help="离线绘制 CSV")
    plot.add_argument("--dir", default="./data")
    plot.add_argument("--files")
    plot.add_argument("--columns")
    plot.add_argument("--rows")
    plot.add_argument("--xcol", default="rel_ms")
    plot.add_argument("--title")
    plot.add_argument("--save")
    plot.add_argument("--error-ref")
    plot.add_argument("--mode", choices=("plot", "full_analysis"), default="plot")
    plot.add_argument("--list", action="store_true")
    plot.set_defaults(handler=_handle_plot)

    fit = commands.add_parser("fit", help="训练并保存拟合模型")
    fit.add_argument("--xy-csv", required=True)
    fit.add_argument("--z-csv", required=True)
    fit.add_argument("--output-model", default="fit_coefs.bin")
    fit.add_argument("--output-plot", default="fit_report.png")
    fit.add_argument("--dim", type=int, choices=(1, 2, 3), default=1)
    fit.add_argument("--poly-order", type=int, choices=(1, 2, 3), default=3)
    fit.add_argument("--fx-type", default="sym_log")
    fit.add_argument("--fy-type", default="sym_log")
    fit.add_argument("--fz-type", default="exp")
    valid = fit.add_mutually_exclusive_group()
    valid.add_argument("--valid-only", dest="valid_only", action="store_true")
    valid.add_argument("--no-valid-only", dest="valid_only", action="store_false")
    fit.set_defaults(valid_only=True)
    split = fit.add_mutually_exclusive_group()
    split.add_argument("--split-sign", dest="split_sign", action="store_true")
    split.add_argument("--no-split-sign", dest="split_sign", action="store_false")
    fit.set_defaults(split_sign=True)
    one = fit.add_mutually_exclusive_group()
    one.add_argument("--one-on-one", dest="one_on_one", action="store_true")
    one.add_argument("--no-one-on-one", dest="one_on_one", action="store_false")
    fit.set_defaults(one_on_one=True)
    fit.add_argument("--write-back")
    fit.add_argument("--force", action="store_true")
    fit.set_defaults(handler=_handle_fit)
    return parser


def _handle_example(args: argparse.Namespace) -> int:
    from . import FixedTerminalRenderer, TangentialSensor

    with TangentialSensor(
        pressure_port=args.pressure_port,
        model_path=args.model,
    ) as sensor:
        renderer = FixedTerminalRenderer()
        while True:
            sample = sensor.read(timeout_s=args.timeout)
            if sample is not None:
                renderer.render(sample)


def _handle_app(args: argparse.Namespace) -> int:
    from .config import FullApplicationConfig
    from .full import FullApplicationRunner, acquisition_loop

    config = FullApplicationConfig(
        save_dir=args.save_dir,
        model_path=args.model,
        pressure_port=args.pressure_port,
        force_port=args.force_port,
        max_time_diff_s=args.max_time_diff_ms / 1000.0,
    )
    FullApplicationRunner(acquisition_loop, config).run()
    return 0


def _parse_columns(value: str | None):
    if not value:
        return None
    columns = [item.strip() for item in value.split(",") if item.strip()]
    return columns or None


def _handle_plot(args: argparse.Namespace) -> int:
    from . import plotting

    if args.list:
        result = [
            {
                "path": str(info.path),
                "modified_at": info.modified_at.isoformat(),
                "size_bytes": info.size_bytes,
                "row_count": info.row_count,
            }
            for info in plotting.list_files(args.dir)
        ]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    files = args.files or "all"
    config = plotting.PlotConfig(
        files=files,
        directory=args.dir,
        columns=_parse_columns(args.columns) or plotting.PlotConfig().columns,
        rows=args.rows,
        x_column=args.xcol,
        title=args.title,
        save_path=args.save,
        error_ref=args.error_ref,
        mode=args.mode,
    )
    result = (
        plotting.plot_full_analysis(config)
        if args.mode == "full_analysis"
        else plotting.plot_csv(config)
    )
    print(f"已保存图像: {result.save_path}")
    if result.error_path is not None:
        print(f"已保存误差: {result.error_path}")
    return 0


def _handle_fit(args: argparse.Namespace) -> int:
    from .training import TrainingConfig, train_model

    result = train_model(TrainingConfig(
        xy_csv=args.xy_csv,
        z_csv=args.z_csv,
        output_model=args.output_model,
        output_plot=args.output_plot,
        dim=args.dim,
        poly_order=args.poly_order,
        fx=args.fx_type,
        fy=args.fy_type,
        fz=args.fz_type,
        valid_only=args.valid_only,
        split_sign=args.split_sign,
        one_on_one=args.one_on_one,
        write_back=args.write_back,
        force=args.force,
    ))
    print(f"模型已保存: {result.model_path}")
    if result.plot_path is not None:
        print(f"报告已保存: {result.plot_path}")
    if result.written_path is not None:
        print(f"CSV已写回: {result.written_path}")
    return 0


def main(argv=None) -> int:
    """解析参数并执行子命令；参数错误由 argparse 返回2。"""
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
