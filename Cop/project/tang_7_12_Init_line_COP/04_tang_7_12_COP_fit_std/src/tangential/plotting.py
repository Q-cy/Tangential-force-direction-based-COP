"""离线 CSV 绘图 API。

本模块只负责读取、分析和绘图，不负责命令行参数解析。Matplotlib 在真正
调用绘图函数时才加载，因此 ``import tangential`` 不会引入绘图库。
"""

from __future__ import annotations

import csv
import datetime as _datetime
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, TypeAlias

import numpy as np

from .api import angle_difference

_ANGLE_COLUMNS = {"ADC_angle", "Force_angle", "Force_cal_angle"}
RowsSpec: TypeAlias = tuple[int | None, int | None] | slice | str | None


@dataclass
class PlotConfig:
    """离线绘图配置。

    ``files`` 可以是路径、路径列表、逗号分隔的路径，或 ``all``、
    ``latest:N``、文件索引/索引范围。``rows`` 可以是 ``(start, stop)``、
    ``slice`` 或 ``"start:stop"``。
    """

    files: str | Path | Sequence[str | Path] | None = None
    directory: str | Path = field(default_factory=lambda: Path.cwd() / "data")
    columns: Sequence[str | int] = ("Fy_cal", "delta_Force_Y")
    rows: RowsSpec = None
    x_column: str | int | None = "rel_ms"
    title: str | None = None
    save_path: str | Path | None = None
    error_ref: str | int | None = None
    mode: str = "plot"
    highlight_valid: bool = True
    show_annotations: bool = True
    force_min: float = 0.2


@dataclass(frozen=True)
class CSVFileInfo:
    """目录扫描结果，不包含任何命令行输出行为。"""

    path: Path
    modified_at: _datetime.datetime
    size_bytes: int
    row_count: int


@dataclass(frozen=True)
class PlotResult:
    """绘图结果及可选误差报告文件。"""

    save_path: Path
    files: tuple[Path, ...]
    error_path: Path | None = None
    errors: tuple[dict[str, Any], ...] = ()


def scan_csv(directory: str | Path | None = None) -> list[Path]:
    """扫描目录中的 CSV，按修改时间从新到旧返回路径。"""

    root = Path(directory) if directory is not None else Path.cwd() / "data"
    if not root.exists():
        raise FileNotFoundError(f"CSV 目录不存在: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"CSV 路径不是目录: {root}")
    paths = [
        path for path in root.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".csv"
        and not path.name.startswith("_")
    ]
    return sorted(paths, key=lambda path: (-path.stat().st_mtime, str(path)))


def list_files(directory: str | Path | None = None) -> list[CSVFileInfo]:
    """返回目录 CSV 的结构化信息，不打印、不退出。"""

    result = []
    for path in scan_csv(directory):
        try:
            with path.open("r", encoding="utf-8", newline="") as stream:
                row_count = max(sum(1 for _ in stream) - 1, 0)
        except OSError as exc:
            raise OSError(f"读取 CSV 文件失败: {path}: {exc}") from exc
        stat = path.stat()
        result.append(
            CSVFileInfo(
                path=path,
                modified_at=_datetime.datetime.fromtimestamp(stat.st_mtime),
                size_bytes=stat.st_size,
                row_count=row_count,
            )
        )
    return result


def _resolve_explicit_path(value: str | Path, directory: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = directory / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {value}")
    if not path.is_file():
        raise IsADirectoryError(f"CSV 路径不是文件: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"不是 CSV 文件: {path}")
    return path


def resolve_csvs(
    files: str | Path | Sequence[str | Path] | None = None,
    directory: str | Path | None = None,
) -> list[Path]:
    """解析文件路径或目录选择表达式，返回有序 CSV 路径列表。"""

    root = Path(directory) if directory is not None else Path.cwd() / "data"
    if files is None:
        return scan_csv(root)

    if isinstance(files, Path):
        return [_resolve_explicit_path(files, root)]
    if isinstance(files, str):
        parts = [part.strip() for part in files.split(",") if part.strip()]
        if not parts:
            raise ValueError("files 不能为空")
    else:
        parts = list(files)
        if not parts:
            raise ValueError("files 不能为空")

    if len(parts) == 1 and isinstance(parts[0], str):
        selector = parts[0]
        if selector == "all":
            return scan_csv(root)
        if selector.startswith("latest:"):
            try:
                count = int(selector.split(":", 1)[1])
            except ValueError as exc:
                raise ValueError(f"latest:N 选择无效: {selector}") from exc
            if count < 0:
                raise ValueError(f"latest:N 的 N 不能为负数: {selector}")
            return scan_csv(root)[:count]

    if all(isinstance(part, (str, Path)) for part in parts):
        path_like = any(
            isinstance(part, Path)
            or ".csv" in str(part).lower()
            or Path(str(part)).exists()
            for part in parts
        )
        if path_like:
            return [_resolve_explicit_path(part, root) for part in parts]

    available = scan_csv(root)
    indices: list[int] = []
    for part in parts:
        text = str(part).strip()
        if "-" in text:
            bounds = text.split("-", 1)
            if len(bounds) != 2:
                raise ValueError(f"文件索引范围无效: {text}")
            try:
                start, stop = (int(value) for value in bounds)
            except ValueError as exc:
                raise ValueError(f"文件索引范围无效: {text}") from exc
            if start < 0 or stop < start:
                raise ValueError(f"文件索引范围无效: {text}")
            indices.extend(range(start, stop + 1))
        else:
            try:
                index = int(text)
            except ValueError as exc:
                raise ValueError(f"文件选择无效: {text}") from exc
            if index < 0:
                raise ValueError(f"文件索引不能为负数: {text}")
            indices.append(index)
    invalid = [index for index in indices if index >= len(available)]
    if invalid:
        raise IndexError(
            f"文件索引越界: {invalid}; 当前目录只有 {len(available)} 个 CSV"
        )
    return [available[index] for index in indices]


# 便于调用方使用更短的正式名称；两者都不包含 CLI 副作用。
scan = scan_csv
resolve = resolve_csvs


def load_csv(path: str | Path) -> tuple[list[str], np.ndarray]:
    """按文件实际表头读取 CSV，返回 ``(header, float64_data)``。"""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    if not csv_path.is_file():
        raise IsADirectoryError(f"CSV 路径不是文件: {csv_path}")
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream)
            try:
                header = [name.strip() for name in next(reader)]
            except StopIteration as exc:
                raise ValueError(f"CSV 文件为空且没有表头: {csv_path}") from exc
            if not header or any(not name for name in header):
                raise ValueError(f"CSV 表头为空或包含空列名: {csv_path}")
            rows = [(line_number, row) for line_number, row in enumerate(reader, 2) if row]
    except UnicodeDecodeError as exc:
        raise ValueError(f"CSV 不是 UTF-8 文本: {csv_path}") from exc
    if not rows:
        return header, np.empty((0, len(header)), dtype=np.float64)
    for line_number, row in rows:
        if len(row) != len(header):
            raise ValueError(
                f"CSV 第 {line_number} 行列数与表头不一致: "
                f"期望 {len(header)}，实际 {len(row)}: {csv_path}"
            )
    try:
        data = np.asarray([[float(value.strip()) for value in row] for _, row in rows])
    except ValueError as exc:
        raise ValueError(f"CSV 含非数值数据: {csv_path}: {exc}") from exc
    return header, data


def resolve_column(column: str | int, header: Sequence[str]) -> int:
    """按实际表头解析列名或从 0 开始的列号。"""

    clean_header = [str(name).strip() for name in header]
    if isinstance(column, (int, np.integer)):
        index = int(column)
    else:
        text = str(column).strip()
        if text.isdigit():
            index = int(text)
        elif text in clean_header:
            return clean_header.index(text)
        else:
            raise ValueError(f"未知列名: {text}，可用列名: {', '.join(clean_header)}")
    if index < 0 or index >= len(clean_header):
        raise ValueError(f"列号越界: {index}，当前 CSV 共 {len(clean_header)} 列")
    return index


def _resolve_columns(columns: Iterable[str | int], header: Sequence[str]) -> list[int]:
    columns = list(columns)
    if not columns:
        raise ValueError("plot 模式至少需要一列 columns")
    return [resolve_column(column, header) for column in columns]


def _row_slice(rows: RowsSpec, size: int) -> tuple[int, int]:
    if rows is None:
        start, stop = 0, size
    elif isinstance(rows, slice):
        start, stop, step = rows.indices(size)
        if step != 1:
            raise ValueError("rows 不支持步长，只支持 start:stop")
    elif isinstance(rows, str):
        parts = rows.split(":")
        if len(parts) != 2:
            raise ValueError(f"行范围无效: {rows}，应为 start:stop")
        try:
            start = int(parts[0]) if parts[0].strip() else 0
            stop = int(parts[1]) if parts[1].strip() else size
        except ValueError as exc:
            raise ValueError(f"行范围无效: {rows}") from exc
    else:
        if len(rows) != 2:
            raise ValueError("rows 应为 (start, stop)")
        start = 0 if rows[0] is None else int(rows[0])
        stop = size if rows[1] is None else int(rows[1])
    if start < 0 or stop < 0 or start > stop:
        raise ValueError(f"行范围无效: [{start}:{stop}]")
    if start >= size or start == stop:
        raise ValueError(f"所选行范围没有数据: [{start}:{stop}]")
    return start, min(stop, size)


def compute_errors(
    reference: Sequence[float], prediction: Sequence[float], *, is_angle: bool = False
) -> dict[str, Any]:
    """计算 MAE、RMSE、MAPE、R² 等误差指标。"""

    ref_values = np.asarray(reference, dtype=np.float64)
    pred_values = np.asarray(prediction, dtype=np.float64)
    if ref_values.shape != pred_values.shape:
        raise ValueError("误差计算的参考值和预测值长度不一致")
    mask = np.isfinite(ref_values) & np.isfinite(pred_values)
    ref = ref_values[mask]
    pred = pred_values[mask]
    if len(ref) < 2:
        return {"count": int(len(ref)), "error": "数据点不足"}
    if is_angle:
        absolute = np.asarray(
            [angle_difference(r, p) for r, p in zip(ref, pred)], dtype=np.float64
        )
        signed = np.sign(pred - ref) * absolute
    else:
        signed = pred - ref
        absolute = np.abs(signed)
    ss_res = float(np.sum(signed ** 2))
    ss_tot = float(np.sum((ref - np.mean(ref)) ** 2))
    nonzero = np.abs(ref) > 1e-6
    mape = (
        float(np.mean(absolute[nonzero] / np.abs(ref[nonzero])) * 100)
        if np.any(nonzero)
        else float("nan")
    )
    return {
        "count": int(len(ref)),
        "MAE": float(np.mean(absolute)),
        "MSE": float(np.mean(signed ** 2)),
        "RMSE": float(np.sqrt(np.mean(signed ** 2))),
        "Max_Error": float(np.max(absolute)),
        "Min_Error": float(np.min(absolute)),
        "MAPE_%": mape,
        "R2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "Error_Std": float(np.std(signed)),
        "Median_Error": float(np.median(absolute)),
    }


def _load_matplotlib():
    """惰性加载 Matplotlib 并强制离线 Agg 后端。"""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _valid_column(header: Sequence[str], data: np.ndarray) -> np.ndarray | None:
    clean = [name.strip() for name in header]
    for name in ("valid", "CoP_state"):
        if name in clean:
            return data[:, clean.index(name)]
    return None


def _split_segments(mask: np.ndarray) -> list[np.ndarray]:
    changes = np.where(np.diff(mask.astype(np.int8)) != 0)[0] + 1
    return [part for part in np.split(np.arange(mask.size), changes) if part.size]


def _plot_segmented(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    color: Any,
    *,
    active: np.ndarray | None,
    show_annotations: bool,
):
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    if active is None:
        ax.plot(x[finite], y[finite], color=color, linewidth=0.8, marker=".", markersize=2, label=label)
        return
    segments = _split_segments(active)
    labeled = {True: False, False: False}
    x_range = float(np.ptp(x[finite])) if np.any(finite) else 1.0
    x_range = x_range or 1.0
    last_annot_x = -math.inf
    for segment in segments:
        seg_active = bool(active[segment[0]])
        seg_finite = finite[segment]
        if not np.any(seg_finite):
            continue
        indices = segment[seg_finite]
        label_used = None
        if not labeled[seg_active]:
            label_used = label if seg_active else f"{label} (inactive)"
            labeled[seg_active] = True
        ax.plot(
            x[indices], y[indices], color=color,
            linewidth=2.0 if seg_active else 0.8,
            alpha=1.0 if seg_active else 0.3,
            marker=".", markersize=2, label=label_used,
        )
        if show_annotations and seg_active:
            first = int(indices[0])
            if abs(x[first] - last_annot_x) > x_range * 0.05:
                ax.annotate(
                    f"{y[first]:.2f}", (x[first], y[first]),
                    xytext=(10, 10), textcoords="offset points", fontsize=5,
                    color=color,
                    arrowprops={"arrowstyle": "-", "color": color, "lw": 0.5},
                    bbox={"boxstyle": "round,pad=0.15", "facecolor": "white",
                          "alpha": 0.7, "edgecolor": "none"},
                )
                last_annot_x = x[first]


def _write_error_csv(path: Path, error_results: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "pred_column", "ref_column", "count", "MAE", "MSE", "RMSE",
            "Max_Error", "Min_Error", "Median_Error", "MAPE_%", "R2", "Error_Std",
        ])
        for item in error_results:
            result = item["results"]
            if "error" in result:
                continue
            writer.writerow([
                item["pred"], item["ref"], result["count"],
                f"{result['MAE']:.6f}", f"{result['MSE']:.6f}",
                f"{result['RMSE']:.6f}", f"{result['Max_Error']:.6f}",
                f"{result['Min_Error']:.6f}", f"{result['Median_Error']:.6f}",
                f"{result['MAPE_%']:.2f}", f"{result['R2']:.6f}",
                f"{result['Error_Std']:.6f}",
            ])


def _output_path(config: PlotConfig, first_path: Path, *, full: bool) -> Path:
    if config.save_path is not None:
        path = Path(config.save_path)
    else:
        name = f"full_analysis_{first_path.stem}.png" if full else f"{first_path.stem}_plot.png"
        path = first_path.parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_csv(config: PlotConfig) -> PlotResult:
    """按 ``plot`` 配置绘制自定义列；``mode=full_analysis``转到全分析图。"""

    if not isinstance(config, PlotConfig):
        raise TypeError("plot_csv 需要 PlotConfig")
    if config.mode == "full_analysis":
        return plot_full_analysis(config)
    if config.mode != "plot":
        raise ValueError(f"未知绘图模式: {config.mode}")
    paths = resolve_csvs(config.files, config.directory)
    if not paths:
        raise FileNotFoundError("没有找到可绘制的 CSV 文件")
    plt = _load_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 6))
    errors: list[dict[str, Any]] = []
    file_colors = plt.cm.tab10(np.linspace(0, 1, max(len(paths), 1)))
    first_path = paths[0]
    for file_index, path in enumerate(paths):
        header, full_data = load_csv(path)
        start, stop = _row_slice(config.rows, full_data.shape[0])
        data = full_data[start:stop]
        columns = _resolve_columns(config.columns, header)
        x_index = resolve_column(config.x_column, header) if config.x_column is not None else 0
        x = data[:, x_index]
        valid = _valid_column(header, data) if config.highlight_valid else None
        ref_index = resolve_column(config.error_ref, header) if config.error_ref is not None else None
        force_ref = data[:, ref_index] if ref_index is not None and config.force_min > 0 else None
        column_colors = plt.cm.tab10(np.linspace(0, 1, max(len(columns), 1)))
        for column_index, data_index in enumerate(columns):
            y = data[:, data_index]
            color = column_colors[column_index] if len(columns) > 1 else file_colors[file_index]
            label = f"{path.name}:{header[data_index]}" if len(paths) > 1 else header[data_index]
            if len(columns) == 1 and len(paths) > 1:
                label = path.name
            active = None
            if valid is not None:
                active = valid != 0
                if force_ref is not None:
                    active = active & (np.abs(force_ref) >= config.force_min)
            _plot_segmented(ax, x, y, label, color, active=active, show_annotations=config.show_annotations)
            if ref_index is None or data_index == ref_index:
                continue
            pair = np.isfinite(data[:, ref_index]) & np.isfinite(y)
            if valid is not None:
                pair &= valid != 0
            if force_ref is not None:
                pair &= np.abs(force_ref) >= config.force_min
            ref_values, pred_values, x_values = data[pair, ref_index], y[pair], x[pair]
            if len(ref_values) < 2:
                errors.append({"pred": label, "ref": header[ref_index], "results": {"count": len(ref_values), "error": "数据点不足"}})
                continue
            is_angle = header[ref_index] in _ANGLE_COLUMNS or header[data_index] in _ANGLE_COLUMNS
            result = compute_errors(ref_values, pred_values, is_angle=is_angle)
            absolute = np.asarray([angle_difference(a, b) for a, b in zip(ref_values, pred_values)]) if is_angle else np.abs(pred_values - ref_values)
            minimum, maximum = int(np.argmin(absolute)), int(np.argmax(absolute))
            ax.scatter([x_values[minimum]], [pred_values[minimum]], color=color, s=50, marker="v", zorder=6)
            ax.scatter([x_values[maximum]], [pred_values[maximum]], color=color, s=50, marker="^", zorder=6)
            ax.annotate(f"Min={absolute[minimum]:.3f}", (x_values[minimum], pred_values[minimum]), fontsize=6, color=color)
            ax.annotate(f"Max={absolute[maximum]:.3f}", (x_values[maximum], pred_values[maximum]), fontsize=6, color=color)
            errors.append({"pred": label, "ref": header[ref_index], "results": result})
    ax.minorticks_on()
    ax.grid(True, which="major", alpha=0.4, linewidth=0.6)
    ax.grid(True, which="minor", alpha=0.15, linewidth=0.3)
    ax.set_xlabel(str(config.x_column if config.x_column is not None else header[x_index]))
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(config.title or first_path.stem, fontsize=12)
    fig.tight_layout()
    output = _output_path(config, first_path, full=False)
    fig.savefig(output, dpi=200)
    plt.close(fig)
    error_path = None
    if errors:
        error_path = output.with_name(f"{output.stem}_error.csv")
        _write_error_csv(error_path, errors)
    return PlotResult(output, tuple(paths), error_path, tuple(errors))


def plot_full_analysis(config: PlotConfig | str | Path) -> PlotResult:
    """绘制 4×2 的 PZT/六维力全分析图，列均按实际表头读取。"""

    if isinstance(config, (str, Path)):
        config = PlotConfig(files=config, mode="full_analysis")
    if not isinstance(config, PlotConfig):
        raise TypeError("plot_full_analysis 需要 PlotConfig 或 CSV 路径")
    paths = resolve_csvs(config.files, config.directory)
    if not paths:
        raise FileNotFoundError("没有找到可绘制的 CSV 文件")
    path = paths[0]
    header, full_data = load_csv(path)
    start, stop = _row_slice(config.rows, full_data.shape[0])
    data = full_data[start:stop]
    clean_header = [name.strip() for name in header]
    indices = {name: index for index, name in enumerate(clean_header)}

    def column(name: str) -> np.ndarray | None:
        index = indices.get(name)
        return None if index is None else data[:, index].astype(np.float64)

    time = column("rel_ms")
    if time is None:
        time = np.arange(len(data), dtype=np.float64)
    series = {
        "adc_angle": column("ADC_angle"), "adc_sum": column("adc_sum"),
        "cop_dx": column("delta_CoP_X"), "cop_dy": column("delta_CoP_Y"),
        "force_angle": column("Force_angle"), "force_cal_angle": column("Force_cal_angle"),
        "force_fz": column("delta_Force_Z"), "force_fx": column("delta_Force_X"),
        "force_fy": column("delta_Force_Y"), "fx_cal": column("Fx_cal"),
        "fy_cal": column("Fy_cal"),
    }
    if not any(value is not None for value in series.values()):
        raise ValueError(f"CSV 缺少 full_analysis 所需的已知列: {path}")
    plt = _load_matplotlib()
    valid = column("valid") if config.highlight_valid else None
    if valid is None and config.highlight_valid:
        valid = column("CoP_state")
    active = (valid != 0) if valid is not None else None
    force_filters = {
        "force_angle": series["force_angle"],
        "force_fx": series["force_fx"],
        "force_fy": series["force_fy"],
    }
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    errors: list[dict[str, Any]] = []

    def draw(axis, key: str, color: str, label: str, *, active_mask=active):
        values = series[key]
        if values is not None:
            mask = active_mask
            if mask is not None and config.force_min > 0 and key in force_filters and force_filters[key] is not None:
                mask = mask & (np.abs(force_filters[key]) >= config.force_min)
            _plot_segmented(axis, time, values, label, color, active=mask, show_annotations=config.show_annotations)

    left1, right1 = axes[0]
    left2, right2 = axes[1]
    left3, right3 = axes[2]
    left4, right4 = axes[3]
    draw(left1, "adc_angle", "b", "PZT Angle")
    draw(left2, "adc_sum", "b", "PZT Fz")
    draw(left3, "cop_dx", "b", "PZT Fx")
    draw(left4, "cop_dy", "c", "PZT Fy")
    left1.set_title("PZT Angle")
    left2.set_title("PZT Fz")
    left3.set_title("PZT Fx")
    left4.set_title("PZT Fy")

    draw(right1, "force_angle", "r", "Measured")
    draw(right1, "force_cal_angle", "g", "Calibrated")
    right1.set_title("Angle: Meas vs Cal")
    draw(right2, "force_fz", "r", "Fz")
    right2.set_title("Fz: Measured")
    draw(right3, "force_fx", "r", "Measured")
    draw(right3, "fx_cal", "g", "Calibrated")
    right3.set_title("Fx: Meas vs Cal")
    draw(right4, "force_fy", "r", "Measured")
    draw(right4, "fy_cal", "c", "Calibrated")
    right4.set_title("Fy: Meas vs Cal")

    pairs = [
        (right1, "force_angle", "force_cal_angle", "Force_cal_angle", True),
        (right3, "force_fx", "fx_cal", "Fx_cal", False),
        (right4, "force_fy", "fy_cal", "Fy_cal", False),
    ]
    for axis, reference_key, prediction_key, prediction_name, is_angle in pairs:
        reference = series[reference_key]
        prediction = series[prediction_key]
        if reference is None or prediction is None:
            continue
        pair = np.isfinite(reference) & np.isfinite(prediction)
        if active is not None:
            pair &= active
        force_value = force_filters[reference_key]
        if config.force_min > 0 and force_value is not None:
            pair &= np.abs(force_value) >= config.force_min
        result = compute_errors(reference[pair], prediction[pair], is_angle=is_angle)
        errors.append({"pred": prediction_name, "ref": reference_key, "results": result})
        if "error" not in result:
            axis.annotate(
                f"MAE={result['MAE']:.4f} MAPE={result['MAPE_%']:.1f}% R²={result['R2']:.3f}",
                xy=(0.02, 0.95), xycoords="axes fraction", fontsize=7,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "wheat", "alpha": 0.7},
            )
    for row in axes:
        for axis in row:
            axis.set_xlabel("Time (ms)", fontsize=9)
            axis.grid(True, alpha=0.3)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(fontsize=8)
    fig.tight_layout()
    output = _output_path(config, path, full=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)
    error_path = None
    if errors:
        error_path = output.with_name(f"{output.stem}_error.csv")
        _write_error_csv(error_path, errors)
    return PlotResult(output, (path,), error_path, tuple(errors))


__all__ = [
    "CSVFileInfo", "PlotConfig", "PlotResult",
    "compute_errors", "list_files", "load_csv", "plot_csv",
    "plot_full_analysis", "resolve", "resolve_column", "resolve_csvs",
    "scan", "scan_csv",
]
