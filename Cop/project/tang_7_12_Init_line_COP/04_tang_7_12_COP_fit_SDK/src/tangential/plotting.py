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

    Attributes:
        files: CSV path selector; ``None`` scans the default directory.
        directory: Base directory for relative paths and selectors.
        columns: Header names or zero-based indices to draw in ``plot`` mode.
        rows: Half-open data-row range, or ``None`` for all rows.
        x_column: Header name/index for the x-axis; ``None`` uses column 0.
        title: Optional figure title.
        save_path: Optional output PNG path; otherwise derived from the first
            input CSV.
        error_ref: Optional reference header/index used for error metrics.
        mode: ``"plot"`` for custom plots or ``"full_analysis"`` for 4×2.
        highlight_valid: Segment by ``valid`` or fallback ``CoP_state``.
        show_annotations: Annotate active-segment starts and error extrema.
        force_min: Minimum absolute force for force-filtered comparisons.

    Side Effects:
        Construction performs no file I/O and imports no plotting backend.
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
    """Metadata returned for one scanned CSV file.

    Attributes:
        path: Resolved CSV path.
        modified_at: Local datetime from filesystem modification time.
        size_bytes: File size in bytes.
        row_count: Number of non-header lines counted in the file.
    """

    path: Path
    modified_at: _datetime.datetime
    size_bytes: int
    row_count: int


@dataclass(frozen=True)
class PlotResult:
    """Paths and diagnostics produced by a plotting operation.

    Attributes:
        save_path: PNG path written by the plot operation.
        files: Ordered input CSV paths used.
        error_path: Optional CSV containing scalar error metrics.
        errors: Dictionaries with ``pred``, ``ref`` and ``results`` keys.
    """

    save_path: Path
    files: tuple[Path, ...]
    error_path: Path | None = None
    errors: tuple[dict[str, Any], ...] = ()


def scan_csv(directory: str | Path | None = None) -> list[Path]:
    """Scan a directory for ordinary CSVs in newest-first order.

    Args:
        directory: Directory to scan; ``None`` means ``Path.cwd()/data``.
    Returns:
        Paths for regular, non-underscore-prefixed ``.csv`` files, sorted by
        descending modification time and then path.
    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """

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
    """Return structured metadata for CSVs without CLI output or exit calls.

    Args:
        directory: Directory passed to :func:`scan_csv`, or ``None`` for the
            default data directory.
    Returns:
        Newest-first :class:`CSVFileInfo` list.
    Raises:
        FileNotFoundError/NotADirectoryError: If the directory is invalid.
        OSError: If a CSV cannot be opened or inspected.
    """

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
    """Resolve and validate one explicit CSV path.

    Args:
        value: Absolute or relative path supplied by the caller.
        directory: Base directory for relative values.
    Returns:
        Absolute resolved regular path with a case-insensitive ``.csv`` suffix.
    Raises:
        FileNotFoundError: If the path does not exist.
        IsADirectoryError: If the path is a directory.
        ValueError: If the suffix is not ``.csv``.
    """
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
    """Resolve CSV paths, selectors and inclusive index ranges.

    Args:
        files: Path, path sequence, comma-separated paths, ``all``,
            ``latest:N``, zero-based index, or inclusive index range. ``None``
            selects all CSVs in ``directory``.
        directory: Base directory for relative paths and selectors; defaults to
            ``Path.cwd()/data``.
    Returns:
        Ordered list of resolved CSV paths. No files are modified.
    Raises:
        FileNotFoundError/IsADirectoryError/ValueError: For invalid paths,
            selectors or directory contents.
        IndexError: If a numeric selector is outside the available list.
    """

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
    """Load a numeric CSV using its actual header and row width.

    Args:
        path: UTF-8 CSV path; the first row is the header.
    Returns:
        ``(header, data)`` where ``header`` is a stripped string list and
        ``data`` is a ``float64`` array of shape ``(n_rows, len(header))``.
        An empty data section has shape ``(0, len(header))``.
    Raises:
        FileNotFoundError: If the file is absent.
        IsADirectoryError: If ``path`` is a directory.
        ValueError: For empty/invalid headers, non-UTF-8 text, inconsistent
            row widths or non-numeric cells.
    """

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
    """Resolve a header name or zero-based numeric column selector.

    Args:
        column: Header name, integer index, or numeric string index.
        header: Actual CSV header sequence.
    Returns:
        Zero-based integer column index.
    Raises:
        ValueError: If the selector is unknown or outside header bounds.
    """

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
    """Resolve all plotted selectors against one actual CSV header.

    Args:
        columns: Iterable of names or zero-based indices.
        header: Actual CSV header sequence.
    Returns:
        Resolved integer indices in the input order.
    Raises:
        ValueError: If no columns are provided or any selector is invalid.
    """
    columns = list(columns)
    if not columns:
        raise ValueError("plot 模式至少需要一列 columns")
    return [resolve_column(column, header) for column in columns]


def _row_slice(rows: RowsSpec, size: int) -> tuple[int, int]:
    """Normalize a half-open plotting row range and reject empty selections.

    Args:
        rows: ``None``, a step-1 ``slice``, ``"start:stop"`` or a
            ``(start, stop)`` pair; bounds are zero-based data-row indices.
        size: Number of available data rows.
    Returns:
        Clamped ``(start, stop)`` suitable for ``data[start:stop]``.
    Raises:
        ValueError: For malformed, negative, stepped, reversed or empty ranges.
    """
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
    """Compute finite-sample regression or wrapped-angle error metrics.

    Args:
        reference: Reference sequence, converted to a ``float64`` array.
        prediction: Predicted sequence with exactly the same shape.
        is_angle: If true, use :func:`angle_difference` for absolute angular
            error; otherwise use ordinary subtraction.
    Returns:
        Dictionary containing ``count`` and, with at least two finite pairs,
        ``MAE``, ``MSE``, ``RMSE``, ``Max_Error``, ``Min_Error``, ``MAPE_%``,
        ``R2``, ``Error_Std`` and ``Median_Error``. With fewer than two pairs,
        returns ``{"count": n, "error": "数据点不足"}``.
    Raises:
        ValueError: If reference and prediction shapes differ.
    """

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
    """Lazily load Matplotlib and force the non-interactive Agg backend.

    Returns:
        Imported ``matplotlib.pyplot`` module.
    Raises:
        ImportError: If the optional plotting dependency is unavailable.
    Side Effects:
        Imports Matplotlib and changes its backend to ``Agg`` for this process.
    """

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _valid_column(header: Sequence[str], data: np.ndarray) -> np.ndarray | None:
    """Find the validity mask column using the current CSV header.

    Args:
        header: Actual CSV header sequence.
        data: Numeric data matrix of shape ``(n_rows, len(header))``.
    Returns:
        One-dimensional ``valid`` column, or the ``CoP_state`` fallback, or
        ``None`` when neither column exists. No file I/O occurs.
    """
    clean = [name.strip() for name in header]
    for name in ("valid", "CoP_state"):
        if name in clean:
            return data[:, clean.index(name)]
    return None


def _split_segments(mask: np.ndarray) -> list[np.ndarray]:
    """Split a boolean activity mask into contiguous index segments.

    Args:
        mask: One-dimensional boolean-like array of length ``n_rows``.
    Returns:
        Nonempty integer index arrays covering the original positions in order.
    """
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
    """Draw finite samples with optional active/inactive segment styling.

    Args:
        ax: Matplotlib axes object receiving the line and annotations.
        x: One-dimensional x values.
        y: One-dimensional y values with the same length as ``x``.
        label: Legend label for the active series.
        color: Matplotlib-compatible line color.
        active: Optional boolean mask; inactive segments are faded.
        show_annotations: Annotate selected active-segment starts when true.
    Returns:
        ``None``. The axes object is mutated as a plotting side effect; rows
        with non-finite x/y values are omitted.
    Raises:
        ValueError: Possible from Matplotlib when supplied arrays or style
            values are incompatible.
    """
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
    """Write scalar error dictionaries to a diagnostic CSV.

    Args:
        path: Destination CSV path; its parent must already be creatable.
        error_results: Dictionaries containing ``pred``, ``ref`` and a
            metric ``results`` mapping. Entries with ``results['error']`` are
            skipped from the output rows.
    Returns:
        ``None``.
    Side Effects:
        Creates or overwrites ``path`` with a fixed metric header and numeric
        values formatted for human-readable reports.
    Raises:
        OSError: If the file cannot be opened or written.
        KeyError: If a successful result lacks a required metric key.
    """
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
    """Resolve and prepare the PNG output path for a plot.

    Args:
        config: Plot configuration; ``save_path`` overrides the derived name.
        first_path: First input CSV, used to derive the default filename.
        full: Select ``full_analysis_<stem>.png`` when true, otherwise
            ``<stem>_plot.png``.
    Returns:
        Destination :class:`Path` whose parent directory exists.
    Raises:
        OSError: If the parent directory cannot be created.
    """
    if config.save_path is not None:
        path = Path(config.save_path)
    else:
        name = f"full_analysis_{first_path.stem}.png" if full else f"{first_path.stem}_plot.png"
        path = first_path.parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_csv(config: PlotConfig) -> PlotResult:
    """Plot selected columns from one or more CSVs and optionally compare errors.

    Args:
        config: :class:`PlotConfig` containing file selectors, actual header
            names/indices, half-open row range, x-axis and output options.
            ``mode='full_analysis'`` delegates to :func:`plot_full_analysis`.
    Returns:
        :class:`PlotResult` with the PNG path, input paths, optional error CSV,
        and per-prediction error dictionaries.
    Side Effects:
        Lazily imports Matplotlib, writes a PNG, and writes a sibling
        ``*_error.csv`` when at least one comparison is requested.
    Raises:
        TypeError: If ``config`` is not :class:`PlotConfig`.
        ValueError: For invalid mode, columns or row range.
        FileNotFoundError/IsADirectoryError: For missing or invalid CSV paths.
        ImportError: If Matplotlib is unavailable.
    """

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
    """Draw the 4×2 PZT/force analysis figure using actual CSV headers.

    Args:
        config: :class:`PlotConfig` or one CSV path. A path creates default
            ``full_analysis`` configuration; ``files`` uses only its first
            resolved CSV. ``rows`` selects a half-open data-row range.
    Returns:
        :class:`PlotResult` containing the 4×2 PNG, the source path, and any
        angle/Fx/Fy comparison metrics plus optional error CSV path.
    Side Effects:
        Lazily imports Matplotlib and writes the PNG and, when comparisons have
        finite pairs, a sibling ``*_error.csv``. Missing optional series are
        skipped, but at least one known series is required.
    Raises:
        TypeError: If ``config`` is neither a path nor :class:`PlotConfig`.
        ValueError: For invalid rows, empty data, missing known series or bad
            CSV content.
        FileNotFoundError/IsADirectoryError/ImportError: For missing input or
            plotting dependency.
    """

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
        """Return one named series from the selected data rows.

        Args:
            name: Exact stripped header name.
        Returns:
            ``float64`` vector of length ``len(data)`` or ``None`` when absent.
            This closure performs no file I/O.
        """
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
        """Draw one optional analysis series on a Matplotlib axis.

        Args:
            axis: Matplotlib axes receiving the line.
            key: Key in the local ``series`` mapping.
            color: Matplotlib-compatible color.
            label: Legend label.
            active_mask: Optional boolean activity mask, defaulting to the
                selected ``valid``/``CoP_state`` mask.
        Returns:
            ``None``; the axis is mutated when the series exists.
        """
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
