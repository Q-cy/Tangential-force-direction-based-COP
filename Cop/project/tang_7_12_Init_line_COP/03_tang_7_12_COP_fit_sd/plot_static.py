"""
静态图绘制工具 —— 从 CSV 中选列、选行，画 matplotlib 图。

两种用法：
  1. 直接改下面「配置区」的变量，然后运行：python plot_static.py
  2. CLI 传参覆盖配置：python plot_static.py -f data_2.csv -c rel_ms,ADC_angle -r 100:500

CLI 参数：
  -f CSV 文件路径
  -c 要画的列（逗号分隔，支持列名或列号），如 "rel_ms,ADC_angle" 或 "1,100,101"
  -r 行范围，如 "100:500" / ":1000" / "200:"
  -x X 轴列名/列号（默认第 0 列）
  -t 图表标题
  -s 保存路径
  -e 误差计算参考列（真值），其他列相对于此列算误差并显示在图上
  -l 仅列出 CSV 的所有列名后退出
"""

import sys
import os
import csv
import datetime
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tang_7_12_InitCOP_realtime_other_package import angle_difference


# ===================================================================
# 配置区：直接改这里的变量，然后 python plot_static.py 即可出图
# CLI 参数会覆盖这里的值
# ===================================================================

# CSV 文件所在目录（自动扫描目录下所有 .csv）
CSV_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"

# 选文件方式，支持混用：
#   文件名（含 .csv）→ 直接用，可以是完整路径或 CSV_DIR 下的文件名
#     "data_20260513_150200.csv"
#     "/full/path/to/data.csv"
#     "data_a.csv,data_b.csv"          ← 逗号分隔多个
#   索引号 → 从目录扫描列表中按索引选（-l 查看索引）
#     "0"        → 第 0 个文件
#     "0,2,4"    → 第 0、2、4 个文件
#     "0-3"      → 第 0 到 3 个文件
#   关键词
#     "latest:N" → 最新的 N 个文件
#     "all"      → 全部文件
CSV_PICK = "COP_0713_1.csv"

# valid 分段显示：True=valid!=0 深色粗线(valid=0 浅色淡化)，False=统一普通样式
HIGHLIGHT_VALID = True

# 是否在高亮段首点标注数值
SHOW_ANNOT = True

# 最小力值阈值（N）：低于此值的行不参与画图和误差计算（需同时满足 valid!=0）
FORCE_MIN = 0.2

# 要绘制的列：支持列名（字符串）或列号（整数从 0 开始）
# 例：["rel_ms", "ADC_angle", "Force_angle"，Fx_cal，delta_Force_X] 或 [1, 100, 102]
PLOT_COLUMNS = ["Fy_cal", "delta_Force_Y"]

# 行范围：None=全程，整数=起/止行号（不含表头）
# 例：ROW_START=0, ROW_END=None  → 全部行
# 例：ROW_START=100, ROW_END=500 → 第 100~500 行
ROW_START = 1000
ROW_END = None

# X 轴列名/列号（None=用第 0 列，即 timestamp）
X_COLUMN = "rel_ms"

# 图表标题（None=自动用文件名生成）
TITLE = None

# 保存路径（None=自动生成在 CSV_DIR 下）
SAVE_PATH = None

# 误差计算参考列（None=不计算误差）。设为真值列名，会对其他列计算相对于此列的误差
# 例：ERROR_REF_COLUMN = "Force_angle"   → 计算其他列 vs Force_angle 的误差
ERROR_REF_COLUMN = "delta_Force_Y"

# ==================== 模式选择 ====================
# "full_analysis" — 5×2 子图：左列 PZT(角度/幅值/Fz/Fx/Fy)，右列 Force(真值 vs 标定)
#                   适合查看全程数据总览和标定效果对比
# "plot"          — 自定义列折线图：按 PLOT_COLUMNS 选列画图，支持误差标注
#                   适合查看特定列的详细变化和数值误差
PLOT_MODE = "full_analysis"


# ===================================================================
# 以下为代码，一般不需要修改
# ===================================================================

# CSV 列名 → 索引映射（与 table.py TABLE_CSV_HEADER 一致）
_COLUMN_NAMES = [
    "timestamp", "rel_ms", "adc_sum",
    *(f"ch{i}" for i in range(1, 85)),
    "Fx", "Fy", "Fz", "Mx", "My", "Mz",
    "press_t", "force_t", "dt",
    "delta_CoP_X", "delta_CoP_Y",
    "delta_Force_X", "delta_Force_Y", "delta_Force_Z",
    "ADC_angle", "Force_angle",
    "Fx_cal", "Fy_cal", "Force_cal_angle",
    "CoP_state", "valid",
]

_NAME_TO_IDX = {name: idx for idx, name in enumerate(_COLUMN_NAMES)}

_ANGLE_COLUMNS = {"ADC_angle", "Force_angle", "Force_cal_angle"}


def _resolve_column(col, header: list) -> int:
    """将列名或列号解析为列索引"""
    if isinstance(col, int):
        return col
    col_str = str(col).strip()
    if col_str.isdigit():
        return int(col_str)
    if col_str in _NAME_TO_IDX:
        return _NAME_TO_IDX[col_str]
    raise ValueError(f"未知列名: {col_str}，请用 -l 查看可用列名")


def _resolve_columns(cols, header: list) -> list:
    return [_resolve_column(c, header) for c in cols]


def _scan_csv_dir(csv_dir: str) -> list:
    """扫描目录下所有 .csv 文件，按修改时间降序排列（最新的在前）"""
    if not os.path.isdir(csv_dir):
        return []
    files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir)
             if f.endswith('.csv') and not f.startswith('_')]
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


def _resolve_pick(csv_dir: str, pick: str) -> list:
    """解析 CSV_PICK 为文件路径列表"""
    all_files = _scan_csv_dir(csv_dir)

    # 含 .csv → 当作文件名处理
    if '.csv' in pick:
        paths = []
        for part in pick.split(','):
            part = part.strip()
            if os.path.isabs(part) and os.path.exists(part):
                paths.append(part)
            elif os.path.exists(os.path.join(csv_dir, part)):
                paths.append(os.path.join(csv_dir, part))
            else:
                print(f"⚠️  文件不存在，跳过: {part}")
        return paths

    # latest:N
    if pick.startswith('latest:'):
        n = int(pick.split(':')[1])
        return all_files[:n]

    # all
    if pick == 'all':
        return all_files

    # 索引号：支持 "0,2,4" 和 "0-3"
    indices = []
    for part in pick.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            indices.extend(range(int(a), int(b) + 1))
        else:
            indices.append(int(part))
    return [all_files[i] for i in indices if i < len(all_files)]


def list_files(csv_dir: str):
    """列出目录下所有 CSV 文件"""
    files = _scan_csv_dir(csv_dir)
    print(f"\n{'='*70}")
    print(f"  CSV 目录: {csv_dir}")
    print(f"  共 {len(files)} 个文件")
    print(f"{'='*70}")
    for idx, fp in enumerate(files):
        fname = os.path.basename(fp)
        mtime = os.path.getmtime(fp)
        ts = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        size_kb = os.path.getsize(fp) / 1024
        # 快速统计行数
        with open(fp, 'r') as f:
            line_cnt = sum(1 for _ in f) - 1  # 减表头
        print(f"  [{idx:>3d}] {fname}")
        print(f"        {ts}  |  {size_kb:.0f} KB  |  {line_cnt} 行")
    print(f"{'='*70}")

    print(f"\n  CSV 列名列表（共 {len(_COLUMN_NAMES)} 列）")
    print(f"{'='*70}")
    for idx, name in enumerate(_COLUMN_NAMES):
        print(f"  [{idx:>3d}]  {name}")
    print(f"{'='*70}\n")


def load_csv(path: str):
    """读取 CSV，返回 (header, data_2d_array)"""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader if row]
    data = np.array(rows, dtype=np.float64)
    return header, data


def _compute_errors(ref_vals: np.ndarray, pred_vals: np.ndarray, is_angle: bool = False) -> dict:
    """计算参考值和预测值之间的各项误差指标"""
    mask = ~np.isnan(ref_vals) & ~np.isnan(pred_vals)
    ref = ref_vals[mask]
    pred = pred_vals[mask]
    if len(ref) < 2:
        return {"count": len(ref), "error": "数据点不足"}

    if is_angle:
        abs_errors = np.array([angle_difference(r, p) for r, p in zip(ref, pred)])
        errors = np.sign(pred - ref) * abs_errors
    else:
        errors = pred - ref
        abs_errors = np.abs(errors)

    # R²
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # MAPE（排除 ref≈0 的点）
    nonzero_mask = np.abs(ref) > 1e-6
    mape = np.mean(abs_errors[nonzero_mask] / np.abs(ref[nonzero_mask])) * 100 if np.any(nonzero_mask) else float('nan')

    return {
        "count": len(ref),
        "MAE": np.mean(abs_errors),
        "MSE": np.mean(errors ** 2),
        "RMSE": np.sqrt(np.mean(errors ** 2)),
        "Max_Error": np.max(abs_errors),
        "Min_Error": np.min(abs_errors),
        "MAPE_%": mape,
        "R2": r2,
        "Error_Std": np.std(errors),
        "Median_Error": np.median(abs_errors),
    }


def _print_error_report(results: dict, ref_name: str, pred_name: str):
    """打印完整误差报告到控制台"""
    print(f"\n{'='*55}")
    print(f"  误差分析: {pred_name} vs {ref_name}")
    print(f"{'='*55}")
    if "error" in results:
        print(f"  {results['error']}")
        return
    print(f"  有效数据点: {results['count']}")
    print(f"  MAE        = {results['MAE']:.6f}")
    print(f"  MSE        = {results['MSE']:.6f}")
    print(f"  RMSE       = {results['RMSE']:.6f}")
    print(f"  Max Error  = {results['Max_Error']:.6f}")
    print(f"  Min Error  = {results['Min_Error']:.6f}")
    print(f"  Median Err = {results['Median_Error']:.6f}")
    print(f"  MAPE       = {results['MAPE_%']:.2f} %")
    print(f"  R²         = {results['R2']:.6f}")
    print(f"  Error Std  = {results['Error_Std']:.6f}")
    print(f"{'='*55}\n")


def _save_error_csv(error_path: str, error_results: list):
    """将误差结果保存为 CSV 文件"""
    with open(error_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["pred_column", "ref_column", "count",
                         "MAE", "MSE", "RMSE", "Max_Error", "Min_Error",
                         "Median_Error", "MAPE_%", "R2", "Error_Std"])
        for r in error_results:
            if "error" in r["results"]:
                continue
            res = r["results"]
            writer.writerow([r["pred"], r["ref"], res["count"],
                             f"{res['MAE']:.6f}", f"{res['MSE']:.6f}",
                             f"{res['RMSE']:.6f}", f"{res['Max_Error']:.6f}",
                             f"{res['Min_Error']:.6f}", f"{res['Median_Error']:.6f}",
                             f"{res['MAPE_%']:.2f}", f"{res['R2']:.6f}",
                             f"{res['Error_Std']:.6f}"])
    print(f"📊 误差 CSV 已保存: {error_path}")


def plot_static(csv_paths: list, plot_cols: list, x_col, row_start=None, row_end=None,
                title=None, save_path=None,
                error_ref_col=None, csv_dir=None):
    """主绘图函数，支持多文件"""
    first_base = os.path.splitext(os.path.basename(csv_paths[0]))[0]
    all_error_results = []

    n_files = len(csv_paths)
    file_colors = plt.cm.tab10(np.linspace(0, 1, max(n_files, 1)))

    fig, ax = plt.subplots(figsize=(14, 6))

    for file_idx, csv_path in enumerate(csv_paths):
        fname = os.path.basename(csv_path)
        print(f"\n📂 [{file_idx}] {fname}")
        header, data = load_csv(csv_path)

        col_indices = _resolve_columns(plot_cols, header)
        x_idx = _resolve_column(x_col, header) if x_col is not None else 0

        r0 = row_start if row_start is not None else 0
        r1 = row_end if row_end is not None else data.shape[0]
        data_slice = data[r0:r1, :]
        print(f"   行: [{r0}:{r1}] → {data_slice.shape[0]} 行")
        print(f"   列: {[header[i] for i in col_indices]}  |  X: {header[x_idx]}")

        x_data = data_slice[:, x_idx]
        n_cols = len(col_indices)

        # valid 列：用于区分有效/无效数据段
        valid_col_idx = None
        if HIGHLIGHT_VALID:
            for name in ("valid", "CoP_state"):
                if name in [h.strip() for h in header]:
                    valid_col_idx = [h.strip() for h in header].index(name)
                    break
        valid_col = data_slice[:, valid_col_idx].astype(np.float64) if valid_col_idx is not None else None

        # 力值掩码：用参考列过滤小力值
        force_ref_col = None
        if FORCE_MIN > 0 and error_ref_col is not None:
            ref_idx = _resolve_column(error_ref_col, header)
            force_ref_col = data_slice[:, ref_idx].astype(np.float64)

        for i, ci in enumerate(col_indices):
            y_data = data_slice[:, ci]
            nan_mask = ~np.isnan(y_data)
            if len(x_data[nan_mask]) == 0:
                continue

            col_color = plt.cm.tab10(np.linspace(0, 1, n_cols))[i] if n_cols > 1 else file_colors[file_idx]
            lbl = f"{fname}:{header[ci]}" if n_files > 1 else header[ci]
            if n_cols == 1 and n_files > 1:
                lbl = fname

            if HIGHLIGHT_VALID and valid_col is not None:
                # 分段绘制：有效数据深色粗线，无效数据浅色淡化
                v_mask = valid_col != 0
                if FORCE_MIN > 0 and force_ref_col is not None:
                    v_mask = v_mask & (np.abs(force_ref_col) >= FORCE_MIN)
                changes = np.where(np.diff(v_mask.astype(int)))[0] + 1
                segments = np.split(np.arange(len(x_data)), changes)
                _labeled = [False, False]
                last_annot_x = -1e9
                x_range = x_data[-1] - x_data[0] if len(x_data) > 1 else 1
                for seg in segments:
                    if len(seg) == 0:
                        continue
                    s, e = seg[0], seg[-1] + 1
                    seg_valid = v_mask[s]
                    seg_nan = nan_mask[s:e]
                    if not np.any(seg_nan):
                        continue
                    lbl_use = None
                    if seg_valid and not _labeled[1]:
                        lbl_use = lbl; _labeled[1] = True
                    elif not seg_valid and not _labeled[0]:
                        lbl_use = f"{lbl} (inactive)"; _labeled[0] = True
                    ax.plot(x_data[s:e][seg_nan], (y_data[s:e][seg_nan]), color=col_color,
                            linewidth=2.0 if seg_valid else 0.8,
                            alpha=1.0 if seg_valid else 0.3,
                            marker='.', markersize=2, label=lbl_use)
                    if SHOW_ANNOT and seg_valid and abs(x_data[s] - last_annot_x) > x_range * 0.05:
                        ax.annotate(f"{y_data[s]:.2f}", xy=(x_data[s], y_data[s]),
                                    xytext=(10, 10), textcoords='offset points',
                                    fontsize=5, color=col_color,
                                    arrowprops=dict(arrowstyle='-', color=col_color, lw=0.5),
                                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'))
                        last_annot_x = x_data[s]
            else:
                ax.plot(x_data[nan_mask], y_data[nan_mask], color=col_color, linewidth=0.8,
                        marker='.', markersize=2, label=lbl)

            # ---- 误差标注 ----
            if error_ref_col is not None:
                ref_idx = _resolve_column(error_ref_col, header)
                if ci == ref_idx:
                    continue
                ref_raw = data_slice[:, ref_idx]
                pred_raw = data_slice[:, ci]
                pair_mask = ~np.isnan(ref_raw) & ~np.isnan(pred_raw)
                if HIGHLIGHT_VALID and valid_col is not None:
                    pair_mask &= (valid_col != 0)
                if FORCE_MIN > 0 and force_ref_col is not None:
                    pair_mask &= (np.abs(force_ref_col) >= FORCE_MIN)
                ref_a = ref_raw[pair_mask]; pred_a = pred_raw[pair_mask]; x_a = x_data[pair_mask]
                if len(ref_a) < 2:
                    continue

                is_angle = header[ref_idx] in _ANGLE_COLUMNS or header[ci] in _ANGLE_COLUMNS
                if is_angle:
                    abs_errs = np.array([angle_difference(r, p) for r, p in zip(ref_a, pred_a)])
                else:
                    abs_errs = np.abs(pred_a - ref_a)
                min_i = int(np.argmin(abs_errs)); max_i = int(np.argmax(abs_errs))
                y_span = np.ptp(pred_a)

                # ▼ Min error 点
                ax.scatter([x_a[min_i]], [pred_a[min_i]], color=col_color, s=50, marker='v', zorder=6)
                ax.annotate(f'Min={abs_errs[min_i]:.3f}',
                           (x_a[min_i], pred_a[min_i] - y_span * 0.06),
                           fontsize=6, color=col_color, ha='center', va='top')

                # ▲ Max error 点
                ax.scatter([x_a[max_i]], [pred_a[max_i]], color=col_color, s=50, marker='^', zorder=6)
                ax.annotate(f'Max={abs_errs[max_i]:.3f}',
                           (x_a[max_i], pred_a[max_i] + y_span * 0.06),
                           fontsize=6, color=col_color, ha='center', va='bottom')

                # 曲线末端标 MAE / MAPE
                results = _compute_errors(ref_a, pred_a, is_angle=is_angle)
                tag = f"{fname}:{header[ci]}" if n_files > 1 else header[ci]
                _print_error_report(results, header[ref_idx], tag)
                all_error_results.append({"pred": tag, "ref": header[ref_idx], "results": results})

                lx, ly = x_data[nan_mask][-1], y_data[nan_mask][-1]
                ax.annotate(f"MAE={results['MAE']:.3f}  MAPE={results['MAPE_%']:.1f}%",
                           (lx, ly), textcoords="offset points", xytext=(10, 5),
                           fontsize=6.5, color=col_color, ha='left',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.65))

        ax.minorticks_on()
        ax.grid(True, which='major', alpha=0.4, linewidth=0.6)
        ax.grid(True, which='minor', alpha=0.15, linewidth=0.3)
        ax.set_xlabel(header[x_idx])

    ax.legend(fontsize=7, loc='best')

    title = title or first_base
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path is None:
        base_dir = csv_dir or os.path.dirname(csv_paths[0])
        save_path = os.path.join(base_dir, f"{first_base}_plot.png")
    plt.savefig(save_path, dpi=200)
    print(f"\n✅ 图片已保存: {save_path}")
    plt.close(fig)

    if all_error_results:
        base_dir = csv_dir or os.path.dirname(csv_paths[0])
        error_path = os.path.join(base_dir, f"{first_base}_error.csv")
        _save_error_csv(error_path, all_error_results)



def plot_full_analysis(csv_path: str, save_path=None, row_start=None, row_end=None,
                       save_dir=None):
    """生成 5×2 full_analysis 图，与 realtime.py plot_full_magnitude_curve 一致"""
    all_error_results = []
    header, data = load_csv(csv_path)
    r0 = row_start if row_start is not None else 0
    r1 = row_end if row_end is not None else data.shape[0]
    data = data[r0:r1, :]

    name_to_idx = {name.strip(): i for i, name in enumerate(header)}

    def _col(name):
        """按列名获取数据，不存在返回 None"""
        idx = name_to_idx.get(name)
        if idx is not None:
            return data[:, idx].astype(np.float64)
        return None

    t = _col("rel_ms")
    if t is None:
        t = np.arange(len(data))

    # 左列数据
    adc_angle = _col("ADC_angle")
    adc_sum = _col("adc_sum")
    cop_dx = _col("delta_CoP_X")
    cop_dy = _col("delta_CoP_Y")

    # 右列数据
    force_angle = _col("Force_angle")
    force_cal_angle = _col("Force_cal_angle")
    force_fz = _col("delta_Force_Z")
    force_fx = _col("delta_Force_X")
    force_fy = _col("delta_Force_Y")
    fx_cal = _col("Fx_cal")
    fy_cal = _col("Fy_cal")

    has_cal_angle = force_cal_angle is not None
    has_fx_cal = fx_cal is not None
    has_fy_cal = fy_cal is not None

    # valid 列：用于区分有效/无效数据段
    valid = _col("valid")
    if valid is None:
        valid = np.ones(len(data))

    # 有效数据掩码：valid!=0 且 力值 >= FORCE_MIN
    v_mask = valid != 0 if HIGHLIGHT_VALID else np.ones(len(t), dtype=bool)
    if FORCE_MIN > 0:
        # 用各子图的参考列过滤小力值
        _force_filters = {}
        for name in ("Force_angle", "delta_Force_X", "delta_Force_Y"):
            col = _col(name)
            if col is not None:
                _force_filters[name] = col

    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    (aL1, aR1), (aL2, aR2), (aL3, aR3), (aL4, aR4) = axes

    def _p(ax, d, c, lbl, first_valid_only=False):
        if d is None or len(d) != len(t):
            return
        if not HIGHLIGHT_VALID:
            ax.plot(t, d, c, linewidth=1.0, label=lbl)
            return
        """分段绘制：无效数据浅色淡化，有效数据深色粗线"""
        mask = v_mask
        changes = np.where(np.diff(mask.astype(int)))[0] + 1
        segments = np.split(np.arange(len(t)), changes)
        _labeled = [False, False]
        last_annot_x = -1e9
        x_range = t[-1] - t[0] if len(t) > 1 else 1
        for seg in segments:
            if len(seg) == 0:
                continue
            s, e = seg[0], seg[-1] + 1
            is_valid = mask[s]
            if first_valid_only and not is_valid:
                continue
            lbl_use = None
            if is_valid and not _labeled[1]:
                lbl_use = lbl; _labeled[1] = True
            elif not is_valid and not _labeled[0]:
                lbl_use = f"{lbl} (inactive)"; _labeled[0] = True
            ax.plot(t[s:e], d[s:e], c,
                    linewidth=2.0 if is_valid else 0.8,
                    alpha=1.0 if is_valid else 0.3,
                    label=lbl_use)
            if SHOW_ANNOT and is_valid and abs(t[s] - last_annot_x) > x_range * 0.05:
                color = c[0] if len(c) > 0 else 'blue'
                ax.annotate(f"{d[s]:.2f}", xy=(t[s], d[s]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=5, color=color,
                            arrowprops=dict(arrowstyle='-', color=color, lw=0.5),
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none'))
                last_annot_x = t[s]

    # 左列：PZT
    _p(aL1, adc_angle, 'b-', 'PZT Angle'); aL1.set_title("PZT Angle"); aL1.grid(True, alpha=0.3)
    if adc_sum is not None:
        _p(aL2, adc_sum, 'b-', 'PZT Fz')
    aL2.set_title("PZT Fz"); aL2.grid(True, alpha=0.3)
    _p(aL3, cop_dx, 'b-', 'PZT Fx'); aL3.set_title("PZT Fx"); aL3.grid(True, alpha=0.3)
    _p(aL4, cop_dy, 'c-', 'PZT Fy'); aL4.set_title("PZT Fy"); aL4.grid(True, alpha=0.3)

    # 右列：Force（含误差计算）
    _p(aR1, force_angle, 'r-', 'Measured')
    if has_cal_angle: _p(aR1, force_cal_angle, 'g--', 'Calibrated')
    aR1.set_title("Angle: Meas vs Cal"); aR1.grid(True, alpha=0.3)
    if has_cal_angle:
        aR1.legend(fontsize=8)
        _vm = v_mask & (np.abs(_force_filters.get("Force_angle", np.ones(len(t)))) >= FORCE_MIN) if FORCE_MIN > 0 else v_mask
        err = _compute_errors(force_angle[_vm], force_cal_angle[_vm], is_angle=True)
        if "error" not in err:
            aR1.annotate(f"MAE={err['MAE']:.2f}° MAPE={err['MAPE_%']:.1f}% R²={err['R2']:.3f}",
                        xy=(0.02, 0.95), xycoords='axes fraction', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))
            all_error_results.append({"pred": "Force_cal_angle", "ref": "Force_angle", "results": err})

    _p(aR2, force_fz, 'r-', 'Fz'); aR2.set_title("Fz: Measured"); aR2.grid(True, alpha=0.3)

    _p(aR3, force_fx, 'r-', 'Measured')
    if has_fx_cal: _p(aR3, fx_cal, 'g--', 'Calibrated')
    aR3.set_title("Fx: Meas vs Cal"); aR3.grid(True, alpha=0.3)
    if has_fx_cal:
        aR3.legend(fontsize=8)
        _vm = v_mask & (np.abs(_force_filters.get("delta_Force_X", np.ones(len(t)))) >= FORCE_MIN) if FORCE_MIN > 0 else v_mask
        err = _compute_errors(force_fx[_vm], fx_cal[_vm])
        if "error" not in err:
            aR3.annotate(f"MAE={err['MAE']:.4f} MAPE={err['MAPE_%']:.1f}% R²={err['R2']:.3f}",
                        xy=(0.02, 0.95), xycoords='axes fraction', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))
            all_error_results.append({"pred": "Fx_cal", "ref": "delta_Force_X", "results": err})

    _p(aR4, force_fy, 'r-', 'Measured')
    if has_fy_cal: _p(aR4, fy_cal, 'c--', 'Calibrated')
    aR4.set_title("Fy: Meas vs Cal"); aR4.grid(True, alpha=0.3)
    if has_fy_cal:
        aR4.legend(fontsize=8)
        _vm = v_mask & (np.abs(_force_filters.get("delta_Force_Y", np.ones(len(t)))) >= FORCE_MIN) if FORCE_MIN > 0 else v_mask
        err = _compute_errors(force_fy[_vm], fy_cal[_vm])
        if "error" not in err:
            aR4.annotate(f"MAE={err['MAE']:.4f} MAPE={err['MAPE_%']:.1f}% R²={err['R2']:.3f}",
                        xy=(0.02, 0.95), xycoords='axes fraction', fontsize=7,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))
            all_error_results.append({"pred": "Fy_cal", "ref": "delta_Force_Y", "results": err})

    for row in axes:
        for ax in row:
            ax.set_xlabel("Time (ms)", fontsize=9)

    plt.tight_layout()

    if save_path is None:
        base_dir = save_dir or os.path.dirname(csv_path)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. data_20260604_131142
        save_path = os.path.join(base_dir, f"full_analysis_{base_name}.png")

    plt.savefig(save_path, dpi=300)
    print(f"📊 已保存：{save_path}")
    plt.close(fig)

    if all_error_results:
        error_path = save_path.replace(".png", "_error.csv")
        _save_error_csv(error_path, all_error_results)


def main():
    parser = argparse.ArgumentParser(description="CSV 静态图绘制")
    parser.add_argument("-d", "--dir", default=None, help="CSV 目录")
    parser.add_argument("-f", "--files", default=None, help="文件选择：文件名/索引/latest:N/all")
    parser.add_argument("-c", "--columns", default=None, help="列名/列号，逗号分隔")
    parser.add_argument("-r", "--rows", default=None, help="行范围，如 100:500")
    parser.add_argument("-x", "--xcol", default=None, help="X 轴列名/列号")
    parser.add_argument("-t", "--title", default=None, help="图表标题")
    parser.add_argument("-s", "--save", default=None, help="保存路径")
    parser.add_argument("-e", "--error-ref", default=None, help="误差计算参考列")
    parser.add_argument("-l", "--list", action="store_true", help="列出目录下 CSV 及列名后退出")
    args = parser.parse_args()

    csv_dir = args.dir or CSV_DIR
    csv_pick = args.files or CSV_PICK

    if args.list:
        list_files(csv_dir)
        return

    csv_paths = _resolve_pick(csv_dir, csv_pick)
    if not csv_paths:
        print(f"❌ 未找到匹配的 CSV 文件: dir={csv_dir}, pick={csv_pick}")
        sys.exit(1)
    print(f"📂 选中 {len(csv_paths)} 个文件:")
    for i, p in enumerate(csv_paths):
        print(f"    [{i}] {os.path.basename(p)}")

    if args.columns is not None:
        plot_cols = [c.strip() for c in args.columns.split(",")]
    else:
        plot_cols = PLOT_COLUMNS

    row_start = ROW_START
    row_end = ROW_END
    if args.rows is not None:
        parts = args.rows.split(":")
        if parts[0]:
            row_start = int(parts[0])
        else:
            row_start = 0
        if len(parts) > 1 and parts[1]:
            row_end = int(parts[1])
        else:
            row_end = None

    x_col = args.xcol or X_COLUMN
    title = args.title or TITLE
    save_path = args.save or SAVE_PATH
    error_ref = args.error_ref or ERROR_REF_COLUMN

    plot_static(csv_paths, plot_cols, x_col, row_start, row_end,
                title=title, save_path=save_path,
                error_ref_col=error_ref, csv_dir=csv_dir)


if __name__ == "__main__":
    if PLOT_MODE == "full_analysis":
        csv_paths = _resolve_pick(CSV_DIR, CSV_PICK)
        if not csv_paths:
            print(f"❌ 未找到匹配的 CSV 文件: dir={CSV_DIR}, pick={CSV_PICK}")
            sys.exit(1)
        plot_full_analysis(csv_paths[0],
                           row_start=ROW_START,
                           row_end=ROW_END)
    else:
        main()
