"""
静态图绘制工具 —— 从 CSV 中选列、选行，画 matplotlib 图。

两种用法：
  1. 直接改下面「配置区」的变量，然后运行：python plot_static.py
  2. CLI 传参覆盖配置：python plot_static.py -f data_2.csv -c rel_ms,ADC_mag -r 100:500

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
CSV_PICK = "data_20260513_150200.csv"

# 要绘制的列：支持列名（字符串）或列号（整数从 0 开始）
# 例：["rel_ms", "ADC_angle", "Force_angle"] 或 [1, 100, 102]
PLOT_COLUMNS = ["Force_cal_angle", "Force_angle"]

# 行范围：None=全程，整数=起/止行号（不含表头）
# 例：ROW_START=0, ROW_END=None  → 全部行
# 例：ROW_START=100, ROW_END=500 → 第 100~500 行
ROW_START = 2000
ROW_END = None

# X 轴列名/列号（None=用第 0 列，即 timestamp）
X_COLUMN = "rel_ms"

# 图表标题（None=自动用文件名生成）
TITLE = None

# 保存路径（None=自动生成在 CSV_DIR 下）
SAVE_PATH = None

# 是否同一子图叠加（True=所有线画在一个图上, False=每列独立子图）
SHARE_AXIS = True

# 误差计算参考列（None=不计算误差）。设为真值列名，会对其他列计算相对于此列的误差
# 例：ERROR_REF_COLUMN = "Force_angle"   → 计算其他列 vs Force_angle 的误差
ERROR_REF_COLUMN = "Force_angle"

# ===================================================================
# 以下为代码，一般不需要修改
# ===================================================================

# CSV 列名 → 索引映射（与 table.py TABLE_CSV_HEADER 一致）
_COLUMN_NAMES = [
    "timestamp", "rel_ms",
    *(f"ch{i}" for i in range(1, 85)),
    "Fx", "Fy", "Fz", "Mx", "My", "Mz",
    "press_t", "force_t", "dt",
    "delta_CoP_X", "delta_CoP_Y",
    "delta_Force_X", "delta_Force_Y", "delta_Force_Z",
    "ADC_angle", "ADC_mag", "Force_angle", "Force_mag",
    "Fx_cal", "Fy_cal", "Force_cal_mag", "Force_cal_angle",
    "CoP_state",
]

_NAME_TO_IDX = {name: idx for idx, name in enumerate(_COLUMN_NAMES)}


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


def _compute_errors(ref_vals: np.ndarray, pred_vals: np.ndarray) -> dict:
    """计算参考值和预测值之间的各项误差指标"""
    mask = ~np.isnan(ref_vals) & ~np.isnan(pred_vals)
    ref = ref_vals[mask]
    pred = pred_vals[mask]
    if len(ref) < 2:
        return {"count": len(ref), "error": "数据点不足"}

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
                title=None, save_path=None, share_axis=False,
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

        for i, ci in enumerate(col_indices):
            y_data = data_slice[:, ci]
            valid_mask = ~np.isnan(y_data)
            valid_x = x_data[valid_mask]
            valid_y = y_data[valid_mask]
            if len(valid_x) == 0:
                continue

            col_color = plt.cm.tab10(np.linspace(0, 1, n_cols))[i] if n_cols > 1 else file_colors[file_idx]
            lbl = f"{fname}:{header[ci]}" if n_files > 1 else header[ci]
            if n_cols == 1 and n_files > 1:
                lbl = fname
            ax.plot(valid_x, valid_y, color=col_color, linewidth=0.8,
                    marker='.', markersize=2, label=lbl)

            # ---- 误差标注 ----
            if error_ref_col is not None:
                ref_idx = _resolve_column(error_ref_col, header)
                if ci == ref_idx:
                    continue
                ref_raw = data_slice[:, ref_idx]
                pred_raw = data_slice[:, ci]
                pair_mask = ~np.isnan(ref_raw) & ~np.isnan(pred_raw)
                ref_a = ref_raw[pair_mask]; pred_a = pred_raw[pair_mask]; x_a = x_data[pair_mask]
                if len(ref_a) < 2:
                    continue

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
                results = _compute_errors(ref_raw, pred_raw)
                tag = f"{fname}:{header[ci]}" if n_files > 1 else header[ci]
                _print_error_report(results, header[ref_idx], tag)
                all_error_results.append({"pred": tag, "ref": header[ref_idx], "results": results})

                lx, ly = valid_x[-1], valid_y[-1]
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


def main():
    parser = argparse.ArgumentParser(description="CSV 静态图绘制")
    parser.add_argument("-d", "--dir", default=None, help="CSV 目录")
    parser.add_argument("-f", "--files", default=None, help="文件选择：文件名/索引/latest:N/all")
    parser.add_argument("-c", "--columns", default=None, help="列名/列号，逗号分隔")
    parser.add_argument("-r", "--rows", default=None, help="行范围，如 100:500")
    parser.add_argument("-x", "--xcol", default=None, help="X 轴列名/列号")
    parser.add_argument("-t", "--title", default=None, help="图表标题")
    parser.add_argument("-s", "--save", default=None, help="保存路径")
    parser.add_argument("--share", action="store_true", help="同一子图叠加")
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
    share = args.share or SHARE_AXIS
    error_ref = args.error_ref or ERROR_REF_COLUMN

    plot_static(csv_paths, plot_cols, x_col, row_start, row_end,
                title=title, save_path=save_path, share_axis=share,
                error_ref_col=error_ref, csv_dir=csv_dir)


if __name__ == "__main__":
    main()
