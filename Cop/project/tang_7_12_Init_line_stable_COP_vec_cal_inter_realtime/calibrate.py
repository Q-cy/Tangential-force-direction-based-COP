"""
CoP 位移 → 切向力 标定模块（查找表版）

存储所有标定点 (dx, dy) → (Fx, Fy)，查询时用最近邻返回对应力值。
纯 numpy 实现，零外部依赖。

用法:
  构建: python calibrate.py <csv_path>
  应用: from calibrate import load_lookup, apply
"""

import os
import sys
import csv
import numpy as np


def build_lookup_from_csv(csv_path: str):
    """读取 CSV，返回 (points, fx_vals, fy_vals)"""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dx = float(row["delta_CoP_X"])
                dy = float(row["delta_CoP_Y"])
                fx = float(row["delta_Force_X"])
                fy = float(row["delta_Force_Y"])
                rows.append((dx, dy, fx, fy))
            except (KeyError, ValueError):
                continue

    if len(rows) < 10:
        raise ValueError(f"有效数据点不足（{len(rows)} < 10），请检查CSV文件")

    data = np.array(rows)
    points = data[:, :2].astype(np.float32)
    fx_vals = data[:, 2].astype(np.float32)
    fy_vals = data[:, 3].astype(np.float32)

    print(f"\n{'='*50}")
    print(f"  查找表构建结果")
    print(f"{'='*50}")
    print(f"  数据点数: {len(data)}")
    print(f"  dx 范围: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
    print(f"  dy 范围: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
    print(f"  Fx 范围: [{fx_vals.min():.4f}, {fx_vals.max():.4f}] N")
    print(f"  Fy 范围: [{fy_vals.min():.4f}, {fy_vals.max():.4f}] N")
    print(f"{'='*50}\n")

    return points, fx_vals, fy_vals


def save_lookup(points: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray, path: str):
    """保存查找表到 .npz"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    np.savez(path, points=points, fx=fx_vals, fy=fy_vals)
    print(f"  查找表已保存至: {path}")


def load_lookup(path: str) -> tuple:
    """加载查找表，返回 (points, fx_vals, fy_vals)"""
    data = np.load(path)
    return data["points"], data["fx"], data["fy"]


def apply(dx: float, dy: float, points: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray) -> tuple:
    """最近邻查找：返回距离 (dx,dy) 最近的标定点对应的 (Fx, Fy)"""
    dists = np.sum((points - np.array([dx, dy], dtype=np.float32)) ** 2, axis=1)
    idx = np.argmin(dists)
    return float(fx_vals[idx]), float(fy_vals[idx])


# ==================== CLI 入口 ====================
DEFAULT_SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"


def _resolve_path(arg: str) -> str:
    if arg.isdigit():
        return os.path.join(DEFAULT_SAVE_DIR, f"data_{arg}.csv")
    if os.path.sep not in arg and not arg.startswith("."):
        return os.path.join(DEFAULT_SAVE_DIR, arg)
    return arg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calibrate.py <csv_path|N|filename>")
        print("  python calibrate.py 1              → data_1.csv")
        print("  python calibrate.py data_1.csv     → SAVE_DIR/data_1.csv")
        sys.exit(1)

    csv_path = _resolve_path(sys.argv[1])
    out_dir = os.path.dirname(csv_path)
    out_path = os.path.join(out_dir, "cal_lookup.npz")

    try:
        points, fx_vals, fy_vals = build_lookup_from_csv(csv_path)
        save_lookup(points, fx_vals, fy_vals, out_path)
    except Exception as e:
        print(f"  查找表构建失败: {e}")
        sys.exit(1)
