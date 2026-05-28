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


def build_lookup_from_csv(csv_path: str, mode: str = "continuous", force_bin: float = 0.2):
    """
    读取 CSV，返回 (points, fx_vals, fy_vals)
    mode="continuous": 只过滤 valid=1，保留所有行
    mode="discrete": 过滤 valid=1，按 (Fx,Fy) 分组求平均，返回网格点
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            try:
                if float(row.get("valid", 1)) != 1:
                    continue
                dx = float(row["delta_CoP_X"])
                dy = float(row["delta_CoP_Y"])
                fx = float(row["delta_Force_X"])
                fy = float(row["delta_Force_Y"])
                rows.append((dx, dy, fx, fy))
            except (KeyError, ValueError):
                continue

    if len(rows) < 2:
        raise ValueError(f"有效数据点不足（当前 {len(rows)} 个，需至少 2 个），请检查CSV文件")

    if mode == "discrete":
        # 按 (Fx, Fy) 分组求平均
        from collections import defaultdict
        groups = defaultdict(list)
        for dx, dy, fx, fy in rows:
            key = (round(fx / force_bin) * force_bin,
                   round(fy / force_bin) * force_bin)
            groups[key].append((dx, dy, fx, fy))

        avg_rows = []
        for _, members in groups.items():
            arr = np.array(members)
            avg_rows.append((arr[:, 0].mean(), arr[:, 1].mean(),
                             arr[:, 2].mean(), arr[:, 3].mean()))
        data = np.array(avg_rows)
        print(f"  离散标定: {len(rows)} 行 → {len(groups)} 组 → {len(data)} 平均点")
    else:
        data = np.array(rows)

    points = data[:, :2].astype(np.float32)
    fx_vals = data[:, 2].astype(np.float32)
    fy_vals = data[:, 3].astype(np.float32)

    print(f"\n{'='*50}")
    print(f"  查找表构建结果 ({mode})")
    print(f"{'='*50}")
    print(f"  数据点数: {len(data)}")
    print(f"  dx 范围: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
    print(f"  dy 范围: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
    print(f"  Fx 范围: [{fx_vals.min():.4f}, {fx_vals.max():.4f}] N")
    print(f"  Fy 范围: [{fy_vals.min():.4f}, {fy_vals.max():.4f}] N")
    print(f"{'='*50}\n")

    return points, fx_vals, fy_vals


def save_lookup(points: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray, path: str):
    """保存查找表到 .bin（C++可直接读取）"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    n = np.int32(len(points))
    with open(path, "wb") as f:
        f.write(n.tobytes())
        f.write(points.astype(np.float32).tobytes())
        f.write(fx_vals.astype(np.float32).tobytes())
        f.write(fy_vals.astype(np.float32).tobytes())
    print(f"  查找表已保存至: {path} ({len(points)} 点, {os.path.getsize(path)} 字节)")


def load_lookup(path: str) -> tuple:
    """加载查找表，返回 (points, fx_vals, fy_vals)"""
    with open(path, "rb") as f:
        n = np.frombuffer(f.read(4), dtype=np.int32)[0]
        points = np.frombuffer(f.read(n * 8), dtype=np.float32).reshape(n, 2)
        fx_vals = np.frombuffer(f.read(n * 4), dtype=np.float32)
        fy_vals = np.frombuffer(f.read(n * 4), dtype=np.float32)
    return points, fx_vals, fy_vals


def apply(dx: float, dy: float, points: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray) -> tuple:
    """最近邻查找：返回距离 (dx,dy) 最近的标定点对应的 (Fx, Fy)"""
    dists = np.sum((points - np.array([dx, dy], dtype=np.float32)) ** 2, axis=1)
    idx = np.argmin(dists)
    return float(fx_vals[idx]), float(fy_vals[idx])


def apply_discrete(dx: float, dy: float, points: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray) -> tuple:
    """双线性插值查找：在 (dx,dy) 网格中找最近4个点做双线性插值"""
    dists = np.sum((points - np.array([dx, dy], dtype=np.float32)) ** 2, axis=1)
    idxs = np.argsort(dists)[:4]

    pts = points[idxs]
    fx4 = fx_vals[idxs]
    fy4 = fy_vals[idxs]

    dx_min, dx_max = pts[:, 0].min(), pts[:, 0].max()
    dy_min, dy_max = pts[:, 1].min(), pts[:, 1].max()

    if dx_max - dx_min < 1e-8 or dy_max - dy_min < 1e-8:
        return float(fx4[0]), float(fy4[0])

    t = (dx - dx_min) / (dx_max - dx_min)
    u = (dy - dy_min) / (dy_max - dy_min)
    t = max(0.0, min(1.0, t))
    u = max(0.0, min(1.0, u))

    # 找四个角点：左下、右下、左上、右上
    bl = np.argmin((pts[:, 0] - dx_min)**2 + (pts[:, 1] - dy_min)**2)
    br = np.argmin((pts[:, 0] - dx_max)**2 + (pts[:, 1] - dy_min)**2)
    tl = np.argmin((pts[:, 0] - dx_min)**2 + (pts[:, 1] - dy_max)**2)
    tr = np.argmin((pts[:, 0] - dx_max)**2 + (pts[:, 1] - dy_max)**2)

    fx = float(fx4[bl]*(1-t)*(1-u) + fx4[br]*t*(1-u) + fx4[tl]*(1-t)*u + fx4[tr]*t*u)
    fy = float(fy4[bl]*(1-t)*(1-u) + fy4[br]*t*(1-u) + fy4[tl]*(1-t)*u + fy4[tr]*t*u)
    return fx, fy


# ==================== 拟合标定 ====================

def build_fit_model(points: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray):
    """二次多项式拟合：Fx/Fy = a0 + a1*dx + a2*dy + a3*dx² + a4*dx*dy + a5*dy²"""
    dx = points[:, 0]
    dy = points[:, 1]
    A = np.column_stack([np.ones(len(points)), dx, dy, dx*dx, dx*dy, dy*dy])  # (N, 6)
    coef_fx, _, _, _ = np.linalg.lstsq(A, fx_vals, rcond=None)
    coef_fy, _, _, _ = np.linalg.lstsq(A, fy_vals, rcond=None)
    return coef_fx, coef_fy


def apply_fit(dx: float, dy: float, coef_fx, coef_fy) -> tuple:
    """拟合标定：用二次多项式计算 (Fx, Fy)"""
    fx = float(coef_fx[0] + coef_fx[1]*dx + coef_fx[2]*dy + coef_fx[3]*dx*dx + coef_fx[4]*dx*dy + coef_fx[5]*dy*dy)
    fy = float(coef_fy[0] + coef_fy[1]*dx + coef_fy[2]*dy + coef_fy[3]*dx*dx + coef_fy[4]*dx*dy + coef_fy[5]*dy*dy)
    return fx, fy


def save_fit_model(coef_fx, coef_fy, path: str):
    """保存拟合系数到 .bin（96字节，C++可直接 fread 读取）"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.array(coef_fx, dtype=np.float64).tobytes())
        f.write(np.array(coef_fy, dtype=np.float64).tobytes())
    print(f"  拟合模型已保存至: {path} ({os.path.getsize(path)} 字节)")


def load_fit_model(path: str) -> tuple:
    """加载拟合系数，返回 (coef_fx, coef_fy)"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(96), dtype=np.float64)
    return data[:6], data[6:12]


# ==================== CLI 入口 ====================
CAL_DEFAULT_SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"


def _resolve_path(arg: str) -> str:
    if arg.isdigit():
        return os.path.join(CAL_DEFAULT_SAVE_DIR, f"data_{arg}.csv")
    if os.path.sep not in arg and not arg.startswith("."):
        return os.path.join(CAL_DEFAULT_SAVE_DIR, arg)
    return arg


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calibrate.py <csv_path|N|filename> [--fit] [--discrete [bin]]")
        print("  python calibrate.py data_20260513_150200.csv                                     → 连续查找表 cal_lookup.bin")
        print("  python calibrate.py 1 --fit                               → 连续查找表 + 拟合")
        print("  python calibrate.py 1 --discrete                          → 离散查找表 (bin=0.2N)")
        print("  python calibrate.py data_20260513_150200.csv --discrete 0.5                      → 离散查找表 (bin=0.5N)")
        print("  python calibrate.py data_20260513_150200.csv --discrete --fit                    → 离散查找表 + 拟合")
        sys.exit(1)

    do_fit = "--fit" in sys.argv
    do_discrete = "--discrete" in sys.argv

    # 解析 force_bin
    force_bin = 0.2
    args = [a for a in sys.argv[1:] if a not in ("--fit", "--discrete")]
    if do_discrete and len(args) > 1:
        try:
            force_bin = float(args[1])
            args = args[:1]
        except ValueError:
            pass

    csv_path = _resolve_path(args[0])
    out_dir = os.path.dirname(csv_path)
    mode = "discrete" if do_discrete else "continuous"

    try:
        points, fx_vals, fy_vals = build_lookup_from_csv(csv_path, mode=mode, force_bin=force_bin)
        save_lookup(points, fx_vals, fy_vals, os.path.join(out_dir, "cal_lookup.bin"))
        if do_fit:
            coef_fx, coef_fy = build_fit_model(points, fx_vals, fy_vals)
            save_fit_model(coef_fx, coef_fy, os.path.join(out_dir, "cal_fit.bin"))
            print(f"  Fx = {coef_fx[0]:.4f} + {coef_fx[1]:.4f}*dx + {coef_fx[2]:.4f}*dy + {coef_fx[3]:.4f}*dx² + {coef_fx[4]:.4f}*dx*dy + {coef_fx[5]:.4f}*dy²")
            print(f"  Fy = {coef_fy[0]:.4f} + {coef_fy[1]:.4f}*dx + {coef_fy[2]:.4f}*dy + {coef_fy[3]:.4f}*dx² + {coef_fy[4]:.4f}*dx*dy + {coef_fy[5]:.4f}*dy²")
    except Exception as e:
        print(f"  构建失败: {e}")
        sys.exit(1)
