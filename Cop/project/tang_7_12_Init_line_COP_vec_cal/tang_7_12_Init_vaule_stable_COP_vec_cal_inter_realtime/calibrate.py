"""
CoP 位移 + 总压力 → 三维力 标定模块

输入: (adc_sum, delta_CoP_X, delta_CoP_Y)
输出: (delta_Force_Z, delta_Force_X, delta_Force_Y)

纯 numpy 实现，零外部依赖。
"""

import os
import sys
import csv
import numpy as np

# ===================== 可调参数 =====================
CAL_CSV_PATH = "/home/qcy/Project/data/2.PZT_tangential/weight/test/data_20260513_150200.csv"  # CSV文件路径
CAL_MODE = "continuous"       # "continuous"=连续标定, "discrete"=离散标定
CAL_DO_FIT = True             # 是否同时生成拟合模型
CAL_FORCE_BIN = 0.2           # discrete模式的力分组间隔(N)


def build_lookup_from_csv(csv_path: str, mode: str = "continuous", force_bin: float = 0.2):
    """
    读取 CSV，返回 (points[N,3], fz_vals, fx_vals, fy_vals)
    输入: (adc_sum, delta_CoP_X, delta_CoP_Y)
    输出: (delta_Force_Z, delta_Force_X, delta_Force_Y)
    过滤: CoP_state=2
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            try:
                if float(row.get("CoP_state", 0)) != 2:
                    continue
                adc_sum = float(row["adc_sum"])
                dx = float(row["delta_CoP_X"])
                dy = float(row["delta_CoP_Y"])
                fz = float(row["delta_Force_Z"])
                fx = float(row["delta_Force_X"])
                fy = float(row["delta_Force_Y"])
                rows.append((adc_sum, dx, dy, fz, fx, fy))
            except (KeyError, ValueError):
                continue

    if len(rows) < 2:
        raise ValueError(f"有效数据点不足（当前 {len(rows)} 个，需至少 2 个），请检查CSV文件")

    if mode == "discrete":
        # 按 (Fz, Fx, Fy) 分组求平均
        from collections import defaultdict
        groups = defaultdict(list)
        for adc_sum, dx, dy, fz, fx, fy in rows:
            key = (round(fz / force_bin) * force_bin,
                   round(fx / force_bin) * force_bin,
                   round(fy / force_bin) * force_bin)
            groups[key].append((adc_sum, dx, dy, fz, fx, fy))

        avg_rows = []
        for _, members in groups.items():
            arr = np.array(members)
            avg_rows.append(arr.mean(axis=0))
        data = np.array(avg_rows)
        print(f"  离散标定: {len(rows)} 行 → {len(groups)} 组 → {len(data)} 平均点")
    else:
        data = np.array(rows)

    points = data[:, :3].astype(np.float32)   # (adc_sum, dx, dy)
    fz_vals = data[:, 3].astype(np.float32)
    fx_vals = data[:, 4].astype(np.float32)
    fy_vals = data[:, 5].astype(np.float32)

    print(f"\n{'='*50}")
    print(f"  查找表构建结果 ({mode})")
    print(f"{'='*50}")
    print(f"  数据点数: {len(data)}")
    print(f"  adc_sum 范围: [{points[:,0].min():.1f}, {points[:,0].max():.1f}]")
    print(f"  dx 范围: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
    print(f"  dy 范围: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")
    print(f"  Fz 范围: [{fz_vals.min():.4f}, {fz_vals.max():.4f}] N")
    print(f"  Fx 范围: [{fx_vals.min():.4f}, {fx_vals.max():.4f}] N")
    print(f"  Fy 范围: [{fy_vals.min():.4f}, {fy_vals.max():.4f}] N")
    print(f"{'='*50}\n")

    return points, fz_vals, fx_vals, fy_vals


def save_lookup(points: np.ndarray, fz_vals: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray, path: str):
    """保存查找表到 .bin（C++可直接读取）"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    n = np.int32(len(points))
    with open(path, "wb") as f:
        f.write(n.tobytes())
        f.write(points.astype(np.float32).tobytes())   # N*3*4 bytes
        f.write(fz_vals.astype(np.float32).tobytes())
        f.write(fx_vals.astype(np.float32).tobytes())
        f.write(fy_vals.astype(np.float32).tobytes())
    print(f"  查找表已保存至: {path} ({len(points)} 点, {os.path.getsize(path)} 字节)")


def load_lookup(path: str) -> tuple:
    """加载查找表，返回 (points[N,3], fz_vals, fx_vals, fy_vals)"""
    with open(path, "rb") as f:
        n = np.frombuffer(f.read(4), dtype=np.int32)[0]
        points = np.frombuffer(f.read(n * 12), dtype=np.float32).reshape(n, 3)
        fz_vals = np.frombuffer(f.read(n * 4), dtype=np.float32)
        fx_vals = np.frombuffer(f.read(n * 4), dtype=np.float32)
        fy_vals = np.frombuffer(f.read(n * 4), dtype=np.float32)
    return points, fz_vals, fx_vals, fy_vals


def apply(adc_sum: float, dx: float, dy: float,
          points: np.ndarray, fz_vals: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray) -> tuple:
    """最近邻查找：返回距离 (adc_sum, dx, dy) 最近的标定点对应的 (Fz, Fx, Fy)"""
    dists = np.sum((points - np.array([adc_sum, dx, dy], dtype=np.float32)) ** 2, axis=1)
    idx = np.argmin(dists)
    return float(fz_vals[idx]), float(fx_vals[idx]), float(fy_vals[idx])


# ==================== 拟合标定 ====================

def build_fit_model(points: np.ndarray, fz_vals: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray):
    """3D二次多项式拟合：10个系数 [1, s, x, y, s², sx, sy, x², xy, y²]"""
    s = points[:, 0]  # adc_sum
    x = points[:, 1]  # delta_CoP_X
    y = points[:, 2]  # delta_CoP_Y
    A = np.column_stack([np.ones(len(points)), s, x, y, s*s, s*x, s*y, x*x, x*y, y*y])  # (N, 10)
    coef_fz, _, _, _ = np.linalg.lstsq(A, fz_vals, rcond=None)
    coef_fx, _, _, _ = np.linalg.lstsq(A, fx_vals, rcond=None)
    coef_fy, _, _, _ = np.linalg.lstsq(A, fy_vals, rcond=None)
    return coef_fz, coef_fx, coef_fy


def apply_fit(adc_sum: float, dx: float, dy: float, coef_fz, coef_fx, coef_fy) -> tuple:
    """拟合标定：用3D二次多项式计算 (Fz, Fx, Fy)"""
    s, x, y = adc_sum, dx, dy
    basis = np.array([1, s, x, y, s*s, s*x, s*y, x*x, x*y, y*y])
    fz = float(np.dot(coef_fz, basis))
    fx = float(np.dot(coef_fx, basis))
    fy = float(np.dot(coef_fy, basis))
    return fz, fx, fy


def save_fit_model(coef_fz, coef_fx, coef_fy, path: str):
    """保存拟合系数到 .bin（240字节，C++可直接 fread 读取）"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.array(coef_fz, dtype=np.float64).tobytes())
        f.write(np.array(coef_fx, dtype=np.float64).tobytes())
        f.write(np.array(coef_fy, dtype=np.float64).tobytes())
    print(f"  拟合模型已保存至: {path} ({os.path.getsize(path)} 字节)")


def load_fit_model(path: str) -> tuple:
    """加载拟合系数，返回 (coef_fz, coef_fx, coef_fy)"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(240), dtype=np.float64)
    return data[:10], data[10:20], data[20:30]


# ==================== 运行 ====================
if __name__ == "__main__":
    out_dir = os.path.dirname(CAL_CSV_PATH)

    try:
        points, fz_vals, fx_vals, fy_vals = build_lookup_from_csv(CAL_CSV_PATH, mode=CAL_MODE, force_bin=CAL_FORCE_BIN)
        save_lookup(points, fz_vals, fx_vals, fy_vals, os.path.join(out_dir, "cal_lookup.bin"))
        if CAL_DO_FIT:
            coef_fz, coef_fx, coef_fy = build_fit_model(points, fz_vals, fx_vals, fy_vals)
            save_fit_model(coef_fz, coef_fx, coef_fy, os.path.join(out_dir, "cal_fit.bin"))
            labels = ["1", "s", "x", "y", "s²", "sx", "sy", "x²", "xy", "y²"]
            for name, coef in [("Fz", coef_fz), ("Fx", coef_fx), ("Fy", coef_fy)]:
                terms = " + ".join(f"{c:.4f}*{l}" for c, l in zip(coef, labels))
                print(f"  {name} = {terms}")
    except Exception as e:
        print(f"  构建失败: {e}")
        sys.exit(1)
