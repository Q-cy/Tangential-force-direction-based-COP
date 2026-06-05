"""
CoP 位移 (+ 总压力) → 力 标定模块

CAL_DIM="2D": 输入 (delta_CoP_X, delta_CoP_Y) → 输出 (delta_Force_X, delta_Force_Y)
CAL_DIM="3D": 输入 (adc_sum, delta_CoP_X, delta_CoP_Y) → 输出 (delta_Force_Z, delta_Force_X, delta_Force_Y)

纯 numpy 实现，零外部依赖。
"""

import os
import sys
import csv
import numpy as np

# ===================== 可调参数 =====================
CAL_CSV_PATH = "/home/qcy/Project/data/2.PZT_tangential/weight/test/data_20260605_191856.csv"
CAL_MODE = "continuous"       # "continuous"=连续标定, "discrete"=离散标定
CAL_DIM = "2D"                # "2D"=仅切向力(Fx,Fy), "3D"=三维力(Fz,Fx,Fy)
CAL_DO_FIT = True             # 是否同时生成拟合模型
CAL_FORCE_BIN = 0.2           # discrete模式的力分组间隔(N)


def build_lookup_from_csv(csv_path: str, mode: str = "continuous", force_bin: float = 0.2, dim: str = "3D"):
    """
    读取 CSV，返回标定数据
    dim="2D": (points[N,2], fx_vals, fy_vals)
    dim="3D": (points[N,3], fz_vals, fx_vals, fy_vals)
    过滤: CoP_state=2
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            try:
                if float(row.get("valid", 0)) == 0:
                    continue
                dx = float(row["delta_CoP_X"])
                dy = float(row["delta_CoP_Y"])
                fx = float(row["delta_Force_X"])
                fy = float(row["delta_Force_Y"])
                if dim == "3D":
                    adc_sum = float(row["adc_sum"])
                    fz = float(row["delta_Force_Z"])
                    rows.append((adc_sum, dx, dy, fz, fx, fy))
                else:
                    rows.append((dx, dy, fx, fy))
            except (KeyError, ValueError):
                continue

    if len(rows) < 2:
        raise ValueError(f"有效数据点不足（当前 {len(rows)} 个，需至少 2 个），请检查CSV文件")

    if mode == "discrete":
        from collections import defaultdict
        groups = defaultdict(list)
        for r in rows:
            if dim == "3D":
                key = (round(r[3] / force_bin) * force_bin,
                       round(r[4] / force_bin) * force_bin,
                       round(r[5] / force_bin) * force_bin)
            else:
                key = (round(r[2] / force_bin) * force_bin,
                       round(r[3] / force_bin) * force_bin)
            groups[key].append(r)
        avg_rows = []
        for _, members in groups.items():
            arr = np.array(members)
            avg_rows.append(arr.mean(axis=0))
        data = np.array(avg_rows)
        print(f"  离散标定: {len(rows)} 行 → {len(groups)} 组 → {len(data)} 平均点")
    else:
        data = np.array(rows)

    if dim == "3D":
        points = data[:, :3].astype(np.float32)
        fz_vals = data[:, 3].astype(np.float32)
        fx_vals = data[:, 4].astype(np.float32)
        fy_vals = data[:, 5].astype(np.float32)
        print(f"\n{'='*50}")
        print(f"  查找表构建结果 ({mode}, 3D)")
        print(f"{'='*50}")
        print(f"  数据点数: {len(data)}")
        print(f"  adc_sum: [{points[:,0].min():.1f}, {points[:,0].max():.1f}]")
        print(f"  dx: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
        print(f"  dy: [{points[:,2].min():.4f}, {points[:,2].max():.4f}]")
        print(f"  Fz: [{fz_vals.min():.4f}, {fz_vals.max():.4f}] N")
        print(f"  Fx: [{fx_vals.min():.4f}, {fx_vals.max():.4f}] N")
        print(f"  Fy: [{fy_vals.min():.4f}, {fy_vals.max():.4f}] N")
        print(f"{'='*50}\n")
        return points, fz_vals, fx_vals, fy_vals
    else:
        points = data[:, :2].astype(np.float32)
        fx_vals = data[:, 2].astype(np.float32)
        fy_vals = data[:, 3].astype(np.float32)
        print(f"\n{'='*50}")
        print(f"  查找表构建结果 ({mode}, 2D)")
        print(f"{'='*50}")
        print(f"  数据点数: {len(data)}")
        print(f"  dx: [{points[:,0].min():.4f}, {points[:,0].max():.4f}]")
        print(f"  dy: [{points[:,1].min():.4f}, {points[:,1].max():.4f}]")
        print(f"  Fx: [{fx_vals.min():.4f}, {fx_vals.max():.4f}] N")
        print(f"  Fy: [{fy_vals.min():.4f}, {fy_vals.max():.4f}] N")
        print(f"{'='*50}\n")
        return points, fx_vals, fy_vals


# ==================== 查找表 ====================

def save_lookup(points, fx_vals, fy_vals, path: str, fz_vals=None):
    """保存查找表到 .bin"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    n = np.int32(len(points))
    with open(path, "wb") as f:
        f.write(n.tobytes())
        f.write(points.astype(np.float32).tobytes())
        if fz_vals is not None:
            f.write(fz_vals.astype(np.float32).tobytes())
        f.write(fx_vals.astype(np.float32).tobytes())
        f.write(fy_vals.astype(np.float32).tobytes())
    print(f"  查找表已保存至: {path} ({len(points)} 点, {os.path.getsize(path)} 字节)")


def load_lookup(path: str, dim: str = "3D") -> tuple:
    """加载查找表"""
    with open(path, "rb") as f:
        n = np.frombuffer(f.read(4), dtype=np.int32)[0]
        if dim == "3D":
            points = np.frombuffer(f.read(n * 12), dtype=np.float32).reshape(n, 3)
            fz_vals = np.frombuffer(f.read(n * 4), dtype=np.float32)
        else:
            points = np.frombuffer(f.read(n * 8), dtype=np.float32).reshape(n, 2)
            fz_vals = None
        fx_vals = np.frombuffer(f.read(n * 4), dtype=np.float32)
        fy_vals = np.frombuffer(f.read(n * 4), dtype=np.float32)
    if dim == "3D":
        return points, fz_vals, fx_vals, fy_vals
    return points, fx_vals, fy_vals


def apply(query, points, fx_vals, fy_vals, fz_vals=None) -> tuple:
    """最近邻查找"""
    dists = np.sum((points - np.array(query, dtype=np.float32)) ** 2, axis=1)
    idx = np.argmin(dists)
    if fz_vals is not None:
        return float(fz_vals[idx]), float(fx_vals[idx]), float(fy_vals[idx])
    return float(fx_vals[idx]), float(fy_vals[idx])


# ==================== 拟合标定 ====================

def build_fit_model(points, fx_vals, fy_vals, fz_vals=None, dim="3D"):
    """多项式拟合"""
    if dim == "3D":
        s = points[:, 0]; x = points[:, 1]; y = points[:, 2]
        A = np.column_stack([np.ones(len(points)), s, x, y, s*s, s*x, s*y, x*x, x*y, y*y])
        coef_fx, _, _, _ = np.linalg.lstsq(A, fx_vals, rcond=None)
        coef_fy, _, _, _ = np.linalg.lstsq(A, fy_vals, rcond=None)
        coef_fz, _, _, _ = np.linalg.lstsq(A, fz_vals, rcond=None)
        return coef_fz, coef_fx, coef_fy
    else:
        x = points[:, 0]; y = points[:, 1]
        A = np.column_stack([np.ones(len(points)), x, y, x*x, x*y, y*y])
        coef_fx, _, _, _ = np.linalg.lstsq(A, fx_vals, rcond=None)
        coef_fy, _, _, _ = np.linalg.lstsq(A, fy_vals, rcond=None)
        return coef_fx, coef_fy


def apply_fit(query, coefs, dim="3D") -> tuple:
    """拟合标定"""
    if dim == "3D":
        s, x, y = query
        basis = np.array([1, s, x, y, s*s, s*x, s*y, x*x, x*y, y*y])
    else:
        x, y = query
        basis = np.array([1, x, y, x*x, x*y, y*y])
    return tuple(float(np.dot(c, basis)) for c in coefs)


def save_fit_model(coefs, path: str):
    """保存拟合系数到 .bin"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        for c in coefs:
            f.write(np.array(c, dtype=np.float64).tobytes())
    print(f"  拟合模型已保存至: {path} ({os.path.getsize(path)} 字节)")


def load_fit_model(path: str, dim: str = "3D") -> tuple:
    """加载拟合系数"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.float64)
    if dim == "3D":
        return data[:10], data[10:20], data[20:30]
    return data[:6], data[6:12]


# ==================== 运行 ====================
if __name__ == "__main__":
    out_dir = os.path.dirname(CAL_CSV_PATH)

    try:
        if CAL_DIM == "3D":
            points, fz_vals, fx_vals, fy_vals = build_lookup_from_csv(
                CAL_CSV_PATH, mode=CAL_MODE, force_bin=CAL_FORCE_BIN, dim="3D")
            save_lookup(points, fx_vals, fy_vals, os.path.join(out_dir, "cal_lookup.bin"), fz_vals=fz_vals)
            if CAL_DO_FIT:
                coefs = build_fit_model(points, fx_vals, fy_vals, fz_vals=fz_vals, dim="3D")
                save_fit_model(coefs, os.path.join(out_dir, "cal_fit.bin"))
                labels = ["1", "s", "x", "y", "s²", "sx", "sy", "x²", "xy", "y²"]
                for name, coef in zip(["Fz", "Fx", "Fy"], coefs):
                    terms = " + ".join(f"{c:.4f}*{l}" for c, l in zip(coef, labels))
                    print(f"  {name} = {terms}")
        else:
            points, fx_vals, fy_vals = build_lookup_from_csv(
                CAL_CSV_PATH, mode=CAL_MODE, force_bin=CAL_FORCE_BIN, dim="2D")
            save_lookup(points, fx_vals, fy_vals, os.path.join(out_dir, "cal_lookup.bin"))
            if CAL_DO_FIT:
                coefs = build_fit_model(points, fx_vals, fy_vals, dim="2D")
                save_fit_model(coefs, os.path.join(out_dir, "cal_fit.bin"))
                labels = ["1", "x", "y", "x²", "xy", "y²"]
                for name, coef in zip(["Fx", "Fy"], coefs):
                    terms = " + ".join(f"{c:.4f}*{l}" for c, l in zip(coef, labels))
                    print(f"  {name} = {terms}")
    except Exception as e:
        print(f"  构建失败: {e}")
        sys.exit(1)
