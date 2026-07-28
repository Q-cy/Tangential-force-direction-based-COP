"""
CoP 位移 (+ 总压力) → 力 标定模块

CAL_DIM="2D": 输入 (delta_CoP_X, delta_CoP_Y) → 输出 (delta_Force_X, delta_Force_Y)
CAL_DIM="3D": 输入 (adc_sum, delta_CoP_X, delta_CoP_Y) → 输出 (delta_Force_Z, delta_Force_X, delta_Force_Y)

仅负责查找表 (lookup / discrete) 标定。拟合标定由 fit.py 统一处理。
"""

import os
import sys
import csv
import numpy as np

# ===================== 可调参数 =====================
CAL_TRAIN_CSV = "/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0615_6.csv"  # 训练数据（构建标定模型）
CAL_CSV_PATH = "/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0615_7.csv"   # 被标定的 CSV
CAL_DATA_MODE = "discrete"  # "continuous"=原始数据, "discrete"=按力值分组
CAL_MODE = "discrete"         # "lookup"=最近邻, "discrete"=双线性插值
CAL_DIM = "2D"                # "2D"=仅切向力(Fx,Fy), "3D"=三维力(Fz,Fx,Fy)
CAL_FORCE_BIN = 0.5           # discrete模式的力分组间隔(N)
CAL_TRAIN_VALID_ONLY = True    # True=构建模型时只用valid!=0的行
CAL_OFFLINE_VALID_ONLY = True  # True=离线标定时只标定valid!=0的行


def build_lookup_from_csv(csv_path: str, mode: str = "continuous", force_bin: float = 0.2, dim: str = "3D", valid_only: bool = True):
    """
    读取 CSV，返回标定数据
    dim="2D": (points[N,2], fx_vals, fy_vals)
    dim="3D": (points[N,3], fz_vals, fx_vals, fy_vals)
    valid_only: True=只用valid!=0的行, False=用所有行
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            try:
                if valid_only and float(row.get("valid", 0)) == 0:
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


def build_discrete_grid(points: np.ndarray, fx_vals: np.ndarray, fy_vals: np.ndarray):
    """将散乱离散点构建为 (dx, dy) 规则网格，用于双线性插值"""
    dx_vals = np.unique(np.round(points[:, 0], 6))
    dy_vals = np.unique(np.round(points[:, 1], 6))

    fx_grid = np.full((len(dy_vals), len(dx_vals)), np.nan, dtype=np.float32)
    fy_grid = np.full((len(dy_vals), len(dx_vals)), np.nan, dtype=np.float32)

    for i in range(len(points)):
        xi = np.argmin(np.abs(dx_vals - points[i, 0]))
        yi = np.argmin(np.abs(dy_vals - points[i, 1]))
        fx_grid[yi, xi] = fx_vals[i]
        fy_grid[yi, xi] = fy_vals[i]

    # 填充 NaN（用最近非NaN邻居）
    for grid in (fx_grid, fy_grid):
        nan_mask = np.isnan(grid)
        if np.all(nan_mask):
            continue
        from scipy.ndimage import distance_transform_edt
        _, indices = distance_transform_edt(nan_mask, return_indices=True)
        grid[nan_mask] = grid[tuple(indices[:, nan_mask])]

    return dx_vals, dy_vals, fx_grid, fy_grid


def apply_discrete(dx: float, dy: float, dx_grid: np.ndarray, dy_grid: np.ndarray,
                   fx_grid: np.ndarray, fy_grid: np.ndarray) -> tuple:
    """规则网格双线性插值：结果连续平滑"""
    # 钳位到网格范围
    dx = np.clip(dx, dx_grid[0], dx_grid[-1])
    dy = np.clip(dy, dy_grid[0], dy_grid[-1])

    # 找所在格子
    xi = np.searchsorted(dx_grid, dx, side='right') - 1
    yi = np.searchsorted(dy_grid, dy, side='right') - 1
    xi = np.clip(xi, 0, len(dx_grid) - 2)
    yi = np.clip(yi, 0, len(dy_grid) - 2)

    # 归一化坐标
    t = (dx - dx_grid[xi]) / (dx_grid[xi + 1] - dx_grid[xi]) if dx_grid[xi + 1] != dx_grid[xi] else 0.0
    u = (dy - dy_grid[yi]) / (dy_grid[yi + 1] - dy_grid[yi]) if dy_grid[yi + 1] != dy_grid[yi] else 0.0

    # 双线性插值
    fx = float(fx_grid[yi, xi]*(1-t)*(1-u) + fx_grid[yi, xi+1]*t*(1-u) +
               fx_grid[yi+1, xi]*(1-t)*u + fx_grid[yi+1, xi+1]*t*u)
    fy = float(fy_grid[yi, xi]*(1-t)*(1-u) + fy_grid[yi, xi+1]*t*(1-u) +
               fy_grid[yi+1, xi]*(1-t)*u + fy_grid[yi+1, xi+1]*t*u)
    return fx, fy


# ==================== 离线标定 ====================
def offline_calibrate_csv(train_csv: str, target_csv: str, dim: str = "2D", data_mode: str = "continuous", mode: str = "lookup", force_bin: float = 0.2, train_valid_only: bool = True, offline_valid_only: bool = True):
    """用训练 CSV 构建标定模型，标定目标 CSV，写回 Fx_cal, Fy_cal 列"""
    # 用训练数据构建标定模型
    if dim == "2D":
        points, fx_vals, fy_vals = build_lookup_from_csv(train_csv, mode=data_mode, force_bin=force_bin, dim="2D", valid_only=train_valid_only)
    else:
        points, fz_vals, fx_vals, fy_vals = build_lookup_from_csv(train_csv, mode=data_mode, force_bin=force_bin, dim="3D", valid_only=train_valid_only)

    # 读取目标 CSV 全部数据
    with open(target_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [name.strip() for name in reader.fieldnames]
        rows = list(reader)

    # 找 Fx_cal, Fy_cal 列索引
    if "Fx_cal" not in header:
        header.extend(["Fx_cal", "Fy_cal", "Force_cal_mag", "Force_cal_angle"])
        for row in rows:
            row["Fx_cal"] = ""
            row["Fy_cal"] = ""
            row["Force_cal_mag"] = ""
            row["Force_cal_angle"] = ""

    # discrete 模式：构建规则网格
    disc_dx_grid = disc_dy_grid = disc_fx_grid = disc_fy_grid = None
    if mode == "discrete":
        disc_dx_grid, disc_dy_grid, disc_fx_grid, disc_fy_grid = build_discrete_grid(points, fx_vals, fy_vals)

    # 对每行应用标定
    cnt = 0
    for row in rows:
        if offline_valid_only and float(row.get("valid", 0)) == 0:
            continue
        try:
            dx = float(row["delta_CoP_X"])
            dy = float(row["delta_CoP_Y"])
        except (KeyError, ValueError):
            continue

        if mode == "discrete":
            cal_fx, cal_fy = apply_discrete(dx, dy, disc_dx_grid, disc_dy_grid, disc_fx_grid, disc_fy_grid)
        else:
            cal_fx, cal_fy = apply([dx, dy], points, fx_vals, fy_vals)

        cal_mag = float(np.hypot(cal_fx, cal_fy))
        cal_angle = float(np.degrees(np.arctan2(cal_fy, cal_fx + 1e-8)))
        if cal_angle < 0:
            cal_angle += 360

        row["Fx_cal"] = f"{cal_fx:.6f}"
        row["Fy_cal"] = f"{cal_fy:.6f}"
        row["Force_cal_mag"] = f"{cal_mag:.6f}"
        row["Force_cal_angle"] = f"{cal_angle:.6f}"
        cnt += 1

    # 写回 CSV
    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  离线标定完成: {cnt} 行已更新 → {target_csv}")


# ==================== 运行 ====================
if __name__ == "__main__":
    out_dir = os.path.dirname(CAL_TRAIN_CSV)

    try:
        if CAL_DIM == "3D":
            points, fz_vals, fx_vals, fy_vals = build_lookup_from_csv(
                CAL_TRAIN_CSV, mode=CAL_DATA_MODE, force_bin=CAL_FORCE_BIN, dim="3D", valid_only=CAL_TRAIN_VALID_ONLY)
            save_lookup(points, fx_vals, fy_vals, os.path.join(out_dir, "cal_lookup.bin"), fz_vals=fz_vals)
        else:
            points, fx_vals, fy_vals = build_lookup_from_csv(
                CAL_TRAIN_CSV, mode=CAL_DATA_MODE, force_bin=CAL_FORCE_BIN, dim="2D", valid_only=CAL_TRAIN_VALID_ONLY)
            save_lookup(points, fx_vals, fy_vals, os.path.join(out_dir, "cal_lookup.bin"))

        # 离线标定：用生成的查找表回写 CSV
        offline_calibrate_csv(CAL_TRAIN_CSV, CAL_CSV_PATH, dim=CAL_DIM, data_mode=CAL_DATA_MODE, mode=CAL_MODE, force_bin=CAL_FORCE_BIN, train_valid_only=CAL_TRAIN_VALID_ONLY, offline_valid_only=CAL_OFFLINE_VALID_ONLY)

    except Exception as e:
        print(f"  构建失败: {e}")
        sys.exit(1)
