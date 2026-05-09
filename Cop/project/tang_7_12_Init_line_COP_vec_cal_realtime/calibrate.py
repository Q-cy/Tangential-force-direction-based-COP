"""
CoP 位移 → 切向力 标定模块

模型: F = K @ d + b  (2x2 线性矩阵)
用法:
  拟合: python calibrate.py <csv_path>
  应用: from calibrate import load_coeffs, apply
"""

import os
import sys
import csv
import numpy as np


def fit_from_csv(csv_path: str):
    """
    从 CSV 读取配对数据 (dx, dy, Fx, Fy)，用最小二乘拟合 2x2 矩阵 K 和偏置 b。
    返回 (K, b, r2, rms)
    """
    rows_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                dx = float(row["delta_CoP_X"])
                dy = float(row["delta_CoP_Y"])
                fx = float(row["delta_Force_X"])
                fy = float(row["delta_Force_Y"])
                rows_data.append((dx, dy, fx, fy))
            except (KeyError, ValueError):
                continue

    if len(rows_data) < 10:
        raise ValueError(f"有效数据点不足（{len(rows_data)} < 10），请检查CSV文件")

    data = np.array(rows_data)  # (N, 4)
    dx_arr = data[:, 0]
    dy_arr = data[:, 1]
    fx_arr = data[:, 2]
    fy_arr = data[:, 3]

    # 构造设计矩阵: [dx, dy, 1] → 拟合 Fx; 同理 Fy
    A = np.column_stack([dx_arr, dy_arr, np.ones(len(data))])

    # 最小二乘拟合 Fx
    coeffs_x, residuals_x, rank_x, _ = np.linalg.lstsq(A, fx_arr, rcond=None)
    k_xx, k_xy, b_x = coeffs_x

    # 最小二乘拟合 Fy
    coeffs_y, residuals_y, rank_y, _ = np.linalg.lstsq(A, fy_arr, rcond=None)
    k_yx, k_yy, b_y = coeffs_y

    K = np.array([[k_xx, k_xy],
                   [k_yx, k_yy]])
    b = np.array([b_x, b_y])

    # R² 计算
    fx_pred = A @ coeffs_x
    fy_pred = A @ coeffs_y

    ss_res_x = np.sum((fx_arr - fx_pred) ** 2)
    ss_tot_x = np.sum((fx_arr - np.mean(fx_arr)) ** 2)
    r2_x = 1 - ss_res_x / ss_tot_x if ss_tot_x > 0 else 0

    ss_res_y = np.sum((fy_arr - fy_pred) ** 2)
    ss_tot_y = np.sum((fy_arr - np.mean(fy_arr)) ** 2)
    r2_y = 1 - ss_res_y / ss_tot_y if ss_tot_y > 0 else 0

    rms_x = np.sqrt(ss_res_x / len(data))
    rms_y = np.sqrt(ss_res_y / len(data))

    print(f"\n{'='*50}")
    print(f"  标定拟合结果")
    print(f"  数据点数: {len(data)}")
    print(f"{'='*50}")
    print(f"  标定矩阵 K (2x2):")
    print(f"    [[{k_xx:8.4f}  {k_xy:8.4f}]")
    print(f"     [{k_yx:8.4f}  {k_yy:8.4f}]]")
    print(f"  偏置 b: [{b_x:.4f}, {b_y:.4f}]")
    print(f"{'='*50}")
    print(f"  Fx: R² = {r2_x:.4f}, RMS = {rms_x:.4f} N")
    print(f"  Fy: R² = {r2_y:.4f}, RMS = {rms_y:.4f} N")
    print(f"{'='*50}\n")

    return K, b, (r2_x, r2_y), (rms_x, rms_y)


def save_coeffs(K: np.ndarray, b: np.ndarray, path: str):
    """保存标定系数到 .npz 文件"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    np.savez(path, K=K, b=b)
    print(f"  标定系数已保存至: {path}")


def load_coeffs(path: str) -> tuple[np.ndarray, np.ndarray]:
    """从 .npz 文件加载标定系数，返回 (K, b)"""
    data = np.load(path)
    return data["K"], data["b"]


def apply(dx: float, dy: float, K: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    应用标定模型: [Fx_cal, Fy_cal] = K @ [dx, dy] + b
    返回 (Fx_cal, Fy_cal)
    """
    cal = K @ np.array([dx, dy]) + b
    return float(cal[0]), float(cal[1])


# ==================== CLI 入口 ====================
DEFAULT_SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"

def _resolve_path(arg: str) -> str:
    """解析用户输入的路径：纯数字→data_N.csv，纯文件名→SAVE_DIR/文件名，否则原样"""
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
        print("  python calibrate.py /full/path.csv → 完整路径")
        sys.exit(1)
        # python /home/qcy/Project/code/Tangential/finger_tang_7_12/Cop/project/tang_7_12_Init_line_COP_vec_cal_realtime/calibrate.py /home/qcy/Project/data/2.PZT_tangential/weight/test/data_1.csv

    csv_path = _resolve_path(sys.argv[1])
    out_dir = os.path.dirname(csv_path)
    out_path = os.path.join(out_dir, "cal_coeffs.npz")

    try:
        K, b, (r2_x, r2_y), (rms_x, rms_y) = fit_from_csv(csv_path)
        save_coeffs(K, b, out_path)
    except Exception as e:
        print(f"  拟合失败: {e}")
        sys.exit(1)
