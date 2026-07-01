# file_name: consistence.py
"""84通道压阻阵列传感器一致性标定
Mode 1: 最大量程均值缩放 — 单CSV，取各通道均值，缩放至统一范围
Mode 2: 多文件多项式拟合 — 多CSV，每文件一力值，逐通道多项式拟合
"""

import csv
import numpy as np


def _read_csv_filtered(csv_path, state_col='CoP_state', state_val=2):
    """读取CSV，返回筛选后的 {col: [values]} 字典和列名列表"""
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames]
        rows = []
        state_idx = None
        for i, h in enumerate(reader.fieldnames):
            if h.strip() == state_col:
                state_idx = i
                break
        # 用 csv.reader 重新读取以获取原始值
        f.seek(0)
        next(csv.reader(f))  # skip header
        reader2 = csv.reader(f)
        for row in reader2:
            try:
                if state_idx is not None and int(float(row[state_idx].strip())) == state_val:
                    rows.append([v.strip() for v in row])
            except (ValueError, IndexError):
                continue
    # 构建列数据
    col_data = {}
    for i, h in enumerate(headers):
        col_data[h] = [float(row[i]) for row in rows if row[i].lower() != 'nan']
    return headers, col_data


def _get_ch_columns(headers, ch_count=84):
    ch_cols = []
    for i in range(1, ch_count + 1):
        col = f'ch{i}'
        if col in headers:
            ch_cols.append(col)
    return ch_cols


# ==================== Mode 1: 最大量程均值缩放 ====================

def calibrate_max_range(csv_path, state_col='CoP_state', state_val=2,
                        target_range=(0, 4000), ch_count=84):
    """读取单CSV，筛选CoP_state==2，取各通道均值，缩放至统一范围。

    Returns:
        dict: {ch_name: {'scale': float, 'offset': float}}
    """
    headers, col_data = _read_csv_filtered(csv_path, state_col, state_val)
    ch_cols = _get_ch_columns(headers, ch_count)
    if len(ch_cols) != ch_count:
        print(f"warning: 期望 {ch_count} 通道，实际找到 {len(ch_cols)}")

    coeffs = {}
    target_min, target_max = target_range
    target_span = target_max - target_min

    for col in ch_cols:
        vals = col_data.get(col, [0])
        if not vals:
            coeffs[col] = {'scale': 1.0, 'offset': target_min}
            continue
        med_val = np.median(vals)

        if med_val > 0:
            scale = target_span / med_val
        else:
            scale = 1.0

        coeffs[col] = {'scale': scale, 'offset': target_min}

    return coeffs


# ==================== Mode 2: 多文件多项式拟合 ====================

def calibrate_poly_fit(csv_path, state_col='CoP_state', state_val=2,
                       force_col='Fz', degree=2, ch_count=84):
    """读取单个CSV，按 Fz 四舍五入分组，逐通道多项式拟合。

    Args:
        csv_path: CSV文件路径
        state_col: 状态列名
        state_val: 筛选值
        force_col: 力值列名
        degree: 多项式阶数
        ch_count: 通道数

    Returns:
        dict: {ch_name: {'coeffs': [c0, c1, ..., cN]}}
              多项式: force = c0 + c1*adc + c2*adc^2 + ...
    """
    headers, col_data = _read_csv_filtered(csv_path, state_col, state_val)
    ch_cols = _get_ch_columns(headers, ch_count)

    # 取 Fz 四舍五入到整数
    fz_vals = col_data.get(force_col, [])
    if not fz_vals:
        print(f"warning: 无 {force_col} 数据")
        return {}

    force_labels = [round(v) for v in fz_vals]

    # 按力标签分组
    groups = {}
    for idx, label in enumerate(force_labels):
        groups.setdefault(label, []).append(idx)

    # 每组内各通道求均值
    ch_data_points = {col: [] for col in ch_cols}
    force_points = []
    for label, indices in groups.items():
        force_points.append(label)
        for col in ch_cols:
            vals = [col_data[col][i] for i in indices if i < len(col_data.get(col, []))]
            ch_data_points[col].append(np.mean(vals) if vals else 0.0)

    # 逐通道多项式拟合
    coeffs = {}
    for col, adc_means in ch_data_points.items():
        if len(adc_means) < degree + 1:
            print(f"warning: {col} 数据点不足 (需要{degree+1}，实际{len(adc_means)})，跳过拟合")
            continue
        # force = poly(adc)
        poly = np.polyfit(adc_means, force_points, degree)
        coeffs[col] = {'coeffs': poly.tolist()}

    return coeffs


# ==================== 应用校准 ====================

def apply_consistence(ch_data, coeffs):
    """对 84 通道原始 ADC 数据应用一致性校准。

    Args:
        ch_data: list/array of 84 raw ADC values
        coeffs: calibrate_xxx 返回的字典

    Returns:
        np.ndarray: 84 通道校准后数据
    """
    ch_data = np.asarray(ch_data, dtype=float)
    calibrated = np.zeros(84, dtype=float)

    for i in range(84):
        col = f'ch{i + 1}'
        raw = ch_data[i]
        if col not in coeffs:
            calibrated[i] = raw
            continue

        c = coeffs[col]
        if 'scale' in c:
            calibrated[i] = raw * c['scale'] + c['offset']
        elif 'coeffs' in c:
            calibrated[i] = np.polyval(c['coeffs'], raw)

    return calibrated


# ==================== 保存/加载 ====================

def save_coeffs(coeffs, path):
    """保存标定系数到 .npy 文件"""
    np.save(path, coeffs, allow_pickle=True)
    print(f"coefficients saved to {path}")


def load_coeffs(path):
    """从 .npy 文件加载标定系数"""
    coeffs = np.load(path, allow_pickle=True).item()
    print(f"coefficients loaded from {path}")
    return coeffs


if __name__ == "__main__":
    import os
    CALIB_MODE = 1  # 1=最大量程均值缩放, 2=按力分组多项式拟合

    data_dir = "/home/qcy/Project/data/2.PZT_tangential/weight/test"
    out_dir = "/home/qcy/Project/data/2.PZT_tangential/weight/png"
    out_path = os.path.join(out_dir, "consistence_coeffs.npy")
    csv_path = os.path.join(data_dir, "COP_test_0630_55.csv")   #55

    if CALIB_MODE == 1:
        coeffs = calibrate_max_range(csv_path)
    elif CALIB_MODE == 2:
        coeffs = calibrate_poly_fit(csv_path)
    else:
        raise ValueError(f"unknown CALIB_MODE: {CALIB_MODE}")

    save_coeffs(coeffs, out_path)
