"""
CoP 压力中心计算核心模块
功能：基线减除、CoP计算、初始点直接锁定、方向向量滤波
修改：去掉直线稳定判断，第一个有效接触点直接作为初始CoP
"""

import numpy as np
import threading


# ===================== 算法参数（仅与CoP计算相关）=====================
THRESHOLD = 200
DIR_SMOOTH_ALPHA = 0.15                 # 方向滤波系数
COP_STABILITY_FRAMES_REQUIRED = 5       # 低压重置帧数（仅用于低压判断）
TOTAL_PRESSURE_LOW_THRESHOLD = 500      # 有效接触压力阈值
SENSOR_ROWS = 12                        # 传感器阵列行数
SENSOR_COLS = 7                         # 传感器阵列列数


# ===================== 线程安全全局状态 =====================
first_frame = None                      # 第一帧基线
first_frame_lock = threading.Lock()     # 线程锁

first_contact_CoP_x = None              # 初始接触点X
first_contact_CoP_y = None              # 初始接触点Y
contact_initialized = False             # 初始点是否已锁定

total_pressure_low_counter = 0           # 压力低于阈值计数器

adc_filtered_dir = None                  # 滤波后的方向向量
grad_table_data = np.zeros((12, 7, 2))   # 梯度表（用于绘图）


# ===================== 基线减除 =====================
def subtract_baseline(current_frame):
    """
    用第一帧作为基线，减去背景
    """
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()
    
    with first_frame_lock:
        if first_frame is None:
            first_frame = current_frame.copy()
    
    diff = current_frame - first_frame
    return np.clip(diff, 0, None)        # 限制差异为非负


# ===================== 重置CoP状态 =====================
def reset_cop_state():
    """
    压力过低/离开接触面 → 重置所有状态
    """
    global adc_filtered_dir, first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter, grad_table_data

    adc_filtered_dir = None
    first_contact_CoP_x = None
    first_contact_CoP_y = None
    contact_initialized = False
    total_pressure_low_counter = 0
    grad_table_data = np.zeros((12, 7, 2))


# ===================== 核心CoP计算 =====================
def compute_pressure_direction(baseline_subtracted_frame):
    """
    输入：基线减除后的84通道压力数据
    输出：方向、幅值、CoP坐标、初始点、偏移量等
    修改：第一个有效接触点直接作为初始CoP
    """
    global adc_filtered_dir, grad_table_data
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter

    rows, cols = SENSOR_ROWS, SENSOR_COLS
    frame_flat = np.asarray(baseline_subtracted_frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)

    # 计算梯度（用于可视化，保留原逻辑）
    grad = np.zeros((rows, cols, 2), dtype=np.float32)
    for y in range(rows):
        for x in range(cols):
            val = frame2d[y, x]
            left = frame2d[y, x-1] if x-1 >= 0 else val
            right = frame2d[y, x+1] if x+1 < cols else val
            up = frame2d[y-1, x] if y-1 >= 0 else val
            down = frame2d[y+1, x] if y+1 < rows else val
            gx = right - left
            gy = up - down
            grad[y, x] = (gx, gy)
    grad_table_data = grad.copy()

    # 总压力判断：有效接触 / 低压
    total_pressure = np.sum(frame2d)
    if total_pressure < TOTAL_PRESSURE_LOW_THRESHOLD:
        total_pressure_low_counter += 1
    else:
        total_pressure_low_counter = 0

    # 连续低压 → 重置所有状态（包括初始CoP）
    if total_pressure_low_counter >= COP_STABILITY_FRAMES_REQUIRED:
        reset_cop_state()
        return 0.0, 0.0, 0, rows-1, 0, cols-1, 0.0, 0.0, 0.0, 0.0

    if total_pressure == 0:
        return 0.0, 0.0, 0, rows-1, 0, cols-1, 0.0, 0.0, 0.0, 0.0

    # 核心：计算CoP中心（原逻辑不变）
    x_grid = np.tile(np.arange(cols), (rows, 1))
    y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols)
    cop_x = np.sum(frame2d * x_grid) / total_pressure
    cop_y = np.sum(frame2d * y_grid) / total_pressure

    delta_CoP_x = 0.0
    delta_CoP_y = 0.0
    base_CoP_x_for_plot = cop_x
    base_CoP_y_for_plot = cop_y

    # ===================== 核心修改：直接锁定第一个有效CoP =====================
    # 只要还没锁定初始点，且当前是有效压力 → 直接将当前CoP设为初始接触点
    if not contact_initialized:
        first_contact_CoP_x = cop_x
        first_contact_CoP_y = cop_y
        contact_initialized = True

    # 初始点已锁定 → 计算相对于初始点的偏移量（原逻辑不变）
    else:
        delta_CoP_x = cop_x - first_contact_CoP_x
        delta_CoP_y = cop_y - first_contact_CoP_y
        base_CoP_x_for_plot = first_contact_CoP_x
        base_CoP_y_for_plot = first_contact_CoP_y

    # 返回值 数量/顺序 完全不变，兼容主程序
    return (cop_x, cop_y,
            0, rows-1, 0, cols-1,              # 绘图用范围
            delta_CoP_x, delta_CoP_y,
            base_CoP_x_for_plot, base_CoP_y_for_plot)