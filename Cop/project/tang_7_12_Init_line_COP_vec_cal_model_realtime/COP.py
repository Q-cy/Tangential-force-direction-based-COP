"""
CoP 压力中心计算核心模块
功能：基线减除、CoP计算、初始稳定点判断、方向向量滤波
"""

import numpy as np
from collections import deque
import threading


# ===================== 算法参数（仅与CoP计算相关）=====================
COP_STABILITY_FRAMES_REQUIRED = 5       # 初始稳定帧数
TOTAL_PRESSURE_LOW_THRESHOLD = 500      # 低压判定阈值
SENSOR_ROWS = 12                        # 传感器阵列行数
SENSOR_COLS = 7                         # 传感器阵列列数


# ===================== 直线方向稳定判断参数 =====================
LINE_DIST_THRESHOLD = 0.1               # 点到直线最大允许距离 (CoP单位)
DIR_DOT_THRESHOLD = 0.7                 # 方向一致性最小点积 cos(夹角)


# ===================== 线程安全全局状态 =====================
first_frame = None                      # 第一帧基线
first_frame_lock = threading.Lock()     # 线程锁

first_contact_CoP_x = None              # 初始接触点X
first_contact_CoP_y = None              # 初始接触点Y
contact_initialized = False             # 初始点是否稳定

# 修正后的缓冲器，用于存储候选的稳定CoP点序列
initial_cop_x_buffer = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)
initial_cop_y_buffer = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)

# 之前的 last_stable_cop_x, last_stable_cop_y, cop_stability_counter 已经不再需要
# 因为新的逻辑直接依赖于 initial_cop_x_buffer 的内容和长度来判断稳定性。

total_pressure_low_counter = 0           # 压力低于阈值计数器

adc_filtered_dir = None                  # 滤波后的方向向量
grad_table_data = np.zeros((12, 7, 2))   # 梯度表（用于绘图）
grad_table_lock = threading.Lock()       # 新增：梯度表读写锁（关键！）


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
    # global 声明要修改全局变量
    global adc_filtered_dir, first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter
    global initial_cop_x_buffer, initial_cop_y_buffer, grad_table_data

    adc_filtered_dir = None
    first_contact_CoP_x = None
    first_contact_CoP_y = None
    contact_initialized = False
    total_pressure_low_counter = 0
    initial_cop_x_buffer.clear()         # 清空缓冲器
    initial_cop_y_buffer.clear()         # 清空缓冲器
    with grad_table_lock:
        grad_table_data.fill(0)


# ===================== 核心CoP计算 =====================
def compute_pressure_direction(baseline_subtracted_frame):
    """
    输入：基线减除后的84通道压力数据
    输出：方向、幅值、CoP坐标、初始点、偏移量等
    """
    global adc_filtered_dir, grad_table_data
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter
    global initial_cop_x_buffer, initial_cop_y_buffer

    rows, cols = SENSOR_ROWS, SENSOR_COLS
    frame_flat = np.asarray(baseline_subtracted_frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)

    # 计算梯度（用于可视化）
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
    with grad_table_lock:
        grad_table_data[:] = grad[:]

    # 总压力判断
    total_pressure = np.sum(frame2d)
    if total_pressure < TOTAL_PRESSURE_LOW_THRESHOLD:
        total_pressure_low_counter += 1
    else:
        total_pressure_low_counter = 0

    # 连续低压 → 重置
    if total_pressure_low_counter >= COP_STABILITY_FRAMES_REQUIRED:
        reset_cop_state()
        # 返回默认值，表示无有效CoP或已重置
        return 0.0, 0.0, 0, rows-1, 0, cols-1, 0.0, 0.0, 0.0, 0.0  # 10个值

    if total_pressure == 0:
        return 0.0, 0.0, 0, rows-1, 0, cols-1, 0.0, 0.0, 0.0, 0.0  # 10个值

    # 已建立初始接触但当前压力过低 → 跳过噪声CoP计算，返回零偏移
    if contact_initialized and total_pressure < TOTAL_PRESSURE_LOW_THRESHOLD:
        return (first_contact_CoP_x, first_contact_CoP_y,
                0, rows-1, 0, cols-1,
                0.0, 0.0,
                first_contact_CoP_x, first_contact_CoP_y)

    # 计算CoP中心
    x_grid = np.tile(np.arange(cols), (rows, 1))
    y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols)
    cop_x = np.sum(frame2d * x_grid) / total_pressure
    cop_y = np.sum(frame2d * y_grid) / total_pressure

    delta_CoP_x = 0.0
    delta_CoP_y = 0.0
    base_CoP_x_for_plot = cop_x
    base_CoP_y_for_plot = cop_y

    # ============ 初始点稳定判断 (修改区域) ============
    if not contact_initialized:
        # 始终将当前CoP添加到缓冲器
        initial_cop_x_buffer.append(cop_x)
        initial_cop_y_buffer.append(cop_y)

        is_current_sequence_stable = True  # 假设当前序列是稳定的，直到被证明不稳定

        # 只有当缓冲器中有至少2个点时，才能开始定义和检查直线
        if len(initial_cop_x_buffer) >= 2:
            # 使用缓冲器中的前两个点定义参考直线和方向
            p0x, p0y = initial_cop_x_buffer[0], initial_cop_y_buffer[0]
            p1x, p1y = initial_cop_x_buffer[1], initial_cop_y_buffer[1]

            dir_ref_x = p1x - p0x
            dir_ref_y = p1y - p0y
            dir_ref_len = np.hypot(dir_ref_x, dir_ref_y)

            # 如果前两个点相同（或非常接近），无法定义有意义的方向，则认为不稳定
            # 这种情况会触发序列重置
            if dir_ref_len < 1e-4:  # 使用一个非常小的阈值避免浮点数问题
                is_current_sequence_stable = False
            else:
                # 归一化参考方向向量
                norm_dir_ref_x = dir_ref_x / dir_ref_len
                norm_dir_ref_y = dir_ref_y / dir_ref_len

                # 遍历缓冲器中所有点（从第三个点开始，因为前两个点定义了线）
                for i in range(2, len(initial_cop_x_buffer)):
                    current_px, current_py = initial_cop_x_buffer[i], initial_cop_y_buffer[i]
                    prev_px, prev_py = initial_cop_x_buffer[i-1], initial_cop_y_buffer[i-1]

                    # --- 1. 检查当前点到参考直线的距离 ---
                    # 直线方程为 (y - y0)(x1 - x0) - (x - x0)(y1 - y0) = 0
                    # 点到直线距离公式 |Ax + By + C| / sqrt(A^2 + B^2)
                    # (x0,y0) 是 p0, (x1,y1) 是 p1, (x,y) 是 current_p
                    # A = -(y1 - y0), B = (x1 - x0), C = -x0(y1-y0) + y0(x1-x0) = y0*x1 - y0*x0 - x0*y1 + x0*y0 = y0*x1 - x0*y1
                    # A = -dir_ref_y, B = dir_ref_x
                    # C = y0*p1x - x0*p1y (或者用更通用的形式 (p0y * p1x - p0x * p1y))
                    # 或者更简洁的：利用向量叉乘的模 = 平行四边形面积
                    # 距离 = |(p1 - p0) x (current_p - p0)| / |p1 - p0|
                    cross_product_val = abs((p1x - p0x) * (current_py - p0y) - (p1y - p0y) * (current_px - p0x))
                    line_dist = cross_product_val / dir_ref_len

                    if line_dist > LINE_DIST_THRESHOLD:
                        is_current_sequence_stable = False
                        break  # 该点距离直线太远，序列不稳定

                    # --- 2. 检查当前CoP移动方向与参考方向的一致性 ---
                    curr_segment_dir_x = current_px - prev_px
                    curr_segment_dir_y = current_py - prev_py
                    curr_segment_len = np.hypot(curr_segment_dir_x, curr_segment_dir_y)

                    if curr_segment_len > 1e-4:  # 只有当有实际移动时才检查方向
                        norm_curr_segment_dir_x = curr_segment_dir_x / curr_segment_len
                        norm_curr_segment_dir_y = curr_segment_dir_y / curr_segment_len

                        dot_product = norm_dir_ref_x * norm_curr_segment_dir_x + norm_dir_ref_y * norm_curr_segment_dir_y

                        if dot_product < DIR_DOT_THRESHOLD:
                            is_current_sequence_stable = False
                            break  # 方向不一致，序列不稳定
                    # 如果 curr_segment_len 接近0，表示当前点与前一个点非常接近，
                    # 这可以视为沿着直线运动的一个“暂停”或“慢速移动”，仍视为稳定。

        # 根据稳定性判断结果处理缓冲器
        if not is_current_sequence_stable:
            # 如果当前序列不稳定，则清空缓冲器，并以当前CoP作为新序列的第一个点
            initial_cop_x_buffer.clear()
            initial_cop_y_buffer.clear()
            initial_cop_x_buffer.append(cop_x)  # 重新开始积累
            initial_cop_y_buffer.append(cop_y)

        # 如果序列稳定且缓冲器已满（达到所需帧数），则确定初始CoP
        elif len(initial_cop_x_buffer) == COP_STABILITY_FRAMES_REQUIRED:
            first_contact_CoP_x = initial_cop_x_buffer[0]  # 取稳定序列的第一个点
            first_contact_CoP_y = initial_cop_y_buffer[0]
            contact_initialized = True

            # 确定初始点后，清空缓冲器，不再需要它
            initial_cop_x_buffer.clear()
            initial_cop_y_buffer.clear()

    # ========== 计算偏移量 ==========
    else:  # contact_initialized 为 True
        delta_CoP_x = cop_x - first_contact_CoP_x
        delta_CoP_y =  first_contact_CoP_y-cop_y 
        base_CoP_x_for_plot = first_contact_CoP_x
        base_CoP_y_for_plot = first_contact_CoP_y

    # 可以在这里更新adc_filtered_dir，但根据当前代码，它在计算cop_x/y之后没有用到
    # 如果需要，可以在这里计算并滤波方向

    return (cop_x, cop_y,
            0, rows-1, 0, cols-1,              # 绘图用的范围 (min_y, max_y, min_x, max_x)
            delta_CoP_x, delta_CoP_y,
            base_CoP_x_for_plot, base_CoP_y_for_plot)