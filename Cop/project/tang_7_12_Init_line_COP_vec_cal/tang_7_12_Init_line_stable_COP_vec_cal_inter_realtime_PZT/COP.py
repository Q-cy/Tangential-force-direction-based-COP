"""
CoP 压力中心计算核心模块
功能：基线减除、CoP计算、初始稳定点判断、方向向量滤波
"""

import numpy as np
from collections import deque
import threading


# ===================== 算法参数（仅与CoP计算相关）=====================
COP_STABILITY_FRAME_CNT = 5            # 初始稳定所需连续帧数
COP_PRESSURE_LOW_THRESH = 500          # 低压判定阈值（ADC原始值之和）
COP_SENSOR_ROW_CNT = 12                # 传感器阵列行数
COP_SENSOR_COL_CNT = 7                 # 传感器阵列列数


# ===================== 直线方向稳定判断参数 =====================
COP_LINE_DIST_THRESH = 0.1             # 点到参考直线最大允许距离（CoP坐标单位）
COP_DIR_DOT_THRESH = 0.7               # 移动方向与参考方向一致性最小点积


# ===================== 二次静置精修参数 =====================
COP_POST_INIT_WINDOW_CNT = 100         # 初始CoP确定后精修监测帧数上限
COP_POST_INIT_STABLE_CNT = 50          # 精修阶段需连续保持不变的帧数
COP_POST_INIT_STABLE_THRESH = 0.1      # 精修判据：CoP偏移距离阈值


# ===================== 线程安全全局状态 =====================
g_cop_base_frame_arr = None            # 第一帧基线（84通道flat数组）
g_cop_base_frame_lock = threading.Lock()  # 基线读写锁

g_cop_contact_init_x = None            # 初始接触点CoP X坐标
g_cop_contact_init_y = None            # 初始接触点CoP Y坐标
g_cop_contact_init_flag = False        # 初始接触点是否已稳定确定

g_cop_init_x_buf = deque(maxlen=COP_STABILITY_FRAME_CNT)  # 候选初始CoP X序列缓冲
g_cop_init_y_buf = deque(maxlen=COP_STABILITY_FRAME_CNT)  # 候选初始CoP Y序列缓冲

g_cop_press_low_cnt = 0                # 连续低压帧计数器

# 二次静置精修状态
g_cop_post_init_frame_cnt = 0          # 精修阶段已监测帧数
g_cop_post_stable_cnt = 0              # 精修阶段连续满足静止判据的帧数
g_cop_post_refined_flag = False        # 精修是否已完成
g_cop_post_cand_x = None               # 精修候选静止点X
g_cop_post_cand_y = None               # 精修候选静止点Y

g_cop_filtered_dir = None              # 滤波后的方向向量（暂未使用）
g_cop_grad_table_arr = np.zeros((COP_SENSOR_ROW_CNT, COP_SENSOR_COL_CNT, 2))  # 梯度表(rows,cols,2)
g_cop_grad_table_lock = threading.Lock()  # 梯度表读写锁


# ===================== 基线减除 =====================
def subtract_baseline(raw_frame_arr):
    """
    用第一帧作为基线，减去背景。返回基线减除后的84通道数据。
    """
    global g_cop_base_frame_arr
    frame_flat_arr = np.array(raw_frame_arr, dtype=np.float32).flatten()

    with g_cop_base_frame_lock:
        if g_cop_base_frame_arr is None:
            g_cop_base_frame_arr = frame_flat_arr.copy()

    diff_arr = frame_flat_arr - g_cop_base_frame_arr
    return np.clip(diff_arr, 0, None)  # 截断负值为0


# ===================== 重置CoP状态 =====================
def reset_cop_state():
    """
    压力过低/离开接触面 → 重置所有状态
    """
    # global 声明要修改全局变量
    global g_cop_filtered_dir, g_cop_contact_init_x, g_cop_contact_init_y, g_cop_contact_init_flag
    global g_cop_press_low_cnt
    global g_cop_init_x_buf, g_cop_init_y_buf, g_cop_grad_table_arr
    global g_cop_post_init_frame_cnt, g_cop_post_stable_cnt, g_cop_post_refined_flag
    global g_cop_post_cand_x, g_cop_post_cand_y

    g_cop_filtered_dir = None
    g_cop_contact_init_x = None
    g_cop_contact_init_y = None
    g_cop_contact_init_flag = False
    g_cop_press_low_cnt = 0
    g_cop_init_x_buf.clear()
    g_cop_init_y_buf.clear()
    g_cop_post_init_frame_cnt = 0
    g_cop_post_stable_cnt = 0
    g_cop_post_refined_flag = False
    g_cop_post_cand_x = None
    g_cop_post_cand_y = None
    with g_cop_grad_table_lock:
        g_cop_grad_table_arr.fill(0)


# ===================== 核心CoP计算 =====================
def compute_pressure_direction(baseline_subtracted_frame):
    """
    输入：基线减除后的84通道压力数据
    输出：方向、幅值、CoP坐标、初始点、偏移量等
    """
    global g_cop_filtered_dir, g_cop_grad_table_arr
    global g_cop_contact_init_x, g_cop_contact_init_y, g_cop_contact_init_flag
    global g_cop_press_low_cnt
    global g_cop_init_x_buf, g_cop_init_y_buf
    global g_cop_post_init_frame_cnt, g_cop_post_stable_cnt, g_cop_post_refined_flag
    global g_cop_post_cand_x, g_cop_post_cand_y

    sensor_rows, sensor_cols = COP_SENSOR_ROW_CNT, COP_SENSOR_COL_CNT
    frame_flat_arr = np.asarray(baseline_subtracted_frame, dtype=np.float32).flatten()
    frame_2d_arr = frame_flat_arr.reshape(sensor_rows, sensor_cols)

    # 计算梯度（用于可视化）
    grad_arr = np.zeros((sensor_rows, sensor_cols, 2), dtype=np.float32)
    for row_idx in range(sensor_rows):
        for col_idx in range(sensor_cols):
            center_val = frame_2d_arr[row_idx, col_idx]
            left_val = frame_2d_arr[row_idx, col_idx-1] if col_idx-1 >= 0 else center_val
            right_val = frame_2d_arr[row_idx, col_idx+1] if col_idx+1 < sensor_cols else center_val
            up_val = frame_2d_arr[row_idx-1, col_idx] if row_idx-1 >= 0 else center_val
            down_val = frame_2d_arr[row_idx+1, col_idx] if row_idx+1 < sensor_rows else center_val
            grad_x = right_val - left_val
            grad_y = up_val - down_val
            grad_arr[row_idx, col_idx] = (grad_x, grad_y)
    with g_cop_grad_table_lock:
        g_cop_grad_table_arr[:] = grad_arr[:]

    # 总压力判断
    total_press_val = np.sum(frame_2d_arr)
    if total_press_val < COP_PRESSURE_LOW_THRESH:
        g_cop_press_low_cnt += 1
    else:
        g_cop_press_low_cnt = 0

    # 连续低压 → 重置
    if g_cop_press_low_cnt >= COP_STABILITY_FRAME_CNT:
        reset_cop_state()
        # 返回默认值，表示无有效CoP或已重置
        return 0.0, 0.0, 0, sensor_rows-1, 0, sensor_cols-1, 0.0, 0.0, 0.0, 0.0  # 10个值

    if total_press_val == 0:
        return 0.0, 0.0, 0, sensor_rows-1, 0, sensor_cols-1, 0.0, 0.0, 0.0, 0.0  # 10个值

    # 已建立初始接触但当前压力过低 → 跳过噪声CoP计算，返回零偏移
    if g_cop_contact_init_flag and total_press_val < COP_PRESSURE_LOW_THRESH:
        return (g_cop_contact_init_x, g_cop_contact_init_y,
                0, sensor_rows-1, 0, sensor_cols-1,
                0.0, 0.0,
                g_cop_contact_init_x, g_cop_contact_init_y)

    # 计算CoP中心
    grid_x_arr = np.tile(np.arange(sensor_cols), (sensor_rows, 1))
    grid_y_arr = np.repeat(np.arange(sensor_rows), sensor_cols).reshape(sensor_rows, sensor_cols)
    cop_curr_x = np.sum(frame_2d_arr * grid_x_arr) / total_press_val
    cop_curr_y = np.sum(frame_2d_arr * grid_y_arr) / total_press_val

    cop_delta_x = 0.0
    cop_delta_y = 0.0
    cop_base_x = cop_curr_x
    cop_base_y = cop_curr_y

    # ============ 初始点稳定判断 ============
    if not g_cop_contact_init_flag:
        g_cop_init_x_buf.append(cop_curr_x)  # 当前CoP加入候选缓冲
        g_cop_init_y_buf.append(cop_curr_y)

        is_seq_stable_flag = True  # 假设当前序列稳定，直到被证明不稳定

        if len(g_cop_init_x_buf) >= 2:
            # 用缓冲器前两个点定义参考直线方向
            ref_p0_x, ref_p0_y = g_cop_init_x_buf[0], g_cop_init_y_buf[0]
            ref_p1_x, ref_p1_y = g_cop_init_x_buf[1], g_cop_init_y_buf[1]

            ref_dir_x = ref_p1_x - ref_p0_x
            ref_dir_y = ref_p1_y - ref_p0_y
            ref_dir_len = np.hypot(ref_dir_x, ref_dir_y)

            if ref_dir_len < 1e-4:  # 前两点重合，无法定义方向
                is_seq_stable_flag = False
            else:
                ref_dir_norm_x = ref_dir_x / ref_dir_len  # 归一化参考方向
                ref_dir_norm_y = ref_dir_y / ref_dir_len

                for buf_idx in range(2, len(g_cop_init_x_buf)):
                    buf_pt_x, buf_pt_y = g_cop_init_x_buf[buf_idx], g_cop_init_y_buf[buf_idx]
                    buf_prev_x, buf_prev_y = g_cop_init_x_buf[buf_idx-1], g_cop_init_y_buf[buf_idx-1]

                    # 1. 当前点到参考直线的距离
                    # 距离 = |(p1-p0) × (pt-p0)| / |p1-p0|
                    cross_prod_val = abs((ref_p1_x - ref_p0_x) * (buf_pt_y - ref_p0_y)
                                        - (ref_p1_y - ref_p0_y) * (buf_pt_x - ref_p0_x))
                    line_dist_val = cross_prod_val / ref_dir_len

                    if line_dist_val > COP_LINE_DIST_THRESH:
                        is_seq_stable_flag = False
                        break  # 偏离直线太远

                    # 2. 当前移动方向与参考方向的一致性
                    seg_dir_x = buf_pt_x - buf_prev_x
                    seg_dir_y = buf_pt_y - buf_prev_y
                    seg_len = np.hypot(seg_dir_x, seg_dir_y)

                    if seg_len > 1e-4:  # 有实际移动才检查方向
                        seg_norm_x = seg_dir_x / seg_len
                        seg_norm_y = seg_dir_y / seg_len
                        dot_prod_val = ref_dir_norm_x * seg_norm_x + ref_dir_norm_y * seg_norm_y
                        if dot_prod_val < COP_DIR_DOT_THRESH:
                            is_seq_stable_flag = False
                            break  # 方向不一致

        if not is_seq_stable_flag:
            # 序列不稳定：清空缓冲，以当前CoP作为新序列起点
            g_cop_init_x_buf.clear()
            g_cop_init_y_buf.clear()
            g_cop_init_x_buf.append(cop_curr_x)
            g_cop_init_y_buf.append(cop_curr_y)

        elif len(g_cop_init_x_buf) == COP_STABILITY_FRAME_CNT:
            g_cop_contact_init_x = g_cop_init_x_buf[0]  # 取稳定序列的第一个点
            g_cop_contact_init_y = g_cop_init_y_buf[0]
            g_cop_contact_init_flag = True

            # 确定初始点后，清空缓冲器，不再需要它
            g_cop_init_x_buf.clear()
            g_cop_init_y_buf.clear()

    # ========== 计算偏移量 ==========
    else:  # g_cop_contact_init_flag 为 True
        # 二次静置精修：检测静止，修正初始CoP
        g_cop_post_init_frame_cnt += 1
        if not g_cop_post_refined_flag and g_cop_post_init_frame_cnt <= COP_POST_INIT_WINDOW_CNT:
            if g_cop_post_cand_x is not None:
                dist_val = np.hypot(cop_curr_x - g_cop_post_cand_x,
                                    cop_curr_y - g_cop_post_cand_y)
                if dist_val <= COP_POST_INIT_STABLE_THRESH:
                    g_cop_post_stable_cnt += 1
                else:
                    g_cop_post_cand_x = cop_curr_x
                    g_cop_post_cand_y = cop_curr_y
                    g_cop_post_stable_cnt = 1
            else:
                g_cop_post_cand_x = cop_curr_x
                g_cop_post_cand_y = cop_curr_y
                g_cop_post_stable_cnt = 1

            if g_cop_post_stable_cnt >= COP_POST_INIT_STABLE_CNT:
                g_cop_contact_init_x = g_cop_post_cand_x
                g_cop_contact_init_y = g_cop_post_cand_y
                g_cop_post_refined_flag = True
        else:
            g_cop_post_refined_flag = True  # 超时或已完成

        cop_delta_x = cop_curr_x - g_cop_contact_init_x
        cop_delta_y = g_cop_contact_init_y - cop_curr_y
        cop_base_x = g_cop_contact_init_x
        cop_base_y = g_cop_contact_init_y

    return (cop_curr_x, cop_curr_y,
            0, sensor_rows-1, 0, sensor_cols-1,  # 绘图范围 (min_y, max_y, min_x, max_x)
            cop_delta_x, cop_delta_y,
            cop_base_x, cop_base_y)