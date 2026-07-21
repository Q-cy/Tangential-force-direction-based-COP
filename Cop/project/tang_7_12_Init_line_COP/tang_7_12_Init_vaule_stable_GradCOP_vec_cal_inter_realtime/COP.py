"""
CoP 压力中心计算核心模块
功能：基线减除、CoP计算、初始稳定点判断、方向向量滤波
"""

import numpy as np
from collections import deque
import threading


# ===================== 算法参数（仅与CoP计算相关）=====================
COP_INIT_MEDIAN_FRAMES = 20             # 初始COP取中位数的帧数
COP_BASELINE_COLLECT_FRAMES = 20        # 基线采集帧数（用于动态阈值计算）
COP_THRESH_K = 5                        # 阈值乘数：K * mean
COP_SENSOR_ROW_CNT = 12                 # 传感器阵列行数
COP_SENSOR_COL_CNT = 7                  # 传感器阵列列数


# ===================== 二次静置精修参数 =====================
COP_POST_INIT_WINDOW_CNT = 600000        # 初始CoP确定后精修监测帧数上限
COP_POST_INIT_STABLE_CNT = 100          # 精修阶段需连续保持不变的帧数
COP_POST_INIT_STABLE_THRESH = 0.1      # 精修判据：CoP偏移距离阈值


# ===================== 线程安全全局状态 =====================
g_cop_base_frame_arr = None            # 第一帧基线（84通道flat数组）
g_cop_base_frame_lock = threading.Lock()  # 基线读写锁

g_cop_contact_init_x = None            # 初始接触点CoP X坐标
g_cop_contact_init_y = None            # 初始接触点CoP Y坐标
g_cop_contact_init_flag = False        # 初始接触点是否已稳定确定

g_cop_init_x_buf = deque(maxlen=COP_INIT_MEDIAN_FRAMES)   # 候选初始CoP X序列缓冲
g_cop_init_y_buf = deque(maxlen=COP_INIT_MEDIAN_FRAMES)   # 候选初始CoP Y序列缓冲

# 二次静置精修状态
g_cop_post_init_frame_cnt = 0          # 精修阶段已监测帧数
g_cop_post_stable_cnt = 0              # 精修阶段连续满足静止判据的帧数
g_cop_post_refined_flag = False        # 精修是否已完成
g_cop_post_cand_x = None               # 精修候选静止点X
g_cop_post_cand_y = None               # 精修候选静止点Y

g_cop_noise_sum_buf = deque(maxlen=COP_BASELINE_COLLECT_FRAMES)  # 基线期total_press_val缓冲
g_cop_dynamic_thresh = None             # 动态计算后的阈值（None=未校准）

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
    global g_cop_init_x_buf, g_cop_init_y_buf
    global g_cop_grad_table_arr
    global g_cop_post_init_frame_cnt, g_cop_post_stable_cnt, g_cop_post_refined_flag
    global g_cop_post_cand_x, g_cop_post_cand_y

    g_cop_filtered_dir = None
    g_cop_contact_init_x = None
    g_cop_contact_init_y = None
    g_cop_contact_init_flag = False
    g_cop_init_x_buf.clear()
    g_cop_init_y_buf.clear()
    g_cop_post_init_frame_cnt = 0
    g_cop_post_stable_cnt = 0
    g_cop_post_refined_flag = False
    g_cop_post_cand_x = None
    g_cop_post_cand_y = None
    with g_cop_grad_table_lock:
        g_cop_grad_table_arr.fill(0)


# ===================== 最大连通域分析 =====================
def find_largest_connected_component(frame_2d, total_press):
    """
    对 12×7 网格做 8-邻域 BFS，找出面积最大的连通域。
    frame_2d: (12, 7) 基线减除后的压力数据
    total_press: 全帧总压力，阈值 = total_press / 84
    返回: (lcc_cx, lcc_cy, cell_count, cells, bbox_cx, bbox_cy) 或 None
    lcc_cx/cy = 压力加权质心, bbox_cx/cy = 几何包围盒中心
    """
    rows, cols = frame_2d.shape
    threshold = total_press / (rows * cols)
    visited = np.zeros((rows, cols), dtype=bool)
    largest_component = None
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    for r in range(rows):
        for c in range(cols):
            if frame_2d[r, c] <= threshold or visited[r, c]:
                continue
            queue = [(r, c)]
            visited[r, c] = True
            cells = []
            min_r, max_r = r, r
            min_c, max_c = c, c
            while queue:
                cr, cc = queue.pop(0)
                cells.append((cr, cc))
                min_r, max_r = min(min_r, cr), max(max_r, cr)
                min_c, max_c = min(min_c, cc), max(max_c, cc)
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        if not visited[nr, nc] and frame_2d[nr, nc] > threshold:
                            visited[nr, nc] = True
                            queue.append((nr, nc))
            cell_count = len(cells)
            total_val = sum(frame_2d[cr, cc] for cr, cc in cells)
            sum_col = sum(cc * frame_2d[cr, cc] for cr, cc in cells)
            sum_row = sum(cr * frame_2d[cr, cc] for cr, cc in cells)
            bbox_cx = (min_c + max_c) * 0.5
            bbox_cy = (min_r + max_r) * 0.5
            if largest_component is None or cell_count > largest_component[0]:
                largest_component = (cell_count, total_val, sum_col, sum_row, cells, bbox_cx, bbox_cy)

    if largest_component is None or largest_component[1] <= 0:
        return None

    _, total_val, sum_col, sum_row, cells, bbox_cx, bbox_cy = largest_component
    return (sum_col / total_val, sum_row / total_val, largest_component[0], cells, bbox_cx, bbox_cy)


# ===================== 核心CoP计算 =====================
def compute_pressure_direction(baseline_subtracted_frame):
    """
    输入：基线减除后的84通道压力数据
    输出：方向、幅值、CoP坐标、初始点、偏移量等
    """
    global g_cop_filtered_dir, g_cop_grad_table_arr
    global g_cop_contact_init_x, g_cop_contact_init_y, g_cop_contact_init_flag
    global g_cop_init_x_buf, g_cop_init_y_buf
    global g_cop_post_init_frame_cnt, g_cop_post_stable_cnt, g_cop_post_refined_flag
    global g_cop_post_cand_x, g_cop_post_cand_y
    global g_cop_noise_sum_buf, g_cop_dynamic_thresh

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

    # 总压力
    total_press_val = np.sum(frame_2d_arr)

    # 动态阈值：启动后收集前N帧的total_press_val，计算 mean + K*std
    if g_cop_dynamic_thresh is None:
        g_cop_noise_sum_buf.append(total_press_val)
        if len(g_cop_noise_sum_buf) >= COP_BASELINE_COLLECT_FRAMES:
            sums = np.array(g_cop_noise_sum_buf)
            g_cop_dynamic_thresh = COP_THRESH_K * float(np.mean(sums))

    # 总压力判断：动态阈值就绪后才启用低压重置
    if g_cop_dynamic_thresh is not None and total_press_val < g_cop_dynamic_thresh:
        if g_cop_contact_init_flag:
            reset_cop_state()
        return (0.0, 0.0, 0, sensor_rows-1, 0, sensor_cols-1, 0.0, 0.0, 0.0, 0.0, 0,
                float('nan'), float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'), float('nan'))

    if total_press_val == 0:
        return (0.0, 0.0, 0, sensor_rows-1, 0, sensor_cols-1, 0.0, 0.0, 0.0, 0.0, 0,
                float('nan'), float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'), float('nan'))

    # 网格坐标
    grid_x_arr = np.tile(np.arange(sensor_cols), (sensor_rows, 1))
    grid_y_arr = np.repeat(np.arange(sensor_rows), sensor_cols).reshape(sensor_rows, sensor_cols)

    # 最大连通域分析
    lcc_result = find_largest_connected_component(frame_2d_arr, total_press_val)
    if lcc_result is not None:
        _lcc_cop_cx, _lcc_cop_cy, lcc_cell_count, lcc_cells, lcc_bbox_cx, lcc_bbox_cy = lcc_result
    else:
        lcc_cells = None
        lcc_bbox_cx = lcc_bbox_cy = float('nan')
        lcc_cell_count = 0

    # LCC 几何中心（包围盒中心）
    lcc_cx = lcc_bbox_cx
    lcc_cy = lcc_bbox_cy

    # COP中心：LCC内压力加权质心
    if lcc_cells is not None:
        cop_curr_x = _lcc_cop_cx
        cop_curr_y = _lcc_cop_cy
    else:
        cop_curr_x = cop_curr_y = float('nan')

    # LCC边界 + 不对称性
    lcc_min_r = lcc_max_r = lcc_min_c = lcc_max_c = float('nan')
    skew_x = skew_y = float('nan')
    skew_pt_x = skew_pt_y = float('nan')
    if lcc_cells is not None:
        lcc_min_r = min(r for r, c in lcc_cells)
        lcc_max_r = max(r for r, c in lcc_cells)
        lcc_min_c = min(c for r, c in lcc_cells)
        lcc_max_c = max(c for r, c in lcc_cells)
        lcc_half_w = max(lcc_max_c - lcc_min_c, 1) * 0.5
        lcc_half_h = max(lcc_max_r - lcc_min_r, 1) * 0.5
        sx = 0.0
        sy = 0.0
        st = 0.0
        for r, c in lcc_cells:
            val = frame_2d_arr[r, c]
            sx += val * ((c - lcc_cx) / lcc_half_w)
            sy += val * ((r - lcc_cy) / lcc_half_h)
            st += val
        if st > 0:
            skew_x = sx / st
            skew_y = sy / st
            skew_pt_x = lcc_cx + skew_x * lcc_half_w
            skew_pt_y = lcc_cy + skew_y * lcc_half_h

    # 无LCC时COP无效，早返（跳过初始点稳定判断）
    if np.isnan(cop_curr_x):
        return (float('nan'), float('nan'), 0, sensor_rows-1, 0, sensor_cols-1,
                0.0, 0.0, float('nan'), float('nan'), 0,
                lcc_cx, lcc_cy, float('nan'), float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'), float('nan'))

    cop_delta_x = 0.0
    cop_delta_y = 0.0
    cop_base_x = cop_curr_x
    cop_base_y = cop_curr_y

    # ============ 初始点稳定判断（中位数判定） ============
    if not g_cop_contact_init_flag:
        g_cop_init_x_buf.append(cop_curr_x)
        g_cop_init_y_buf.append(cop_curr_y)
        if len(g_cop_init_x_buf) >= COP_INIT_MEDIAN_FRAMES:
            g_cop_contact_init_x = float(np.median(g_cop_init_x_buf))
            g_cop_contact_init_y = float(np.median(g_cop_init_y_buf))
            g_cop_contact_init_flag = True
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

    cop_state = 2 if g_cop_post_refined_flag else 1

    return (cop_curr_x, cop_curr_y,
            0, sensor_rows-1, 0, sensor_cols-1,
            cop_delta_x, cop_delta_y,
            cop_base_x, cop_base_y,
            cop_state,
            lcc_cx, lcc_cy, skew_x, skew_y, skew_pt_x, skew_pt_y,
            lcc_min_r, lcc_max_r, lcc_min_c, lcc_max_c)