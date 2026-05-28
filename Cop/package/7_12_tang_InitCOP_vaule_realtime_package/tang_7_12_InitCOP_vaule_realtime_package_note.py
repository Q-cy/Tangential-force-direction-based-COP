import numpy as np
from collections import deque

# ===================== 算法参数 =====================
COP_INIT_MEDIAN_FRAMES = 1                # 初始COP取中位数的帧数（1=立即确定）
NOISE_COLLECT_FRAMES = 20                 # 动态阈值基线采集帧数
THRESH_K = 5                              # 阈值 = K * mean(total_pressure)
SENSOR_ROWS = 12                          # 传感器阵列行数
SENSOR_COLS = 7                           # 传感器阵列列数

# ===================== 吸附中心参数 =====================
SNAP_CENTER_X, SNAP_CENTER_Y = 3.0, 5.5   # 吸附目标（阵列中心）
SNAP_RANGE_X = 0.0                         # X方向吸附范围（0=禁用吸附）
SNAP_RANGE_Y = 0.0                         # Y方向吸附范围（0=禁用吸附）

# ===================== 二次静置精修参数 =====================
POST_INIT_WINDOW_CNT = 600000             # 初始CoP确定后精修监测帧数上限
POST_INIT_STABLE_CNT = 500                # 精修阶段需连续保持不变的帧数
POST_INIT_STABLE_THRESH = 0.1             # 精修判据：CoP偏移距离阈值

# ===================== 全局状态 =====================
first_contact_CoP_x = None                # 初始接触点CoP X坐标
first_contact_CoP_y = None                # 初始接触点CoP Y坐标
contact_initialized = False               # 初始接触点是否已稳定确定

# 候选初始CoP缓冲（中位数滤波用）
cop_init_x_buf = deque(maxlen=COP_INIT_MEDIAN_FRAMES)
cop_init_y_buf = deque(maxlen=COP_INIT_MEDIAN_FRAMES)

# 动态阈值
noise_sum_buf = deque(maxlen=NOISE_COLLECT_FRAMES)
dynamic_thresh = None                     # 动态计算后的阈值（None=未校准）

# 二次静置精修状态
post_init_frame_cnt = 0                   # 精修阶段已监测帧数
post_stable_cnt = 0                       # 精修阶段连续满足静止判据的帧数
post_refined_flag = False                 # 精修是否已完成
post_cand_x = None                        # 精修候选静止点X
post_cand_y = None                        # 精修候选静止点Y


# ===================== 重置CoP状态 =====================
def reset_cop_state():
    """
    压力过低/离开接触面 → 重置所有状态
    """
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global post_init_frame_cnt, post_stable_cnt, post_refined_flag
    global post_cand_x, post_cand_y

    first_contact_CoP_x = None
    first_contact_CoP_y = None
    contact_initialized = False
    cop_init_x_buf.clear()
    cop_init_y_buf.clear()
    post_init_frame_cnt = 0
    post_stable_cnt = 0
    post_refined_flag = False
    post_cand_x = None
    post_cand_y = None


# ===================== CoP压力中心计算 =====================
def compute_pressure_direction(baseline_subtracted_frame):
    """
    输入：84通道压力数据（原始ADC或基线减除后）
    输出：14个值的元组
      (cop_x, cop_y, row_min, row_max, col_min, col_max,
       delta_CoP_x, delta_CoP_y, base_x, base_y,
       magnitude, state, total_pressure, dynamic_thresh)

    state:
      0 = 未初始化（正在采集初始CoP）
      1 = 已初始化，精修中
      2 = 已初始化，精修完成
    """
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global post_init_frame_cnt, post_stable_cnt, post_refined_flag
    global post_cand_x, post_cand_y
    global noise_sum_buf, dynamic_thresh

    rows, cols = SENSOR_ROWS, SENSOR_COLS
    frame_flat = np.asarray(baseline_subtracted_frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)

    total_pressure = np.sum(frame2d)

    # 动态阈值：启动后收集前N帧的total_pressure，计算 K * mean
    if dynamic_thresh is None:
        noise_sum_buf.append(total_pressure)
        if len(noise_sum_buf) >= NOISE_COLLECT_FRAMES:
            sums = np.array(noise_sum_buf)
            dynamic_thresh = THRESH_K * float(np.mean(sums))

    # 低压重置（含 total_pressure == 0）
    if total_pressure == 0 or (dynamic_thresh is not None and total_pressure < dynamic_thresh):
        if contact_initialized and dynamic_thresh is not None:
            reset_cop_state()
        return 0.0, 0.0, 0, rows-1, 0, cols-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, dynamic_thresh

    # 计算CoP中心（加权质心法）
    x_grid = np.tile(np.arange(cols), (rows, 1))
    y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols)
    cop_x = np.sum(frame2d * x_grid) / total_pressure
    cop_y = np.sum(frame2d * y_grid) / total_pressure

    delta_CoP_x = 0.0
    delta_CoP_y = 0.0
    base_x = cop_x
    base_y = cop_y

    # ============ 初始点稳定判断（中位数判定） ============
    if not contact_initialized:
        cop_init_x_buf.append(cop_x)
        cop_init_y_buf.append(cop_y)

        if len(cop_init_x_buf) >= COP_INIT_MEDIAN_FRAMES:
            first_contact_CoP_x = float(np.median(cop_init_x_buf))
            first_contact_CoP_y = float(np.median(cop_init_y_buf))
            contact_initialized = True
            cop_init_x_buf.clear()
            cop_init_y_buf.clear()
            # 吸附到阵列中心（如果在吸附范围内）
            if (abs(first_contact_CoP_x - SNAP_CENTER_X) <= SNAP_RANGE_X and
                abs(first_contact_CoP_y - SNAP_CENTER_Y) <= SNAP_RANGE_Y):
                first_contact_CoP_x = SNAP_CENTER_X
                first_contact_CoP_y = SNAP_CENTER_Y

    # ========== 计算偏移量 ==========
    else:
        # 二次静置精修：检测静止，修正初始CoP
        post_init_frame_cnt += 1
        if not post_refined_flag and post_init_frame_cnt <= POST_INIT_WINDOW_CNT:
            if post_cand_x is not None:
                dist_val = np.hypot(cop_x - post_cand_x, cop_y - post_cand_y)
                if dist_val <= POST_INIT_STABLE_THRESH:
                    post_stable_cnt += 1
                else:
                    post_cand_x = cop_x
                    post_cand_y = cop_y
                    post_stable_cnt = 1
            else:
                post_cand_x = cop_x
                post_cand_y = cop_y
                post_stable_cnt = 1

            if post_stable_cnt >= POST_INIT_STABLE_CNT:
                first_contact_CoP_x = post_cand_x
                first_contact_CoP_y = post_cand_y
                post_refined_flag = True
        else:
            post_refined_flag = True  # 超时或已完成

        delta_CoP_x = cop_x - first_contact_CoP_x
        delta_CoP_y = first_contact_CoP_y - cop_y
        base_x = first_contact_CoP_x
        base_y = first_contact_CoP_y

    magnitude = np.hypot(delta_CoP_x, delta_CoP_y)
    if not contact_initialized:
        state = 0
    elif not post_refined_flag:
        state = 1
    else:
        state = 2

    return (cop_x, cop_y,
            0, rows-1, 0, cols-1,
            delta_CoP_x, delta_CoP_y,
            base_x, base_y,
            magnitude, state,
            total_pressure, dynamic_thresh)


# ===================== 角度计算核心 =====================
def compute_vector_angle(x: float, y: float) -> tuple[float, float]:
    """计算向量(x,y)的角度(0~360°)和幅值"""
    epsilon = 1e-8
    mag = np.hypot(x, y)
    angle = np.degrees(np.arctan2(y, x + epsilon))
    if angle < 0:
        angle += 360
    return angle, mag

def compute_PZT_angle(Px: float, Py: float) -> tuple[float, float]:
    """计算压阻传感器(Px,Py)的角度(0~360°)和幅值"""
    return compute_vector_angle(Px, Py)


# ===================== 核心入口函数 =====================
def get_pzt_angle(adc_data):
    """
    输入：84通道ADC原始数据
    输出：(角度, 幅值, 状态, cop_x, cop_y, base_x, base_y, 总压力, 动态阈值)
    """
    if len(adc_data) != 84:
        raise ValueError("ADC数据长度必须为84")
    result = compute_pressure_direction(adc_data)
    cop_x, cop_y = result[0], result[1]
    dx, dy = result[6], result[7]
    base_x, base_y = result[8], result[9]
    magnitude = result[10]
    state = int(result[11])
    total_press = result[12]
    threshold = result[13]
    pzt_angle, _ = compute_PZT_angle(dx, dy)
    return pzt_angle, magnitude, state, cop_x, cop_y, base_x, base_y, total_press, threshold
