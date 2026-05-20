import numpy as np
import threading
from collections import deque

# ===================== 算法核心参数=====================
TOTAL_PRESSURE_LOW_THRESHOLD = 500      # 有效接触压力阈值
COP_STABILITY_FRAMES_REQUIRED = 10      # 初始稳定所需帧数（取中位数）
SENSOR_ROWS = 12                        # 传感器阵列行数
SENSOR_COLS = 7                         # 传感器阵列列数

# ===================== 二次静置精修参数 =====================
POST_INIT_WINDOW_CNT = 100              # 初始CoP确定后精修监测帧数上限
POST_INIT_STABLE_CNT = 50               # 精修阶段需连续保持不变的帧数
POST_INIT_STABLE_THRESH = 0.1           # 精修判据：CoP偏移距离阈值

# ===================== 线程安全全局状态 =====================
first_frame = None                      # 第一帧基线
first_frame_lock = threading.Lock()     # 线程锁

first_contact_CoP_x = None              # 初始接触点X
first_contact_CoP_y = None              # 初始接触点Y
contact_initialized = False             # 初始点是否已锁定

total_pressure_low_counter = 0          # 压力低于阈值计数器

# 候选初始CoP缓冲
cop_init_x_buf = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)  # 候选初始CoP X序列
cop_init_y_buf = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)  # 候选初始CoP Y序列

# 二次静置精修状态
post_init_frame_cnt = 0                 # 精修阶段已监测帧数
post_stable_cnt = 0                     # 精修阶段连续满足静止判据的帧数
post_refined_flag = False               # 精修是否已完成
post_cand_x = None                      # 精修候选静止点X
post_cand_y = None                      # 精修候选静止点Y


# ===================== 基线减除 =====================
def subtract_baseline(current_frame):
    """用第一帧作为基线，减去背景噪声"""
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()

    with first_frame_lock:
        if first_frame is None:
            first_frame = current_frame.copy()

    diff = current_frame - first_frame
    return np.clip(diff, 0, None)


# ===================== 重置CoP状态 =====================
def reset_cop_state():
    """压力低于阈值 → 重置所有CoP状态"""
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter
    global post_init_frame_cnt, post_stable_cnt, post_refined_flag
    global post_cand_x, post_cand_y

    first_contact_CoP_x = None
    first_contact_CoP_y = None
    contact_initialized = False
    total_pressure_low_counter = 0
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
    输入：基线减除后的84通道压力数据
    输出：(cop_x, cop_y, min_y, max_y, min_x, max_x, delta_x, delta_y, base_x, base_y)
    """
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter
    global post_init_frame_cnt, post_stable_cnt, post_refined_flag
    global post_cand_x, post_cand_y

    rows, cols = SENSOR_ROWS, SENSOR_COLS
    frame_flat = np.asarray(baseline_subtracted_frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)

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

    # 计算CoP中心坐标
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

        if len(cop_init_x_buf) >= COP_STABILITY_FRAMES_REQUIRED:
            xs = list(cop_init_x_buf)
            ys = list(cop_init_y_buf)
            first_contact_CoP_x = float(np.median(xs))
            first_contact_CoP_y = float(np.median(ys))
            print(f"[CoP Init] 前{COP_STABILITY_FRAMES_REQUIRED}帧坐标:")
            for i in range(len(xs)):
                print(f"  frame {i}: x={xs[i]:.3f}, y={ys[i]:.3f}")
            print(f"  中位数: x={first_contact_CoP_x:.3f}, y={first_contact_CoP_y:.3f}")
            contact_initialized = True
            cop_init_x_buf.clear()
            cop_init_y_buf.clear()

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

    return (cop_x, cop_y,
            0, rows-1, 0, cols-1,
            delta_CoP_x, delta_CoP_y,
            base_x, base_y)


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
    return compute_vector_angle(Px, -Py)


# ===================== 核心入口函数 =====================
def get_pzt_angle(adc_data):
    """
    输入84个ADC值，输出压阻传感器角度
    :param adc_data: list/np.array，长度为84的ADC原始数据
    :return: float，压阻传感器角度（0~360°）
    :raises ValueError: ADC数据长度不为84时抛出
    """
    if len(adc_data) != 84:
        raise ValueError("ADC数据长度必须为84")

    # 1. 基线减除（消除背景噪声）
    baseline_subtracted = subtract_baseline(adc_data)

    # 2. 计算CoP（返回10个值，取第7、8个为偏移量）
    result = compute_pressure_direction(baseline_subtracted)
    dx, dy = result[6], result[7]

    # 3. 计算压阻传感器角度
    pzt_angle, _ = compute_PZT_angle(dx, dy)

    return pzt_angle


# ===================== 重置基线（校准用） =====================
def reset_baseline():
    """重置基线和CoP状态，用于重新校准传感器"""
    global first_frame
    with first_frame_lock:
        first_frame = None
    reset_cop_state()
