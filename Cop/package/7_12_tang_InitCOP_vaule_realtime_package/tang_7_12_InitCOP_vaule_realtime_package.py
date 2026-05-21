import numpy as np
import threading
from collections import deque

# ===================== 算法参数=====================
TOTAL_PRESSURE_LOW_THRESHOLD = 15000
COP_STABILITY_FRAMES_REQUIRED = 15
SENSOR_ROWS = 12
SENSOR_COLS = 7

# ===================== 二次静置精修参数 =====================
POST_INIT_WINDOW_CNT = 100000000000
POST_INIT_STABLE_CNT = 200
POST_INIT_STABLE_THRESH = 0.1

# ===================== 线程安全全局状态 =====================
first_frame = None
first_frame_lock = threading.Lock()

first_contact_CoP_x = None
first_contact_CoP_y = None
contact_initialized = False

total_pressure_low_counter = 0

# 候选初始CoP缓冲
cop_init_x_buf = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)
cop_init_y_buf = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)

# 二次静置精修状态
post_init_frame_cnt = 0
post_stable_cnt = 0
post_refined_flag = False
post_cand_x = None
post_cand_y = None


# ===================== 基线减除 =====================
def subtract_baseline(current_frame):
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()

    with first_frame_lock:
        if first_frame is None:
            first_frame = current_frame.copy()

    diff = current_frame - first_frame
    return np.clip(diff, 0, None)


# ===================== 重置CoP状态 =====================
def reset_cop_state():
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
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter
    global post_init_frame_cnt, post_stable_cnt, post_refined_flag
    global post_cand_x, post_cand_y

    rows, cols = SENSOR_ROWS, SENSOR_COLS
    frame_flat = np.asarray(baseline_subtracted_frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)

    total_pressure = np.sum(frame2d)
    if total_pressure < TOTAL_PRESSURE_LOW_THRESHOLD:
        total_pressure_low_counter += 1
    else:
        total_pressure_low_counter = 0

    if total_pressure_low_counter >= COP_STABILITY_FRAMES_REQUIRED:
        reset_cop_state()
        return 0.0, 0.0, 0, rows-1, 0, cols-1, 0.0, 0.0, 0.0, 0.0

    if total_pressure == 0:
        return 0.0, 0.0, 0, rows-1, 0, cols-1, 0.0, 0.0, 0.0, 0.0

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
        # 二次静置精修
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
            post_refined_flag = True

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
    epsilon = 1e-8
    mag = np.hypot(x, y)
    angle = np.degrees(np.arctan2(y, x + epsilon))
    if angle < 0:
        angle += 360
    return angle, mag

def compute_PZT_angle(Px: float, Py: float) -> tuple[float, float]:
    return compute_vector_angle(Px, -Py)


# ===================== 核心入口函数 =====================
def get_pzt_angle(adc_data):
    if len(adc_data) != 84:
        raise ValueError("ADC数据长度必须为84")
    baseline_subtracted = subtract_baseline(adc_data)
    result = compute_pressure_direction(baseline_subtracted)
    dx, dy = result[6], result[7]
    pzt_angle, _ = compute_PZT_angle(dx, dy)

    return pzt_angle


# ===================== 重置基线（校准用） =====================
def reset_baseline():
    global first_frame
    with first_frame_lock:
        first_frame = None
    reset_cop_state()

