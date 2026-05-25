import numpy as np
from collections import deque

COP_INIT_MEDIAN_FRAMES = 5
NOISE_COLLECT_FRAMES = 10
THRESH_K = 5
SENSOR_ROWS = 12
SENSOR_COLS = 7

POST_INIT_WINDOW_CNT = 60000
POST_INIT_STABLE_CNT = 100
POST_INIT_STABLE_THRESH = 0.1

first_contact_CoP_x = None
first_contact_CoP_y = None
contact_initialized = False

cop_init_x_buf = deque(maxlen=COP_INIT_MEDIAN_FRAMES)
cop_init_y_buf = deque(maxlen=COP_INIT_MEDIAN_FRAMES)

noise_sum_buf = deque(maxlen=NOISE_COLLECT_FRAMES)
dynamic_thresh = None

post_init_frame_cnt = 0
post_stable_cnt = 0
post_refined_flag = False
post_cand_x = None
post_cand_y = None


def reset_cop_state():
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


def compute_pressure_direction(raw_frame):
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global post_init_frame_cnt, post_stable_cnt, post_refined_flag
    global post_cand_x, post_cand_y
    global noise_sum_buf, dynamic_thresh

    rows, cols = SENSOR_ROWS, SENSOR_COLS
    frame_flat = np.asarray(raw_frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)

    total_pressure = np.sum(frame2d)

    if dynamic_thresh is None:
        noise_sum_buf.append(total_pressure)
        if len(noise_sum_buf) >= NOISE_COLLECT_FRAMES:
            sums = np.array(noise_sum_buf)
            dynamic_thresh = THRESH_K * float(np.mean(sums))

    if total_pressure == 0 or (dynamic_thresh is not None and total_pressure < dynamic_thresh):
        if contact_initialized and dynamic_thresh is not None:
            reset_cop_state()
        return 0.0, 0.0, 0, rows-1, 0, cols-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, dynamic_thresh

    x_grid = np.tile(np.arange(cols), (rows, 1))
    y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols)
    cop_x = np.sum(frame2d * x_grid) / total_pressure
    cop_y = np.sum(frame2d * y_grid) / total_pressure

    delta_CoP_x = 0.0
    delta_CoP_y = 0.0
    base_x = cop_x
    base_y = cop_y

    if not contact_initialized:
        cop_init_x_buf.append(cop_x)
        cop_init_y_buf.append(cop_y)

        if len(cop_init_x_buf) >= COP_INIT_MEDIAN_FRAMES:
            first_contact_CoP_x = float(np.median(cop_init_x_buf))
            first_contact_CoP_y = float(np.median(cop_init_y_buf))
            contact_initialized = True
            cop_init_x_buf.clear()
            cop_init_y_buf.clear()

    else:
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


def compute_vector_angle(x: float, y: float) -> tuple[float, float]:
    epsilon = 1e-8
    mag = np.hypot(x, y)
    angle = np.degrees(np.arctan2(y, x + epsilon))
    if angle < 0:
        angle += 360
    return angle, mag


def compute_PZT_angle(Px: float, Py: float) -> tuple[float, float]:
    return compute_vector_angle(Px, Py)


def get_pzt_angle(adc_data):
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


