"""
CoP 压力中心计算核心模块 (移植自 multi_dim_force.rs)
功能：EMA基线减除、滞回接触检测、多源方向融合（asymmetry+drift+motion）
"""

import numpy as np
import threading


# ===================== 算法参数 =====================
COP_SENSOR_ROW_CNT = 12
COP_SENSOR_COL_CNT = 7
COP_SENSOR_COUNT = COP_SENSOR_ROW_CNT * COP_SENSOR_COL_CNT

# 接触检测（双阈值滞回）
COP_CONTACT_ENTER_TOTAL = 520.0         # 进入：总压阈值
COP_CONTACT_ENTER_PEAK = 50.0           # 进入：峰值阈值
COP_CONTACT_EXIT_TOTAL = 260.0          # 退出：总压阈值
COP_CONTACT_EXIT_PEAK = 28.0            # 退出：峰值阈值
COP_CONTACT_ENTER_FRAMES = 2            # 进入需连续帧数
COP_CONTACT_EXIT_FRAMES = 8             # 退出需连续帧数

# 基线（EMA）
COP_BASELINE_IDLE_ALPHA = 0.035         # 空闲时基线更新速率
COP_BASELINE_BOOTSTRAP_ALPHA = 1.0      # 启动时（=直接复制首帧）
COP_BASELINE_NOISE_FLOOR = 5.0          # 基线减除后额外扣除的噪声底

# 活跃cell筛选
COP_ACTIVE_CELL_MIN = 18.0              # 活跃cell最低值
COP_ACTIVE_CELL_PEAK_RATIO = 0.14       # 活跃cell = max(peak*ratio, min)
COP_MIN_ACTIVE_CELLS = 3                # 最少活跃cell数

# 锚点 + 平滑 + 融合权重
COP_ANCHOR_LERP_ALPHA = 0.018           # 锚点向当前COP漂移速率
COP_VECTOR_SMOOTH_ALPHA = 0.16          # 方向向量平滑速率
COP_ASYMMETRY_WEIGHT = 1.1              # 压力不对称性权重
COP_DRIFT_WEIGHT = 0.65                 # COP漂移权重
COP_MOTION_WEIGHT = 0.25                # 帧间运动权重


# ===================== 线程安全全局状态 =====================
g_cop_baseline_arr = None               # EMA基线（84通道flat）
g_cop_baseline_lock = threading.Lock()

g_cop_contact_active = False            # 是否处于接触状态
g_cop_enter_counter = 0                 # 进入连续帧计数
g_cop_exit_counter = 0                  # 退出连续帧计数

g_cop_anchor_x = None                   # 锚点COP X (EMA)
g_cop_anchor_y = None                   # 锚点COP Y (EMA)
g_cop_last_x = None                     # 上一帧COP X
g_cop_last_y = None                     # 上一帧COP Y
g_cop_smooth_x = 0.0                    # 平滑后的组合方向X
g_cop_smooth_y = 0.0                    # 平滑后的组合方向Y


# ===================== 辅助函数 =====================
def _compute_vector_angle(x, y):
    mag = float(np.hypot(x, y))
    if mag <= 1e-8:
        return 0.0, 0.0
    angle = float(np.degrees(np.arctan2(y, x)))
    if angle < 0:
        angle += 360.0
    return angle, mag


def _pressure_metrics(frame):
    """返回 (total, peak)"""
    total = float(np.sum(frame))
    peak = float(np.max(frame))
    return total, peak


# ===================== 基线减除（EMA） =====================
def subtract_baseline(raw_frame_arr, is_idle=False):
    """
    EMA基线减除。
    - 首次调用或基线为None: 直接用当前帧作为基线(bootstrap)
    - is_idle=True: 以低速alpha更新基线（空闲时追踪漂移）
    返回: 基线减除后clip>=0的84通道数据
    """
    global g_cop_baseline_arr
    frame = np.array(raw_frame_arr, dtype=np.float32).flatten()

    with g_cop_baseline_lock:
        if g_cop_baseline_arr is None:
            g_cop_baseline_arr = frame.copy()
        elif is_idle:
            g_cop_baseline_arr += (frame - g_cop_baseline_arr) * COP_BASELINE_IDLE_ALPHA

        baseline = g_cop_baseline_arr.copy()

    diff = frame - baseline - COP_BASELINE_NOISE_FLOOR
    return np.clip(diff, 0.0, None)


# ===================== 接触检测（滞回） =====================
def _is_contact_enter(frame):
    total, peak = _pressure_metrics(frame)
    return total >= COP_CONTACT_ENTER_TOTAL and peak >= COP_CONTACT_ENTER_PEAK


def _is_contact_exit(frame):
    total, peak = _pressure_metrics(frame)
    return total <= COP_CONTACT_EXIT_TOTAL or peak <= COP_CONTACT_EXIT_PEAK


def update_contact_state(raw_frame, subtracted_frame):
    """
    更新接触状态（滞回）。返回当前是否处于接触中。
    """
    global g_cop_contact_active, g_cop_enter_counter, g_cop_exit_counter
    global g_cop_anchor_x, g_cop_anchor_y, g_cop_last_x, g_cop_last_y
    global g_cop_smooth_x, g_cop_smooth_y

    if g_cop_contact_active:
        if _is_contact_exit(subtracted_frame):
            g_cop_exit_counter += 1
            if g_cop_exit_counter >= COP_CONTACT_EXIT_FRAMES:
                # 退出时用当前原始帧更新基线
                subtract_baseline(raw_frame, is_idle=True)
                _reset_tracking_state()
                return False
        else:
            g_cop_exit_counter = 0
        return True

    # 未接触：检测进入
    if _is_contact_enter(subtracted_frame):
        g_cop_enter_counter += 1
        if g_cop_enter_counter >= COP_CONTACT_ENTER_FRAMES:
            g_cop_contact_active = True
            g_cop_enter_counter = 0
            g_cop_exit_counter = 0
            return True
        return False

    # 空闲：慢速更新基线
    g_cop_enter_counter = 0
    subtract_baseline(raw_frame, is_idle=True)
    return False


def _reset_tracking_state():
    global g_cop_contact_active, g_cop_enter_counter, g_cop_exit_counter
    global g_cop_anchor_x, g_cop_anchor_y, g_cop_last_x, g_cop_last_y
    global g_cop_smooth_x, g_cop_smooth_y

    g_cop_contact_active = False
    g_cop_enter_counter = 0
    g_cop_exit_counter = 0
    g_cop_anchor_x = None
    g_cop_anchor_y = None
    g_cop_last_x = None
    g_cop_last_y = None
    g_cop_smooth_x = 0.0
    g_cop_smooth_y = 0.0


# ===================== 活跃cell + COP + asymmetry =====================
def compute_contact_stats(subtracted_frame):
    """
    筛选活跃cell，计算COP和压力不对称性。
    返回: dict 或 None (活跃cell不足时)
    """
    rows, cols = COP_SENSOR_ROW_CNT, COP_SENSOR_COL_CNT
    frame = np.asarray(subtracted_frame, dtype=np.float32).flatten()
    frame2d = frame.reshape(rows, cols)

    total = float(np.sum(frame2d))
    if total <= 0:
        return None
    peak = float(np.max(frame2d))
    if peak <= 0:
        return None

    active_threshold = max(peak * COP_ACTIVE_CELL_PEAK_RATIO, COP_ACTIVE_CELL_MIN)

    active_total = 0.0
    active_cells = 0
    weighted_col = 0.0
    weighted_row = 0.0
    min_row, max_row = rows, -1
    min_col, max_col = cols, -1

    for r in range(rows):
        for c in range(cols):
            val = frame2d[r, c]
            if val < active_threshold:
                continue
            active_cells += 1
            active_total += val
            weighted_col += val * c
            weighted_row += val * r
            min_row = min(min_row, r)
            max_row = max(max_row, r)
            min_col = min(min_col, c)
            max_col = max(max_col, c)

    if active_cells < COP_MIN_ACTIVE_CELLS or active_total <= 0:
        return None

    cop_x = weighted_col / active_total
    cop_y = weighted_row / active_total

    bbox_cx = (min_col + max_col) * 0.5
    bbox_cy = (min_row + max_row) * 0.5
    half_w = max(max_col - min_col, 1) * 0.5
    half_h = max(max_row - min_row, 1) * 0.5

    asym_x = 0.0
    asym_y = 0.0
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            val = frame2d[r, c]
            if val < active_threshold:
                continue
            asym_x += val * ((c - bbox_cx) / half_w)
            asym_y += val * ((r - bbox_cy) / half_h)

    return {
        'total': total,
        'peak': peak,
        'active_total': active_total,
        'active_cells': active_cells,
        'min_row': min_row, 'max_row': max_row,
        'min_col': min_col, 'max_col': max_col,
        'cop_x': cop_x, 'cop_y': cop_y,
        'bbox_cx': bbox_cx, 'bbox_cy': bbox_cy,
        'asym_x': asym_x / active_total,
        'asym_y': asym_y / active_total,
    }


# ===================== 核心方向计算 =====================
def compute_pressure_direction(subtracted_frame):
    """
    输入：基线减除后的84通道压力数据（flat或2d均可）
    输出: (angle_deg, magnitude, planar_x, planar_y, confidence, contact_active,
           cop_x, cop_y, bbox_cx, bbox_cy)
    """
    global g_cop_anchor_x, g_cop_anchor_y, g_cop_last_x, g_cop_last_y
    global g_cop_smooth_x, g_cop_smooth_y

    rows, cols = COP_SENSOR_ROW_CNT, COP_SENSOR_COL_CNT

    stats = compute_contact_stats(subtracted_frame)
    if stats is None:
        # 弱接触或无有效cell
        if g_cop_anchor_x is not None:
            angle, mag = _compute_vector_angle(g_cop_smooth_x, -g_cop_smooth_y)
            return (angle, mag, g_cop_smooth_x, -g_cop_smooth_y, 0.0, g_cop_contact_active,
                    float('nan'), float('nan'), float('nan'), float('nan'))
        return (0.0, 0.0, 0.0, 0.0, 0.0, g_cop_contact_active,
                float('nan'), float('nan'), float('nan'), float('nan'))

    cop_x, cop_y = stats['cop_x'], stats['cop_y']

    # 锚点初始化：首次接触直接设为当前COP
    if g_cop_anchor_x is None:
        g_cop_anchor_x = cop_x
        g_cop_anchor_y = cop_y
        g_cop_last_x = cop_x
        g_cop_last_y = cop_y
        return (0.0, 0.0, 0.0, 0.0, 0.0, g_cop_contact_active,
                cop_x, cop_y, stats['bbox_cx'], stats['bbox_cy'])

    anchor_x = g_cop_anchor_x
    anchor_y = g_cop_anchor_y
    last_x = g_cop_last_x if g_cop_last_x is not None else cop_x
    last_y = g_cop_last_y if g_cop_last_y is not None else cop_y

    drift_x = cop_x - anchor_x
    drift_y = cop_y - anchor_y
    motion_x = cop_x - last_x
    motion_y = cop_y - last_y

    combined_x = (stats['asym_x'] * COP_ASYMMETRY_WEIGHT +
                  drift_x * COP_DRIFT_WEIGHT +
                  motion_x * COP_MOTION_WEIGHT)
    combined_y = (stats['asym_y'] * COP_ASYMMETRY_WEIGHT +
                  drift_y * COP_DRIFT_WEIGHT +
                  motion_y * COP_MOTION_WEIGHT)

    g_cop_smooth_x += (combined_x - g_cop_smooth_x) * COP_VECTOR_SMOOTH_ALPHA
    g_cop_smooth_y += (combined_y - g_cop_smooth_y) * COP_VECTOR_SMOOTH_ALPHA

    g_cop_anchor_x = anchor_x + drift_x * COP_ANCHOR_LERP_ALPHA
    g_cop_anchor_y = anchor_y + drift_y * COP_ANCHOR_LERP_ALPHA
    g_cop_last_x = cop_x
    g_cop_last_y = cop_y

    planar_x = g_cop_smooth_x
    planar_y = -g_cop_smooth_y
    angle_deg, magnitude = _compute_vector_angle(planar_x, planar_y)

    # 置信度
    active_span_r = (stats['max_row'] - stats['min_row'] + 1) / rows
    active_span_c = (stats['max_col'] - stats['min_col'] + 1) / cols
    activity = min(stats['active_cells'] / COP_SENSOR_COUNT, 1.0)
    span = min((active_span_r + active_span_c) * 0.5, 1.0)
    pressure_ratio = min(stats['active_total'] / max(stats['total'], 1.0), 1.0)
    peak_ratio = min(stats['peak'] / (stats['active_total'] / stats['active_cells'] + 1.0), 1.0)
    confidence = (activity * 0.35 + span * 0.2 + pressure_ratio * 0.3 + peak_ratio * 0.15)
    confidence = min(max(confidence, 0.0), 1.0)

    return (angle_deg, magnitude, planar_x, planar_y, confidence, g_cop_contact_active,
            cop_x, cop_y, stats['bbox_cx'], stats['bbox_cy'])


# ===================== 重置基线 =====================
def reset_baseline():
    global g_cop_baseline_arr
    with g_cop_baseline_lock:
        g_cop_baseline_arr = None
    _reset_tracking_state()


# ===================== 高层入口（对应Rust get_pzt_analysis） =====================
def get_pzt_analysis(raw_adc_data):
    """
    完整的单帧处理管线。
    输入: 84通道原始ADC数据
    输出: (angle_deg, magnitude, planar_x, planar_y, confidence, contact_active,
           cop_x, cop_y, bbox_cx, bbox_cy)
    """
    if len(raw_adc_data) != COP_SENSOR_COUNT:
        raise ValueError(f"ADC数据长度必须为{COP_SENSOR_COUNT}")

    subtracted = subtract_baseline(raw_adc_data, is_idle=False)
    if not update_contact_state(raw_adc_data, subtracted):
        return (0.0, 0.0, 0.0, 0.0, 0.0, False,
                float('nan'), float('nan'), float('nan'), float('nan'))

    return compute_pressure_direction(subtracted)
