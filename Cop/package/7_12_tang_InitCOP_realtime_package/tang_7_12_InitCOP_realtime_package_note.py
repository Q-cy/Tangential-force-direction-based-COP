import numpy as np
import threading

# ===================== 算法核心参数=====================
TOTAL_PRESSURE_LOW_THRESHOLD = 500      # 有效接触压力阈值
COP_STABILITY_FRAMES_REQUIRED = 5       # 低压重置帧数
SENSOR_ROWS = 12                        # 传感器阵列行数
SENSOR_COLS = 7                         # 传感器阵列列数

# ===================== 线程安全全局状态 =====================
first_frame = None                      # 第一帧基线
first_frame_lock = threading.Lock()     # 线程锁

first_contact_CoP_x = None              # 初始接触点X
first_contact_CoP_y = None              # 初始接触点Y
contact_initialized = False             # 初始点是否已锁定

total_pressure_low_counter = 0          # 压力低于阈值计数器

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

    first_contact_CoP_x = None
    first_contact_CoP_y = None
    contact_initialized = False
    total_pressure_low_counter = 0

# ===================== CoP压力中心计算 =====================
def compute_pressure_direction(baseline_subtracted_frame):
    """
    输入：基线减除后的84通道压力数据
    输出：CoP偏移量(dx, dy)
    """
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter

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
        return 0.0, 0.0

    if total_pressure == 0:
        return 0.0, 0.0

    # 计算CoP中心坐标
    x_grid = np.tile(np.arange(cols), (rows, 1))
    y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols)
    cop_x = np.sum(frame2d * x_grid) / total_pressure
    cop_y = np.sum(frame2d * y_grid) / total_pressure

    delta_CoP_x = 0.0
    delta_CoP_y = 0.0

    # 锁定第一个有效接触点作为初始CoP
    if not contact_initialized:
        first_contact_CoP_x = cop_x
        first_contact_CoP_y = cop_y
        contact_initialized = True
    else:
        # 计算相对于初始点的偏移量
        delta_CoP_x = cop_x - first_contact_CoP_x
        delta_CoP_y = cop_y - first_contact_CoP_y

    return delta_CoP_x, delta_CoP_y

# ===================== 角度计算核心 =====================
def compute_vector_angle(x: float, y: float) -> tuple[float, float]:
    """计算向量(x,y)的角度(0~360°)和幅值"""
    epsilon = 1e-8
    mag = np.hypot(x, y)                                   # 向量模长
    angle = np.degrees(np.arctan2(y, x + epsilon))         # 弧度转角度
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
    
    # 2. 计算CoP相对于初始点的偏移量(dx, dy)
    dx, dy = compute_pressure_direction(baseline_subtracted)
    
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
