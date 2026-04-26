import numpy as np
import threading

# ===================== 算法参数=====================
TOTAL_PRESSURE_LOW_THRESHOLD = 500     
COP_STABILITY_FRAMES_REQUIRED = 5       
SENSOR_ROWS = 12                        
SENSOR_COLS = 7                         

# ===================== 线程安全全局状态 =====================
first_frame = None                     
first_frame_lock = threading.Lock()     

first_contact_CoP_x = None              
first_contact_CoP_y = None              
contact_initialized = False             

total_pressure_low_counter = 0          

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

    first_contact_CoP_x = None
    first_contact_CoP_y = None
    contact_initialized = False
    total_pressure_low_counter = 0

# ===================== CoP压力中心计算 =====================
def compute_pressure_direction(baseline_subtracted_frame):
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global total_pressure_low_counter

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
        return 0.0, 0.0

    if total_pressure == 0:
        return 0.0, 0.0

    x_grid = np.tile(np.arange(cols), (rows, 1))
    y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols)
    cop_x = np.sum(frame2d * x_grid) / total_pressure
    cop_y = np.sum(frame2d * y_grid) / total_pressure

    delta_CoP_x = 0.0
    delta_CoP_y = 0.0

    if not contact_initialized:
        first_contact_CoP_x = cop_x
        first_contact_CoP_y = cop_y
        contact_initialized = True
    else:
        delta_CoP_x = cop_x - first_contact_CoP_x
        delta_CoP_y = cop_y - first_contact_CoP_y

    return delta_CoP_x, delta_CoP_y

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
    dx, dy = compute_pressure_direction(baseline_subtracted)
    pzt_angle, _ = compute_PZT_angle(dx, dy)
    
    return pzt_angle

# ===================== 重置基线（校准用） =====================
def reset_baseline():
    global first_frame
    with first_frame_lock:
        first_frame = None
    reset_cop_state()
