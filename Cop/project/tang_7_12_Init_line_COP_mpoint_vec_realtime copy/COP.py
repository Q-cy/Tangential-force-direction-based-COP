import numpy as np
from collections import deque
import threading
import cv2 # 引入OpenCV

# ===================== 算法参数（仅与CoP计算相关）=====================
COP_STABILITY_FRAMES_REQUIRED = 5       # 初始稳定帧数 (每个ROI独立判断)
TOTAL_PRESSURE_LOW_THRESHOLD = 500      # 整体压力低于此阈值，重置所有CoP状态
PRESSURE_THRESHOLD_BINARIZATION = 100   # 压力值高于此阈值才被视为接触点（用于二值化）
MIN_ROI_PIXELS = 3                      # 连通区域最小像素数，低于此值视为噪声
SENSOR_ROWS = 12                        # 传感器阵列行数
SENSOR_COLS = 7                         # 传感器阵列列数

# ===================== 单个CoP的直线方向稳定判断参数 =====================
LINE_DIST_THRESHOLD = 0.1               # 点到直线最大允许距离 (CoP单位)
DIR_DOT_THRESHOLD = 0.7                 # 方向一致性最小点积 cos(夹角)

# ===================== 线程安全全局状态 =====================
first_frame = None                      # 第一帧基线
first_frame_lock = threading.Lock()     # 线程锁

# 存储所有活动ROI的状态，键为 watershed 分割出的标签ID (从2开始)
# 值是一个字典：{'initial_cop_x', 'initial_cop_y', 'is_initialized', 'x_buffer', 'y_buffer', 'current_pressure_sum'}
roi_states = {}
roi_states_lock = threading.Lock() # 保护 roi_states 的访问

total_pressure_low_counter = 0           # 整体压力低于阈值计数器

grad_table_data = np.zeros((SENSOR_ROWS, SENSOR_COLS, 2))   # 梯度表（用于绘图）
grad_table_lock = threading.Lock()       # 梯度表读写锁


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
def reset_cop_states():
    """
    压力过低/离开接触面 → 重置所有CoP状态
    """
    global roi_states, total_pressure_low_counter
    
    with roi_states_lock:
        roi_states.clear()
    total_pressure_low_counter = 0
    with grad_table_lock:
        grad_table_data.fill(0)


# ===================== 核心多点CoP计算 =====================
def detect_multiple_cops(baseline_subtracted_frame):
    """
    输入：基线减除后的84通道压力数据
    输出：一个列表，每个元素是一个字典，包含一个检测到的CoP的详细信息
    字典结构: {
        'id': int,                 # ROI的唯一ID
        'cop_x': float,
        'cop_y': float,
        'initial_cop_x': float,
        'initial_cop_y': float,
        'delta_cop_x': float,
        'delta_cop_y': float,
        'pressure_sum': float,     # 该CoP区域的总压力
        'is_initialized': bool     # 该CoP的初始点是否已稳定
    }
    """
    global grad_table_data, roi_states, total_pressure_low_counter

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

    # ==================== 整体压力判断和重置 ====================
    total_pressure = np.sum(frame2d)
    if total_pressure < TOTAL_PRESSURE_LOW_THRESHOLD:
        total_pressure_low_counter += 1
    else:
        total_pressure_low_counter = 0

    if total_pressure_low_counter >= COP_STABILITY_FRAMES_REQUIRED:
        reset_cop_states()
        return [] # 没有有效CoP

    if total_pressure == 0:
        return [] # 没有压力

    # ==================== 图像处理实现多CoP检测 ====================
    # 1. 归一化并二值化
    # 转换为 0-255 范围，因为 OpenCV 图像处理通常是 uint8 类型
    normalized_frame = cv2.normalize(frame2d, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    # 基于阈值二值化
    _, binary_frame = cv2.threshold(normalized_frame, PRESSURE_THRESHOLD_BINARIZATION, 255, cv2.THRESH_BINARY)

    # 2. 连通域过滤（去噪）
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_frame, 8, cv2.CV_32S)
    
    cleaned_binary_map = np.zeros_like(binary_frame, dtype=np.uint8)
    for i in range(1, num_labels): # 忽略背景标签0
        if stats[i, cv2.CC_STAT_AREA] >= MIN_ROI_PIXELS:
            cleaned_binary_map[labels == i] = 255

    # 确保 cleaned_binary_map 至少有一个前景像素，否则后续处理会失败
    if np.sum(cleaned_binary_map) == 0:
        reset_cop_states() # 没有足够大的连通区域，重置
        return []

    # 3. 分水岭分割
    # 距离变换
    dist_transform = cv2.distanceTransform(cleaned_binary_map, cv2.DIST_L2, 5)
    # 寻找前景标记（局部极大值作为种子）
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, cv2.THRESH_BINARY) # 可以调整0.7
    sure_fg = np.uint8(sure_fg)

    # 寻找背景标记
    sure_bg = cv2.dilate(cleaned_binary_map, kernel=np.ones((3,3), np.uint8), iterations=3) # 扩张一点作为背景
    _, sure_bg = cv2.threshold(sure_bg, 1, 255, cv2.THRESH_BINARY_INV) # 反色，把背景设为白色

    # 未知区域
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 标记连通区域
    num_markers, markers = cv2.connectedComponents(sure_fg)
    # 将所有标记 +1，确保背景是1，而非0
    markers = markers + 1
    # 将未知区域标记为0
    markers[unknown == 255] = 0

    # 进行分水岭变换 (需要3通道图像作为输入)
    # 这里我们使用原始压力图的副本作为输入，但颜色信息不重要，只需通道数匹配
    color_frame = np.stack([normalized_frame, normalized_frame, normalized_frame], axis=-1)
    markers = cv2.watershed(color_frame, markers)

    detected_cops = []
    current_roi_ids = set() # 记录本帧检测到的ROI ID

    with roi_states_lock:
        # 清理不再存在的ROI
        for roi_id in list(roi_states.keys()):
            if roi_id not in markers: # 如果该ROI在当前帧markers中不存在
                del roi_states[roi_id]

        for roi_id_val in np.unique(markers):
            if roi_id_val <= 1: # 忽略背景和未知区域 (通常markers背景是1，分水岭分割的物体从2开始)
                continue

            roi_mask = (markers == roi_id_val)
            roi_pressure_sum = np.sum(frame2d[roi_mask])

            if roi_pressure_sum < TOTAL_PRESSURE_LOW_THRESHOLD / 5.0: # 单个ROI的压力阈值
                continue

            # 计算当前CoP
            roi_x_indices, roi_y_indices = np.where(roi_mask)
            cop_x_roi = np.sum(frame2d[roi_mask] * roi_y_indices) / roi_pressure_sum # 注意x,y与行列表达的转换
            cop_y_roi = np.sum(frame2d[roi_mask] * roi_x_indices) / roi_pressure_sum # CoP_x是列索引，CoP_y是行索引

            # ============== 更新ROI状态和稳定性判断 ==============
            if roi_id_val not in roi_states:
                roi_states[roi_id_val] = {
                    'initial_cop_x': None,
                    'initial_cop_y': None,
                    'is_initialized': False,
                    'x_buffer': deque(maxlen=COP_STABILITY_FRAMES_REQUIRED),
                    'y_buffer': deque(maxlen=COP_STABILITY_FRAMES_REQUIRED),
                    'current_pressure_sum': 0.0 # 用于排序
                }
            
            roi_state = roi_states[roi_id_val]
            roi_state['current_pressure_sum'] = roi_pressure_sum
            roi_state['x_buffer'].append(cop_x_roi)
            roi_state['y_buffer'].append(cop_y_roi)

            delta_CoP_x = 0.0
            delta_CoP_y = 0.0
            base_CoP_x_for_plot = cop_x_roi
            base_CoP_y_for_plot = cop_y_roi

            if not roi_state['is_initialized']:
                is_current_sequence_stable = True
                if len(roi_state['x_buffer']) >= 2:
                    p0x, p0y = roi_state['x_buffer'][0], roi_state['y_buffer'][0]
                    p1x, p1y = roi_state['x_buffer'][1], roi_state['y_buffer'][1]

                    dir_ref_x = p1x - p0x
                    dir_ref_y = p1y - p0y
                    dir_ref_len = np.hypot(dir_ref_x, dir_ref_y)

                    if dir_ref_len < 1e-4: # 避免除以零
                        is_current_sequence_stable = False
                    else:
                        norm_dir_ref_x = dir_ref_x / dir_ref_len
                        norm_dir_ref_y = dir_ref_y / dir_ref_len

                        for i in range(2, len(roi_state['x_buffer'])):
                            current_px, current_py = roi_state['x_buffer'][i], roi_state['y_buffer'][i]
                            prev_px, prev_py = roi_state['x_buffer'][i-1], roi_state['y_buffer'][i-1]

                            # 1. 检查点到直线距离
                            cross_product_val = abs((p1x - p0x) * (current_py - p0y) - (p1y - p0y) * (current_px - p0x))
                            line_dist = cross_product_val / dir_ref_len
                            if line_dist > LINE_DIST_THRESHOLD:
                                is_current_sequence_stable = False
                                break

                            # 2. 检查移动方向与参考方向的一致性
                            curr_segment_dir_x = current_px - prev_px
                            curr_segment_dir_y = current_py - prev_py
                            curr_segment_len = np.hypot(curr_segment_dir_x, curr_segment_dir_y)
                            if curr_segment_len > 1e-4:
                                norm_curr_segment_dir_x = curr_segment_dir_x / curr_segment_len
                                norm_curr_segment_dir_y = curr_segment_dir_y / curr_segment_len
                                dot_product = norm_dir_ref_x * norm_curr_segment_dir_x + norm_dir_ref_y * norm_curr_segment_dir_y
                                if dot_product < DIR_DOT_THRESHOLD:
                                    is_current_sequence_stable = False
                                    break
                
                if not is_current_sequence_stable:
                    roi_state['x_buffer'].clear() # 清空并重新开始
                    roi_state['y_buffer'].clear()
                    roi_state['x_buffer'].append(cop_x_roi)
                    roi_state['y_buffer'].append(cop_y_roi)
                elif len(roi_state['x_buffer']) == COP_STABILITY_FRAMES_REQUIRED:
                    roi_state['initial_cop_x'] = roi_state['x_buffer'][0]
                    roi_state['initial_cop_y'] = roi_state['y_buffer'][0]
                    roi_state['is_initialized'] = True
                    roi_state['x_buffer'].clear() # 确定后不再需要缓冲区
                    roi_state['y_buffer'].clear()
            
            if roi_state['is_initialized']:
                delta_CoP_x = cop_x_roi - roi_state['initial_cop_x']
                delta_CoP_y = roi_state['initial_cop_y'] - cop_y_roi # Y轴方向需要注意
                base_CoP_x_for_plot = roi_state['initial_cop_x']
                base_CoP_y_for_plot = roi_state['initial_cop_y']

            detected_cops.append({
                'id': roi_id_val,
                'cop_x': cop_x_roi,
                'cop_y': cop_y_roi,
                'initial_cop_x': base_CoP_x_for_plot,
                'initial_cop_y': base_CoP_y_for_plot,
                'delta_cop_x': delta_CoP_x,
                'delta_cop_y': delta_CoP_y,
                'pressure_sum': roi_pressure_sum,
                'is_initialized': roi_state['is_initialized']
            })
            current_roi_ids.add(roi_id_val)
        
        # 清理不再检测到的ROI（如果markers中没有，则从roi_states中删除）
        keys_to_remove = [k for k in roi_states.keys() if k not in current_roi_ids]
        for k in keys_to_remove:
            del roi_states[k]

    return detected_cops

# 辅助函数，用于外部模块获取整体CoP初始化状态
def get_global_contact_initialized():
    with roi_states_lock:
        # 如果有任何一个CoP已经稳定，则视为全局已初始化
        return any(state['is_initialized'] for state in roi_states.values())
