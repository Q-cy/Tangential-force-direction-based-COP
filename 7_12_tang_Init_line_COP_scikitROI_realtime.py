import serial
import serial.tools.list_ports
import time
import struct
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from collections import deque
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle # 导入Rectangle用于绘制矩形
from skimage.measure import label, regionprops # 新增导入

# ==================== Config ====================
BAUDRATE_PRESS = 921600         # 压力传感器串口波特率
BAUDRATE_FORCE = 460860         # 六轴力传感器串口波特率 
SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test" # 数据保存目录
TARGET_HZ = 100.0               # 目标数据采集和处理频率 (Hz)
PLOT_INTERVAL_MS = 100          # Matplotlib实时绘图的更新间隔 (毫秒)，例如100ms即每秒更新10次
MAX_SYNC_DT = 0.015             # 压力传感器和力传感器数据同步允许的最大时间差 (秒)
PRESS_BUFFER_SIZE = 500         # 压力传感器数据缓冲区大小
FORCE_BUFFER_SIZE = 200         # 力传感器数据缓冲区大小
# THRESHOLD = 100               # 压力阈值，低于此值的压力被认为是零，不参与CoP和梯度计算 (已移除，改为动态计算)

# 新增动态阈值配置参数
THRESHOLD_MULTIPLIER = 1         # 动态阈值乘数：例如，如果平均ADC是200，阈值将是100
MIN_THRESHOLD_FLOOR = 500        # 最小阈值地板：即使平均值很低，阈值也不会低于此值，防止噪声被计入

# Smoothing 参数-
DIR_SMOOTH_ALPHA = 0.15         # 方向向量指数平滑的 alpha 值 (0-1)，越大越平滑
ERROR_PLOT_LEN = 100            # 角度误差绘图历史队列长度
MAG_PLOT_LEN = 100              # 幅度绘图历史队列长度

# CoP 稳定性判断参数 (现在用于基于直线的判断)
COP_STABILITY_FRAMES_REQUIRED = 5 # 认定为稳定的所需总帧数 (包括定义直线的前两帧)
COP_STABILITY_TOLERANCE = 0.1      # CoP点到所定义直线的最大允许垂直距离（网格单位）

# 新增/修改 ROI 配置参数
# ROI_HALF_SIZE = 1             # 此变量在连通域分析中不再直接使用，但保留以防万一
ROI_EXPANSION_MARGIN = 1        # 连通域边界框向外扩展的单元格数量

# ================================================================

# ==================== Global Variables ====================
# Baseline subtraction 相关的全局变量
first_frame = None                                        # 存储第一帧数据，用于基线减除
first_frame_lock = threading.Lock()                       # 保护 first_frame 的互斥锁，确保线程安全访问

# CoP-based direction calculation related global variables
# first_contact_CoP_x/y 用于记录第一次有效接触时的CoP, 作为切向力为0的参考点
first_contact_CoP_x = None
first_contact_CoP_y = None
contact_initialized = False                               # 标记是否已经建立了基准CoP

# 新增全局变量，用于基于直线的CoP稳定判断 (此部分保持原样)
cop_line_check_buffer = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED) # 存储CoP点，用于直线拟合
current_line_segment_p1 = None                            # 定义稳定直线的第一个点 (x, y)
current_line_segment_p2 = None                            # 定义稳定直线的第二个点 (x, y)
cop_line_stability_active = False                         # 标记是否已定义直线并正在进行稳定性检查
line_equation_coeffs = None                               # 存储直线方程的 (A, B, C) 系数 (Ax + By + C = 0)

# 梯度方向相关的全局变量
adc_filtered_dir = None                                   # 滤波后的ADC（CoP偏移）方向向量 (fx, fy)
grad_table_data = np.zeros((12, 7, 2))                    # 存储每个压力点 (gx, gy) 梯度的全局变量，用于梯度表格
                                                          # 这一行被保留，但其内容将在 compute_gradient_in_region 中被实时更新

# 历史数据队列，用于实时绘图
angle_error_history = deque(maxlen=ERROR_PLOT_LEN)        # 存储ADC和力传感器角度误差的历史
adc_mag_history = deque(maxlen=MAG_PLOT_LEN)              # 存储ADC（CoP偏移）幅值的历史
force_mag_history = deque(maxlen=MAG_PLOT_LEN)            # 存储力传感器幅值的历史
frame_count_history = deque(maxlen=MAG_PLOT_LEN)          # 存储帧计数，作为历史图表的X轴

# ==================== 原始值历史记录 ====================
# 用于存储原始（未经基线减除、滤波等处理）ADC总和和力传感器幅值，用于独立绘图
raw_adc_sum_history = deque(maxlen=MAG_PLOT_LEN)
raw_force_mag_history = deque(maxlen=MAG_PLOT_LEN)

# ==================== 全程数据保存 ====================
# 用于在程序结束后绘制完整的数据曲线
full_time_list = []                                       # 记录所有数据点的时间戳
full_adc_mag_list = []                                    # 记录所有ADC（CoP偏移）幅值
full_force_mag_list = []                                  # 记录所有力传感器幅值

# ==================== Baseline Subtraction ====================
def subtract_baseline(current_frame):
    """
    对当前帧数据进行基线减除。
    Args:
        current_frame (list/np.array): 当前传感器帧的原始数据。
    Returns:
        np.array: 减去基线后的数据，负值被裁剪为0。
    """
    global first_frame
    # 将当前帧转换为一维NumPy数组，数据类型为float32
    current_frame = np.array(current_frame, dtype=np.float32).flatten()
    with first_frame_lock:                                        # 使用互斥锁确保在多线程环境下对 first_frame 的安全访问
        if first_frame is None:                                   # 如果是第一帧，则将其设为基线
            first_frame = current_frame.copy()
    diff = current_frame - first_frame                            # 当前帧减去基线
    diff = np.clip(diff, 0, None)                                 # 将所有负值裁剪为0（压力不能为负）
    return diff

# ==================== CoP-based Direction Calculation ====================
def compute_gradient_in_region(frame):
    """
    计算给定压力帧的CoP，并根据CoP偏移量计算方向向量。
    同时计算每个点的梯度，用于在ax6中显示。
    Args:
        frame (np.array): 经过基线减除的1D压力数据 (84个传感器单元)。
    Returns:
        tuple: (x方向CoP偏移分量, y方向CoP偏移分量, CoP偏移幅值, Current_CoP_x, Current_CoP_y, 
               ROI_r1, ROI_r2, ROI_c1, ROI_c2, Delta_CoP_x, Delta_CoP_y, Base_CoP_x_for_plot, Base_CoP_y_for_plot)。
               如果无有效压力，则返回 (0.0, 0.0, 0.0, 0.0, 0.0, 0, 11, 0, 6, 0.0, 0.0, 0.0, 0.0)。
    """
    global adc_filtered_dir, grad_table_data 
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    # 新增全局变量声明，用于基于直线的CoP稳定判断 (此部分保持原样)
    global cop_line_check_buffer, current_line_segment_p1, current_line_segment_p2, \
           cop_line_stability_active, line_equation_coeffs

    rows, cols = 12, 7                                            # 传感器矩阵的行数和列数 (12行, 7列)
    frame_flat = np.array(frame, dtype=np.float32).flatten()      # 确保是浮点数一维数组
    frame2d = frame_flat.reshape(rows, cols)                      # 重塑为二维矩阵 (12行x7列)
    
    # --- 动态阈值计算 ---
    # 计算所有正压力的平均值
    positive_pressures = frame_flat[frame_flat > 0]
    if len(positive_pressures) > 0:
        current_avg_adc = np.mean(positive_pressures)
        # 应用乘数并确保不低于最小地板值
        calculated_threshold = max(MIN_THRESHOLD_FLOOR, current_avg_adc * THRESHOLD_MULTIPLIER)
    else:
        # 如果没有正压力，使用最小地板值作为阈值，防止在无接触时阈值过低
        calculated_threshold = MIN_THRESHOLD_FLOOR 
    
    threshold = calculated_threshold # 将动态计算出的阈值赋值给局部变量 threshold
    # print(f"Dynamic Threshold: {threshold:.2f}") # 用于调试，可以取消注释查看实时阈值

    # 定义默认的ROI边界 (整个传感器网格)，以防没有有效压力或连通域
    default_r1, default_r2 = 0, rows - 1 # 0到11
    default_c1, default_c2 = 0, cols - 1 # 0到6

    # 预设返回值，避免多次重复代码
    # 注意：如果无有效压力，ROI边界返回整个网格，而不是0。
    default_return_values = (0.0, 0.0, 0.0, 0.0, 0.0, 
                             default_r1, default_r2, default_c1, default_c2, 
                             0.0, 0.0, 0.0, 0.0)

    # 1. 粗定位 CoP (使用整个 12x7 阵列)
    mask_rough = frame2d > threshold
    total_pressure_rough = np.sum(frame2d[mask_rough])
    
    cop_x_rough, cop_y_rough = None, None # 粗CoP初始化为None

    if total_pressure_rough < 1e-3: # 如果没有足够的粗定位总压力 (或总压力非常低)
        adc_filtered_dir = None                                   # 重置滤波方向
        contact_initialized = False                               # 如果失去接触，重置CoP基准
        first_contact_CoP_x = None
        first_contact_CoP_y = None
        
        # 重置所有直线CoP稳定性追踪变量 (保持原样)
        cop_line_check_buffer.clear()
        current_line_segment_p1 = None
        current_line_segment_p2 = None
        cop_line_stability_active = False
        line_equation_coeffs = None
        
        grad_table_data = np.zeros((rows, cols, 2), dtype=np.float32) # 重置梯度数据
        return default_return_values

    # 定义X和Y坐标网格
    x_grid = np.tile(np.arange(cols), (rows, 1)) # X坐标从0到6，对应7列
    y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols) # Y坐标从0到11，对应12行
    
    cop_x_rough = np.sum(frame2d[mask_rough] * x_grid[mask_rough]) / total_pressure_rough
    cop_y_rough = np.sum(frame2d[mask_rough] * y_grid[mask_rough]) / total_pressure_rough

    # --- 2. 基于连通域分析确定精细 ROI ---
    r1, c1, r2, c2 = default_r1, default_c1, default_r2, default_c2 # 初始化ROI边界为整个网格

    # 创建二值图像：压力高于阈值为1，否则为0
    binary_image = (frame2d > threshold).astype(int)
    
    # 标注连通域
    labeled_image = label(binary_image)
    
    # 获取区域属性
    regions = regionprops(labeled_image)
    
    largest_region = None
    max_area = 0
    
    # 找到最大的连通区域，或者粗CoP所在的区域
    # 为了简化，我们找到包含粗CoP的区域，如果没有则找最大的
    target_region = None
    if cop_x_rough is not None and cop_y_rough is not None:
        # 将浮点CoP坐标转换为整数索引
        rough_cop_r_int = int(round(cop_y_rough))
        rough_cop_c_int = int(round(cop_x_rough))

        # 确保索引在有效范围内
        rough_cop_r_int = np.clip(rough_cop_r_int, 0, rows - 1)
        rough_cop_c_int = np.clip(rough_cop_c_int, 0, cols - 1)

        # 检查 rough_cop_r_int, rough_cop_c_int 是否在 labeled_image 的有效区域内
        if labeled_image[rough_cop_r_int, rough_cop_c_int] != 0: # 如果该点属于某个连通域
            label_at_cop = labeled_image[rough_cop_r_int, rough_cop_c_int]
            for props in regions:
                if props.label == label_at_cop:
                    target_region = props
                    break
    
    # 如果没有找到粗CoP所在的区域，则找最大的连通区域
    if target_region is None:
        for props in regions:
            if props.area > max_area:
                max_area = props.area
                largest_region = props
        target_region = largest_region
    
    if target_region is not None:
        # 获取边界框 (min_row, min_col, max_row, max_col)
        # 注意 skimage 的 bbox 是 (min_row, min_col, max_row+1, max_col+1)
        # 即 max_row 和 max_col 是不包含的，我们需要将其转换为包含的索引
        min_r, min_c, max_r_exclusive, max_c_exclusive = target_region.bbox
        
        # 将不包含的索引转换为包含的索引
        max_r_inclusive = max_r_exclusive - 1
        max_c_inclusive = max_c_exclusive - 1

        # 扩展 ROI 边界，并确保在传感器网格内
        r1 = max(0, min_r - ROI_EXPANSION_MARGIN)
        c1 = max(0, min_c - ROI_EXPANSION_MARGIN)
        r2 = min(rows - 1, max_r_inclusive + ROI_EXPANSION_MARGIN)
        c2 = min(cols - 1, max_c_inclusive + ROI_EXPANSION_MARGIN)
        
        # 确保 ROI 至少为 1x1
        if r1 > r2: r2 = r1
        if c1 > c2: c2 = c1
    else:
        # 如果没有找到任何连通域，ROI使用默认的整个网格
        r1, c1, r2, c2 = default_r1, default_c1, default_r2, default_c2


    # --- 3. 在精细 ROI 内计算 CoP ---
    # 创建 ROI 掩码
    fine_roi_mask = np.zeros_like(frame2d, dtype=bool)
    fine_roi_mask[r1 : r2+1, c1 : c2+1] = True # +1 因为切片是独占结束的
    
    # 结合压力阈值掩码和 ROI 掩码
    mask_fine = np.logical_and(frame2d > threshold, fine_roi_mask)
    valid_indices_count_fine = np.sum(mask_fine)

    if valid_indices_count_fine == 0:
        # 如果 ROI 内没有有效压力点，重置相关变量，并返回0。
        adc_filtered_dir = None
        contact_initialized = False
        first_contact_CoP_x = None
        first_contact_CoP_y = None
        
        cop_line_check_buffer.clear()
        current_line_segment_p1 = None
        current_line_segment_p2 = None
        cop_line_stability_active = False
        line_equation_coeffs = None
        
        grad_table_data = np.zeros((rows, cols, 2), dtype=np.float32)
        return default_return_values

    total_pressure_fine = np.sum(frame2d[mask_fine])
    if total_pressure_fine < 1e-3: # 再次检查精算总压力
        adc_filtered_dir = None
        contact_initialized = False
        first_contact_CoP_x = None
        first_contact_CoP_y = None
        
        cop_line_check_buffer.clear()
        current_line_segment_p1 = None
        current_line_segment_p2 = None
        cop_line_stability_active = False
        line_equation_coeffs = None
        
        grad_table_data = np.zeros((rows, cols, 2), dtype=np.float32)
        return default_return_values

    cop_x_fine = np.sum(frame2d[mask_fine] * x_grid[mask_fine]) / total_pressure_fine
    cop_y_fine = np.sum(frame2d[mask_fine] * y_grid[mask_fine]) / total_pressure_fine
    
    # ====================== 后续计算使用精算 CoP ======================
    # 将 cop_x_fine, cop_y_fine 赋值给 cop_x, cop_y，以便后续逻辑使用
    cop_x, cop_y = cop_x_fine, cop_y_fine
    # total_pressure = total_pressure_fine # 后续也使用精算的总压力 (此处实际未被使用)

    # ======== 计算每个点的梯度 (借鉴第二段代码，使用整个frame2d) ========
    # 这里我们直接使用 numpy.gradient 来计算，它更高效且内置了边界处理。
    # frame_float 用于梯度计算
    frame_float = frame2d.astype(float)
    
    # gy 是Y方向的梯度 (行方向), gx 是X方向的梯度 (列方向)
    # numpy.gradient 返回的 (dy, dx)
    gy, gx = np.gradient(frame_float)
    
    # 将结果存储到 grad_table_data
    # grad_table_data 的结构是 (rows, cols, 2)
    # 其中 grad_table_data[r, c, 0] 存储 gx (X方向梯度)
    # grad_table_data[r, c, 1] 存储 gy (Y方向梯度)
    grad_table_data[:, :, 0] = gx
    grad_table_data[:, :, 1] = gy
    # ======================================================

    # --- CoP-based Direction Calculation ---
    delta_CoP_x = 0.0
    delta_CoP_y = 0.0
    base_CoP_x_for_plot = cop_x # 默认的CoP绘图基准点是当前的精算CoP
    base_CoP_y_for_plot = cop_y

    if not contact_initialized:
        # 将当前精算CoP添加到用于直线检查的缓冲区中 (保持原样)
        cop_line_check_buffer.append((cop_x, cop_y))

        # 至少需要2个点来定义一条直线
        if len(cop_line_check_buffer) >= 2:
            # 如果直线尚未激活，尝试使用缓冲区中的前两个点来定义它
            if not cop_line_stability_active:
                p1_candidate = cop_line_check_buffer[0]
                p2_candidate = cop_line_check_buffer[1]

                # 检查直线是否退化 (两点过于接近)
                if np.hypot(p2_candidate[0] - p1_candidate[0], p2_candidate[1] - p1_candidate[1]) < 1e-6:
                    # 如果两点几乎相同，无法定义有意义的直线进行线性稳定性检查。
                    # 清空缓冲区，以当前CoP (cop_line_check_buffer[-1]) 开始新的序列。
                    cop_line_check_buffer.clear()
                    cop_line_check_buffer.append((cop_x, cop_y)) # 当前CoP开始一个新的序列
                    current_line_segment_p1 = None
                    current_line_segment_p2 = None
                    line_equation_coeffs = None
                    cop_line_stability_active = False # 确保标志为False
                else:
                    # 定义直线: Ax + By + C = 0
                    current_line_segment_p1 = p1_candidate
                    current_line_segment_p2 = p2_candidate
                    
                    A = current_line_segment_p2[1] - current_line_segment_p1[1]
                    B = current_line_segment_p1[0] - current_line_segment_p2[0]
                    C = -A * current_line_segment_p1[0] - B * current_line_segment_p1[1]
                    line_equation_coeffs = (A, B, C)
                    cop_line_stability_active = True # 直线现在处于激活状态
            
            # 如果直线已激活，检查缓冲区中所有点到已定义直线的垂直距离
            if cop_line_stability_active and line_equation_coeffs is not None:
                A, B, C = line_equation_coeffs
                is_stable_along_line = True
                
                for point_x, point_y in cop_line_check_buffer:
                    numerator = abs(A * point_x + B * point_y + C)
                    denominator = np.hypot(A, B)
                    
                    if denominator < 1e-6: # 鲁棒性检查，如果之前未捕获到退化直线
                        distance = 0.0
                    else:
                        distance = numerator / denominator

                    if distance >= COP_STABILITY_TOLERANCE:
                        is_stable_along_line = False
                        break # 发现一个点偏离直线过远，停止检查
                
                if is_stable_along_line:
                    # 缓冲区中的所有点都靠近直线，且缓冲区已收集到足够帧数
                    if len(cop_line_check_buffer) >= COP_STABILITY_FRAMES_REQUIRED:
                        # 锁定初始CoP为当前稳定序列的第一个点
                        first_contact_CoP_x = current_line_segment_p1[0]
                        first_contact_CoP_y = current_line_segment_p1[1]
                        contact_initialized = True
                        
                        # 成功初始化后，重置直线稳定性追踪变量
                        cop_line_check_buffer.clear()
                        current_line_segment_p1 = None
                        current_line_segment_p2 = None
                        cop_line_stability_active = False
                        line_equation_coeffs = None
                        
                        # 计算相对于新设置的first_contact_CoP的偏移量
                        delta_CoP_x = cop_x - first_contact_CoP_x
                        delta_CoP_y = cop_y - first_contact_CoP_y
                        base_CoP_x_for_plot = first_contact_CoP_x
                        base_CoP_y_for_plot = first_contact_CoP_y
                    # 否则: 尚未达到所需帧数，但当前点序列沿直线稳定。继续收集。
                else:
                    # 缓冲区中的某个点偏离直线过远。重置并重新开始直线检测。
                    cop_line_check_buffer.clear()
                    cop_line_check_buffer.append((cop_x, cop_y)) # 以当前CoP开始新的序列
                    current_line_segment_p1 = None
                    current_line_segment_p2 = None
                    cop_line_stability_active = False
                    line_equation_coeffs = None
            # 如果 cop_line_stability_active 为 False，表示我们正在等待至少2个点
            # 或者直线是退化的。在这种情况下，delta_CoP 保持为0。
        
        # 如果尚未锁定 contact_initialized，delta_CoP_x/y 应为0，绘图基准点是当前CoP
        if not contact_initialized:
            delta_CoP_x = 0.0
            delta_CoP_y = 0.0
            base_CoP_x_for_plot = cop_x
            base_CoP_y_for_plot = cop_y
            
    else: # contact_initialized 为 True，已经有基准点
        # 计算当前CoP相对于基准点的偏移
        delta_CoP_x = cop_x - first_contact_CoP_x
        delta_CoP_y = cop_y - first_contact_CoP_y
        base_CoP_x_for_plot = first_contact_CoP_x # 绘图时基准点是第一次接触的CoP
        base_CoP_y_for_plot = first_contact_CoP_y
        
        # 一旦初始CoP被锁定，就停止直线稳定性检查并清空相关变量
        cop_line_check_buffer.clear()
        current_line_segment_p1 = None
        current_line_segment_p2 = None
        cop_line_stability_active = False
        line_equation_coeffs = None

    # 计算CoP偏移向量的幅值
    vec_mag = np.hypot(delta_CoP_x, delta_CoP_y)

    fx, fy = 0.0, 0.0 # 归一化后的CoP偏移方向向量分量
    if vec_mag > 1e-6: # 只有当偏移幅值足够大时才计算方向
        raw_x = delta_CoP_x / vec_mag
        # 重点：将Y轴向下为正的delta_CoP_y转换为Y轴向上为正的raw_y，以匹配标准数学角度和力传感器Fy
        raw_y = -delta_CoP_y / vec_mag # 这里调整Y方向，使其与力传感器Fy的约定一致 (Y向上为正)
        
        # 指数加权移动平均 (EWMA) 滤波CoP偏移方向
        if adc_filtered_dir is None:
            fx, fy = raw_x, raw_y                                     # 如果是第一帧，直接使用原始方向
        else:
            fx = (1 - DIR_SMOOTH_ALPHA) * adc_filtered_dir[0] + DIR_SMOOTH_ALPHA * raw_x
            fy = (1 - DIR_SMOOTH_ALPHA) * adc_filtered_dir[1] + DIR_SMOOTH_ALPHA * raw_y
            n = np.hypot(fx, fy)                                      # 重新归一化滤波后的向量
            if n > 1e-6:
                fx /= n
                fy /= n
        adc_filtered_dir = (fx, fy)                                   # 更新滤波后的方向
    else:
        adc_filtered_dir = None                                       # 如果偏移量过小，则认为没有明确方向
        
    # 返回滤波后的CoP偏移方向分量、CoP偏移幅值、精算后的CoP、ROI边界，CoP偏移量和绘图基准点
    # 注意返回的ROI边界是 (r1, r2, c1, c2)
    return fx, fy, vec_mag, cop_x, cop_y, r1, r2, c1, c2, delta_CoP_x, delta_CoP_y, base_CoP_x_for_plot, base_CoP_y_for_plot

# ==================== Angle Calculation ====================
def compute_gradient_angle_single(x, y):
    """
    计算给定(x, y)向量的角度和幅值。此处的x, y应为Y轴向上为正的坐标分量。
    Args:
        x (float): X分量。
        y (float): Y分量。
    Returns:
        tuple: (角度(度), 幅值)。
    """
    epsilon = 1e-8                                                # 避免除以零的小量
    angle = np.degrees(np.arctan2(y, x + epsilon))                # 使用arctan2计算角度，结果为(-180, 180]
    if angle < 0:
        angle += 360                                              # 将角度转换为 [0, 360) 范围
    mag = np.hypot(x, y)                                          # 计算向量幅值
    return angle, mag

def compute_force_angle(Fx, Fy):
    """
    计算给定力向量(Fx, Fy)的角度和幅值。Fx, Fy通常假定为标准笛卡尔坐标系（Y轴向上为正）。
    Args:
        Fx (float): X方向力。
        Fy (float): Y方向力。
    Returns:
        tuple: (角度(度), 幅值)。
               如果幅值过小，返回 (0.0, 0.0)。
    """
    epsilon = 1e-8                                                # 避免除以零的小量
    mag = np.hypot(Fx, Fy)                                        # 计算力向量幅值
    if mag < 1e-8:                                                # 如果幅值过小，认为没有力
        return 0.0, 0.0
    angle = np.degrees(np.arctan2(Fy, Fx + epsilon))              # 使用arctan2计算角度
    if angle < 0:
        angle += 360                                              # 将角度转换为 [0, 360) 范围
    return angle, mag

# ==================== 6-axis Force Sensor Class ====================
class SixAxisForceSensor:
    """
    六轴力传感器数据读取和处理类。
    """
    def __init__(self):
        self.ser = None                                           # 串口对象
        self.port = "/dev/ttyUSB0"                                # 默认串口路径
        self.zero_data = [0.0]*6                                  # 零点校准数据
        self.open_port()                                          # 初始化时尝试打开串口

    def open_port(self):
        """尝试打开串口。"""
        try:
            self.ser = serial.Serial(self.port, BAUDRATE_FORCE, timeout=0.05)
            time.sleep(0.1)                                       # 等待串口稳定
            self.ser.reset_input_buffer()                         # 清空输入缓冲区
        except:
            self.ser = None                                       # 打开失败则设为None

    def reconnect(self):
        """重新连接串口。"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()                                      # 尝试关闭现有串口
        except:
            pass
        time.sleep(0.2)                                           # 等待
        self.open_port()                                          # 重新打开串口

    def read(self):
        """
        从传感器读取六轴力矩数据。
        Returns:
            list: [Fx, Fy, Fz, Mx, My, Mz] 经过零点校准和单位转换后的力矩数据。
                  如果读取失败或数据无效，返回 None。
        """
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.write(b'\x49\xAA\x0D\x0A')                   # 发送读取命令
            time.sleep(0.005)
            resp = self.ser.read(28)                              # 读取响应数据
            if len(resp)!=28 or resp[:2]!=b'\x49\xAA':            # 检查数据长度和起始字节
                return None
            
            # 解析二进制数据为浮点数
            Fx = struct.unpack('<f', resp[2:6])[0]
            Fy = struct.unpack('<f', resp[6:10])[0]
            Fz = struct.unpack('<f', resp[10:14])[0]
            Mx = struct.unpack('<f', resp[14:18])[0]
            My = struct.unpack('<f', resp[18:22])[0]
            Mz = struct.unpack('<f', resp[22:26])[0]
            
            # 将原始单位转换为牛顿和牛米
            Fx *= 9.8; Fy *= 9.8; Fz *= 9.8; Mx *= 9.8; My *= 9.8; Mz *= 9.8
            
            # 减去零点校准值
            Fx -= self.zero_data[0]; Fy -= self.zero_data[1]; Fz -= self.zero_data[2]
            Mx -= self.zero_data[3]; My -= self.zero_data[4]; Mz -= self.zero_data[5]
            
            # 返回四舍五入到两位小数的数据
            return [round(v, 2) for v in [Fx, Fy, Fz, Mx, My, Mz]]
        except Exception as e:
            # print(f"Force sensor read error: {e}")
            return None

    def calibrate_zero(self):
        """
        进行零点校准。读取多组数据并取平均值作为零点。
        """
        vals = []
        print("Calibrating force sensor zero point...")
        for _ in range(20):                                       # 读取20次数据
            d = self.read()
            if d:
                vals.append(d)
            time.sleep(0.05)
        if len(vals)>=5:                                          # 至少有5组有效数据才进行校准
            self.zero_data = np.mean(np.array(vals), axis=0).tolist()
            print(f"Force sensor zero data: {self.zero_data}")
        else:
            print("Failed to calibrate force sensor zero point. Using default zeros.")

# ==================== Pressure Sensor Class ====================
class PressureSensor:
    """
    压力传感器数据读取和处理类。
    """
    def __init__(self):
        self.ser = None                                           # 串口对象
        self.port = None                                          # 自动发现的串口路径
        self.last = None                                          # 存储上一次解码的数据
        self.auto_find_port()                                     # 初始化时自动查找并连接串口

    def auto_find_port(self):
        """
        自动查找并连接压力传感器串口。
        """
        ports = list(serial.tools.list_ports.comports())          # 获取所有可用串口
        for p,_,_ in ports:
            if p == "/dev/ttyUSB0":                               # 跳过力传感器的串口
                continue
            try:
                self.ser = serial.Serial(p, BAUDRATE_PRESS, timeout=0.01)
                self.port = p                                     # 记录找到的串口
                time.sleep(0.1)                                   # 等待串口稳定
                self.ser.reset_input_buffer()                     # 清空输入缓冲区
                print(f"Pressure sensor found on {p}")
                return
            except:
                continue
        raise Exception("Pressure sensor not found")              # 如果所有串口都尝试失败

    def reconnect(self):
        """重新连接串口。"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.auto_find_port()

    def read_data(self):
        """
        从传感器读取原始二进制压力数据。
        Returns:
            bytes: 原始二进制数据包，如果读取失败或无效，返回 None。
        """
        if not self.ser or not self.ser.is_open:
            return None
        try:
            cmd = [0x55,0xAA,9,0,0x34,0,0xFB,0,0x1C,0,0,0xA8,0,0x35] # 压力传感器读取命令
            self.ser.write(bytearray(cmd))
            time.sleep(0.005)
            resp = self.ser.read(256)                             # 读取响应数据
            idx = resp.find(b'\xaa\x55')                          # 查找数据包起始标志
            if idx == -1 or len(resp[idx:])<182:                  # 检查数据包完整性
                return None
            return resp[idx+14:idx+14+168]                        # 提取有效数据部分 (168字节)
        except Exception as e:
            # print(f"Pressure sensor read error: {e}")
            return None

    def decode(self, raw):
        """
        解码原始二进制数据为传感器数值。
        Args:
            raw (bytes): 原始二进制压力数据。
        Returns:
            list: 包含84个传感器单元的压力值列表。
        """
        arr = []
        for i in range(0,168,2):                                  # 每两个字节为一个传感器值
            arr.append(struct.unpack("<H", raw[i:i+2])[0])        # <H 表示小端无符号短整数
        out = []
        # 注意这里的数据顺序，需要与压力传感器物理布局一致
        # 原代码中 out.extend(arr[i*7:(i+1)*7]) 意味着它读取了12行，每行7个
        # 如果你的传感器是 12行x7列，那么这个映射是正确的
        for i in range(12):                                       # 12行
            out.extend(arr[i*7:(i+1)*7])                          # 每行7个传感器
        
        self.last = out.copy()                                    # 更新last数据，用于可能的后续处理或错误恢复
        return out

# ==================== Timestamped Buffer Class ====================
class TimestampedBuffer:
    """
    带时间戳的环形缓冲区，用于存储传感器数据。
    """
    def __init__(self, maxlen=300):
        self.buf = deque(maxlen=maxlen)                           # 双端队列，固定最大长度
        self.lock = threading.Lock()                              # 保护缓冲区操作的互斥锁

    def append(self, item):
        """向缓冲区末尾添加带时间戳的数据项。"""
        with self.lock:
            self.buf.append(item)

    def get_latest(self):
        """获取缓冲区中最新的数据项。"""
        with self.lock:
            return self.buf[-1] if self.buf else None

    def find_closest(self, ts):
        """
        在缓冲区中查找时间戳最接近给定时间戳的数据项。
        Args:
            ts (float): 目标时间戳。
        Returns:
            dict: 最接近的数据项，包含 {"t": timestamp, "data": data}。
        """
        with self.lock:
            best = None
            best_dt = 1e9                                         # 初始最小时间差设为极大值
            for item in self.buf:
                dt = abs(item["t"]-ts)                            # 计算时间差
                if dt < best_dt:                                  # 找到更接近的
                    best_dt = dt
                    best = item
            return best

# ==================== Reader Threads ====================
class PressureReaderThread(threading.Thread):
    """压力传感器数据读取线程。"""
    def __init__(self, sensor, buf, stop_event):
        super().__init__(daemon=True)                             # 设为守护线程，主程序退出时自动结束
        self.sensor = sensor
        self.buf = buf                                            # 目标缓冲区
        self.stop = stop_event                                    # 停止事件，用于控制线程退出
        self.fail = 0                                             # 连续读取失败计数器

    def run(self):
        while not self.stop.is_set():                             # 循环直到停止事件被设置
            t = time.perf_counter()                               # 记录当前时间
            raw = self.sensor.read_data()                         # 读取原始数据
            if raw is None:
                self.fail +=1
                if self.fail >=30:                                # 连续失败30次尝试重连
                    print("Pressure sensor read failed repeatedly, trying to reconnect...")
                    self.sensor.reconnect()
                    self.fail=0
                time.sleep(0.002)
                continue
            try:
                data = self.sensor.decode(raw)                    # 解码数据
                self.buf.append({"t":t, "data":data})             # 将带时间戳的数据添加到缓冲区
                self.fail=0
            except Exception as e:
                # print(f"Pressure decode error: {e}") # 可以打印解码错误
                time.sleep(0.001)

class ForceReaderThread(threading.Thread):
    """力传感器数据读取线程。"""
    def __init__(self, sensor, buf, stop_event):
        super().__init__(daemon=True)
        self.sensor = sensor
        self.buf = buf
        self.stop = stop_event
        self.fail=0

    def run(self):
        while not self.stop.is_set():
            t = time.perf_counter()
            data = self.sensor.read()
            if data is None:
                self.fail +=1
                if self.fail >=30:                                # 连续失败30次尝试重连
                    print("Force sensor read failed repeatedly, trying to reconnect...")
                    self.sensor.reconnect()
                    self.fail=0
                time.sleep(0.002)
                continue
            self.buf.append({"t":t, "data":data})
            self.fail=0

# ==================== Real-time Plot (Integrated Layout) ====================
class RealTimePlot:
    """
    实时绘图类，使用Matplotlib绘制传感器数据。
    集成了多个子图，以GridSpec布局。
    """
    def __init__(self):
        # 设置Matplotlib字体和负号显示
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 定义传感器矩阵的行数和列数，作为实例属性
        self.rows = 12
        self.cols = 7

        # 创建主画布
        self.fig = plt.figure(figsize=(16, 12))
        
        # --- 主GridSpec：4行2列，宽度比例1:1 ---
        gs_outer = GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[6, 1, 1, 1], hspace=0.2, wspace=0.3)
        
        # --- 图1和图2 (方向和幅值箭头图) 的嵌套GridSpec (水平排列) ---
        gs_arrows = gs_outer[0, 0].subgridspec(1, 2, wspace=0.3)
        
        self.ax1 = plt.subplot(gs_arrows[0, 0]) # 图1占据 gs_arrows 的左半部分
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal') # 保持宽高比
        self.ax1.axis('off')         # 关闭坐标轴
        self.ax1.set_title("Direction Arrows (CoP Offset & Force)", fontsize=10)
        
        self.ax2 = plt.subplot(gs_arrows[0, 1]) # 图2占据 gs_arrows 的右半部分
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_aspect('equal')
        self.ax2.axis('off')
        self.ax2.set_title("Magnitude Arrows (CoP Offset & Force)", fontsize=10)
        
        # --- 图3a, 3b, 4 (曲线图) 直接占据 gs_outer 的相应行和第一列 ---
        
        self.ax3a = plt.subplot(gs_outer[1, 0]) # 图3a占据 gs_outer 的第二行，第一列
        self.ax3a.set_title("Raw ADC Sum (Original Values)", fontsize=10)
        self.ax3a.set_xlabel("Frame", fontsize=10)
        self.ax3a.set_ylabel("ADC Sum", fontsize=10)
        self.ax3a.grid(True, alpha=0.3)
        self.raw_adc_line, = self.ax3a.plot([], [], 'b-', linewidth=1.5, label="Raw ADC Sum")
        self.ax3a.legend(fontsize=8)
        
        self.ax3b = plt.subplot(gs_outer[2, 0]) # 图3b占据 gs_outer 的第三行，第一列
        self.ax3b.set_title("Raw Force Magnitude (Original Values)", fontsize=10)
        self.ax3b.set_xlabel("Frame", fontsize=10)
        self.ax3b.set_ylabel("Force Magnitude (N)", fontsize=10)
        self.ax3b.grid(True, alpha=0.3)
        self.raw_force_line, = self.ax3b.plot([], [], 'r-', linewidth=1.5, label="Raw Force Mag")
        self.ax3b.legend(fontsize=8)
        
        self.ax4 = plt.subplot(gs_outer[3, 0]) # 图4占据 gs_outer 的第四行，第一列
        self.ax4.set_title("Angle Error between CoP Offset and Force", fontsize=10)
        self.ax4.set_xlabel("Frame", fontsize=10)
        self.ax4.set_ylabel("Angle Error (deg)", fontsize=10)
        self.ax4.set_ylim(0, 180)
        self.ax4.grid(True, alpha=0.3)
        self.error_line, = self.ax4.plot([], [], 'g-o', linewidth=1.5, markersize=2, label="Angle Error |CoP Offset - Force|")
        self.ax4.legend(fontsize=8)
        
        # --- 右边2个子图 (压力表和梯度箭头图)，使用一个嵌套GridSpec占据 gs_outer 的整个右列 ---
        gs_right_tables = gs_outer[:, 1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.3) 
        
        self.ax5 = self.fig.add_subplot(gs_right_tables[0, 0]) 
        self.ax5.set_title("Baseline-Subtracted Pressure (12×7)", fontsize=10)
        self.ax5.axis('off')
        self.table_plot = None
        
        # ax6 的标题和布局来自第二段代码
        self.ax6 = self.fig.add_subplot(gs_right_tables[0, 1]) 
        self.ax6.set_title("Gradient Arrows (gx, gy) 12×7", fontsize=10) # 标题修改
        self.ax6.axis('off')

        # 实时数据存储变量
        self.adc_angle = 0
        self.adc_mag = 0
        self.force_angle = 0
        self.force_mag = 0
        self.raw_adc_sum = 0
        self.raw_force_mag = 0
        self.table_data = np.zeros((self.rows, self.cols))
        self.cop_x = 0.0
        self.cop_y = 0.0
        # ROI边界现在是r1, r2, c1, c2
        self.r1, self.r2, self.c1, self.c2 = 0, self.rows - 1, 0, self.cols - 1 # 初始化为整个网格
        self.delta_CoP_x = 0.0
        self.delta_CoP_y = 0.0
        self.base_CoP_x_for_plot = 0.0
        self.base_CoP_y_for_plot = 0.0

        self.lock = threading.Lock()
        self.fixed_arrow = 0.35
        self.epsilon = 1e-8
        self.frame_counter = 0

        self.ani = FuncAnimation(self.fig, self.update_all, interval=PLOT_INTERVAL_MS, cache_frame_data=False)

    def set_data(self, adc_a, adc_m, f_a, f_m, diff_frame, raw_adc_sum, raw_force_mag, cop_x, cop_y, r1, r2, c1, c2,
                 delta_CoP_x_val, delta_CoP_y_val, base_cop_x_plot, base_cop_y_plot):
        """
        更新绘图所需的所有数据。
        """
        with self.lock:
            self.adc_angle = adc_a
            self.adc_mag = adc_m
            self.force_angle = f_a
            self.force_mag = f_m
            self.raw_adc_sum = raw_adc_sum
            self.raw_force_mag = raw_force_mag
            self.table_data = diff_frame.reshape(self.rows, self.cols)
            self.cop_x = cop_x
            self.cop_y = cop_y
            self.r1, self.r2, self.c1, self.c2 = r1, r2, c1, c2 # 更新ROI边界
            self.delta_CoP_x = delta_CoP_x_val
            self.delta_CoP_y = delta_CoP_y_val
            self.base_CoP_x_for_plot = base_cop_x_plot
            self.base_CoP_y_for_plot = base_cop_y_plot
            self.frame_counter += 1
            
            # 更新历史数据队列
            diff = abs(adc_a - f_a)
            error = min(diff, 360 - diff)
            angle_error_history.append(error)
            
            adc_mag_history.append(adc_m)
            force_mag_history.append(f_m)
            frame_count_history.append(self.frame_counter)
            
            raw_adc_sum_history.append(raw_adc_sum)
            raw_force_mag_history.append(raw_force_mag)

    def update_all(self, frame):
        """
        FuncAnimation 的更新函数，每次更新绘制所有子图。
        """
        # 更新所有子图
        self.update_direction_arrows()
        self.update_magnitude_arrows()
        self.update_raw_adc_sum()
        self.update_raw_force_mag()
        self.update_angle_error()
        self.update_pressure_table()
        self.update_gradient_table() # 此处会尝试绘制梯度箭头，现在 grad_table_data 不再全零
        
        return []

    def update_direction_arrows(self):
        """更新方向箭头图 (ax1)。"""
        with self.lock:
            a = self.adc_angle
            f = self.force_angle
        
        self.ax1.clear()
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal')
        self.ax1.axis('off')
        self.ax1.set_title("Direction Arrows (CoP Offset & Force)", fontsize=10)
        
        # ADC(CoP偏移)方向箭头（黑色）
        th_adc = np.deg2rad(a)
        self.ax1.arrow(0.5, 0.5, 0.4*np.cos(th_adc), 0.4*np.sin(th_adc), 
                      head_width=0.12, fc='k', ec='k', lw=2.5, length_includes_head=True)
        self.ax1.text(0.5, 0.1, f"CoP Offset: {a:.1f}°", ha='center', va='center', fontsize=8, color='black')
        
        # Force方向箭头（红色）
        th_force = np.deg2rad(f)
        self.ax1.arrow(0.5, 0.5, self.fixed_arrow*np.cos(th_force), self.fixed_arrow*np.sin(th_force), 
                      head_width=0.1, fc='r', ec='r', lw=2, length_includes_head=True)
        self.ax1.text(0.5, 0.9, f"Force: {f:.1f}°", ha='center', va='center', fontsize=8, color='red')

    def update_magnitude_arrows(self):
        """更新幅值箭头图 (ax2)。"""
        with self.lock:
            a = self.adc_angle
            m = self.adc_mag
            fa = self.force_angle
            fm = self.force_mag
        
        self.ax2.clear()
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_aspect('equal')
        self.ax2.axis('off')
        self.ax2.set_title("Magnitude Arrows (CoP Offset & Force)", fontsize=10)
        
        # ADC(CoP偏移)幅值箭头（黑色），长度按CoP偏移幅值线性映射
        th_adc = np.deg2rad(a)
        max_adc_length = 0.45
        l_adc = (m / 10.0) * max_adc_length if m > self.epsilon else 0.0
        l_adc = min(l_adc, max_adc_length)
        self.ax2.arrow(0.5, 0.5, l_adc*np.cos(th_adc), l_adc*np.sin(th_adc), 
                      head_width=0.12, fc='k', ec='k', lw=2.5, length_includes_head=True)
        self.ax2.text(0.5, 0.1, f"CoP Offset: {m:.2f}", ha='center', va='center', fontsize=8, color='black')

        # Force幅值箭头（红色），长度按Force幅值线性映射
        th_force = np.deg2rad(fa)
        max_force_length = 0.4
        l_force = (abs(fm) / 20.0) * max_force_length if abs(fm) > self.epsilon else 0.0
        l_force = min(l_force, max_force_length)
        
        if l_force > 0.02:
            head_width = max(0.06, l_force * 0.15)
            head_length = max(0.04, l_force * 0.1)
            self.ax2.arrow(0.5, 0.5, l_force*np.cos(th_force), l_force*np.sin(th_force), 
                          head_width=head_width, head_length=head_length, 
                          fc='red', ec='darkred', 
                          lw=3.5, length_includes_head=True, 
                          alpha=1.0, joinstyle='round', capstyle='round')
        else:
            self.ax2.plot([0.5, 0.5 + l_force*np.cos(th_force)], 
                         [0.5, 0.5 + l_force*np.sin(th_force)], 
                         'r-', lw=3.5, alpha=1.0)
        
        self.ax2.text(0.5, 0.9, f"Force: {fm:.1f}", ha='center', va='center', fontsize=8, color='red')

    def update_raw_adc_sum(self):
        """更新原始ADC总和曲线图 (ax3a)。"""
        if len(raw_adc_sum_history) > 0:
            xs = list(range(len(raw_adc_sum_history)))
            ys = list(raw_adc_sum_history)
            self.raw_adc_line.set_data(xs, ys)
            
            if len(ys) > 0:
                self.ax3a.set_ylim(min(ys) * 0.95, max(ys) * 1.05 if max(ys) > 0 else 1)
            self.ax3a.set_xlim(0, max(len(xs), 1))

    def update_raw_force_mag(self):
        """更新原始Force幅值曲线图 (ax3b)。"""
        if len(raw_force_mag_history) > 0:
            xs = list(range(len(raw_force_mag_history)))
            ys = list(raw_force_mag_history)
            self.raw_force_line.set_data(xs, ys)
            
            if len(ys) > 0:
                self.ax3b.set_ylim(min(ys) * 0.95, max(ys) * 1.05 if max(ys) > 0 else 1)
            self.ax3b.set_xlim(0, max(len(xs), 1))

    def update_angle_error(self):
        """更新角度误差曲线图 (ax4)。"""
        if len(angle_error_history) > 0:
            xs = list(range(len(angle_error_history)))
            ys = list(angle_error_history)
            self.error_line.set_data(xs, ys)
            
            self.ax4.set_xlim(0, max(len(xs), 1))

    def update_pressure_table(self):
        """
        更新压力表 (ax5)，显示初始CoP、动态CoP和CoP偏移向量。
        """
        with self.lock:
            data = self.table_data.copy()
            cop_x_plot = self.cop_x
            cop_y_plot = self.cop_y
            r1, r2, c1, c2 = self.r1, self.r2, self.c1, self.c2 # 使用实例属性的ROI边界
            # 获取CoP偏移量和基准CoP，用于在ax5上绘制
            delta_cop_x = self.delta_CoP_x
            delta_cop_y = self.delta_CoP_y
            base_cop_x = self.base_CoP_x_for_plot
            base_cop_y = self.base_CoP_y_for_plot


        self.ax5.clear()
        self.ax5.set_title("Baseline-Subtracted Pressure (12×7)", fontsize=10)
        self.ax5.axis('off')

        nrows, ncols = self.rows, self.cols
        
        self.ax5.set_xlim(-0.5, ncols - 0.5) # X轴对应列 (0到6)
        self.ax5.set_ylim(nrows - 0.5, -0.5) # Y轴对应行 (0到11), 反转使0在顶部
        self.ax5.set_aspect('equal')
        self.ax5.grid(False)

        vmax = np.max(data) if np.max(data) != 0 else 1
        norm = data / vmax
        cmap = plt.colormaps['Reds']
        colors = cmap(norm)
        cell_text = [[f"{v:.0f}" for v in row] for row in data]

        self.table_plot = self.ax5.table(
            cellText=cell_text, 
            cellColours=colors,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        self.table_plot.auto_set_font_size(False)
        self.table_plot.set_fontsize(8)
        
        for i in range(nrows):
            for j in range(ncols):
                cell = self.table_plot[(i, j)]
                cell.set_height(1/nrows)
                cell.set_width(1/ncols)

        # ====== 在此添加初始CoP和CoP偏移向量的绘图逻辑（来自第二段代码） ======

        # 绘制初始CoP (Base CoP)
        if not np.isnan(base_cop_x) and not np.isnan(base_cop_y):
            self.ax5.plot(base_cop_x, base_cop_y, 'bx', markersize=10, 
                          label=f'Initial CoP ({base_cop_x:.1f}, {base_cop_y:.1f})', zorder=10)
        
        # 绘图时直接使用浮点数的CoP坐标 (Current CoP)
        self.ax5.scatter(cop_x_plot, cop_y_plot, s=150, color='green', marker='o', zorder=10, 
                         label=f"Current CoP ({cop_x_plot:.1f}, {cop_y_plot:.1f})")

        # 绘制CoP偏移向量 (从Base CoP指向Current CoP)
        # 只有当偏移量足够大时才绘制箭头，避免噪声箭头
        if np.hypot(delta_cop_x, delta_cop_y) > 0.05: # 阈值与ax6保持一致
            self.ax5.arrow(base_cop_x, base_cop_y, 
                          delta_cop_x,
                          delta_cop_y,
                          head_width=0.3, head_length=0.3, fc='purple', ec='purple', linewidth=2, zorder=5,
                          label=f'CoP Offset (dx={delta_cop_x:.1f}, dy={delta_cop_y:.1f})')

        # =========================================================================

        # 绘制基于连通域分析的 ROI 框 (蓝色虚线框)
        rect_x = c1 - 0.5  # ROI左边缘的列索引 - 0.5 (对齐网格线)
        rect_y = r1 - 0.5  # ROI上边缘的行索引 - 0.5 (对齐网格线)
        rect_width = (c2 - c1 + 1) # ROI宽度 (包含的列数)
        rect_height = (r2 - r1 + 1) # ROI高度 (包含的行数)
        
        roi_rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                             linewidth=3, edgecolor='blue', facecolor='none', linestyle='--', zorder=5, label="Dynamic ROI")
        self.ax5.add_patch(roi_rect)
        self.ax5.legend(loc='upper left', fontsize=8)


    def update_gradient_table(self):
        """
        更新梯度箭头图 (ax6)，显示每个点的梯度。
        此实现完全参考了第二段代码的 `update_gradient_table` 逻辑。
        """
        with self.lock:
            # 这里的data是全局变量grad_table_data，它在compute_gradient_in_region中被更新
            data = grad_table_data.copy() 
            cop_x_plot = self.cop_x      # CoP x坐标
            cop_y_plot = self.cop_y      # CoP y坐标
            r1, r2, c1, c2 = self.r1, self.r2, self.c1, self.c2 # ROI边界 (r1=min_row, r2=max_row, c1=min_col, c2=max_col)
            # 获取CoP偏移量和基准CoP，用于在ax6上绘制
            delta_cop_x = self.delta_CoP_x
            delta_cop_y = self.delta_CoP_y
            base_cop_x = self.base_CoP_x_for_plot
            base_cop_y = self.base_CoP_y_for_plot

        # 清空并重新设置子图
        self.ax6.clear()
        # 标题保持不变，显示梯度箭头
        self.ax6.set_title("Gradient Arrows (gx, gy) 12×7", fontsize=10)
        
        nrows, ncols = self.rows, self.cols 
        # 设置坐标轴范围，使每个单元格在0到cols-1和0到rows-1之间
        self.ax6.set_xlim(-0.5, ncols - 0.5)  # X轴对应列 (0到6)
        self.ax6.set_ylim(nrows - 0.5, -0.5)  # Y轴对应行 (0到11), 反转使0在顶部
        self.ax6.set_aspect('equal') # 保持宽高比
        self.ax6.axis('off') # 关闭坐标轴
        
        # 在背景绘制12x7的表格网格 (细线)，颜色改为黑色
        for r_grid in range(nrows + 1): # Use r_grid to avoid conflict with `r` in loop below
            self.ax6.axhline(r_grid - 0.5, color='black', linestyle='-', linewidth=0.5, zorder=0)
        for c_grid in range(ncols + 1): # Use c_grid
            self.ax6.axvline(c_grid - 0.5, color='black', linestyle='-', linewidth=0.5, zorder=0)

        # 遍历每个传感器单元绘制梯度箭头
        for r in range(nrows):
            for c in range(ncols):
                gx, gy = data[r, c, 0], data[r, c, 1] # 获取该单元格的梯度分量
                
                mag = np.hypot(gx, gy) # 计算梯度幅值
                
                # 只绘制幅值大于一定阈值的箭头，避免绘制微小噪声 (阈值1.0借鉴第二段代码)
                if mag > 1.0: 
                    # 归一化梯度分量，以便统一箭头长度
                    gx_norm = gx / mag
                    gy_norm = gy / mag
                    
                    # quiver(X, Y, U, V) 函数默认将箭头从 (X, Y) 绘制到 (X+U, Y+V)。
                    # 调整 quiver 参数以使箭杆更粗、箭头头更明显 (参数值借鉴第二段代码)
                    self.ax6.quiver(c, r, gx_norm, gy_norm, 
                                    color='k', 
                                    scale=2.5, # 调整 scale 使箭头的视觉长度合适 (1/2.5 = 0.4 cell units)
                                    width=0.02, # 箭杆粗细 (数据单位)
                                    headwidth=6, headlength=8, headaxislength=7, # 箭头头尺寸 (点)
                                    angles='xy', scale_units='xy', zorder=5)

        # --- 在梯度表上绘制CoP绿点 ---
        if not np.isnan(cop_x_plot) and not np.isnan(cop_y_plot):
            self.ax6.scatter(cop_x_plot, cop_y_plot, s=150, color='green', marker='o', zorder=10, 
                            label=f"Current CoP ({cop_x_plot:.1f}, {cop_y_plot:.1f})") # CoP绿色点

        # --- 在梯度表上绘制初始CoP (Base CoP) ---
        if not np.isnan(base_cop_x) and not np.isnan(base_cop_y):
            self.ax6.plot(base_cop_x, base_cop_y, 'bx', markersize=10, 
                          label=f'Base CoP ({base_cop_x:.1f}, {base_cop_y:.1f})', zorder=10)

        # --- 在梯度表上绘制CoP偏移向量 (从Base CoP指向Current CoP) ---
        if np.hypot(delta_cop_x, delta_cop_y) > 0.05:
            self.ax6.arrow(base_cop_x, base_cop_y, 
                          delta_cop_x,
                          delta_cop_y,
                          head_width=0.3, head_length=0.3, fc='purple', ec='purple', linewidth=2, zorder=5,
                          label=f'CoP Offset (dx={delta_cop_x:.1f}, dy={delta_cop_y:.1f})')


        # --- 在梯度表上绘制动态ROI蓝框 ---
        rect_x = c1 - 0.5
        rect_y = r1 - 0.5
        rect_width = (c2 - c1 + 1)
        rect_height = (r2 - r1 + 1)
        
        # ROI框的粗细和颜色与压力表上的保持一致
        roi_rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                             linewidth=3, edgecolor='blue', facecolor='none', linestyle='--', zorder=5, label="Dynamic ROI")
        self.ax6.add_patch(roi_rect) # 添加矩形到子图

        # --- 绘制整个 12x7 区域的粗黑色外框 ---
        full_grid_rect = Rectangle((-0.5, -0.5), ncols, nrows,
                                   linewidth=2, edgecolor='black', facecolor='none', zorder=1) 
        self.ax6.add_patch(full_grid_rect)
        self.ax6.legend(loc='upper left', fontsize=8) # 添加图例


    def show(self):
        """显示实时绘图窗口。"""
        plt.tight_layout() 
        plt.show()

# ==================== Data Collector Class ====================
class DataCollector:
    """
    数据收集器类，负责从传感器读取数据，进行处理，并更新实时绘图及保存数据到CSV。
    """
    def __init__(self, force_sensor, press_sensor, csv_path):
        self.force_sensor = force_sensor
        self.press_sensor = press_sensor
        self.csv_path = csv_path
        self.running = True
        self.stop_event = threading.Event()
        self.press_buf = TimestampedBuffer(PRESS_BUFFER_SIZE)
        self.force_buf = TimestampedBuffer(FORCE_BUFFER_SIZE)
        self.plot = None
        self.start_time = None

    def set_plot(self, p):
        """设置绘图对象。"""
        self.plot = p

    def start(self):
        """启动数据收集线程。"""
        self.press_thread = PressureReaderThread(self.press_sensor, self.press_buf, self.stop_event)
        self.force_thread = ForceReaderThread(self.force_sensor, self.force_buf, self.stop_event)
        self.press_thread.start()
        self.force_thread.start()
        threading.Thread(target=self.run_collect, daemon=True).start()

    def run_collect(self):
        """
        主数据收集和处理循环。
        """
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            header = ["timestamp","rel_ms"] + [f"ch{i+1}" for i in range(84)] + \
                     ["Fx","Fy","Fz","Mx","My","Mz","press_t","force_t","dt",
                      "ADC_angle","ADC_mag","Force_angle","Force_mag", 
                      "CoP_X", "CoP_Y", "ROI_r1", "ROI_r2", "ROI_c1", "ROI_c2",
                      "Delta_CoP_X", "Delta_CoP_Y", "Base_CoP_X_Plot", "Base_CoP_Y_Plot"]
            writer.writerow(header)
            period = 1.0 / TARGET_HZ

            while self.running:
                t0 = time.perf_counter()
                if self.start_time is None:
                    self.start_time = t0
                rel_ms = int((t0 - self.start_time) * 1000)

                p_item = self.press_buf.get_latest()
                f_item = self.force_buf.find_closest(p_item["t"]) if p_item else None
                if not p_item or not f_item or abs(p_item["t"] - f_item["t"]) > MAX_SYNC_DT:
                    time.sleep(0.001)
                    continue

                p_data = p_item["data"]
                f_data = f_item["data"]
                
                raw_adc_sum = np.sum(p_data)
                
                diff_frame = subtract_baseline(p_data)

                # compute_gradient_in_region 现在会同时计算CoP偏移方向和更新grad_table_data
                dir_x, dir_y, vec_mag, cop_x, cop_y, r1, r2, c1, c2, delta_CoP_x, delta_CoP_y, base_cop_x_for_plot, base_cop_y_for_plot = compute_gradient_in_region(diff_frame)
                adc_angle, _ = compute_gradient_angle_single(dir_x, dir_y)
                
                fx, fy = f_data[0], f_data[1]
                f_angle, f_mag = compute_force_angle(fx, fy)
                
                raw_force_mag = np.hypot(fx, fy)

                full_time_list.append(rel_ms)
                full_adc_mag_list.append(vec_mag)
                full_force_mag_list.append(f_mag)

                if self.plot:
                    self.plot.set_data(adc_angle, vec_mag, f_angle, f_mag, diff_frame, raw_adc_sum, raw_force_mag, 
                                       cop_x, cop_y, r1, r2, c1, c2, # 将ROI边界传递给绘图对象
                                       delta_CoP_x, delta_CoP_y, base_cop_x_for_plot, base_cop_y_for_plot)

                ts_str = time.strftime("%Y%m%d%H%M%S%f")[:-3]
                row = [ts_str, rel_ms] + p_data + f_data + [
                    round(p_item["t"],6), round(f_item["t"],6), round(abs(p_item["t"]-f_item["t"])*1000,3),
                    adc_angle, vec_mag, f_angle, f_mag,
                    cop_x, cop_y, r1, r2, c1, c2, # ROI边界现在也被写入CSV
                    delta_CoP_x, delta_CoP_y, 
                    base_cop_x_for_plot, base_cop_y_for_plot
                ]
                writer.writerow(row)
                csv_file.flush()
                
                elapsed = time.perf_counter() - t0
                time.sleep(max(0, period-elapsed))

    def stop(self):
        """停止数据收集线程。"""
        self.running = False
        self.stop_event.set()
        time.sleep(0.3)

# ==================== 程序结束绘制全程静态图 ====================
def plot_full_magnitude_curve():
    """
    在程序结束后绘制全程的ADC(CoP偏移)和力传感器幅值曲线。
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    if len(full_time_list) > 0 and len(full_adc_mag_list) > 0:
        ax1.plot(full_time_list, full_adc_mag_list, 'b-', linewidth=1.5, label='CoP Offset Magnitude')
        ax1.set_title("CoP Offset Magnitude Over Time", fontsize=14)
        ax1.set_xlabel("Time (ms)", fontsize=12)
        ax1.set_ylabel("CoP Offset Magnitude (Units)", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.set_ylim(0, max(full_adc_mag_list) * 1.1 if max(full_adc_mag_list) > 0 else 1)
    
    if len(full_time_list) > 0 and len(full_force_mag_list) > 0:
        ax2.plot(full_time_list, full_force_mag_list, 'r-', linewidth=1.5, label='Force Magnitude')
        ax2.set_title("Force Magnitude Over Time", fontsize=14)
        ax2.set_xlabel("Time (ms)", fontsize=12)
        ax2.set_ylabel("Force Magnitude (N)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.set_ylim(0, max(full_force_mag_list) * 1.1 if max(full_force_mag_list) > 0 else 1)
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "full_magnitude_curve_cop.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

# ==================== Main Program Execution ====================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    i = 1
    while os.path.exists(os.path.join(SAVE_DIR, f"data_{i}.csv")):
        i += 1
    csv_path = os.path.join(SAVE_DIR, f"data_{i}.csv")

    force_sensor = SixAxisForceSensor()
    press_sensor = PressureSensor()
    force_sensor.calibrate_zero()

    plot = RealTimePlot()
    collector = DataCollector(force_sensor, press_sensor, csv_path)
    collector.set_plot(plot)
    collector.start()

    print("✅ CoP offset direction calculation enabled")
    print("✅ Real-time angle error plot enabled")
    # 更新打印信息以匹配连通域ROI的描述
    print("✅ 12×7 real-time pressure table enabled (with Initial CoP, dynamic CoP, and dynamic Connected-Component ROI)") 
    print("✅ 12×7 real-time gradient arrows enabled (with shafts and head at moving end, within dynamic ROI)")
    print("✅ Raw ADC & Force magnitude plots enabled (original values)")
    print(f"✅ Saving data to: {csv_path}")
    
    # CoP稳定性提示信息保留第一段代码的描述
    print(f"CoP stability requires {COP_STABILITY_FRAMES_REQUIRED} consecutive frames "
          f"where all points are within {COP_STABILITY_TOLERANCE} grid units "
          f"perpendicular distance to the line defined by the first two CoP points in the sequence.")
    print(f"Dynamic threshold for pressure detection: Calculated as (Average Positive ADC) * {THRESHOLD_MULTIPLIER}, with a minimum floor of {MIN_THRESHOLD_FLOOR}.")
    
    plot.show()
    
    collector.stop()
    
    print("\n📊 Generating full-time magnitude curve...")
    plot_full_magnitude_curve()
    
    print("✅ Program finished")