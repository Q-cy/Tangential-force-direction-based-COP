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

# ==================== Config ====================
BAUDRATE_PRESS = 921600         # 压力传感器串口波特率
BAUDRATE_FORCE = 460860         # 六轴力传感器串口波特率 # 修正了这里，之前是460800
SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test" # 数据保存目录
TARGET_HZ = 100.0               # 目标数据采集和处理频率 (Hz)
PLOT_INTERVAL_MS = 100          # Matplotlib实时绘图的更新间隔 (毫秒)，例如100ms即每秒更新10次
MAX_SYNC_DT = 0.015             # 压力传感器和力传感器数据同步允许的最大时间差 (秒)
PRESS_BUFFER_SIZE = 500         # 压力传感器数据缓冲区大小
FORCE_BUFFER_SIZE = 500         # 力传感器数据缓冲区大小
THRESHOLD = 1000                # 压力阈值，低于此值的压力被认为是零，不参与CoP和梯度计算

# Smoothing 参数
DIR_SMOOTH_ALPHA = 0.15         # 方向向量指数平滑的 alpha 值 (0-1)，越大越平滑
ERROR_PLOT_LEN = 100            # 角度误差绘图历史队列长度
MAG_PLOT_LEN = 100              # 幅度绘图历史队列长度
# ================================================================

# ==================== Global Variables ====================
# Baseline subtraction 相关的全局变量
first_frame = None                                        # 存储第一帧数据，用于基线减除
first_frame_lock = threading.Lock()                       # 保护 first_frame 的互斥锁，确保线程安全访问

# 梯度方向相关的全局变量
adc_filtered_dir = None                                   # 滤波后的ADC（压力梯度）方向向量 (fx, fy)

# 历史数据队列，用于实时绘图
angle_error_history = deque(maxlen=ERROR_PLOT_LEN)        # 存储ADC和力传感器角度误差的历史
grad_table_data = np.zeros((11, 6, 2))                    # 存储每个压力点 (gx, gy) 梯度的全局变量，用于梯度表格, (12-1)x(7-1)
adc_mag_history = deque(maxlen=MAG_PLOT_LEN)              # 存储ADC（压力梯度）幅值的历史
force_mag_history = deque(maxlen=MAG_PLOT_LEN)            # 存储力传感器幅值的历史
frame_count_history = deque(maxlen=MAG_PLOT_LEN)          # 存储帧计数，作为历史图表的X轴

# ==================== 原始值历史记录 ====================
# 用于存储原始（未经基线减除、滤波等处理）ADC总和和力传感器幅值，用于独立绘图
raw_adc_sum_history = deque(maxlen=MAG_PLOT_LEN)
raw_force_mag_history = deque(maxlen=MAG_PLOT_LEN)

# ==================== 全程数据保存 ====================
# 用于在程序结束后绘制完整的数据曲线
full_time_list = []                                       # 记录所有数据点的时间戳
full_adc_mag_list = []                                    # 记录所有ADC（压力梯度）幅值
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

# ==================== Gradient in Symmetric Region ====================
def compute_gradient_in_region(frame):
    """
    计算给定压力帧的梯度，并返回CoP区域的平均梯度方向。
    引入了边界处理，使得边缘点的梯度也能计算。
    Args:
        frame (np.array): 经过基线减除的1D压力数据 (84个传感器单元)。
    Returns:
        tuple: (x方向梯度分量, y方向梯度分量, 梯度幅值, CoP_x, CoP_y, ROI_r1, ROI_r2, ROI_c1, ROI_c2)。
               如果无有效压力，则返回 (0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)。
    """
    global adc_filtered_dir, grad_table_data
    
    # 原始压力传感器矩阵的尺寸
    original_rows, original_cols = 12, 7
    # 梯度矩阵的尺寸，比压力矩阵小一行一列
    grad_rows = original_rows - 1 # 11
    grad_cols = original_cols - 1 # 6
    
    frame_flat = np.array(frame, dtype=np.float32).flatten()      # 确保是浮点数一维数组
    frame2d = frame_flat.reshape(original_rows, original_cols)    # 重塑为二维矩阵
    threshold = THRESHOLD                                         # 压力阈值

    # 第一阶段：计算 (rows-1) x (cols-1) 矩阵中每个点的梯度 (gx, gy)
    grad = np.zeros((grad_rows, grad_cols, 2), dtype=np.float32)  # 存储每个点的 (gx, gy)
    
    # === MODIFICATION START: 梯度计算方式修改 ===
    # 遍历 (rows-1) x (cols-1) 的区域，即不计算最右列和最下行的梯度
    for y in range(grad_rows): # 遍历到 original_rows - 2
        for x in range(grad_cols): # 遍历到 original_cols - 2
            gx = 0.0 # 默认gx为0
            gy = 0.0 # 默认gy为0

            # gx: 右边点 - 当前点。x+1 在 original_cols - 1 范围内
            gx = frame2d[y, x+1] - frame2d[y, x]
            
            # gy: 当前点 - 下面点。y+1 在 original_rows - 1 范围内
            gy = frame2d[y, x] - frame2d[y+1, x]
            
            grad[y, x] = (gx, gy)
    # === MODIFICATION END ===

    grad_table_data = grad.copy()                                 # 更新全局梯度数据，用于显示梯度表格/图 (现在是11x6)

    # 找出所有压力值高于阈值的传感器单元的索引 (基于原始完整压力帧)
    valid_indices = np.where(frame_flat > threshold)[0].tolist()
    if not valid_indices:                                         # 如果没有有效的压力点
        adc_filtered_dir = None                                   # 重置滤波方向
        # 即使没有有效压力点，也要返回 CoP 和 ROI 的默认值以避免绘图错误
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, grad_rows-1, 0, grad_cols-1

    # ====================== CoP 压力中心计算 (基于原始完整压力帧) ======================
    mask = frame2d > threshold                                    # 创建布尔掩码，标记有效压力点
    # 创建x坐标和y坐标网格，用于加权平均 (基于原始完整压力帧)
    x_grid = np.tile(np.arange(original_cols), (original_rows, 1))
    y_grid = np.repeat(np.arange(original_rows), original_cols).reshape(original_rows, original_cols)
    
    total_pressure = np.sum(frame2d[mask])                        # 计算总压力
    if total_pressure < 1e-3:                                     # 如果总压力过小，也视为无有效压力
        adc_filtered_dir = None
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0, grad_rows-1, 0, grad_cols-1

    cop_x = np.sum(frame2d[mask] * x_grid[mask]) / total_pressure # 计算CoP的X坐标
    cop_y = np.sum(frame2d[mask] * y_grid[mask]) / total_pressure # 计算CoP的Y坐标

    # 将浮点数的CoP坐标四舍五入到最近的整数，作为中心单元格的索引 (基于原始完整压力帧)
    center_r = int(round(cop_y))
    center_c = int(round(cop_x))
    # ==========================================================================

    # 计算CoP到各个边界的距离，用于确定对称扩展的最大范围 (基于原始完整压力帧)
    d_up    = center_r
    d_down  = (original_rows - 1) - center_r
    d_left  = center_c
    d_right = (original_cols - 1) - center_c

    # 确定在行和列方向上可以对称扩展的最大距离，受限于最近的边界
    expand_r = min(d_up, d_down)
    expand_c = min(d_left, d_right)

    # 初始ROI边界 (基于原始完整压力帧的对称区域)
    final_r1_full = center_r - expand_r
    final_r2_full = center_r + expand_r
    final_c1_full = center_c - expand_c
    final_c2_full = center_c + expand_c

    # 将ROI边界裁剪到梯度矩阵的有效范围 (0到grad_rows-1, 0到grad_cols-1)
    final_r1 = max(0, final_r1_full)
    final_r2 = min(grad_rows - 1, final_r2_full) # 梯度矩阵的最大行索引
    final_c1 = max(0, final_c1_full)
    final_c2 = min(grad_cols - 1, final_c2_full) # 梯度矩阵的最大列索引

    # 如果裁剪后ROI变得无效 (例如，起始索引大于结束索引)
    if final_r1 > final_r2 or final_c1 > final_c2:
        adc_filtered_dir = None
        # 即使ROI无效，也返回CoP和ROI的默认值
        return 0.0, 0.0, 0.0, cop_x, cop_y, 0, grad_rows-1, 0, grad_cols-1

    # 第二阶段：在CoP为中心的对称矩形区域内 (裁剪到梯度矩阵范围)，对梯度进行加权求和
    sum_gx = 0.0
    sum_gy = 0.0
    sum_w  = 0.0

    # 遍历裁剪后的ROI区域内的每个传感器单元
    for y in range(final_r1, final_r2 + 1):
        for x in range(final_c1, final_c2 + 1):
            val = frame2d[y, x]                                   # 获取当前单元格的压力值 (来自原始完整压力帧)
            if val < 1:                                           # 如果压力值过小，不参与梯度加权
                continue
            sum_gx += val * grad[y, x, 0]                         # 压力加权X方向梯度 (使用裁剪后的梯度数据)
            sum_gy += val * grad[y, x, 1]                         # 压力加权Y方向梯度 (使用裁剪后的梯度数据)
            sum_w  += val                                         # 压力加权总和

    if sum_w < 1e-3:                                              # 如果加权总和过小，视为无有效梯度
        adc_filtered_dir = None
        return 0.0, 0.0, 0.0, cop_x, cop_y, final_r1, final_r2, final_c1, final_c2

    vec_mag = np.hypot(sum_gx, sum_gy)                            # 计算加权后总梯度的幅值
    if vec_mag < 1e-3:                                            # 如果梯度幅值过小
        adc_filtered_dir = None
        return 0.0, 0.0, 0.0, cop_x, cop_y, final_r1, final_r2, final_c1, final_c2

    raw_x = sum_gx / vec_mag                                      # 归一化X方向梯度分量
    raw_y = sum_gy / vec_mag                                      # 归一化Y方向梯度分量

    # 指数加权移动平均 (EWMA) 滤波梯度方向
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
    
    # 返回滤波后的梯度方向分量、幅值，以及CoP和ROI边界
    return fx, fy, vec_mag, cop_x, cop_y, final_r1, final_r2, final_c1, final_c2

# ==================== Angle Calculation ====================
def compute_gradient_angle_single(x, y):
    """
    计算给定(x, y)向量的角度和幅值。
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
    计算给定力向量(Fx, Fy)的角度和幅值。
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
        if not self.ser:
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
        except:
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
        if not self.ser:
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
        except:
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
        self.press_rows = 12
        self.press_cols = 7
        self.grad_rows = self.press_rows - 1 # 梯度网格的行数
        self.grad_cols = self.press_cols - 1 # 梯度网格的列数

        # 创建主画布，左右各占一半宽度
        self.fig = plt.figure(figsize=(16, 12))
        
        # 创建GridSpec：总共5行2列
        # 左边：方向箭头、幅值箭头、原始ADC和Force幅值曲线、角度误差曲线
        # 右边：压力表和梯度箭头图
        gs = GridSpec(5, 2, width_ratios=[1, 1], height_ratios=[1, 1, 0.8, 0.8, 1], hspace=0.3, wspace=0.3)
        
        # --- 左边5个子图 ---
        # 图1：方向箭头图 (显示ADC和力的归一化方向)
        self.ax1 = plt.subplot(gs[0, 0])
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal') # 保持宽高比
        self.ax1.axis('off')         # 关闭坐标轴
        self.ax1.set_title("Direction Arrows (ADC & Force)", fontsize=10)
        
        # 图2：幅值箭头图 (显示ADC和力的幅值，箭头长度按比例)
        self.ax2 = plt.subplot(gs[1, 0])
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_aspect('equal')
        self.ax2.axis('off')
        self.ax2.set_title("Magnitude Arrows (ADC & Force)", fontsize=10)
        
        # 图3a：原始ADC总和曲线图
        self.ax3a = plt.subplot(gs[2, 0])
        self.ax3a.set_title("Raw ADC Sum (Original Values)", fontsize=10)
        self.ax3a.set_xlabel("Frame", fontsize=10)
        self.ax3a.set_ylabel("ADC Sum", fontsize=10)
        self.ax3a.grid(True, alpha=0.3)
        self.raw_adc_line, = self.ax3a.plot([], [], 'b-', linewidth=1.5, label="Raw ADC Sum") # 初始化曲线对象
        self.ax3a.legend(fontsize=8)
        
        # 图3b：原始Force幅值曲线图
        self.ax3b = plt.subplot(gs[3, 0])
        self.ax3b.set_title("Raw Force Magnitude (Original Values)", fontsize=10)
        self.ax3b.set_xlabel("Frame", fontsize=10)
        self.ax3b.set_ylabel("Force Magnitude (N)", fontsize=10)
        self.ax3b.grid(True, alpha=0.3)
        self.raw_force_line, = self.ax3b.plot([], [], 'r-', linewidth=1.5, label="Raw Force Mag") # 初始化曲线对象
        self.ax3b.legend(fontsize=8)
        
        # 图4：角度误差曲线图
        self.ax4 = plt.subplot(gs[4, 0])
        self.ax4.set_title("Angle Error between ADC and Force", fontsize=10)
        self.ax4.set_xlabel("Frame", fontsize=10)
        self.ax4.set_ylabel("Angle Error (deg)", fontsize=10)
        self.ax4.set_ylim(0, 180) # 角度误差范围
        self.ax4.grid(True, alpha=0.3)
        self.error_line, = self.ax4.plot([], [], 'g-o', linewidth=1.5, markersize=2, label="Angle Error |ADC - Force|") # 初始化曲线对象
        self.ax4.legend(fontsize=8)
        
        # --- 右边2个子图 ---
        gs_right = gs[:, 1].subgridspec(2, 1, hspace=0.2)
        
        self.ax5 = self.fig.add_subplot(gs_right[0, 0]) # 压力表占据右侧GridSpec的第一行
        self.ax5.set_title(f"Baseline-Subtracted Pressure ({self.press_rows}×{self.press_cols})", fontsize=10)
        self.ax5.axis('off')
        self.table_plot = None # Matplotlib表格对象
        
        self.ax6 = self.fig.add_subplot(gs_right[1, 0]) # 梯度表占据右侧GridSpec的第二行
        self.ax6.set_title(f"Gradient Arrows (gx, gy) ({self.grad_rows}×{self.grad_cols})", fontsize=10) # 梯度表显示 (11x6)
        self.ax6.axis('off')

        # 实时数据存储变量
        self.adc_angle = 0
        self.adc_mag = 0
        self.force_angle = 0
        self.force_mag = 0
        self.raw_adc_sum = 0
        self.raw_force_mag = 0
        self.table_data = np.zeros((self.press_rows, self.press_cols)) # 压力数据是 (12x7)
        self.cop_x = 0.0 # CoP X坐标
        self.cop_y = 0.0 # CoP Y坐标
        self.final_r1, self.final_r2, self.final_c1, self.final_c2 = 0, 0, 0, 0 # ROI边界

        self.lock = threading.Lock() # 保护数据更新的互斥锁
        self.fixed_arrow = 0.35      # 箭头固定长度
        self.epsilon = 1e-8          # 小的浮点数容差
        self.frame_counter = 0       # 帧计数

        # FuncAnimation 用于周期性更新图表
        self.ani = FuncAnimation(self.fig, self.update_all, interval=PLOT_INTERVAL_MS, cache_frame_data=False)

    def set_data(self, adc_a, adc_m, f_a, f_m, diff_frame, raw_adc_sum, raw_force_mag, cop_x, cop_y, r1, r2, c1, c2):
        """
        更新绘图所需的所有数据。
        Args:
            adc_a (float): ADC（压力梯度）角度。
            adc_m (float): ADC（压力梯度）幅值。
            f_a (float): 力传感器角度。
            f_m (float): 力传感器幅值。
            diff_frame (np.array): 基线减除后的压力数据。
            raw_adc_sum (float): 原始ADC总和。
            raw_force_mag (float): 原始力传感器幅值。
            cop_x (float): CoP的X坐标。
            cop_y (float): CoP的Y坐标。
            r1, r2, c1, c2 (int): 梯度计算区域的行/列边界。 (这些值已裁剪到梯度矩阵范围)
        """
        with self.lock: # 锁定以确保线程安全
            self.adc_angle = adc_a
            self.adc_mag = adc_m
            self.force_angle = f_a
            self.force_mag = f_m
            self.raw_adc_sum = raw_adc_sum
            self.raw_force_mag = raw_force_mag
            self.table_data = diff_frame.reshape(self.press_rows, self.press_cols) # 压力数据仍为 12x7
            self.cop_x = cop_x
            self.cop_y = cop_y
            self.final_r1, self.final_r2, self.final_c1, self.final_c2 = r1, r2, c1, c2
            self.frame_counter += 1
            
            # 更新历史数据队列
            diff = abs(adc_a - f_a)
            error = min(diff, 360 - diff) # 确保角度误差在0-180度之间
            angle_error_history.append(error)
            
            adc_mag_history.append(adc_m)
            force_mag_history.append(f_m)
            frame_count_history.append(self.frame_counter)
            
            raw_adc_sum_history.append(raw_adc_sum)
            raw_force_mag_history.append(raw_force_mag)

    def update_all(self, frame):
        """
        FuncAnimation 的更新函数，每次更新绘制所有子图。
        Args:
            frame (int): 当前帧索引 (由FuncAnimation自动提供)。
        Returns:
            list: Matplotlib Artists 对象列表 (通常为空列表或要刷新的对象)。
        """
        # 更新所有子图
        self.update_direction_arrows()
        self.update_magnitude_arrows()
        self.update_raw_adc_sum()
        self.update_raw_force_mag()
        self.update_angle_error()
        self.update_pressure_table()
        self.update_gradient_table()
        
        return []

    def update_direction_arrows(self):
        """更新方向箭头图 (ax1)。"""
        with self.lock:
            a = self.adc_angle
            f = self.force_angle
        
        self.ax1.clear() # 清空当前子图
        # 重新设置坐标轴范围和标题
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal')
        self.ax1.axis('off')
        self.ax1.set_title("Direction Arrows (ADC & Force)", fontsize=10)
        
        # ADC方向箭头（黑色）
        th_adc = np.deg2rad(a) # 角度转弧度
        self.ax1.arrow(0.5, 0.5, 0.4*np.cos(th_adc), 0.4*np.sin(th_adc), 
                      head_width=0.12, fc='k', ec='k', lw=2.5, length_includes_head=True)
        self.ax1.text(0.5, 0.1, f"ADC: {a:.1f}°", ha='center', va='center', fontsize=8, color='black')
        
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
        self.ax2.set_title("Magnitude Arrows (ADC & Force)", fontsize=10)
        
        # ADC幅值箭头（黑色），长度按ADC幅值线性映射
        th_adc = np.deg2rad(a)
        max_adc_length = 0.45 # 最大箭头长度
        l_adc = (m / 5000000.0) * max_adc_length if m > self.epsilon else 0.0 # 假设ADC最大值5M
        l_adc = min(l_adc, max_adc_length) # 确保不超过边界
        self.ax2.arrow(0.5, 0.5, l_adc*np.cos(th_adc), l_adc*np.sin(th_adc), 
                      head_width=0.12, fc='k', ec='k', lw=2.5, length_includes_head=True)
        self.ax2.text(0.5, 0.1, f"ADC: {m:.0f}", ha='center', va='center', fontsize=8, color='black')
        
        # Force幅值箭头（红色），长度按Force幅值线性映射
        th_force = np.deg2rad(fa)
        max_force_length = 0.4 # 最大箭头长度
        l_force = (abs(fm) / 20.0) * max_force_length if abs(fm) > self.epsilon else 0.0 # 假设Force最大值20N
        l_force = min(l_force, max_force_length)
        
        # 红色箭头显示：优化参数，确保箭头和杆子都可见
        if l_force > 0.02: # 只有长度足够时才绘制完整的箭头
            head_width = max(0.06, l_force * 0.15) # 动态调整箭头宽度
            head_length = max(0.04, l_force * 0.1) # 动态调整箭头长度
            # 注意：arrow的长度参数是整个箭头的长度，不是杆子长度
            self.ax2.arrow(0.5, 0.5, l_force*np.cos(th_force), l_force*np.sin(th_force), 
                          head_width=head_width, head_length=head_length, 
                          fc='red', ec='darkred', 
                          lw=3.5, length_includes_head=True, 
                          alpha=1.0, joinstyle='round', capstyle='round')
        else: # 当长度很小时，用线段表示
            self.ax2.plot([0.5, 0.5 + l_force*np.cos(th_force)], 
                         [0.5, 0.5 + l_force*np.sin(th_force)], 
                         'r-', lw=3.5, alpha=1.0)
        
        self.ax2.text(0.5, 0.9, f"Force: {fm:.1f}", ha='center', va='center', fontsize=8, color='red')

    def update_raw_adc_sum(self):
        """更新原始ADC总和曲线图 (ax3a)。"""
        if len(raw_adc_sum_history) > 0:
            xs = list(range(len(raw_adc_sum_history)))
            ys = list(raw_adc_sum_history)
            self.raw_adc_line.set_data(xs, ys) # 更新曲线数据
            
            # 自动调整y轴范围
            if len(ys) > 0:
                self.ax3a.set_ylim(min(ys) * 0.95, max(ys) * 1.05 if max(ys) > 0 else 1)
            self.ax3a.set_xlim(0, max(len(xs), 1)) # 自动调整x轴范围

    def update_raw_force_mag(self):
        """更新原始Force幅值曲线图 (ax3b)。"""
        if len(raw_force_mag_history) > 0:
            xs = list(range(len(raw_force_mag_history)))
            ys = list(raw_force_mag_history)
            self.raw_force_line.set_data(xs, ys)
            
            # 自动调整y轴范围
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
        """更新压力表 (ax5)，显示CoP和ROI。"""
        with self.lock:
            data = self.table_data.copy()
            cop_x_plot = self.cop_x # CoP x坐标 (基于12x7)
            cop_y_plot = self.cop_y # CoP y坐标 (基于12x7)
            r1, r2, c1, c2 = self.final_r1, self.final_r2, self.final_c1, self.final_c2 # ROI边界 (已裁剪到11x6)

        # 清空并重新设置子图
        self.ax5.clear()
        self.ax5.set_title(f"Baseline-Subtracted Pressure ({self.press_rows}×{self.press_cols})", fontsize=10)
        self.ax5.axis('off') # 关闭坐标轴

        nrows, ncols = self.press_rows, self.press_cols # 使用压力传感器原始尺寸
        
        # 坐标轴设置
        self.ax5.set_xlim(-0.5, ncols - 0.5)
        self.ax5.set_ylim(nrows - 0.5, -0.5)
        self.ax5.set_aspect('equal')
        self.ax5.grid(False)  # 确保网格关闭

        # 准备表格数据
        vmax = np.max(data) if np.max(data) != 0 else 1
        norm = data / vmax
        cmap = plt.colormaps['Reds']
        colors = cmap(norm)
        cell_text = [[f"{v:.0f}" for v in row] for row in data]

        # 创建表格
        self.table_plot = self.ax5.table(
            cellText=cell_text, 
            cellColours=colors,
            cellLoc='center',
            loc='center', # 居中放置
            bbox=[0, 0, 1, 1] # 占据整个子图区域，具体单元格尺寸由Matplotlib自动调整
        )
        self.table_plot.auto_set_font_size(False)
        self.table_plot.set_fontsize(8)
        
        # 调整每个单元格的尺寸
        for i in range(nrows):
            for j in range(ncols):
                cell = self.table_plot[(i, j)]
                cell.set_height(1/nrows) # 相对高度
                cell.set_width(1/ncols)  # 相对宽度

        # --- 绘制CoP蓝点 (CoP基于12x7，直接绘制) ---
        self.ax5.scatter(cop_x_plot, cop_y_plot, s=150, color='blue', marker='o', zorder=10, 
                         label=f"CoP ({cop_x_plot:.1f}, {cop_y_plot:.1f})")
        
        # --- 绘制CoP周围的ROI蓝框 (ROI边界已裁剪到11x6，在12x7背景上绘制) ---
        rect_x = c1 - 0.5
        rect_y = r1 - 0.5
        rect_width = (c2 - c1 + 1)
        rect_height = (r2 - r1 + 1)
        
        roi_rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                             linewidth=3, edgecolor='blue', facecolor='none', linestyle='--', zorder=5, label="ROI for Gradient")
        self.ax5.add_patch(roi_rect) # 添加矩形到子图

        # --- 绘制整个 12x7 区域的粗黑色外框 ---
        full_grid_rect = Rectangle((-0.5, -0.5), ncols, nrows,
                                   linewidth=2, edgecolor='black', facecolor='none', zorder=1)
        self.ax5.add_patch(full_grid_rect)

    def update_gradient_table(self):
        """更新梯度箭头图 (ax6)。"""
        with self.lock:
            data = grad_table_data.copy() # 获取全局梯度数据 (现在是11x6)
            cop_x_plot = self.cop_x # CoP x坐标 (基于12x7)
            cop_y_plot = self.cop_y # CoP y坐标 (基于12x7)
            r1, r2, c1, c2 = self.final_r1, self.final_r2, self.final_c1, self.final_c2 # ROI边界 (已裁剪到11x6)

        # 清空并重新设置子图
        self.ax6.clear()
        self.ax6.set_title(f"Gradient Arrows (gx, gy) ({self.grad_rows}×{self.grad_cols})", fontsize=10)
        
        nrows_grad, ncols_grad = self.grad_rows, self.grad_cols # 使用梯度矩阵尺寸
        # 设置坐标轴范围，使每个单元格在0到ncols_grad-1和0到nrows_grad-1之间
        self.ax6.set_xlim(-0.5, ncols_grad - 0.5)
        self.ax6.set_ylim(nrows_grad - 0.5, -0.5) # Y轴反转，使r=0在上方
        self.ax6.set_aspect('equal') # 保持宽高比
        self.ax6.axis('off') # 关闭坐标轴
        
        # 在背景绘制 (11x6) 的表格网格 (细线)
        for r in range(nrows_grad + 1):
            self.ax6.axhline(r - 0.5, color='black', linestyle='-', linewidth=0.5, zorder=0)
        for c in range(ncols_grad + 1):
            self.ax6.axvline(c - 0.5, color='black', linestyle='-', linewidth=0.5, zorder=0)

        # 遍历每个传感器单元绘制梯度箭头 (基于11x6)
        for r in range(nrows_grad):
            for c in range(ncols_grad):
                gx, gy = data[r, c, 0], data[r, c, 1] # 获取该单元格的梯度分量
                
                mag = np.hypot(gx, gy) # 计算梯度幅值
                
                # 只绘制幅值大于一定阈值的箭头
                if mag > 1.0:
                    gx_norm = gx / mag
                    gy_norm = gy / mag
                    
                    # 箭头的起始点在单元格中心 (c, r)
                    self.ax6.quiver(c, r, gx_norm, gy_norm, 
                                    color='k', 
                                    scale=2.5, # 调整 scale 使箭头的视觉长度合适
                                    width=0.02, # 箭杆粗细
                                    headwidth=6, headlength=8, headaxislength=7, # 箭头头尺寸
                                    angles='xy', scale_units='xy', zorder=5)

        # --- 在梯度表上绘制CoP蓝点 ---
        # 裁剪CoP坐标，使其在11x6梯度表的显示范围内
        cop_x_grad_plot = min(cop_x_plot, ncols_grad - 1)
        cop_y_grad_plot = min(cop_y_plot, nrows_grad - 1)
        self.ax6.scatter(cop_x_grad_plot, cop_y_grad_plot, s=150, color='blue', marker='o', zorder=10, 
                         label=f"CoP ({cop_x_plot:.1f}, {cop_y_plot:.1f})")

        # --- 在梯度表上绘制CoP周围的ROI蓝框 (ROI边界已裁剪到11x6，直接绘制) ---
        rect_x = c1 - 0.5
        rect_y = r1 - 0.5
        rect_width = (c2 - c1 + 1)
        rect_height = (r2 - r1 + 1)
        
        roi_rect = Rectangle((rect_x, rect_y), rect_width, rect_height,
                             linewidth=3, edgecolor='blue', facecolor='none', linestyle='--', zorder=5, label="ROI for Gradient")
        self.ax6.add_patch(roi_rect) # 添加矩形到子图

        # --- 绘制整个 11x6 区域的粗黑色外框 ---
        full_grid_rect = Rectangle((-0.5, -0.5), ncols_grad, nrows_grad,
                                   linewidth=2, edgecolor='black', facecolor='none', zorder=1)
        self.ax6.add_patch(full_grid_rect)


    def show(self):
        """显示实时绘图窗口。"""
        plt.tight_layout() # 自动调整子图参数，使之填充整个 figure 区域
        plt.show()

# ==================== Data Collector Class ====================
class DataCollector:
    """
    数据收集器类，负责从传感器读取数据，进行处理，并更新实时绘图及保存数据到CSV。
    """
    def __init__(self, force_sensor, press_sensor, csv_path):
        self.force_sensor = force_sensor                          # 六轴力传感器实例
        self.press_sensor = press_sensor                          # 压力传感器实例
        self.csv_path = csv_path                                  # CSV文件保存路径
        self.running = True                                       # 控制数据收集循环的标志
        self.stop_event = threading.Event()                       # 用于线程间通信的停止事件
        self.press_buf = TimestampedBuffer(PRESS_BUFFER_SIZE)     # 压力数据缓冲区
        self.force_buf = TimestampedBuffer(FORCE_BUFFER_SIZE)     # 力数据缓冲区
        self.plot = None                                          # RealTimePlot实例
        self.start_time = None                                    # 数据收集开始时间

    def set_plot(self, p):
        """设置绘图对象。"""
        self.plot = p

    def start(self):
        """启动数据收集线程。"""
        # 创建并启动压力和力传感器读取线程
        self.press_thread = PressureReaderThread(self.press_sensor, self.press_buf, self.stop_event)
        self.force_thread = ForceReaderThread(self.force_sensor, self.force_buf, self.stop_event)
        self.press_thread.start()
        self.force_thread.start()
        # 启动主数据处理和CSV写入线程
        threading.Thread(target=self.run_collect, daemon=True).start()

    def run_collect(self):
        """
        主数据收集和处理循环。
        从缓冲区获取同步数据，计算梯度和角度，更新绘图，并写入CSV。
        """
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            # CSV文件头
            header = ["timestamp","rel_ms"] + [f"ch{i+1}" for i in range(84)] + ["Fx","Fy","Fz","Mx","My","Mz","press_t","force_t","dt","ADC_angle","ADC_mag","Force_angle","Force_mag", "CoP_X", "CoP_Y", "ROI_r1", "ROI_r2", "ROI_c1", "ROI_c2"]
            writer.writerow(header)
            period = 1.0 / TARGET_HZ                              # 每帧的目标时间间隔

            while self.running:
                t0 = time.perf_counter()                          # 记录当前帧开始时间
                if self.start_time is None:
                    self.start_time = t0                          # 第一次进入循环时记录开始时间
                rel_ms = int((t0 - self.start_time) * 1000)       # 相对于开始时间的毫秒数

                p_item = self.press_buf.get_latest()              # 获取最新的压力数据
                # 查找时间戳最接近压力数据的力数据，实现数据同步
                f_item = self.force_buf.find_closest(p_item["t"]) if p_item else None
                if not p_item or not f_item or abs(p_item["t"] - f_item["t"]) > MAX_SYNC_DT: # 确保数据有效且同步时间差在允许范围内
                    time.sleep(0.001)
                    continue

                p_data = p_item["data"]
                f_data = f_item["data"]
                
                # 计算原始ADC总和（未做baseline subtraction）
                raw_adc_sum = np.sum(p_data)
                
                # 基线减除
                diff_frame = subtract_baseline(p_data)

                # 计算梯度、CoP和ROI边界
                dir_x, dir_y, vec_mag, cop_x, cop_y, r1, r2, c1, c2 = compute_gradient_in_region(diff_frame)
                adc_angle, _ = compute_gradient_angle_single(dir_x, dir_y)
                
                # 获取力传感器数据
                fx, fy = f_data[0], f_data[1]
                f_angle, f_mag = compute_force_angle(fx, fy)
                
                # 计算原始Force幅值（直接用Fx,Fy的平方和开方）
                raw_force_mag = np.hypot(fx, fy)

                # ==================== 【保存全程数据】 ====================
                # 将当前帧的处理结果添加到全局列表中，用于程序结束后的统计绘图
                full_time_list.append(rel_ms)
                full_adc_mag_list.append(vec_mag)
                full_force_mag_list.append(f_mag)

                if self.plot:
                    # 更新实时绘图的数据
                    self.plot.set_data(adc_angle, vec_mag, f_angle, f_mag, diff_frame, raw_adc_sum, raw_force_mag, cop_x, cop_y, r1, r2, c1, c2)

                # 格式化时间戳并准备写入CSV
                ts_str = time.strftime("%Y%m%d%H%M%S%f")[:-3] # 精确到毫秒
                row = [ts_str, rel_ms] + p_data + f_data + [
                    round(p_item["t"],6), round(f_item["t"],6), round(abs(p_item["t"]-f_item["t"])*1000,3),
                    adc_angle, vec_mag, f_angle, f_mag,
                    cop_x, cop_y, r1, r2, c1, c2 # 新增CoP和ROI信息
                ]
                writer.writerow(row)                              # 写入CSV行
                csv_file.flush()                                  # 立即写入磁盘，防止数据丢失
                
                # 控制循环频率
                elapsed = time.perf_counter() - t0                # 计算本帧处理耗时
                time.sleep(max(0, period-elapsed))                # 暂停以达到目标帧率

    def stop(self):
        """停止数据收集线程。"""
        self.running = False
        self.stop_event.set()                                     # 设置停止事件，通知读取线程退出
        time.sleep(0.3)                                           # 等待线程安全退出

# ==================== 程序结束绘制全程静态图 ====================
def plot_full_magnitude_curve():
    """
    在程序结束后绘制全程的ADC和力传感器幅值曲线。
    """
    # 创建一个包含两个子图的Figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1：ADC幅值曲线
    if len(full_time_list) > 0 and len(full_adc_mag_list) > 0:
        ax1.plot(full_time_list, full_adc_mag_list, 'b-', linewidth=1.5, label='ADC Magnitude')
        ax1.set_title("ADC Magnitude Over Time", fontsize=14)
        ax1.set_xlabel("Time (ms)", fontsize=12)
        ax1.set_ylabel("ADC Magnitude", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.set_ylim(0, max(full_adc_mag_list) * 1.1 if max(full_adc_mag_list) > 0 else 1) # 自动调整Y轴范围
    
    # 图2：Force幅值曲线
    if len(full_time_list) > 0 and len(full_force_mag_list) > 0:
        ax2.plot(full_time_list, full_force_mag_list, 'r-', linewidth=1.5, label='Force Magnitude')
        ax2.set_title("Force Magnitude Over Time", fontsize=14)
        ax2.set_xlabel("Time (ms)", fontsize=12)
        ax2.set_ylabel("Force Magnitude (N)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.set_ylim(0, max(full_force_mag_list) * 1.1 if max(full_force_mag_list) > 0 else 1) # 自动调整Y轴范围
    
    plt.tight_layout() # 自动调整布局
    save_path = os.path.join(SAVE_DIR, "full_magnitude_curve.png") # 保存图片
    plt.savefig(save_path, dpi=300)
    plt.show() # 显示图片

# ==================== Main Program Execution ====================
if __name__ == "__main__":
    # 创建保存目录（如果不存在）
    os.makedirs(SAVE_DIR, exist_ok=True)
    # 查找未使用的CSV文件名 (data_1.csv, data_2.csv ...)
    i = 1
    while os.path.exists(os.path.join(SAVE_DIR, f"data_{i}.csv")):
        i += 1
    csv_path = os.path.join(SAVE_DIR, f"data_{i}.csv")

    # 初始化传感器
    force_sensor = SixAxisForceSensor()
    press_sensor = PressureSensor()
    force_sensor.calibrate_zero() # 校准力传感器零点

    # 初始化实时绘图和数据收集器
    plot = RealTimePlot()
    collector = DataCollector(force_sensor, press_sensor, csv_path)
    collector.set_plot(plot)
    collector.start() # 启动数据收集

    print("✅ Region gradient & symmetric area enabled")
    print("✅ Real-time angle error plot enabled")
    print("✅ 12×7 real-time pressure table enabled")
    print("✅ 11×6 real-time gradient arrows enabled (with shafts and head at moving end)") # 更新提示
    print("✅ Raw ADC & Force magnitude plots enabled (original values)")
    print(f"✅ Saving data to: {csv_path}")
    
    plot.show()  # 运行实时绘图界面，此调用会阻塞直到窗口关闭
    
    collector.stop() # 停止数据收集线程
    
    # ==================== 【结束后自动出图】 ====================
    print("\n📊 Generating full-time magnitude curve...")
    plot_full_magnitude_curve() # 绘制全程幅值曲线图
    
    print("✅ Program finished")