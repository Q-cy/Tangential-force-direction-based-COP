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

# ==================== Config ====================
BAUDRATE_PRESS = 921600
BAUDRATE_FORCE = 460800
SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"
TARGET_HZ = 100.0
PLOT_INTERVAL_MS = 100
MAX_SYNC_DT = 0.015
PRESS_BUFFER_SIZE = 500
FORCE_BUFFER_SIZE = 500
THRESHOLD = 1000

# Smoothing
DIR_SMOOTH_ALPHA = 0.15
ERROR_PLOT_LEN = 100
MAG_PLOT_LEN = 100
# ================================================================

# ==================== Global ====================
first_frame = None
first_frame_lock = threading.Lock()
adc_filtered_dir = None

angle_error_history = deque(maxlen=ERROR_PLOT_LEN)
grad_table_data = np.zeros((12, 7, 2))
adc_mag_history = deque(maxlen=MAG_PLOT_LEN)
force_mag_history = deque(maxlen=MAG_PLOT_LEN)
frame_count_history = deque(maxlen=MAG_PLOT_LEN)

# ==================== 原始值历史记录 ====================
# 原始ADC总和（未处理）
raw_adc_sum_history = deque(maxlen=MAG_PLOT_LEN)
# 原始Force X,Y分量平方和（未处理）
raw_force_mag_history = deque(maxlen=MAG_PLOT_LEN)

# ==================== 全程数据保存 ====================
full_time_list = []
full_adc_mag_list = []
full_force_mag_list = []

# ==================== Baseline Subtraction ====================
def subtract_baseline(current_frame):
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()
    with first_frame_lock:
        if first_frame is None:
            first_frame = current_frame.copy()
    diff = current_frame - first_frame
    diff = np.clip(diff, 0, None)
    return diff

# ==================== Gradient in Symmetric Region ====================
def compute_gradient_in_region(frame):
    global adc_filtered_dir, grad_table_data
    rows, cols = 12, 7
    frame_flat = np.array(frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)
    threshold = THRESHOLD

    grad = np.zeros((rows, cols, 2), dtype=np.float32)
    for y in range(rows):
        for x in range(cols):
            val = frame2d[y, x]
            left  = frame2d[y, x-1] if x-1 >= 0 else val
            right = frame2d[y, x+1] if x+1 < cols else val
            up    = frame2d[y-1, x] if y-1 >= 0 else val
            down  = frame2d[y+1, x] if y+1 < rows else val
            gx = 0.5 * (right - left)
            gy = 0.5 * (up - down)
            grad[y, x] = (gx, gy)

    grad_table_data = grad.copy()

    valid_indices = np.where(frame_flat > threshold)[0].tolist()
    if not valid_indices:
        adc_filtered_dir = None
        return 0.0, 0.0, 0.0

    coords = [(idx // cols, idx % cols) for idx in valid_indices]
    rs, cs = zip(*coords)
    min_r, max_r = min(rs), max(rs)
    min_c, max_c = min(cs), max(cs)

    sub_mat = frame2d[min_r:max_r+1, min_c:max_c+1]
    best_local_r = np.argmax(sub_mat.sum(axis=1))
    best_local_c = np.argmax(sub_mat.sum(axis=0))
    center_r = min_r + best_local_r
    center_c = min_c + best_local_c

    d_up    = center_r
    d_down  = (rows - 1) - center_r
    d_left  = center_c
    d_right = (cols - 1) - center_c

    expand_r = min(d_up, d_down)
    expand_c = min(d_left, d_right)

    final_r1 = center_r - expand_r
    final_r2 = center_r + expand_r
    final_c1 = center_c - expand_c
    final_c2 = center_c + expand_c

    sum_gx = 0.0
    sum_gy = 0.0
    sum_w  = 0.0

    for y in range(final_r1, final_r2 + 1):
        for x in range(final_c1, final_c2 + 1):
            val = frame2d[y, x]
            if val < 1:
                continue
            sum_gx += val * grad[y, x, 0]
            sum_gy += val * grad[y, x, 1]
            sum_w  += val

    if sum_w < 1e-3:
        adc_filtered_dir = None
        return 0.0, 0.0, 0.0

    vec_mag = np.hypot(sum_gx, sum_gy)
    if vec_mag < 1e-3:
        adc_filtered_dir = None
        return 0.0, 0.0, 0.0

    raw_x = sum_gx / vec_mag
    raw_y = sum_gy / vec_mag

    if adc_filtered_dir is None:
        fx, fy = raw_x, raw_y
    else:
        fx = (1 - DIR_SMOOTH_ALPHA) * adc_filtered_dir[0] + DIR_SMOOTH_ALPHA * raw_x
        fy = (1 - DIR_SMOOTH_ALPHA) * adc_filtered_dir[1] + DIR_SMOOTH_ALPHA * raw_y
        n = np.hypot(fx, fy)
        if n > 1e-6:
            fx /= n
            fy /= n

    adc_filtered_dir = (fx, fy)
    return fx, fy, vec_mag

# ==================== Angle Calculation ====================
def compute_gradient_angle_single(x, y):
    epsilon = 1e-8
    angle = np.degrees(np.arctan2(y, x + epsilon))
    if angle < 0:
        angle += 360
    mag = np.hypot(x, y)
    return angle, mag

def compute_force_angle(Fx, Fy):
    epsilon = 1e-8
    mag = np.hypot(Fx, Fy)
    if mag < 1e-8:
        return 0.0, 0.0
    angle = np.degrees(np.arctan2(Fy, Fx + epsilon))
    if angle < 0:
        angle += 360
    return angle, mag

# ==================== 6-axis Force Sensor ====================
class SixAxisForceSensor:
    def __init__(self):
        self.ser = None
        self.port = "/dev/ttyUSB0"
        self.zero_data = [0.0]*6
        self.open_port()
    def open_port(self):
        try:
            self.ser = serial.Serial(self.port, BAUDRATE_FORCE, timeout=0.05)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        except:
            self.ser = None
    def reconnect(self):
        try:
            self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.open_port()
    def read(self):
        if not self.ser:
            return None
        try:
            self.ser.write(b'\x49\xAA\x0D\x0A')
            time.sleep(0.005)
            resp = self.ser.read(28)
            if len(resp)!=28 or resp[:2]!=b'\x49\xAA':
                return None
            Fx = struct.unpack('<f', resp[2:6])[0]
            Fy = struct.unpack('<f', resp[6:10])[0]
            Fz = struct.unpack('<f', resp[10:14])[0]
            Mx = struct.unpack('<f', resp[14:18])[0]
            My = struct.unpack('<f', resp[18:22])[0]
            Mz = struct.unpack('<f', resp[22:26])[0]
            Fx *= 9.8; Fy *= 9.8; Fz *= 9.8; Mx *= 9.8; My *= 9.8; Mz *= 9.8
            Fx -= self.zero_data[0]; Fy -= self.zero_data[1]; Fz -= self.zero_data[2]
            Mx -= self.zero_data[3]; My -= self.zero_data[4]; Mz -= self.zero_data[5]
            return [round(v, 2) for v in [Fx, Fy, Fz, Mx, My, Mz]]
        except:
            return None
    def calibrate_zero(self):
        vals = []
        for _ in range(20):
            d = self.read()
            if d:
                vals.append(d)
            time.sleep(0.05)
        if len(vals)>=5:
            self.zero_data = np.mean(np.array(vals), axis=0).tolist()

# ==================== Pressure Sensor ====================
class PressureSensor:
    def __init__(self):
        self.ser = None
        self.port = None
        self.last = None
        self.auto_find_port()
    def auto_find_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p,_,_ in ports:
            if p == "/dev/ttyUSB0":
                continue
            try:
                self.ser = serial.Serial(p, BAUDRATE_PRESS, timeout=0.01)
                self.port = p
                time.sleep(0.1)
                self.ser.reset_input_buffer()
                return
            except:
                continue
        raise Exception("Pressure sensor not found")
    def reconnect(self):
        try:
            self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.auto_find_port()
    def read_data(self):
        if not self.ser:
            return None
        try:
            cmd = [0x55,0xAA,9,0,0x34,0,0xFB,0,0x1C,0,0,0xA8,0,0x35]
            self.ser.write(bytearray(cmd))
            time.sleep(0.005)
            resp = self.ser.read(256)
            idx = resp.find(b'\xaa\x55')
            if idx == -1 or len(resp[idx:])<182:
                return None
            return resp[idx+14:idx+14+168]
        except:
            return None
    def decode(self, raw):
        arr = []
        for i in range(0,168,2):
            arr.append(struct.unpack("<H", raw[i:i+2])[0])
        out = []
        for i in range(12):
            out.extend(arr[i*7:(i+1)*7])
        
        # 直接更新last，不做任何异常值处理
        self.last = out.copy()
        return out

# ==================== Timestamp Buffer ====================
class TimestampedBuffer:
    def __init__(self, maxlen=300):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
    def append(self, item):
        with self.lock:
            self.buf.append(item)
    def get_latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None
    def find_closest(self, ts):
        with self.lock:
            best = None
            best_dt = 1e9
            for item in self.buf:
                dt = abs(item["t"]-ts)
                if dt < best_dt:
                    best_dt = dt
                    best = item
            return best

# ==================== Reader Threads ====================
class PressureReaderThread(threading.Thread):
    def __init__(self, sensor, buf, stop_event):
        super().__init__(daemon=True)
        self.sensor = sensor
        self.buf = buf
        self.stop = stop_event
        self.fail = 0
    def run(self):
        while not self.stop.is_set():
            t = time.perf_counter()
            raw = self.sensor.read_data()
            if raw is None:
                self.fail +=1
                if self.fail >=30:
                    self.sensor.reconnect()
                    self.fail=0
                time.sleep(0.002)
                continue
            try:
                data = self.sensor.decode(raw)
                self.buf.append({"t":t, "data":data})
                self.fail=0
            except:
                time.sleep(0.001)

class ForceReaderThread(threading.Thread):
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
                if self.fail >=30:
                    self.sensor.reconnect()
                    self.fail=0
                time.sleep(0.002)
                continue
            self.buf.append({"t":t, "data":data})
            self.fail=0

# ==================== Real-time Plot (Integrated Layout) ====================
class RealTimePlot:
    def __init__(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        self.fig = plt.figure(figsize=(16, 12))
        
        # 主GridSpec：4行2列，宽度比例1:1
        # height_ratios: [图1&图2所在行的高度, 图3a所在行高度, 图3b所在行高度, 图4所在行高度]
        # 修改 height_ratios 为 [6, 1, 1, 1] 使 ax3a, ax3b, ax4 等高且铺满剩余空间
        # 修改 hspace 以减少图1/2和图3a之间的空白 -> 修正：增加 hspace 避免标题重叠
        gs_outer = GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[6, 1, 1, 1], hspace=0.2, wspace=0.3)
        
        # 嵌套GridSpec for 图1 & 图2: 放在 gs_outer 的第一行第一列 (左上角)
        # 这个嵌套的GridSpec有1行2列，用于让图1和图2左右并排
        gs_arrows = gs_outer[0, 0].subgridspec(1, 2, wspace=0.3)
        
        # 图1：方向箭头图 (占据嵌套GridSpec的左边)
        self.ax1 = plt.subplot(gs_arrows[0, 0])
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal')
        self.ax1.axis('off')
        self.ax1.set_title("Direction Arrows (ADC & Force)", fontsize=10)
        
        # 图2：幅值箭头图 (占据嵌套GridSpec的右边)
        self.ax2 = plt.subplot(gs_arrows[0, 1])
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_aspect('equal')
        self.ax2.axis('off')
        self.ax2.set_title("Magnitude Arrows (ADC & Force)", fontsize=10)
        
        # 图3a：原始ADC总和图 (占据主GridSpec的第二行，第一列)
        self.ax3a = plt.subplot(gs_outer[1, 0])
        self.ax3a.set_title("Raw ADC Sum (Original Values)", fontsize=10)
        self.ax3a.set_xlabel("Frame", fontsize=10)
        self.ax3a.set_ylabel("ADC Sum", fontsize=10)
        self.ax3a.grid(True, alpha=0.3)
        self.raw_adc_line, = self.ax3a.plot([], [], 'b-', linewidth=1.5, label="Raw ADC Sum")
        self.ax3a.legend(fontsize=8)
        
        # 图3b：原始Force幅值图 (占据主GridSpec的第三行，第一列)
        self.ax3b = plt.subplot(gs_outer[2, 0])
        self.ax3b.set_title("Raw Force Magnitude (Original Values)", fontsize=10)
        self.ax3b.set_xlabel("Frame", fontsize=10)
        self.ax3b.set_ylabel("Force Magnitude (N)", fontsize=10)
        self.ax3b.grid(True, alpha=0.3)
        self.raw_force_line, = self.ax3b.plot([], [], 'r-', linewidth=1.5, label="Raw Force Mag")
        self.ax3b.legend(fontsize=8)
        
        # 图4：角度误差图 (占据主GridSpec的第四行，第一列)
        self.ax4 = plt.subplot(gs_outer[3, 0])
        self.ax4.set_title("Angle Error between ADC and Force", fontsize=10)
        self.ax4.set_xlabel("Frame", fontsize=10)
        self.ax4.set_ylabel("Angle Error (deg)", fontsize=10)
        self.ax4.set_ylim(0, 180)
        self.ax4.grid(True, alpha=0.3)
        self.error_line, = self.ax4.plot([], [], 'g-o', linewidth=1.5, markersize=2, label="Angle Error |ADC - Force|")
        self.ax4.legend(fontsize=8)
        
        # 新增：嵌套GridSpec，用于右侧的两张表，使其等高并铺满右侧区域
        # gs_outer[:, 1] 表示 gs_outer 的所有行，但只使用第二列
        gs_right_tables = gs_outer[:, 1].subgridspec(2, 1, height_ratios=[1, 1], hspace=0.3)
        
        # 表1：压力表 (占据右侧嵌套GridSpec的第一行，即上半部分)
        self.ax5 = plt.subplot(gs_right_tables[0, 0])
        self.ax5.set_title("Baseline-Subtracted Pressure (12×7)", fontsize=10)
        self.ax5.axis('off')
        self.table_plot = None
        
        # 表2：梯度表 (占据右侧嵌套GridSpec的第二行，即下半部分)
        self.ax6 = plt.subplot(gs_right_tables[1, 0])
        self.ax6.set_title("Gradient (gx, gy) 12×7", fontsize=10)
        self.ax6.axis('off')
        self.grad_table_plot = None
        
        self.adc_angle = 0
        self.adc_mag = 0
        self.force_angle = 0
        self.force_mag = 0
        self.raw_adc_sum = 0  # 原始ADC总和
        self.raw_force_mag = 0  # 原始Force幅值
        self.table_data = np.zeros((12, 7))
        self.lock = threading.Lock()
        self.fixed_arrow = 0.35
        self.epsilon = 1e-8
        self.frame_counter = 0
        
        # 箭头图的子元素
        self.center_arrow_adc = None
        self.center_arrow_force = None
        self.mag_arrow_adc = None
        self.mag_arrow_force = None
        
        self.ani = FuncAnimation(self.fig, self.update_all, interval=PLOT_INTERVAL_MS, cache_frame_data=False)

    def set_data(self, adc_a, adc_m, f_a, f_m, diff_frame, raw_adc_sum, raw_force_mag):
        with self.lock:
            self.adc_angle = adc_a
            self.adc_mag = adc_m
            self.force_angle = f_a
            self.force_mag = f_m
            self.raw_adc_sum = raw_adc_sum  # 保存原始ADC总和
            self.raw_force_mag = raw_force_mag  # 保存原始Force幅值
            self.table_data = diff_frame.reshape(12,7)
            self.frame_counter += 1
            
            # 正确更新所有历史数据
            diff = abs(adc_a - f_a)
            error = min(diff, 360 - diff)
            angle_error_history.append(error)
            
            # 处理ADC幅值历史（处理后的值）
            adc_mag_history.append(adc_m)
            
            # 处理Force幅值历史（处理后的值）
            force_mag_history.append(f_m)
            
            # 处理帧计数历史
            frame_count_history.append(self.frame_counter)
            
            # 修复：正确更新原始值历史记录
            raw_adc_sum_history.append(raw_adc_sum)
            raw_force_mag_history.append(raw_force_mag)

    def update_all(self, frame):
        # 更新图1：方向箭头
        self.update_direction_arrows()
        
        # 更新图2：幅值箭头
        self.update_magnitude_arrows()
        
        # 更新图3a：原始ADC总和
        self.update_raw_adc_sum()
        
        # 更新图3b：原始Force幅值
        self.update_raw_force_mag()
        
        # 更新图4：角度误差
        self.update_angle_error()
        
        # 更新表1：压力表
        self.update_pressure_table()
        
        # 更新表2：梯度表
        self.update_gradient_table()
        
        return []

    def update_direction_arrows(self):
        with self.lock:
            a = self.adc_angle
            f = self.force_angle
        
        self.ax1.clear()
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal')
        self.ax1.axis('off')
        self.ax1.set_title("Direction Arrows (ADC & Force)", fontsize=10)
        
        # ADC方向箭头（黑色）
        th_adc = np.deg2rad(a)
        self.ax1.arrow(0.5, 0.5, 0.4*np.cos(th_adc), 0.4*np.sin(th_adc), 
                      head_width=0.12, fc='k', ec='k', lw=2.5, length_includes_head=True)
        self.ax1.text(0.5, 0.1, f"ADC: {a:.1f}°", ha='center', va='center', fontsize=8, color='black')
        
        # Force方向箭头（红色）
        th_force = np.deg2rad(f)
        self.ax1.arrow(0.5, 0.5, self.fixed_arrow*np.cos(th_force), self.fixed_arrow*np.sin(th_force), 
                      head_width=0.1, fc='r', ec='r', lw=2, length_includes_head=True)
        self.ax1.text(0.5, 0.9, f"Force: {f:.1f}°", ha='center', va='center', fontsize=8, color='red')

    def update_magnitude_arrows(self):
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
        
        # ADC幅值箭头（黑色，线性变化长度，按比例到边界）
        th_adc = np.deg2rad(a)
        # 线性映射：ADC值范围0-100000映射到0-0.45长度（到图片边界）
        max_adc_length = 0.45  # 最大长度到图片边界
        l_adc = (m / 5000000.0) * max_adc_length if m > self.epsilon else 0.0
        # 确保不超过边界
        l_adc = min(l_adc, max_adc_length)
        self.ax2.arrow(0.5, 0.5, l_adc*np.cos(th_adc), l_adc*np.sin(th_adc), 
                      head_width=0.12, fc='k', ec='k', lw=2.5, length_includes_head=True)
        self.ax2.text(0.5, 0.1, f"ADC: {m:.0f}", ha='center', va='center', fontsize=8, color='black')
        
        # Force幅值箭头（红色，线性变化长度，按比例到边界）
        th_force = np.deg2rad(fa)
        # 线性映射：Force值范围0-100映射到0-0.4长度（到图片边界）
        max_force_length = 0.4  # 最大长度到图片边界
        l_force = (abs(fm) / 20.0) * max_force_length if abs(fm) > self.epsilon else 0.0
        # 确保不超过边界
        l_force = min(l_force, max_force_length)
        
        # 红色箭头显示：优化参数，确保箭头和杆子都可见
        if l_force > 0.02:  # 降低最小显示阈值
            # 使用更合适的箭头参数
            head_width = max(0.06, l_force * 0.15)  # 动态调整箭头宽度
            head_length = max(0.04, l_force * 0.1)  # 动态调整箭头长度
            stem_length = l_force - head_length  # 杆子长度
            
            if stem_length > 0.01:  # 确保有足够的杆子长度
                self.ax2.arrow(0.5, 0.5, l_force*np.cos(th_force), l_force*np.sin(th_force), 
                              head_width=head_width, head_length=head_length, 
                              fc='red', ec='darkred', 
                              lw=3.5,  # 粗线宽确保杆子可见
                              length_includes_head=True, 
                              alpha=1.0,
                              joinstyle='round',
                              capstyle='round')
            else:
                # 当杆子太短时，用线段+小三角形
                self.ax2.plot([0.5, 0.5 + l_force*np.cos(th_force)], 
                             [0.5, 0.5 + l_force*np.sin(th_force)], 
                             'r-', lw=3.5, alpha=1.0)
                # 添加小箭头头
                arrow_x = 0.5 + l_force*np.cos(th_force)
                arrow_y = 0.5 + l_force*np.sin(th_force)
                self.ax2.scatter(arrow_x, arrow_y, c='red', s=30, marker='^', zorder=10)
        else:
            # 当长度很小时，用线段表示
            self.ax2.plot([0.5, 0.5 + l_force*np.cos(th_force)], 
                         [0.5, 0.5 + l_force*np.sin(th_force)], 
                         'r-', lw=3.5, alpha=1.0)
        
        self.ax2.text(0.5, 0.9, f"Force: {fm:.1f}", ha='center', va='center', fontsize=8, color='red')

    def update_raw_adc_sum(self):
        # 正确获取并更新原始ADC总和数据
        if len(raw_adc_sum_history) > 0:
            xs = list(range(len(raw_adc_sum_history)))
            ys = list(raw_adc_sum_history)
            self.raw_adc_line.set_data(xs, ys)
            
            # 自动调整y轴范围
            if len(ys) > 0:
                self.ax3a.set_ylim(min(ys) * 0.95, max(ys) * 1.05)
            self.ax3a.set_xlim(0, max(len(xs), 1))

    def update_raw_force_mag(self):
        # 正确获取并更新原始Force幅值数据
        if len(raw_force_mag_history) > 0:
            xs = list(range(len(raw_force_mag_history)))
            ys = list(raw_force_mag_history)
            self.raw_force_line.set_data(xs, ys)
            
            # 自动调整y轴范围
            if len(ys) > 0:
                self.ax3b.set_ylim(min(ys) * 0.95, max(ys) * 1.05)
            self.ax3b.set_xlim(0, max(len(xs), 1))

    def update_angle_error(self):
        # 正确获取并更新角度误差数据
        if len(angle_error_history) > 0:
            xs = list(range(len(angle_error_history)))
            ys = list(angle_error_history)
            self.error_line.set_data(xs, ys)
            
            # 自动调整x轴范围
            self.ax4.set_xlim(0, max(len(xs), 1))

    def update_pressure_table(self):
        with self.lock:
            data = self.table_data.copy()
        
        if self.table_plot:
            self.table_plot.remove()
        
        self.ax5.clear()
        self.ax5.set_title("Baseline-Subtracted Pressure (12×7)", fontsize=10)
        self.ax5.axis('off')

        # 计算表格尺寸：12行7列
        nrows, ncols = 12, 7
        
        vmax = np.max(data) if np.max(data) != 0 else 1
        norm = data / vmax
        cmap = plt.colormaps['Reds']
        colors = cmap(norm)
        cell_text = [[f"{v:.0f}" for v in row] for row in data]

        # 创建表格，使用 bbox=[0, 0, 1, 1] 确保表格铺满整个ax5区域
        self.table_plot = self.ax5.table(
            cellText=cell_text, 
            cellColours=colors,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1] # 让表格自己铺满ax5区域
        )
        self.table_plot.auto_set_font_size(False)
        self.table_plot.set_fontsize(8)
        # 设置每个单元格的尺寸一致
        for i in range(nrows):
            for j in range(ncols):
                self.table_plot[(i, j)].set_width(1.0 / ncols) # 单元格宽度占总宽度的比例
                self.table_plot[(i, j)].set_height(1.0 / nrows) # 单元格高度占总高度的比例

    def update_gradient_table(self):
        with self.lock:
            data = grad_table_data.copy()

        if self.grad_table_plot:
            self.grad_table_plot.remove()
        
        self.ax6.clear()
        self.ax6.set_title("Gradient (gx, gy) 12×7", fontsize=10)
        self.ax6.axis('off')

        # 计算表格尺寸：12行7列
        nrows, ncols = 12, 7
        
        cell_text = []
        for row in data:
            txt_row = [f"gx:{g[0]:+.0f}\ngy:{g[1]:+.0f}" for g in row]
            cell_text.append(txt_row)

        # 创建表格，使用 bbox=[0, 0, 1, 1] 确保表格铺满整个ax6区域
        self.grad_table_plot = self.ax6.table(
            cellText=cell_text, 
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1] # 让表格自己铺满ax6区域
        )
        self.grad_table_plot.auto_set_font_size(False)
        self.grad_table_plot.set_fontsize(7)
        # 设置每个单元格的尺寸一致
        for i in range(nrows):
            for j in range(ncols):
                self.grad_table_plot[(i, j)].set_width(1.0 / ncols)
                self.grad_table_plot[(i, j)].set_height(1.0 / nrows)

    def show(self):
        plt.tight_layout()
        plt.show()

# ==================== Data Collector ====================
class DataCollector:
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
        self.plot = p

    def start(self):
        # 修复: 将 'self.sensor==self.press_sensor' 改为 'sensor=self.press_sensor'
        self.press_thread = PressureReaderThread(sensor=self.press_sensor, buf=self.press_buf, stop_event=self.stop_event)
        # 修复: 将 'self.sensor=self.force_sensor' 改为 'sensor=self.force_sensor'
        self.force_thread = ForceReaderThread(sensor=self.force_sensor, buf=self.force_buf, stop_event=self.stop_event)
        self.press_thread.start()
        self.force_thread.start()
        threading.Thread(target=self.run_collect, daemon=True).start()

    def run_collect(self):
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.writer(csv_file)
            header = ["timestamp","rel_ms"] + [f"ch{i+1}" for i in range(84)] + ["Fx","Fy","Fz","Mx","My","Mz","press_t","force_t","dt","ADC_angle","ADC_mag","Force_angle","Force_mag"]
            writer.writerow(header)
            period = 1.0 / TARGET_HZ

            while self.running:
                t0 = time.perf_counter()
                if self.start_time is None:
                    self.start_time = t0
                rel_ms = int((t0 - self.start_time) * 1000)

                p_item = self.press_buf.get_latest()
                f_item = self.force_buf.find_closest(p_item["t"]) if p_item else None
                if not p_item or not f_item:
                    time.sleep(0.001)
                    continue

                p_data = p_item["data"]
                f_data = f_item["data"]
                
                # 计算原始ADC总和（未做baseline subtraction）
                raw_adc_sum = np.sum(p_data)
                
                diff_frame = subtract_baseline(p_data)

                dir_x, dir_y, vec_mag = compute_gradient_in_region(diff_frame)
                adc_angle, _ = compute_gradient_angle_single(dir_x, dir_y)
                fx, fy = f_data[0], f_data[1]
                f_angle, f_mag = compute_force_angle(fx, fy)
                
                # 计算原始Force幅值（直接用Fx,Fy的平方和开方）
                raw_force_mag = np.hypot(fx, fy)

                # ==================== 【保存全程数据】 ====================
                full_time_list.append(rel_ms)
                full_adc_mag_list.append(vec_mag)
                full_force_mag_list.append(f_mag)

                if self.plot:
                    # 传递原始值给plot
                    self.plot.set_data(adc_angle, vec_mag, f_angle, f_mag, diff_frame, raw_adc_sum, raw_force_mag)

                ts_str = time.strftime("%Y%m%d%H%M%S%f")[:-3]
                row = [ts_str, rel_ms] + p_data + f_data + [
                    round(p_item["t"],6), round(f_item["t"],6), round(abs(p_item["t"]-f_item["t"])*1000,3),
                    adc_angle, vec_mag, f_angle, f_mag
                ]
                writer.writerow(row)
                csv_file.flush()
                elapsed = time.perf_counter() - t0
                time.sleep(max(0, period-elapsed))

    def stop(self):
        self.running = False
        self.stop_event.set()
        time.sleep(0.3)

# ==================== 程序结束绘制全程静态图 ====================
def plot_full_magnitude_curve():
    # 修复：创建两个独立的图表在同一画布上
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 图1：ADC幅值
    if len(full_time_list) > 0 and len(full_adc_mag_list) > 0:
        ax1.plot(full_time_list, full_adc_mag_list, 'b-', linewidth=1.5, label='ADC Magnitude')
        ax1.set_title("ADC Magnitude Over Time", fontsize=14)
        ax1.set_xlabel("Time (ms)", fontsize=12)
        ax1.set_ylabel("ADC Magnitude", fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=12)
        ax1.set_ylim(0, max(full_adc_mag_list) * 1.1 if max(full_adc_mag_list) > 0 else 1)
    
    # 图2：Force幅值
    if len(full_time_list) > 0 and len(full_force_mag_list) > 0:
        ax2.plot(full_time_list, full_force_mag_list, 'r-', linewidth=1.5, label='Force Magnitude')
        ax2.set_title("Force Magnitude Over Time", fontsize=14)
        ax2.set_xlabel("Time (ms)", fontsize=12)
        ax2.set_ylabel("Force Magnitude (N)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        ax2.set_ylim(0, max(full_force_mag_list) * 1.1 if max(full_force_mag_list) > 0 else 1)
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, "full_magnitude_curve.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

# ==================== Main ====================
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

    print("✅ Region gradient & symmetric area enabled")
    print("✅ Real-time angle error plot enabled")
    print("✅ 12×7 real-time pressure table enabled")
    print("✅ 12×7 real-time gradient table enabled")
    print("✅ Raw ADC & Force magnitude plots enabled (original values)")
    print(f"✅ Saving to: {csv_path}")
    
    plot.show()  # 运行实时界面
    
    collector.stop()
    
    # ==================== 【结束后自动出图】 ====================
    print("\n📊 Generating full-time magnitude curve...")
    plot_full_magnitude_curve()
    
    print("✅ Program finished")