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

# ==================== 配置区 ====================
BAUDRATE_PRESS = 921600
BAUDRATE_FORCE = 460800
SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"

TARGET_HZ = 100.0
PLOT_INTERVAL_MS = 100
MAX_SYNC_DT = 0.015
PRESS_BUFFER_SIZE = 500
FORCE_BUFFER_SIZE = 500
THRESHOLD = 1000
# ================================================

# ==================== 全局基线 ====================
first_frame = None
first_frame_lock = threading.Lock()

# ==================== 算法函数（完全保留你的逻辑） ====================
def subtract_baseline(current_frame):
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()
    with first_frame_lock:
        if first_frame is None:
            first_frame = current_frame.copy()
    diff_frame = current_frame - first_frame
    return diff_frame

def get_pressure_region_indices(diff_frame, threshold):
    rows = 12
    cols = 7
    valid_indices = []
    for idx in range(rows * cols):
        if diff_frame[idx] > threshold:
            valid_indices.append(idx)
    if not valid_indices:
        return [], 4, 3
    coords = [(idx // cols, idx % cols) for idx in valid_indices]
    min_col = min(c for r, c in coords)
    max_col = max(c for r, c in coords)
    min_row = min(r for r, c in coords)
    max_row = max(r for r, c in coords)
    region_indices = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            idx = r * cols + c
            region_indices.append(idx)
    frame_mat = diff_frame.reshape(rows, cols)
    sub_mat = frame_mat[min_row:max_row+1, min_col:max_col+1]
    row_sums = sub_mat.sum(axis=1)
    best_local_row = np.argmax(row_sums)
    col_sums = sub_mat.sum(axis=0)
    best_local_col = np.argmax(col_sums)
    center_row = min_row + best_local_row
    center_col = min_col + best_local_col
    return region_indices, center_row, center_col

def compute_diff_adjacent(frame, region_indices, center_row, center_col):
    frame = np.array(frame)
    cols_total = 7
    if not region_indices:
        return 0.0
    left_points = [idx for idx in region_indices if (idx % cols_total) < center_col]
    right_points = [idx for idx in region_indices if (idx % cols_total) > center_col]
    left_sum = np.sum(frame[left_points]) if left_points else 0
    right_sum = np.sum(frame[right_points]) if right_points else 0
    return right_sum - left_sum

def compute_diff_7step(frame, region_indices, center_row, center_col):
    frame = np.array(frame)
    cols_total = 7
    if not region_indices:
        return 0.0
    upper_points = [idx for idx in region_indices if (idx // cols_total) < center_row]
    lower_points = [idx for idx in region_indices if (idx // cols_total) > center_row]
    upper_sum = np.sum(frame[upper_points]) if upper_points else 0
    lower_sum = np.sum(frame[lower_points]) if lower_points else 0
    return upper_sum - lower_sum

def compute_gradient_angle_single(x, y):
    epsilon = 1e-8
    angle = np.degrees(np.arctan2(-y, -x + epsilon))
    if angle < 0:
        angle += 360
    magnitude = np.sqrt(x**2 + y**2)
    return angle, magnitude

def compute_force_angle(Fx, Fy):
    epsilon = 1e-8
    force_magnitude = np.sqrt(Fx**2 + Fy**2)
    force_angle = np.degrees(np.arctan2(Fy, Fx + epsilon))
    if force_angle < 0:
        force_angle += 360
    return force_angle, force_magnitude

# ==================== 六维力传感器 ====================
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
            Fx*=9.8; Fy*=9.8; Fz*=9.8; Mx*=9.8; My*=9.8; Mz*=9.8
            Fx-=self.zero_data[0]; Fy-=self.zero_data[1]; Fz-=self.zero_data[2]
            Mx-=self.zero_data[3]; My-=self.zero_data[4]; Mz-=self.zero_data[5]
            return [round(v,2) for v in [Fx,Fy,Fz,Mx,My,Mz]]
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

# ==================== 压力传感器 ====================
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
        raise Exception("压力传感器未找到")
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
        if self.last is not None:
            for i in range(84):
                if abs(out[i]-self.last[i])>3000:
                    out[i]=self.last[i]
        self.last = out.copy()
        return out

# ==================== 数据缓存 ====================
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

# ==================== 采集线程 ====================
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

# ==================== 实时绘图 ====================
class RealTimePlot:
    def __init__(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        self.adc_angle = 0
        self.adc_mag = 0
        self.force_angle = 0
        self.force_mag = 0
        self.lock = threading.Lock()
        self.fixed_arrow = 0.35
        self.epsilon = 1e-8

        # 图1：固定长度方向
        self.fig1, self.axes1 = plt.subplots(10,6,figsize=(12,6), gridspec_kw={'width_ratios':[1,1,1,1,1,0.8]})
        for r in range(10):
            for c in range(5):
                ax=self.axes1[r,c]
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                ax.set_aspect('equal')
                ax.axis('off')
        self.center1 = self.axes1[4:6,2:3][0,0]
        self.gs1 = GridSpec(10,6,width_ratios=[1,1,1,1,1,0.8])
        self.force1 = self.fig1.add_subplot(self.gs1[:,5])
        self.force1.set_xlim(0,1); self.force1.set_ylim(0,1)
        self.force1.set_aspect('equal'); self.force1.axis('off')
        self.force1.set_title("Force",fontsize=12)
        for r in range(10):
            self.axes1[r,5].remove()
        # ✅ 修复：关闭缓存，消除警告
        self.ani1 = FuncAnimation(self.fig1, self.update1, interval=PLOT_INTERVAL_MS, cache_frame_data=False)

        # 图2：变长度（强度）
        self.fig2, self.axes2 = plt.subplots(10,6,figsize=(12,6), gridspec_kw={'width_ratios':[1,1,1,1,1,0.8]})
        for r in range(10):
            for c in range(5):
                ax=self.axes2[r,c]
                ax.set_xlim(0,1); ax.set_ylim(0,1)
                ax.set_aspect('equal'); ax.axis('off')
        self.center2 = self.axes2[4:6,2:3][0,0]
        self.gs2 = GridSpec(10,6,width_ratios=[1,1,1,1,1,0.8])
        self.force2 = self.fig2.add_subplot(self.gs2[:,5])
        self.force2.set_xlim(0,1); self.force2.set_ylim(0,1)
        self.force2.set_aspect('equal'); self.force2.axis('off')
        self.force2.set_title("Force",fontsize=12)
        for r in range(10):
            self.axes2[r,5].remove()
        # ✅ 修复：关闭缓存，消除警告
        self.ani2 = FuncAnimation(self.fig2, self.update2, interval=PLOT_INTERVAL_MS, cache_frame_data=False)

    def set_data(self, adc_a, adc_m, f_a, f_m):
        with self.lock:
            self.adc_angle = adc_a
            self.adc_mag = adc_m
            self.force_angle = f_a
            self.force_mag = f_m

    def update1(self, frame):
        with self.lock:
            a = self.adc_angle
            f = self.force_angle
        self.center1.clear()
        self.center1.set_xlim(0,1); self.center1.set_ylim(0,1)
        self.center1.set_aspect('equal'); self.center1.axis('off')
        th = np.deg2rad(a)
        self.center1.arrow(0.5,0.5, 0.4*np.cos(th),0.4*np.sin(th), head_width=0.15,fc='k',ec='k',lw=3)
        self.force1.clear()
        self.force1.set_xlim(0,1); self.force1.set_ylim(0,1)
        self.force1.set_aspect('equal'); self.force1.axis('off')
        thf = np.deg2rad(f)
        self.force1.arrow(0.5,0.5, self.fixed_arrow*np.cos(thf), self.fixed_arrow*np.sin(thf), head_width=0.12,fc='r',ec='r',lw=2)
        self.force1.set_title("Force",fontsize=12)
        return []

    def update2(self, frame):
        with self.lock:
            a = self.adc_angle
            m = self.adc_mag
            fa = self.force_angle
            fm = self.force_mag
        self.center2.clear()
        self.center2.set_xlim(0,1); self.center2.set_ylim(0,1)
        self.center2.set_aspect('equal'); self.center2.axis('off')
        th = np.deg2rad(a)
        l = 0.4 * min(m/5000, 1.0) if m>self.epsilon else 0
        self.center2.arrow(0.5,0.5, l*np.cos(th), l*np.sin(th), head_width=0.15,fc='k',ec='k',lw=3)
        self.force2.clear()
        self.force2.set_xlim(0,1); self.force2.set_ylim(0,1)
        self.force2.set_aspect('equal'); self.force2.axis('off')
        thf = np.deg2rad(fa)
        lf = 0.35 * min(fm/50,1.0) if fm>self.epsilon else 0
        self.force2.arrow(0.5,0.5, lf*np.cos(thf), lf*np.sin(thf), head_width=0.12,fc='r',ec='r',lw=2)
        self.force2.set_title("Force",fontsize=12)
        return []

    def show(self):
        plt.show()

# ==================== 数据采集 + 保存 ====================
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
        self.frame = 0
        self.start_time = None

    def set_plot(self, p):
        self.plot = p

    def start(self):
        self.press_thread = PressureReaderThread(self.press_sensor, self.press_buf, self.stop_event)
        self.force_thread = ForceReaderThread(self.force_sensor, self.force_buf, self.stop_event)
        self.press_thread.start()
        self.force_thread.start()
        threading.Thread(target=self.run_collect, daemon=True).start()

    def run_collect(self):
        # ✅ 修复：文件对象必须用正确的变量名！
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
                if not p_item:
                    time.sleep(0.001)
                    continue

                f_item = self.force_buf.find_closest(p_item["t"])
                if not f_item:
                    time.sleep(0.001)
                    continue

                sync_dt = abs(p_item["t"] - f_item["t"])
                if sync_dt > MAX_SYNC_DT:
                    time.sleep(0.001)
                    continue

                # 你的算法（完全不变）
                p_data = p_item["data"]
                f_data = f_item["data"]
                diff_frame = subtract_baseline(p_data)
                region_indices, center_r, center_c = get_pressure_region_indices(diff_frame, THRESHOLD)
                x_diff = compute_diff_adjacent(diff_frame, region_indices, center_r, center_c)
                y_diff = compute_diff_7step(diff_frame, region_indices, center_r, center_c)
                adc_angle, adc_mag = compute_gradient_angle_single(x_diff, y_diff)
                fx, fy = f_data[0], f_data[1]
                f_angle, f_mag = compute_force_angle(fx, fy)

                if self.plot:
                    self.plot.set_data(adc_angle, adc_mag, f_angle, f_mag)

                # 写入CSV
                ts_str = time.strftime("%Y%m%d%H%M%S%f")[:-3]
                row = [ts_str, rel_ms] + p_data + f_data + [
                    round(p_item["t"], 6),
                    round(f_item["t"], 6),
                    round(sync_dt * 1000, 3),
                    adc_angle, adc_mag, f_angle, f_mag
                ]
                writer.writerow(row)
                csv_file.flush()  # ✅ 修复：正确的文件对象

                self.frame += 1
                elapsed = time.perf_counter() - t0
                sleep_time = max(0, period - elapsed)
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
        self.stop_event.set()
        time.sleep(0.3)

# ==================== 主程序 ====================
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

    print(f"✅ 实时采集已启动 → 保存到: {csv_path}")
    plot.show()
    collector.stop()
    print("✅ 采集结束")