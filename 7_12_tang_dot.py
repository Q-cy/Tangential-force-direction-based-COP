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

# ==================== 配置区 ====================
BAUDRATE_PRESS = 921600
BAUDRATE_FORCE = 460800
SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"

TARGET_HZ = 100.0                  # 融合/保存频率
PLOT_INTERVAL_MS = 100             # 动图刷新间隔
MAX_SYNC_DT = 0.015                # 时间戳最大匹配误差 15ms
PRESS_BUFFER_SIZE = 500
FORCE_BUFFER_SIZE = 500
PRINT_EVERY_N = 100
# ================================================


# ==================== 全局基线（严格沿用你的离线逻辑） ====================
first_frame = None
first_frame_lock = threading.Lock()


# ==================== 1. 基线扣除 + 找最大ADC ====================
def subtract_baseline(current_frame):
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()

    with first_frame_lock:
        if first_frame is None:
            first_frame = current_frame.copy()

    # 基线扣除
    diff_frame = current_frame - first_frame
    abs_diff = np.abs(diff_frame)

    # 按绝对值从大到小排序
    candidate_cols = np.argsort(-abs_diff)

    best_col = None
    for col in candidate_cols:
        # 同时满足：左右不越界 + 上下7不越界
        if (col - 1 >= 0) and (col + 1 < len(diff_frame)) and \
           (col - 7 >= 0) and (col + 7 < len(diff_frame)):
            best_col = col
            break

    # 极端情况兜底
    if best_col is None:
        best_col = 42

    # 输出：diff_frame + 安全最佳列
    return diff_frame, abs_diff[best_col], best_col


# ==================== 2. 计算 diff_adjacent ====================
def compute_diff_adjacent(frame, max_col):
    frame = np.array(frame)
    if max_col - 1 < 0 or max_col + 1 >= len(frame):
        return 0.0
    diff = frame[max_col + 1] - frame[max_col - 1]
    return np.array(diff)


# ==================== 3. 计算 diff_7step ====================
def compute_diff_7step(frame, max_col):
    frame = np.array(frame)
    if max_col - 7 < 0 or max_col + 7 >= len(frame):
        return 0.0
    diff = frame[max_col + 7] - frame[max_col - 7]
    return np.array(diff)


# ==================== 4. 计算 ADC 角度 ====================
def compute_gradient_angle(x_diff, y_diff):
    epsilon = 1e-8
    angle = np.degrees(np.arctan2(-y_diff, -x_diff + epsilon))
    if angle < 0:
        angle += 360
    return angle


# ==================== 5. 计算 Force 角度 ====================
def compute_force_angle(Fx, Fy):
    epsilon = 1e-8
    angle = np.degrees(np.arctan2(Fy, Fx + epsilon))
    angle = angle + 360 if angle < 0 else angle
    return angle


# ==================== 六维力传感器 ====================
class SixAxisForceSensor:
    def __init__(self):
        self.ser = None
        self.port = "/dev/ttyUSB0"
        self.zero_data = [0.0] * 6
        self.open_port()

    def open_port(self):
        try:
            self.ser = serial.Serial(
                port=self.port,
                baudrate=BAUDRATE_FORCE,
                bytesize=8,
                stopbits=1,
                parity='N',
                timeout=0.05
            )
            time.sleep(0.1)
            self.ser.reset_input_buffer()
            print(f"✅ 六维力传感器已连接: {self.port}")
        except Exception as e:
            print(f"❌ 六维力传感器打开失败: {e}")
            self.ser = None

    def reconnect(self):
        try:
            if self.ser:
                self.ser.close()
        except:
            pass
        self.ser = None
        time.sleep(0.2)
        self.open_port()

    def read(self):
        if not self.ser:
            return None

        try:
            cmd = b'\x49\xAA\x0D\x0A'
            self.ser.reset_input_buffer()
            self.ser.write(cmd)
            time.sleep(0.005)
            resp = self.ser.read(28)

            if len(resp) != 28 or resp[:2] != b'\x49\xAA':
                return None

            Fx = struct.unpack('<f', resp[2:6])[0]
            Fy = struct.unpack('<f', resp[6:10])[0]
            Fz = struct.unpack('<f', resp[10:14])[0]
            Mx = struct.unpack('<f', resp[14:18])[0]
            My = struct.unpack('<f', resp[18:22])[0]
            Mz = struct.unpack('<f', resp[22:26])[0]

            # 单位换算
            Fx *= 9.8
            Fy *= 9.8
            Fz *= 9.8
            Mx *= 9.8
            My *= 9.8
            Mz *= 9.8

            # 去皮
            Fx -= self.zero_data[0]
            Fy -= self.zero_data[1]
            Fz -= self.zero_data[2]
            Mx -= self.zero_data[3]
            My -= self.zero_data[4]
            Mz -= self.zero_data[5]

            return [round(v, 2) for v in [Fx, Fy, Fz, Mx, My, Mz]]

        except Exception:
            return None

    def calibrate_zero(self):
        print("\n🔄 六维力传感器去皮中...")
        vals = []
        for _ in range(20):
            d = self.read()
            if d:
                vals.append(d)
            time.sleep(0.05)

        if len(vals) >= 5:
            arr = np.array(vals, dtype=np.float32)
            self.zero_data = np.mean(arr, axis=0).tolist()
            print(f"✅ 去皮成功: {[round(v, 2) for v in self.zero_data]}")
        else:
            print("❌ 去皮失败，继续使用默认零点")


# ==================== 压力传感器 ====================
class PressureSensor:
    def __init__(self):
        self.ser = None
        self.port = None
        self.last = None
        self.auto_find_port()

    def auto_find_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p, _, _ in ports:
            if p == "/dev/ttyUSB0":
                continue
            try:
                self.ser = serial.Serial(
                    port=p,
                    baudrate=BAUDRATE_PRESS,
                    bytesize=8,
                    stopbits=1,
                    parity='N',
                    timeout=0.01
                )
                self.port = p
                time.sleep(0.1)
                self.ser.reset_input_buffer()
                print(f"✅ 压力传感器已连接: {p}")
                return
            except:
                continue
        raise Exception("❌ 未找到压力传感器串口")

    def reconnect(self):
        try:
            if self.ser:
                self.ser.close()
        except:
            pass
        self.ser = None
        self.port = None
        time.sleep(0.2)
        self.auto_find_port()

    def read_data(self):
        if not self.ser:
            return None

        try:
            cmd = bytes([
                0x55, 0xAA, 0x09, 0x00, 0x34, 0x00, 0xFB,
                0x00, 0x1C, 0x00, 0x00, 0xA8, 0x00, 0x35
            ])
            self.ser.reset_input_buffer()
            self.ser.write(cmd)
            time.sleep(0.005)
            resp = self.ser.read(256)

            idx = resp.find(b'\xaa\x55')
            if idx == -1 or len(resp[idx:]) < 182:
                return None

            return resp[idx + 14: idx + 14 + 168]

        except Exception:
            return None

    def decode(self, raw):
        arr = []
        for i in range(0, 168, 2):
            arr.append(struct.unpack("<H", raw[i:i+2])[0])

        out = []
        for i in range(12):
            out.extend(arr[i * 7:(i + 1) * 7])

        # 去毛刺：跳变过大则沿用上一帧
        if self.last is not None:
            for i in range(84):
                if abs(out[i] - self.last[i]) > 3000:
                    out[i] = self.last[i]

        self.last = out.copy()
        return out


# ==================== 时间戳缓存 ====================
class TimestampedBuffer:
    def __init__(self, maxlen=300):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, item):
        with self.lock:
            self.buf.append(item)

    def get_latest(self):
        with self.lock:
            if not self.buf:
                return None
            return self.buf[-1]

    def find_closest(self, ts):
        with self.lock:
            if not self.buf:
                return None

            best = None
            best_dt = None
            for item in self.buf:
                dt = abs(item["t"] - ts)
                if best_dt is None or dt < best_dt:
                    best = item
                    best_dt = dt
            return best


# ==================== 压力采集线程 ====================
class PressureReaderThread(threading.Thread):
    def __init__(self, sensor, out_buffer, stop_event):
        super().__init__(daemon=True)
        self.sensor = sensor
        self.out_buffer = out_buffer
        self.stop_event = stop_event
        self.fail_count = 0

    def run(self):
        while not self.stop_event.is_set():
            t_now = time.perf_counter()
            raw = self.sensor.read_data()

            if raw is None:
                self.fail_count += 1
                if self.fail_count >= 30:
                    print("⚠️ 压力传感器连续读取失败，尝试重连...")
                    try:
                        self.sensor.reconnect()
                    except Exception as e:
                        print(f"❌ 压力传感器重连失败: {e}")
                    self.fail_count = 0
                time.sleep(0.002)
                continue

            try:
                decoded = self.sensor.decode(raw)
                self.out_buffer.append({
                    "t": t_now,
                    "data": decoded
                })
                self.fail_count = 0
            except Exception:
                time.sleep(0.001)


# ==================== 力传感器采集线程 ====================
class ForceReaderThread(threading.Thread):
    def __init__(self, sensor, out_buffer, stop_event):
        super().__init__(daemon=True)
        self.sensor = sensor
        self.out_buffer = out_buffer
        self.stop_event = stop_event
        self.fail_count = 0

    def run(self):
        while not self.stop_event.is_set():
            t_now = time.perf_counter()
            data = self.sensor.read()

            if data is None:
                self.fail_count += 1
                if self.fail_count >= 30:
                    print("⚠️ 六维力传感器连续读取失败，尝试重连...")
                    try:
                        self.sensor.reconnect()
                    except Exception as e:
                        print(f"❌ 六维力传感器重连失败: {e}")
                    self.fail_count = 0
                time.sleep(0.002)
                continue

            self.out_buffer.append({
                "t": t_now,
                "data": data
            })
            self.fail_count = 0


# ==================== 实时双箭头绘图 ====================
class RealTimeAnglePlot:
    def __init__(self):
        self.adc_angle = 0.0
        self.force_angle = 0.0
        self.running = True
        self.lock = threading.Lock()

        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
        try:
            self.fig.canvas.manager.set_window_title("实时角度监测")
        except:
            pass

        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal')
        self.ax1.axis('off')
        self.ax1.set_title("ADC Angle")
        self.arrow_adc = self.ax1.arrow(
            0.5, 0.5, 0.001, 0.001,
            head_width=0.12, head_length=0.12,
            fc='k', ec='k', linewidth=2
        )
        self.txt_adc = self.ax1.text(0.5, 0.1, '', ha='center', fontsize=10)

        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_aspect('equal')
        self.ax2.axis('off')
        self.ax2.set_title("Force Angle")
        self.arrow_force = self.ax2.arrow(
            0.5, 0.5, 0.001, 0.001,
            head_width=0.12, head_length=0.12,
            fc='r', ec='r', linewidth=2
        )
        self.txt_force = self.ax2.text(0.5, 0.1, '', ha='center', fontsize=10)

        self.fig.canvas.mpl_connect('close_event', self.on_close)

    def on_close(self, event):
        print("🛑 绘图窗口关闭")
        self.running = False

    def set_angles(self, adc_angle, force_angle):
        with self.lock:
            self.adc_angle = float(adc_angle)
            self.force_angle = float(force_angle)

    def _redraw_arrow(self, ax, old_arrow, angle_deg, color):
        try:
            old_arrow.remove()
        except:
            pass

        theta = np.deg2rad(angle_deg)
        dx = 0.25 * np.cos(theta)
        dy = 0.25 * np.sin(theta)

        new_arrow = ax.arrow(
            0.5, 0.5, dx, dy,
            head_width=0.12, head_length=0.12,
            fc=color, ec=color, linewidth=2
        )
        return new_arrow

    def update(self, frame):
        if not self.running:
            return []

        with self.lock:
            adc_angle = self.adc_angle
            force_angle = self.force_angle

        self.arrow_adc = self._redraw_arrow(self.ax1, self.arrow_adc, adc_angle, 'k')
        self.arrow_force = self._redraw_arrow(self.ax2, self.arrow_force, force_angle, 'r')

        self.txt_adc.set_text(f"ADC: {adc_angle:.1f}°")
        self.txt_force.set_text(f"Force: {force_angle:.1f}°")

        return [self.arrow_adc, self.txt_adc, self.arrow_force, self.txt_force]

    def start_animation(self):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            interval=PLOT_INTERVAL_MS,
            blit=False,
            cache_frame_data=False
        )
        plt.tight_layout()
        plt.show()


# ==================== 数据融合/保存 ====================
class DataCollector:
    def __init__(self, force_sensor, press_sensor, csv_path):
        self.force_sensor = force_sensor
        self.press_sensor = press_sensor
        self.csv_path = csv_path

        self.running = True
        self.stop_event = threading.Event()
        self.plot = None

        self.press_buffer = TimestampedBuffer(maxlen=PRESS_BUFFER_SIZE)
        self.force_buffer = TimestampedBuffer(maxlen=FORCE_BUFFER_SIZE)

        self.press_thread = PressureReaderThread(
            self.press_sensor, self.press_buffer, self.stop_event
        )
        self.force_thread = ForceReaderThread(
            self.force_sensor, self.force_buffer, self.stop_event
        )

        self.frame_idx = 0
        self.start_time = None

    def set_plot(self, plot):
        self.plot = plot

    def start(self):
        self.press_thread.start()
        self.force_thread.start()
        self.thread = threading.Thread(target=self.collect_data, daemon=True)
        self.thread.start()

    def collect_data(self):
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # 这里特意保持和你离线代码兼容：
            # 第0列 timestamp
            # 第1列 rel_ms
            # 第2~85列 84个压力通道
            # 第86列 Fx
            # 第87列 Fy
            # 后面再加其他列
            header = (
                ["timestamp", "rel_ms"]
                + [f"ch{i+1}" for i in range(84)]
                + ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
                + ["press_t", "force_t", "sync_dt_ms",
                   "max_adc", "max_col", "x_diff", "y_diff",
                   "ADC_angle", "Force_angle"]
            )
            writer.writerow(header)

            print(f"\n📄 保存到：{self.csv_path}")
            print("✅ 双线程采集 + 时间戳对齐缓存 已启动")
            print(f"✅ 输出频率：{TARGET_HZ:.1f} Hz")
            print(f"✅ 最大同步误差：{MAX_SYNC_DT*1000:.1f} ms")
            print("✅ 实时双箭头动图已启用")
            print("ℹ️ 关闭绘图窗口即可退出\n")

            period = 1.0 / TARGET_HZ

            while self.running and not self.stop_event.is_set():
                loop_t0 = time.perf_counter()

                if self.start_time is None:
                    self.start_time = loop_t0
                rel_ms = int((loop_t0 - self.start_time) * 1000)

                p_item = self.press_buffer.get_latest()
                if p_item is None:
                    time.sleep(0.001)
                    continue

                f_item = self.force_buffer.find_closest(p_item["t"])
                if f_item is None:
                    time.sleep(0.001)
                    continue

                sync_dt = abs(p_item["t"] - f_item["t"])
                if sync_dt > MAX_SYNC_DT:
                    time.sleep(0.001)
                    continue

                p_data = p_item["data"]
                f_data = f_item["data"]

                # ==================== 严格按你的离线逻辑计算角度 ====================
                current_frame = p_data
                diff_frame, max_adc, max_col = subtract_baseline(current_frame)

                x_diff = compute_diff_adjacent(diff_frame, max_col)
                y_diff = compute_diff_7step(diff_frame, max_col)

                adc_angle = compute_gradient_angle(x_diff, y_diff)

                fx = f_data[0]
                fy = f_data[1]
                force_angle = compute_force_angle(fx, fy)

                # 更新绘图
                if self.plot:
                    self.plot.set_angles(adc_angle, force_angle)

                # 保存
                ts_str = time.strftime("%Y%m%d%H%M%S") + f"{int(time.time()*1000)%1000:03d}"

                row = (
                    [ts_str, rel_ms]
                    + p_data
                    + f_data
                    + [
                        round(p_item["t"], 6),
                        round(f_item["t"], 6),
                        round(sync_dt * 1000, 3),
                        float(max_adc),
                        int(max_col),
                        float(x_diff),
                        float(y_diff),
                        float(adc_angle),
                        float(force_angle)
                    ]
                )
                writer.writerow(row)
                f.flush()

                if self.frame_idx % PRINT_EVERY_N == 0:
                    print(
                        f"[{ts_str}] "
                        f"Frame={self.frame_idx} | "
                        f"ADC={float(adc_angle):7.2f}° | "
                        f"Force={float(force_angle):7.2f}° | "
                        f"dt={sync_dt*1000:6.2f} ms | "
                        f"max_adc={float(max_adc):8.2f} | "
                        f"max_col={int(max_col):2d}"
                    )

                self.frame_idx += 1

                if self.plot and not self.plot.running:
                    break

                elapsed = time.perf_counter() - loop_t0
                sleep_t = period - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

            print("\n🛑 数据采集停止")

    def stop(self):
        self.running = False
        self.stop_event.set()

        try:
            self.thread.join(timeout=1.0)
        except:
            pass

        try:
            self.press_thread.join(timeout=1.0)
        except:
            pass

        try:
            self.force_thread.join(timeout=1.0)
        except:
            pass

        try:
            if self.force_sensor.ser:
                self.force_sensor.ser.close()
        except:
            pass

        try:
            if self.press_sensor.ser:
                self.press_sensor.ser.close()
        except:
            pass


# ==================== 主程序 ====================
if __name__ == '__main__':
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 自动编号保存
    i = 1
    while os.path.exists(os.path.join(SAVE_DIR, f"data_{i}.csv")):
        i += 1
    csv_path = os.path.join(SAVE_DIR, f"data_{i}.csv")

    force_sensor = SixAxisForceSensor()
    press_sensor = PressureSensor()

    force_sensor.calibrate_zero()

    plot = RealTimeAnglePlot()

    collector = DataCollector(force_sensor, press_sensor, csv_path)
    collector.set_plot(plot)
    collector.start()

    try:
        plot.start_animation()
    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        collector.stop()
        plt.close('all')
        print("✅ 程序结束")