"""
数据采集模块
功能：压力传感器/六维力传感器串口读取、解码、缓存、重连
"""
import serial
import serial.tools.list_ports
import time
import struct
from collections import deque
import threading
import numpy as np

from eskin_ffi import EskinDevice

DATA_BAUDRATE_FORCE = 460800  # 六维力传感器串口波特率

# ===================== 压力传感器 =====================
class PressureSensor:
    def __init__(self):
        self.dev = EskinDevice()
        self.dev.open("/dev/ttyUSB0")
        self.dev.start_stream()
        self._raw_queue = deque(maxlen=100)
        self._running = True
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        while self._running:
            try:
                self.dev.read_sample(timeout_ms=50)
                raw = self.dev.read_stream_frame(timeout_ms=1)
                if raw:
                    self._raw_queue.append(raw)
            except RuntimeError:
                pass

    def read_data(self):
        return self._raw_queue.popleft() if self._raw_queue else None

    def decode(self, raw):
        raw_after = raw[14:-1]
        arr = [struct.unpack("<H", raw_after[i:i+2])[0] for i in range(0, 168, 2)]
        out = []
        for i in range(12):
            out.extend(arr[i*7:(i+1)*7])
        return out

    def close(self):
        self._running = False
        self.dev.stop_stream()

# ===================== 六维力传感器 =====================
class SixAxisForceSensor:
    def __init__(self):
        self.ser = None
        self.port = None
        self.zero_data = [0.0]*6
        self.auto_find_port()

    def auto_find_port(self):
        """自动寻找可用串口（排除 /dev/ttyUSB0）"""
        ports = list(serial.tools.list_ports.comports())
        for p, _, _ in ports:
            if p == "/dev/ttyUSB0":
                continue
            try:
                self.ser = serial.Serial(p, DATA_BAUDRATE_FORCE, timeout=0.05)
                self.port = p
                time.sleep(0.1)
                self.ser.reset_input_buffer()
                return
            except:
                continue
        raise Exception("未找到六维力传感器")

    def open_port(self):
        try:
            self.ser = serial.Serial(self.port, DATA_BAUDRATE_FORCE, timeout=0.05)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        except:
            self.ser = None

    def reconnect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.open_port()

    def read(self):
        """读取力/力矩数据（清空缓存 + 帧头校验）"""
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.reset_input_buffer()  # 清空残留，防帧错位
            self.ser.write(b'\x49\xAA\x0D\x0A')
            time.sleep(0.008)
            resp = self.ser.read(28)
            if len(resp) != 28 or resp[:2] != b'\x49\xAA' or resp[-2:] != b'\x0D\x0A':
                return None
            Fx = struct.unpack('<f', resp[2:6])[0]
            Fy = struct.unpack('<f', resp[6:10])[0]
            Fz = struct.unpack('<f', resp[10:14])[0]
            Mx = struct.unpack('<f', resp[14:18])[0]
            My = struct.unpack('<f', resp[18:22])[0]
            Mz = struct.unpack('<f', resp[22:26])[0]
            Fx *= 9.8; Fy *= 9.8; Fz *= 9.8
            Mx *= 9.8; My *= 9.8; Mz *= 9.8
            Fx -= self.zero_data[0]; Fy -= self.zero_data[1]; Fz -= self.zero_data[2]
            Mx -= self.zero_data[3]; My -= self.zero_data[4]; Mz -= self.zero_data[5]
            return [round(v, 2) for v in [Fx, Fy, Fz, Mx, My, Mz]]
        except Exception as e:
            return None

    def calibrate_zero(self):
        """零点校准"""
        vals = []
        for _ in range(20):
            d = self.read()
            if d:
                vals.append(d)
            time.sleep(0.05)
        if len(vals) >= 5:
            self.zero_data = np.mean(np.array(vals), axis=0).tolist()

# ===================== 带时间戳的线程安全缓存 =====================
class TimestampedBuffer:
    def __init__(self, maxlen=500):
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
                dt = abs(item["t"] - ts)
                if dt < best_dt:
                    best_dt = dt
                    best = item
            return best