"""
数据采集模块
功能：压力传感器/六维力传感器串口读取、解码、缓存、重连
"""
import serial
import serial.tools.list_ports
import time
import struct
import csv
from collections import deque
import threading
import numpy as np

BAUDRATE_PRESS = 921600
BAUDRATE_FORCE = 460860

# ===================== 压力传感器 =====================
class PressureSensor:
    def __init__(self):
        self.ser = None
        self.port = None
        self.last = None
        self.auto_find_port()

    def auto_find_port(self):
        """自动寻找可用串口"""
        ports = list(serial.tools.list_ports.comports())
        for p, _, _ in ports:
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
        raise Exception("未找到压力传感器")

    def reconnect(self):
        """断开重连"""
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.auto_find_port()

    def read_data(self):
        """读取一帧原始数据"""
        if not self.ser or not self.ser.is_open:
            return None
        try:
            cmd = [0x55,0xAA,9,0,0x34,0,0xFB,0,0x1C,0,0,0xA8,0,0x35]
            self.ser.write(bytearray(cmd))
            time.sleep(0.005)
            resp = self.ser.read(256)
            idx = resp.find(b'\xaa\x55')
            if idx == -1 or len(resp[idx:]) < 182:
                return None
            return resp[idx+14:idx+14+168]
        except Exception as e:
            return None

    def decode(self, raw):
        """解码为84通道数组"""
        arr = []
        for i in range(0, 168, 2):
            arr.append(struct.unpack("<H", raw[i:i+2])[0])
        out = []
        for i in range(12):
            out.extend(arr[i*7:(i+1)*7])
        self.last = out.copy()
        return out

# ===================== 六维力传感器 =====================
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
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.open_port()

    def read(self):
        """读取力/力矩数据"""
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.write(b'\x49\xAA\x0D\x0A')
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