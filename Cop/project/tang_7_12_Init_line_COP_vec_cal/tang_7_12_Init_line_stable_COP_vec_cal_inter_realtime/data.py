"""
数据采集模块
功能：压力传感器/六维力传感器串口读取、解码、缓存、重连
"""
import serial
import serial.tools.list_ports
import time
import struct
import csv
import os
from collections import deque
import threading
import numpy as np

DATA_BAUDRATE_PRESS = 921600  # 压力传感器串口波特率
DATA_BAUDRATE_FORCE = 460860  # 六维力传感器串口波特率
DATA_DEBUG_DUMP_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"

# ===================== 压力传感器 =====================
class PressureSensor:                              # 为什么定义成类而不是函数？1.包含很多函数，2.方便管理状态，有很多global变量
    def __init__(self):
        self.ser = None                            # self.变量是默认global
        self.port = None
        self.last = None
        self._dump_cnt = 0                         # hex dump 计数器
        self.auto_find_port()

    def auto_find_port(self):                      #类里的函数，第一个参数必须是self，调用的时候代表对象自己
        """自动寻找可用串口"""
        ports = list(serial.tools.list_ports.comports())
        for p, _, _ in ports:
            if p == "/dev/ttyUSB0":
                continue
            try:
                self.ser = serial.Serial(p, DATA_BAUDRATE_PRESS, timeout=0.01)
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
        """读取一帧原始数据：write 后循环读满一帧，丢弃多余数据"""
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.reset_input_buffer()
            cmd = [0x55,0xAA,9,0,0x34,0,0xFB,0,0x1C,0,0,0xA8,0,0x35]
            self.ser.write(bytearray(cmd))
            # 循环读取直到收满一帧（182 字节），超时 50ms
            resp = b''
            t0 = time.perf_counter()
            while len(resp) < 182 and time.perf_counter() - t0 < 0.05:
                chunk = self.ser.read(256)
                if chunk:
                    resp += chunk

            if not resp:
                return None
            idx = resp.find(b'\xaa\x55')
            if idx == -1 or len(resp[idx:]) < 182:
                return None

            self._last_resp = resp
            self._last_idx = idx

            return resp[idx+14:idx+14+168]
        except Exception:
            return None

    def dump_last_raw(self, decoded_values=None):
        """将最近一帧的原始 hex 写入文件（供 decode 检测到异常时调用）"""
        resp = getattr(self, '_last_resp', None)
        if resp is None:
            return None

        self._dump_cnt += 1
        fname = os.path.join(DATA_DEBUG_DUMP_DIR,
                             f"_hex_dump_{self._dump_cnt:03d}_{time.strftime('%H%M%S')}.txt")
        with open(fname, 'w') as f:
            f.write(f"# dump #{self._dump_cnt}  time={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            all_pos = getattr(self, '_last_all_pos', [])
            f.write(f"# find() idx={self._last_idx}  all_pos={all_pos}  len(resp)={len(resp)}\n")
            f.write(f"# used bytes: resp[{self._last_idx}+14 : {self._last_idx}+14+168]\n\n")

            # 全量 hex dump，每行 16 字节
            for i in range(0, len(resp), 16):
                chunk = resp[i:i+16]
                hex_str = ' '.join(f'{b:02X}' for b in chunk)
                ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in chunk)
                f.write(f"  {i:04X}: {hex_str:<48s} {ascii_str}\n")

            f.write(f"\n# data-only hex (168 bytes from idx+14):\n")
            data_bytes = resp[self._last_idx+14:self._last_idx+14+168]
            f.write(' '.join(f'{b:02X}' for b in data_bytes) + '\n')

            # 解码预览
            if decoded_values is not None:
                f.write(f"\n# decoded 84ch, max={max(decoded_values)}, min={min(decoded_values)}\n")
                spike_idx = [i for i, v in enumerate(decoded_values) if v > 20000]
                if spike_idx:
                    f.write(f"# SPIKE channels: {[(i, decoded_values[i]) for i in spike_idx]}\n")
                for row in range(12):
                    chs = decoded_values[row*7:(row+1)*7]
                    f.write('  ' + ' '.join(f'{v:6d}' for v in chs) + '\n')

        return fname

    def decode(self, raw):
        """解码为84通道数组，检测到跳变 >20000 时自动保存 hex dump"""
        arr = [struct.unpack("<H", raw[i:i+2])[0] for i in range(0, 168, 2)]
        out = []
        for i in range(12):
            out.extend(arr[i*7:(i+1)*7])

        # 逐通道跳变检测
        if self.last is not None and len(self.last) == 84:
            for i in range(84):
                if out[i] - self.last[i] > 20000:
                    out[i] = self.last[i]
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