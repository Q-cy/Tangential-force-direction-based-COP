"""
数据采集模块
功能：压力传感器/六维力传感器串口读取、解码、缓存、重连
"""
import serial
import time
import struct
from collections import deque
import threading

DATA_BAUDRATE_PRESS = 921600  # 压力传感器串口波特率
DATA_BAUDRATE_FORCE = 460800  # 六维力传感器串口波特率

# 固定串口路径
PRESSURE_SENSOR_PORT = "/dev/ttyUSB0"   # 压阻
FORCE_SENSOR_PORT = "/dev/ttyUSB1"      # 六维力

# ===================== 压力传感器 =====================
class PressureSensor:
    CMD_BYTES = bytes([0x55, 0xAA, 0x09, 0x00, 0x34, 0x00,
                       0xFB, 0x00, 0x1C, 0x00, 0x00, 0xA8, 0x00, 0x35])
    FRAME_LEN = 183              # 14B header + 168B payload + 1B status
    MAX_RX_BUF = 4096            # 缓存区大小

    def __init__(self):
        self.ser = None
        self.port = PRESSURE_SENSOR_PORT
        self.open_port()
        self._rx_buf = bytearray()
        self._rx_lock = threading.Lock()        # 保护 _rx_buf 跨线程访问
        self._running = True
        self._tx_thread = threading.Thread(target=self._tx_loop, daemon=True)
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._tx_thread.start()
        self._rx_thread.start()

    def open_port(self):
        """打开固定路径的串口;失败抛错(由 main.py 捕获)"""
        self.ser = serial.Serial(self.port, DATA_BAUDRATE_PRESS, timeout=0.01)
        time.sleep(0.1)
        self.ser.reset_input_buffer()

    # TODO(deprecated): reconnect 死代码 - 当前未使用（保留作为扩展点）
    def reconnect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.open_port()

    @staticmethod
    def crc8_itu(data: bytes) -> int:
        """CRC-8-ITU 校验(多项式 0x07, 初始 0x00, final XOR 0x55)"""
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc ^ 0x55

    def _tx_loop(self):
        """固定 10ms 一次 READ: 与设备"读一帧删一帧"模型同步"""
        while self._running:
            try:
                self.ser.write(self.CMD_BYTES)
            except Exception:
                pass
            time.sleep(0.005)

    def _rx_loop(self):
        """持续读字节追加到 _rx_buf (解析在 read_data 时做)"""
        while self._running:
            try:
                # 用 in_waiting 读全部可用字节, 防止硬件 FIFO 溢出丢字节
                waiting = self.ser.in_waiting
                if waiting > 0:
                    chunk = self.ser.read(waiting)  # 一次读完, 无阻塞
                else:
                    time.sleep(0.001)  # 短暂 sleep, 避免 busy loop
                    continue
                if chunk:
                    with self._rx_lock:  # 保护 _rx_buf
                        # 溢出策略: 保留最近的完整帧, 按 AA 55 重新对齐
                        if len(self._rx_buf) + len(chunk) > self.MAX_RX_BUF:
                            first_aa55 = self._rx_buf.find(b'\xaa\x55')
                            if first_aa55 >= 0:
                                second_aa55 = self._rx_buf.find(b'\xaa\x55', first_aa55 + 2)
                                if second_aa55 >= 0:
                                    # 删到第二个 AA 55 之前 (丢最旧 1 帧, 保留对齐)
                                    del self._rx_buf[:second_aa55]
                                else:
                                    self._rx_buf.clear()
                            else:
                                self._rx_buf.clear()
                        # 再次检查 (极端: 新 chunk > MAX_RX_BUF)
                        if len(self._rx_buf) + len(chunk) > self.MAX_RX_BUF:
                            self._rx_buf.clear()
                        self._rx_buf.extend(chunk)
            except Exception:
                pass

    def read_data(self):
        """读一帧删一帧: 解析 _rx_buf 头部, 成功返回 168B payload 并删除, 失败整段/单字节删除"""
        with self._rx_lock:  # 整个解析逻辑在锁内, 防止与 _rx_loop race
            for _ in range(100):  # safety: 防止坏数据卡死
                if len(self._rx_buf) < self.FRAME_LEN:
                    return None  # 不够 183B, 等下次 read 补齐

                if self._rx_buf[0:2] != b'\xaa\x55':
                    del self._rx_buf[:1]   # 不是真同步头, 滑 1 字节
                    continue

                # buf[0:2] == AA 55, 长度够 183B
                if self._rx_buf[13] != 0:
                    del self._rx_buf[:self.FRAME_LEN]   # status 错, 整段错误帧丢弃
                    continue
                if self.crc8_itu(bytes(self._rx_buf[:182])) != self._rx_buf[182]:
                    del self._rx_buf[:self.FRAME_LEN]   # CRC 错, 整段错误帧丢弃
                    continue

                # 校验通过, 提取 168B payload
                payload = bytes(self._rx_buf[14:182])
                del self._rx_buf[:self.FRAME_LEN]
                return payload

            # 解析 100 次还没拿到 (数据全坏), 清空防卡死
            self._rx_buf.clear()
            return None

    def decode(self, raw):
        """raw = 168 字节 payload (84 个 uint16 LE, 12×7)"""
        arr = [struct.unpack("<H", raw[i:i+2])[0] for i in range(0, 168, 2)]
        out = []
        for i in range(12):
            out.extend(arr[i*7:(i+1)*7])
        return out

    def close(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()

# ===================== 六维力传感器 =====================
class SixAxisForceSensor:
    def __init__(self):
        self.ser = None
        self.port = FORCE_SENSOR_PORT
        self.zero_data = [0.0]*6
        self.open_port()

    def open_port(self):
        """打开固定路径的串口;失败抛错(由 main.py 捕获)"""
        self.ser = serial.Serial(self.port, DATA_BAUDRATE_FORCE, timeout=0.05)
        time.sleep(0.1)
        self.ser.reset_input_buffer()

    # TODO(deprecated): reconnect 死代码 - 当前未使用（保留作为扩展点）
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
        """零点校准（1帧）"""
        d = self.read()
        if d:
            self.zero_data = d

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

    # TODO(deprecated): find_closest 死代码 - 当前未使用（主循环只用 get_latest）
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