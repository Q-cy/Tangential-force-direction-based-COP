"""
数据采集模块
功能：压力传感器/六维力传感器串口读取、解码、缓存、重连
"""
import serial
import time
import struct
from collections import deque
import threading
import numpy as np

DATA_BAUDRATE_PRESS = 921600  # 压力传感器串口波特率
DATA_BAUDRATE_FORCE = 460800  # 六维力传感器串口波特率

# 固定串口路径
PRESSURE_SENSOR_PORT = "/dev/ttyUSB0"   # 压阻
FORCE_SENSOR_PORT = "/dev/ttyUSB1"      # 六维力

# 压阻阵列几何（14 行 × 5 列 = 70 通道，行优先）
PRESSURE_ROWS = 14
PRESSURE_COLS = 5
PRESSURE_CELLS = PRESSURE_ROWS * PRESSURE_COLS        # 70
PRESSURE_PAYLOAD_LEN = PRESSURE_CELLS * 2             # 140
PRESSURE_FRAME_LEN = 14 + PRESSURE_PAYLOAD_LEN + 1    # 155: 14B 头 + 140B payload + 1B CRC

# 活跃接触区: 只关注 2×2 块（1-based 坐标 (4,2)(4,3)(5,2)(5,3) → 0-based 行[3,4] 列[1,2]）
# 读取时掩码, 其余 cell 置 0, 所有下游操作只看该区域
PRESSURE_ACTIVE_R0, PRESSURE_ACTIVE_R1 = 3, 5   # 0-based 行 [3, 5)
PRESSURE_ACTIVE_C0, PRESSURE_ACTIVE_C1 = 1, 3   # 0-based 列 [1, 3)

# ===================== 压力传感器 =====================
class PressureSensor:
    # 请求帧: data[11-12]=读取数据长度(140 LE=0x8C 0x00), data[13]=CRC-8-ITU
    CMD_BYTES = bytes([0x55, 0xAA, 0x09, 0x00, 0x34, 0x00,
                       0xFB, 0x00, 0x1C, 0x00, 0x00, 0x8C, 0x00, 0xCF])
    FRAME_LEN = PRESSURE_FRAME_LEN   # 155 = 14B 头 + 140B payload + 1B CRC
    MAX_RX_BUF = 4096                # 缓存区大小

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
        """读一帧删一帧: 解析 _rx_buf 头部, 成功返回 140B payload 并删除, 失败整段/单字节删除"""
        with self._rx_lock:  # 整个解析逻辑在锁内, 防止与 _rx_loop race
            for _ in range(100):  # safety: 防止坏数据卡死
                if len(self._rx_buf) < self.FRAME_LEN:
                    return None  # 不够 155B, 等下次 read 补齐

                if self._rx_buf[0:2] != b'\xaa\x55':
                    del self._rx_buf[:1]   # 不是真同步头, 滑 1 字节
                    continue

                # buf[0:2] == AA 55, 长度够 155B
                if self._rx_buf[13] != 0:
                    del self._rx_buf[:self.FRAME_LEN]   # status 错, 整段错误帧丢弃
                    continue
                crc_idx = 14 + PRESSURE_PAYLOAD_LEN    # CRC 位置 = 帧尾
                if self.crc8_itu(bytes(self._rx_buf[:crc_idx])) != self._rx_buf[crc_idx]:
                    del self._rx_buf[:self.FRAME_LEN]   # CRC 错, 整段错误帧丢弃
                    continue

                # 校验通过, 提取 payload
                payload = bytes(self._rx_buf[14:crc_idx])
                del self._rx_buf[:self.FRAME_LEN]
                return payload

            # 解析 100 次还没拿到 (数据全坏), 清空防卡死
            self._rx_buf.clear()
            return None

    def decode(self, raw):
        """raw = 140 字节 payload (70 个 uint16 LE, 14×5, 行优先)。
        只保留 2×2 活跃接触区（行3-4、列1-2, 0-based），其余 cell 置 0。"""
        arr = [struct.unpack("<H", raw[i:i+2])[0] for i in range(0, PRESSURE_PAYLOAD_LEN, 2)]
        grid = np.asarray(arr, dtype=np.float64).reshape(PRESSURE_ROWS, PRESSURE_COLS)
        mask = np.zeros((PRESSURE_ROWS, PRESSURE_COLS), dtype=np.float64)
        mask[PRESSURE_ACTIVE_R0:PRESSURE_ACTIVE_R1, PRESSURE_ACTIVE_C0:PRESSURE_ACTIVE_C1] = 1.0
        return (grid * mask).flatten().tolist()

    def close(self):
        self._running = False
        if self._tx_thread.is_alive():
            self._tx_thread.join(timeout=1)
        if self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1)
        if self.ser and self.ser.is_open:
            self.ser.close()

# ===================== 六维力传感器 =====================
class SixAxisForceSensor:
    CMD_BYTES = b'\x49\xAA\x0D\x0A'   # 读一帧命令
    FRAME_LEN = 28                    # 2B 帧头 + 24B 数据 + 2B 帧尾
    MAX_RX_BUF = 512                  # 持久化接收缓冲区上限
    CMD_INTERVAL_S = 0.005            # 命令发送最小间隔 (与设备 ~8ms 处理延迟匹配)

    def __init__(self):
        self.ser = None
        self.port = FORCE_SENSOR_PORT
        self.zero_data = [0.0]*6
        self._rx_buf = bytearray()          # 持久化接收缓冲 (粘包/分包统一处理)
        self._rx_lock = threading.Lock()    # 保护 _rx_buf
        self._io_lock = threading.Lock()    # 保护串口写/读 + zero_data (多线程并发访问)
        self._last_cmd_t = 0.0
        self.open_port()

    def open_port(self):
        """打开固定路径的串口;失败抛错(由 main.py 捕获)"""
        self.ser = serial.Serial(self.port, DATA_BAUDRATE_FORCE, timeout=0.01)
        time.sleep(0.1)
        self.ser.reset_input_buffer()

    def _fill_rx_buf(self):
        """把串口当前可读字节追加进 _rx_buf (调用方须持 _io_lock);
        溢出时按最近的 49 AA 截断, 保留可对齐的最旧帧"""
        try:
            waiting = self.ser.in_waiting
            if waiting <= 0:
                return
            chunk = self.ser.read(waiting)
            if not chunk:
                return
            if len(self._rx_buf) + len(chunk) > self.MAX_RX_BUF:
                idx = self._rx_buf.rfind(b'\x49\xaa')
                if idx >= 0:
                    del self._rx_buf[:idx]
                else:
                    self._rx_buf.clear()
            self._rx_buf.extend(chunk)
        except Exception:
            pass

    def _try_pop_frame(self):
        """从 _rx_buf 解析一帧: 帧头 49 AA + 28B + 帧尾 0D 0A; 成功取帧并返回, 无完整帧返回 None"""
        with self._rx_lock:
            for _ in range(16):   # safety: 坏数据滑字节, 防止卡死
                if len(self._rx_buf) < self.FRAME_LEN:
                    return None
                if self._rx_buf[0:2] != b'\x49\xaa':
                    del self._rx_buf[:1]          # 非帧头, 滑 1 字节
                    continue
                if self._rx_buf[26:28] != b'\x0d\x0a':
                    # 伪帧头: 跳到下一个 49 AA; 没有则滑 1 字节
                    nxt = self._rx_buf.find(b'\x49\xaa', 2)
                    if nxt >= 0:
                        del self._rx_buf[:nxt]
                    else:
                        del self._rx_buf[:1]
                    continue
                resp = bytes(self._rx_buf[:self.FRAME_LEN])
                del self._rx_buf[:self.FRAME_LEN]
                return resp
            self._rx_buf.clear()   # 解析多次仍失败, 清空防卡死
            return None

    def _parse_frame(self, resp: bytes):
        """28B 帧解析为 [Fx,Fy,Fz,Mx,My,Mz] (N, 去零点, 保留 2 位小数)"""
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

    def read(self):
        """写命令 + 从持久化缓冲解析一帧; 无完整帧返回 None。
        不再每次 reset_input_buffer (readme 所述丢帧根源): 字节持续累积, 解析出完整帧再移除。"""
        if not self.ser or not self.ser.is_open:
            return None
        with self._io_lock:   # 防止与 rezero 线程并发访问串口/zero_data
            try:
                now = time.perf_counter()
                if now - self._last_cmd_t >= self.CMD_INTERVAL_S:
                    self.ser.write(self.CMD_BYTES)
                    self._last_cmd_t = now
                self._fill_rx_buf()
            except Exception:
                return None
            resp = self._try_pop_frame()
        if resp is None:
            return None
        return self._parse_frame(resp)

    def calibrate_zero(self):
        """零点校准（1帧）"""
        d = self.read()
        if d:
            with self._io_lock:
                self.zero_data = d

    def add_zero_bias(self, fx: float, fy: float):
        """累加 Fx/Fy 零点偏差 (锁内修改 zero_data, 供 rezero 线程调用)"""
        with self._io_lock:
            self.zero_data[0] += fx
            self.zero_data[1] += fy

    def close(self):
        """关闭串口 (main.py 退出路径调用)"""
        if self.ser and self.ser.is_open:
            self.ser.close()

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
        """返回时间上最接近 ts 的帧; 缓冲为空返回 None (F5 严格同步用)"""
        with self.lock:
            best = None
            best_dt = 1e9
            for item in self.buf:
                dt = abs(item["t"] - ts)
                if dt < best_dt:
                    best_dt = dt
                    best = item
            return best


def match_closest(buf: "TimestampedBuffer", ts: float, max_diff_s: float):
    """在 buf 中找与 ts 最接近的帧; 时间差超过 max_diff_s 返回 None (F5 严格同步)"""
    item = buf.find_closest(ts)
    if item is None or abs(item["t"] - ts) > max_diff_s:
        return None
    return item