"""
数据采集模块
功能：压力传感器/六维力传感器串口读取、解码、缓存、重连
"""
import serial
import time
import struct
import os
import select
from collections import deque
import queue
import threading

DATA_BAUDRATE_PRESS = 921600  # 压力传感器串口波特率
DATA_BAUDRATE_FORCE = 460800  # 六维力传感器串口波特率

# 固定串口路径
PRESSURE_SENSOR_PORT = "/dev/ttyUSB0"   # 压阻
FORCE_SENSOR_PORT = "/dev/ttyUSB1"      # 六维力

PRESSURE_TARGET_HZ = 200
PRESSURE_PERIOD_S = 1.0 / PRESSURE_TARGET_HZ
PRESSURE_RESPONSE_TIMEOUT_S = 0.050
PRESSURE_FRAME_QUEUE_SIZE = 256

# ===================== 压力传感器 =====================
class PressureSensor:
    CMD_BYTES = bytes([0x55, 0xAA, 0x09, 0x00, 0x34, 0x00,
                       0xFB, 0x00, 0x1C, 0x00, 0x00, 0xA8, 0x00, 0x35])
    FRAME_LEN = 183              # 默认响应: 4B头 + 178B payload + 1B CRC
    EXPECTED_SENSOR_BYTES = 168
    MIN_PAYLOAD_LEN = 10
    MAX_PAYLOAD_LEN = 512
    MAX_RX_BUF = 8192
    RX_BUF_RETAIN = 4096
    READ_CHUNK_SIZE = 1024

    def __init__(self, serial_instance=None, period_s=PRESSURE_PERIOD_S,
                 response_timeout_s=PRESSURE_RESPONSE_TIMEOUT_S,
                 queue_size=PRESSURE_FRAME_QUEUE_SIZE,
                 readiness_waiter=None):
        if period_s <= 0 or response_timeout_s <= 0 or queue_size <= 0:
            raise ValueError("压力采集周期、响应超时和队列长度必须大于 0")

        self.ser = None
        self.port = PRESSURE_SENSOR_PORT
        self._rx_buf = bytearray()
        self._rx_lock = threading.Lock()
        self._frame_queue = queue.Queue(maxsize=queue_size)
        self._period_s = float(period_s)
        self._response_timeout_s = float(response_timeout_s)
        # 仅供无文件描述符的测试 fake 使用。生产路径保持 None，必须走
        # select.select(fd, ...) + os.read(fd, ...)。
        self._readiness_waiter = readiness_waiter
        self._stop_event = threading.Event()
        self._error = None
        self._request_seq = 0
        self._stats_lock = threading.Lock()
        self._stats = {
            "requests": 0,
            "frames": 0,
            "response_timeouts": 0,
            "crc_errors": 0,
            "length_errors": 0,
            "status_errors": 0,
            "framing_bytes": 0,
            "serial_read_errors": 0,
            "serial_write_errors": 0,
            "serial_flush_errors": 0,
            "queue_drops": 0,
            "schedule_skips": 0,
        }
        self._tx_intervals = deque(maxlen=1000)
        self._rx_intervals = deque(maxlen=1000)
        self._latencies = deque(maxlen=1000)
        self._last_tx_t = None
        self._last_rx_t = None

        if serial_instance is None:
            self.open_port()
        else:
            self.ser = serial_instance

        self._io_thread = threading.Thread(
            target=self._io_loop, name="pressure-io", daemon=True
        )
        self._io_thread.start()

    def open_port(self):
        """打开固定路径的串口;失败抛错(由 main.py 捕获)"""
        self.ser = serial.Serial(
            self.port,
            DATA_BAUDRATE_PRESS,
            timeout=0,
            write_timeout=0,
        )
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

    def _add_stat(self, name, amount=1):
        if not hasattr(self, "_stats_lock"):
            return
        with self._stats_lock:
            self._stats[name] += amount

    def _append_rx(self, chunk):
        """追加串口字节；仿照 C++ 解析器，过大时保留最近 4096B。"""
        if not chunk:
            return
        with self._rx_lock:
            self._rx_buf.extend(chunk)
            if len(self._rx_buf) > self.MAX_RX_BUF:
                drop_count = len(self._rx_buf) - self.RX_BUF_RETAIN
                del self._rx_buf[:drop_count]
                self._add_stat("framing_bytes", drop_count)

    def _record_tx(self, tx_t):
        with self._stats_lock:
            if self._last_tx_t is not None:
                self._tx_intervals.append(tx_t - self._last_tx_t)
            self._last_tx_t = tx_t
            self._stats["requests"] += 1

    def _record_frame(self, rx_t, latency_s):
        with self._stats_lock:
            if self._last_rx_t is not None:
                self._rx_intervals.append(rx_t - self._last_rx_t)
            self._last_rx_t = rx_t
            self._latencies.append(latency_s)
            self._stats["frames"] += 1

    def _queue_frame(self, frame):
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
            self._add_stat("queue_drops")
            self._frame_queue.put_nowait(frame)

    def _clear_cycle_buffers(self):
        """每轮清空串口输入/输出和解析器，与 C++ TCIOFLUSH 对齐。"""
        with self._rx_lock:
            self._rx_buf.clear()
        for reset in (self.ser.reset_input_buffer, self.ser.reset_output_buffer):
            try:
                reset()
            except Exception:
                self._add_stat("serial_flush_errors")

    def _read_chunk(self, timeout_s):
        """等待并批量读取当前可用字节，语义与 C++ SerialPort::read 一致。

        生产串口使用非阻塞 fd：select 负责等待，os.read 只读取当下已经
        到达的字节，最多 1024B。readiness_waiter 只为无 fd 的测试 fake
        提供最小注入点，生产路径不会使用它。
        """
        wait_s = min(0.010, max(0.0, timeout_s))
        try:
            if self._readiness_waiter is not None:
                if not self._readiness_waiter(wait_s):
                    return b""
                return self.ser.read(self.READ_CHUNK_SIZE)

            fd = self.ser.fileno()
            readable, _, _ = select.select([fd], [], [], wait_s)
            if not readable:
                return b""
            return os.read(fd, self.READ_CHUNK_SIZE)
        except Exception:
            self._add_stat("serial_read_errors")
            self._stop_event.wait(0.001)
            return b""

    def _io_loop(self):
        """C++式逐轮轮询：flush、单请求、等待响应、补足目标周期。"""
        try:
            while not self._stop_event.is_set():
                cycle_start = time.perf_counter()
                self._clear_cycle_buffers()
                tx_t = time.perf_counter()
                request_seq = self._request_seq
                self._request_seq += 1
                self._record_tx(tx_t)
                try:
                    written = self.ser.write(self.CMD_BYTES)
                    if written != len(self.CMD_BYTES):
                        self._add_stat("serial_write_errors")
                except Exception:
                    self._add_stat("serial_write_errors")

                response_deadline = time.perf_counter() + self._response_timeout_s
                payload = None
                while (
                    not self._stop_event.is_set()
                    and time.perf_counter() < response_deadline
                ):
                    remaining = response_deadline - time.perf_counter()
                    chunk = self._read_chunk(remaining)
                    if chunk:
                        self._append_rx(chunk)
                        payload = self.read_data()
                        if payload is not None:
                            break

                if payload is not None:
                    rx_t = time.perf_counter()
                    latency_s = rx_t - tx_t
                    self._record_frame(rx_t, latency_s)
                    self._queue_frame({
                        "request_seq": request_seq,
                        "tx_t": tx_t,
                        "rx_t": rx_t,
                        "latency_s": latency_s,
                        "raw": payload,
                    })
                elif not self._stop_event.is_set():
                    self._add_stat("response_timeouts")

                # 单轮只接受一个响应；其余粘包/残片不带入下一请求。
                with self._rx_lock:
                    self._rx_buf.clear()

                elapsed = time.perf_counter() - cycle_start
                if elapsed < self._period_s:
                    self._stop_event.wait(self._period_s - elapsed)
                else:
                    self._add_stat("schedule_skips")
        except Exception as exc:
            self._error = exc
            self._stop_event.set()

    def read_data(self):
        """按响应长度字段流式解析一个168B传感器 payload。"""
        with self._rx_lock:
            while True:
                header_pos = self._rx_buf.find(b'\xaa\x55')
                if header_pos < 0:
                    keep = 1 if self._rx_buf[-1:] == b'\xaa' else 0
                    drop_count = len(self._rx_buf) - keep
                    if drop_count > 0:
                        del self._rx_buf[:drop_count]
                        self._add_stat("framing_bytes", drop_count)
                    return None
                if header_pos > 0:
                    del self._rx_buf[:header_pos]
                    self._add_stat("framing_bytes", header_pos)

                if len(self._rx_buf) < 4:
                    return None
                payload_len = self._rx_buf[2] | (self._rx_buf[3] << 8)
                if not self.MIN_PAYLOAD_LEN <= payload_len <= self.MAX_PAYLOAD_LEN:
                    self._add_stat("length_errors")
                    del self._rx_buf[:2]
                    continue

                total_len = 4 + payload_len + 1
                if len(self._rx_buf) < total_len:
                    return None

                expected_crc = self.crc8_itu(bytes(self._rx_buf[:4 + payload_len]))
                if expected_crc != self._rx_buf[4 + payload_len]:
                    self._add_stat("crc_errors")
                    del self._rx_buf[:2]
                    continue

                sensor_len = payload_len - 10
                if sensor_len != self.EXPECTED_SENSOR_BYTES:
                    self._add_stat("length_errors")
                    del self._rx_buf[:total_len]
                    continue
                if self._rx_buf[13] != 0:
                    self._add_stat("status_errors")
                    del self._rx_buf[:total_len]
                    continue

                payload = bytes(self._rx_buf[14:14 + sensor_len])
                del self._rx_buf[:total_len]
                return payload

    def read_frame(self, timeout_s=0.1):
        """返回带请求/接收时间的完整压力帧；超时返回 None。"""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s 不能小于 0")
        while True:
            try:
                wait_s = 0.1 if timeout_s is None else timeout_s
                return self._frame_queue.get(timeout=wait_s)
            except queue.Empty:
                if self._error is not None:
                    raise RuntimeError(
                        f"压力串口 I/O 异常: {self._error}"
                    ) from self._error
                if timeout_s is not None or self._stop_event.is_set():
                    return None

    def get_timing_stats(self):
        """返回累计计数及最近最多 1000 帧的时序样本快照。"""
        with self._stats_lock:
            result = dict(self._stats)
            result["tx_intervals_s"] = list(self._tx_intervals)
            result["rx_intervals_s"] = list(self._rx_intervals)
            result["latencies_s"] = list(self._latencies)
            return result

    def decode(self, raw):
        """raw = 168 字节 payload (84 个 uint16 LE, 12×7)"""
        arr = [struct.unpack("<H", raw[i:i+2])[0] for i in range(0, 168, 2)]
        out = []
        for i in range(12):
            out.extend(arr[i*7:(i+1)*7])
        return out

    def close(self):
        self._stop_event.set()
        if self._io_thread.is_alive():
            self._io_thread.join(timeout=1)
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
            # zero_data 与帧解析放在同一把锁内，避免运行期重新归零时
            # 一帧的六个通道混用新旧零点。
            return self._parse_frame(resp)

    def calibrate_zero(self, sample_count: int = 10, timeout_s: float = 1.0) -> bool:
        """收集多帧完成启动零点校准，成功返回 True。

        校准在线程启动前执行，此时串口只有本方法一个消费者。设备没有在
        timeout_s 内返回足够帧时不修改零点，并由上层禁用力传感器通道。
        """
        if sample_count <= 0 or timeout_s <= 0:
            raise ValueError("sample_count 和 timeout_s 必须大于 0")

        samples = []
        deadline = time.perf_counter() + timeout_s
        while len(samples) < sample_count and time.perf_counter() < deadline:
            d = self.read()
            if d is not None:
                samples.append(d)
            else:
                time.sleep(0.001)

        if len(samples) < sample_count:
            return False

        zero = [sum(values) / len(samples) for values in zip(*samples)]
        with self._io_lock:
            self.zero_data = zero
        return True

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
        self._next_seq = 0

    def append(self, item):
        """追加一帧并自动赋予当前 buffer 内单调递增的 seq。"""
        with self.lock:
            stored = dict(item)
            stored["seq"] = self._next_seq
            self._next_seq += 1
            self.buf.append(stored)
            return stored["seq"]

    def get_latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def get_after(self, seq):
        """按 seq 顺序返回尚未处理的帧快照。"""
        with self.lock:
            return [item for item in self.buf if item["seq"] > seq]

    def find_closest(self, ts, max_diff_s=None, min_seq=-1):
        """返回未使用帧中时间最接近 ts 的一帧。

        min_seq 是已消费的最后序号，候选帧必须满足 seq > min_seq。
        max_diff_s 非 None 时，超出时间窗口直接返回 None。
        """
        with self.lock:
            best = None
            best_dt = float("inf")
            for item in self.buf:
                if item["seq"] <= min_seq:
                    continue
                dt = abs(item["t"] - ts)
                if dt < best_dt:
                    best_dt = dt
                    best = item
            if best is not None and max_diff_s is not None and best_dt > max_diff_s:
                return None
            return best


def match_closest(buf: "TimestampedBuffer", ts: float, max_diff_s: float,
                  min_seq: int = -1):
    """在 buf 中找与 ts 最接近的帧; 时间差超过 max_diff_s 返回 None (F5 严格同步)"""
    return buf.find_closest(ts, max_diff_s=max_diff_s, min_seq=min_seq)
