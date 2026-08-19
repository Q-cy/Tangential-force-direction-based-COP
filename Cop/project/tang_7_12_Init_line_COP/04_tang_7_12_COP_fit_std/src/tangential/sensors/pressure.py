"""压力阵列串口驱动：200 Hz 目标、单请求在途、独立采集进程。"""

import multiprocessing
import os
import queue
import select
import serial
import struct
import threading
import time
from collections import deque

DATA_BAUDRATE_PRESS = 921600
PRESSURE_SENSOR_PORT = "/dev/ttyUSB0"
PRESSURE_TARGET_HZ = 200
PRESSURE_PERIOD_S = 1.0 / PRESSURE_TARGET_HZ
PRESSURE_RESPONSE_TIMEOUT_S = 0.050
PRESSURE_FRAME_QUEUE_SIZE = 256


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
                 readiness_waiter=None, port=None, _use_process=None,
                 _mp_context=None, _process_factory=None, _startup_timeout_s=2.0,
                 _frame_sink=None, _status_sink=None):
        if period_s <= 0 or response_timeout_s <= 0 or queue_size <= 0:
            raise ValueError("压力采集周期、响应超时和队列长度必须大于 0")

        self.ser = None
        self.port = port or PRESSURE_SENSOR_PORT
        self._rx_buf = bytearray()
        self._rx_lock = threading.Lock()
        self._frame_queue = queue.Queue(maxsize=queue_size)
        self._frame_sink = _frame_sink
        self._status_sink = _status_sink
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
        self._last_stats_publish_t = None
        self._closed = False
        self._io_thread = None
        self._process = None
        self._ipc_frame_queue = None
        self._ipc_status_queue = None
        self._ipc_startup_queue = None
        self._mp_stop_event = None
        self._process_startup_timeout_s = float(_startup_timeout_s)
        self._process_factory = _process_factory
        self._mp_context = _mp_context

        if _use_process is None:
            use_process = serial_instance is None and _frame_sink is None
        else:
            use_process = bool(_use_process)
        self._use_process = use_process

        if self._use_process:
            self._start_process(queue_size)
        else:
            if serial_instance is None:
                self.open_port()
            else:
                self.ser = serial_instance
            self._io_thread = threading.Thread(
                target=self._io_loop, name="pressure-io", daemon=True
            )
            self._io_thread.start()

    def _start_process(self, queue_size):
        """启动独立采集进程，并同步等待子进程完成串口初始化。"""
        context = self._mp_context or multiprocessing.get_context("spawn")
        self._ipc_frame_queue = context.Queue(maxsize=queue_size)
        # 统计不是数据通道，使用小的有界队列即可；子进程不会因统计拥塞而阻塞采集。
        self._ipc_status_queue = context.Queue(maxsize=8)
        self._ipc_startup_queue = context.Queue(maxsize=1)
        self._mp_stop_event = context.Event()
        process_factory = self._process_factory or context.Process
        self._process = process_factory(
            target=_pressure_process_main,
            args=(
                self.port,
                self._period_s,
                self._response_timeout_s,
                queue_size,
                self._ipc_frame_queue,
                self._ipc_status_queue,
                self._ipc_startup_queue,
                self._mp_stop_event,
            ),
        )
        try:
            self._process.daemon = True
        except (AttributeError, AssertionError):
            pass
        try:
            self._process.start()
            message = self._ipc_startup_queue.get(
                timeout=self._process_startup_timeout_s
            )
        except Exception:
            self._stop_process(join_timeout=0.5)
            self._close_ipc_queues()
            raise

        kind, detail = message
        if kind != "ready":
            self._stop_process(join_timeout=0.5)
            self._close_ipc_queues()
            raise RuntimeError(f"压力串口启动失败: {detail}")

    def _stop_process(self, join_timeout=1.0):
        if self._mp_stop_event is not None:
            self._mp_stop_event.set()
        if self._process is None:
            return
        try:
            self._process.join(timeout=join_timeout)
        except Exception:
            pass
        try:
            alive = self._process.is_alive()
        except Exception:
            alive = False
        if alive:
            # 只终止由本 PressureSensor 创建的子进程。
            try:
                self._process.terminate()
                self._process.join(timeout=join_timeout)
            except Exception:
                pass

    def _close_ipc_queues(self):
        ipc_queues = (
            self._ipc_frame_queue,
            self._ipc_status_queue,
            self._ipc_startup_queue,
        )
        for ipc_queue in ipc_queues:
            if ipc_queue is None:
                continue
            try:
                # 不等待 multiprocessing.Queue 的 feeder 线程排空，避免 close()
                # 因积压数据而阻塞采集退出。
                ipc_queue.cancel_join_thread()
            except Exception:
                pass
            try:
                ipc_queue.close()
            except Exception:
                continue
        self._ipc_frame_queue = None
        self._ipc_status_queue = None
        self._ipc_startup_queue = None

    def _drain_process_status(self):
        if self._ipc_status_queue is None:
            return
        while True:
            try:
                kind, payload = self._ipc_status_queue.get_nowait()
            except queue.Empty:
                return
            if kind == "stats":
                self._apply_stats(payload)
            elif kind == "error":
                self._error = RuntimeError(f"压力串口子进程异常: {payload}")

    def _apply_stats(self, snapshot):
        with self._stats_lock:
            for name, value in snapshot.items():
                if name in ("tx_intervals_s", "rx_intervals_s", "latencies_s"):
                    continue
                if name in self._stats:
                    self._stats[name] = value
            if "tx_intervals_s" in snapshot:
                self._tx_intervals = deque(snapshot["tx_intervals_s"], maxlen=1000)
            if "rx_intervals_s" in snapshot:
                self._rx_intervals = deque(snapshot["rx_intervals_s"], maxlen=1000)
            if "latencies_s" in snapshot:
                self._latencies = deque(snapshot["latencies_s"], maxlen=1000)

    def _stats_snapshot(self):
        with self._stats_lock:
            result = dict(self._stats)
            result["tx_intervals_s"] = list(self._tx_intervals)
            result["rx_intervals_s"] = list(self._rx_intervals)
            result["latencies_s"] = list(self._latencies)
            return result

    def _publish_stats(self, force=False):
        """子进程非阻塞同步统计，统计拥塞不能阻塞压力轮询。"""
        if self._status_sink is None:
            return
        now = time.perf_counter()
        if not force and self._last_stats_publish_t is not None:
            if now - self._last_stats_publish_t < 0.2:
                return
        try:
            self._status_sink.put_nowait(("stats", self._stats_snapshot()))
            self._last_stats_publish_t = now
        except queue.Full:
            pass

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
        if getattr(self, "_frame_sink", None) is not None:
            try:
                self._frame_sink.put_nowait(frame)
            except queue.Full:
                self._add_stat("queue_drops")
            return
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
                    self._publish_stats()
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
                self._publish_stats()
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
        if self._use_process:
            deadline = (
                None if timeout_s is None
                else time.monotonic() + timeout_s
            )
            while True:
                self._drain_process_status()
                if self._error is not None:
                    raise self._error
                if self._process is not None and not self._process.is_alive():
                    if self._stop_event.is_set():
                        return None
                    self._error = RuntimeError("压力串口子进程已退出")
                    raise self._error
                if deadline is None:
                    wait_s = 0.1
                else:
                    wait_s = max(0.0, min(0.1, deadline - time.monotonic()))
                try:
                    return self._ipc_frame_queue.get(timeout=wait_s)
                except queue.Empty:
                    if deadline is not None and time.monotonic() >= deadline:
                        return None
                    if self._stop_event.is_set():
                        return None
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
        self._drain_process_status()
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
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._use_process:
            self._stop_process(join_timeout=1)
            self._drain_process_status()
            self._close_ipc_queues()
        else:
            if self._io_thread is not None and self._io_thread.is_alive():
                self._io_thread.join(timeout=1)
            if self.ser and self.ser.is_open:
                self.ser.close()


def _pressure_process_main(port, period_s, response_timeout_s, queue_size,
                           frame_queue, status_queue, startup_queue,
                           stop_event):
    """spawn 子进程入口：子进程内部仍使用 PressureSensor 的本地线程模式。"""
    sensor = None
    try:
        sensor = PressureSensor(
            serial_instance=None,
            period_s=period_s,
            response_timeout_s=response_timeout_s,
            queue_size=queue_size,
            port=port,
            _use_process=False,
            _status_sink=status_queue,
        )
        startup_queue.put(("ready", None))
        while not stop_event.is_set():
            frame = sensor.read_frame(timeout_s=0.05)
            if frame is not None:
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    # 这是进程间数据通道溢出，不是子进程内部线程队列溢出。
                    sensor._add_stat("queue_drops")
                sensor._publish_stats()
            if sensor._error is not None:
                try:
                    status_queue.put_nowait(("error", str(sensor._error)))
                except queue.Full:
                    pass
                break
            if sensor._io_thread is not None and not sensor._io_thread.is_alive():
                try:
                    status_queue.put_nowait(("error", "压力 I/O 线程意外退出"))
                except queue.Full:
                    pass
                break
    except Exception as exc:
        try:
            startup_queue.put(("error", f"{type(exc).__name__}: {exc}"))
        except Exception:
            pass
    finally:
        if sensor is not None:
            sensor._publish_stats(force=True)
            sensor.close()
