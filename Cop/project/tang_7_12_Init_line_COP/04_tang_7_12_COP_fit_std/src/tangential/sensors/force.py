"""六维力传感器串口驱动与软件校零。"""

import multiprocessing
import os
import queue
import select
import serial
import struct
import threading
import time
from collections import deque

from .pressure import PRESSURE_PERIOD_S, PRESSURE_TARGET_HZ

DATA_BAUDRATE_FORCE = 460800
FORCE_SENSOR_PORT = "/dev/ttyUSB1"
FORCE_TARGET_HZ = PRESSURE_TARGET_HZ
FORCE_PERIOD_S = PRESSURE_PERIOD_S
FORCE_RESPONSE_TIMEOUT_S = 0.050
FORCE_FRAME_QUEUE_SIZE = 256


class SixAxisForceSensor:
    """六维力采集器。

    生产路径由 spawn 子进程承载串口 I/O；注入 serial_instance 时保留线程
    后端，方便协议和调度测试。子进程只传递未扣零点的物理量，zero_data
    只存在父进程，避免归零线程与串口读取线程竞争。
    """

    CMD_BYTES = b"\x49\xAA\x0D\x0A"
    FRAME_LEN = 28
    MAX_RX_BUF = 4096
    READ_CHUNK_SIZE = 1024

    def __init__(self, serial_instance=None, period_s=FORCE_PERIOD_S,
                 response_timeout_s=FORCE_RESPONSE_TIMEOUT_S,
                 queue_size=FORCE_FRAME_QUEUE_SIZE, readiness_waiter=None,
                 port=None, _use_process=None, _mp_context=None,
                 _process_factory=None, _startup_timeout_s=2.0,
                 _status_sink=None):
        if period_s <= 0 or response_timeout_s <= 0 or queue_size <= 0:
            raise ValueError("六维力采集周期、响应超时和队列长度必须大于 0")

        self.ser = None
        self.port = port or FORCE_SENSOR_PORT
        self.zero_data = [0.0] * 6
        self._zero_lock = threading.Lock()
        self._rx_buf = bytearray()
        self._rx_lock = threading.Lock()
        self._frame_queue = queue.Queue(maxsize=queue_size)
        self._period_s = float(period_s)
        self._response_timeout_s = float(response_timeout_s)
        self._readiness_waiter = readiness_waiter
        self._status_sink = _status_sink
        self._stop_event = threading.Event()
        self._error = None
        self._request_seq = 0
        self._stats_lock = threading.Lock()
        self._stats = {
            "requests": 0,
            "frames": 0,
            "response_timeouts": 0,
            "framing_errors": 0,
            "tail_errors": 0,
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
            use_process = serial_instance is None
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
                target=self._io_loop, name="force-io", daemon=True
            )
            self._io_thread.start()

    def _start_process(self, queue_size):
        context = self._mp_context or multiprocessing.get_context("spawn")
        self._ipc_frame_queue = context.Queue(maxsize=queue_size)
        self._ipc_status_queue = context.Queue(maxsize=8)
        self._ipc_startup_queue = context.Queue(maxsize=1)
        self._mp_stop_event = context.Event()
        process_factory = self._process_factory or context.Process
        self._process = process_factory(
            target=_force_process_main,
            args=(self.port, self._period_s, self._response_timeout_s,
                  queue_size, self._ipc_frame_queue, self._ipc_status_queue,
                  self._ipc_startup_queue, self._mp_stop_event),
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
            raise RuntimeError(f"六维力串口启动失败: {detail}")

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
            try:
                self._process.terminate()
                self._process.join(timeout=join_timeout)
            except Exception:
                pass

    def _close_ipc_queues(self):
        for ipc_queue in (self._ipc_frame_queue, self._ipc_status_queue,
                          self._ipc_startup_queue):
            if ipc_queue is None:
                continue
            try:
                ipc_queue.cancel_join_thread()
            except Exception:
                pass
            try:
                ipc_queue.close()
            except Exception:
                pass
        self._ipc_frame_queue = None
        self._ipc_status_queue = None
        self._ipc_startup_queue = None

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
                self._error = RuntimeError(f"六维力串口子进程异常: {payload}")

    def _stats_snapshot(self):
        with self._stats_lock:
            result = dict(self._stats)
            result["tx_intervals_s"] = list(self._tx_intervals)
            result["rx_intervals_s"] = list(self._rx_intervals)
            result["latencies_s"] = list(self._latencies)
            return result

    def _publish_stats(self, force=False):
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
        self.ser = serial.Serial(
            self.port, DATA_BAUDRATE_FORCE, timeout=0, write_timeout=0
        )
        time.sleep(0.1)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            self._add_stat("serial_flush_errors")

    def _add_stat(self, name, amount=1):
        if not hasattr(self, "_stats_lock") or not hasattr(self, "_stats"):
            return
        with self._stats_lock:
            self._stats[name] = self._stats.get(name, 0) + amount

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
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                self._add_stat("queue_drops")

    def _clear_parser(self):
        with self._rx_lock:
            self._rx_buf.clear()

    def _flush_input_output(self):
        self._clear_parser()
        for reset in (self.ser.reset_input_buffer, self.ser.reset_output_buffer):
            try:
                reset()
            except Exception:
                self._add_stat("serial_flush_errors")

    def _flush_output(self):
        try:
            self.ser.reset_output_buffer()
        except Exception:
            self._add_stat("serial_flush_errors")

    def _read_chunk(self, timeout_s):
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

    def _append_rx(self, chunk):
        if not chunk:
            return
        with self._rx_lock:
            self._rx_buf.extend(chunk)
            if len(self._rx_buf) > self.MAX_RX_BUF:
                drop = len(self._rx_buf) - self.MAX_RX_BUF
                del self._rx_buf[:drop]
                self._add_stat("framing_bytes", drop)

    def _try_pop_frame(self):
        """从持久缓存取出一个合法28B帧，支持噪声、分包、粘包和坏尾。"""
        with self._rx_lock:
            while True:
                header_pos = self._rx_buf.find(b"\x49\xaa")
                if header_pos < 0:
                    keep = 1 if self._rx_buf[-1:] == b"\x49" else 0
                    drop = len(self._rx_buf) - keep
                    if drop:
                        del self._rx_buf[:drop]
                        self._add_stat("framing_errors")
                        self._add_stat("framing_bytes", drop)
                    return None
                if header_pos:
                    del self._rx_buf[:header_pos]
                    self._add_stat("framing_errors")
                    self._add_stat("framing_bytes", header_pos)
                if len(self._rx_buf) < self.FRAME_LEN:
                    return None
                if self._rx_buf[26:28] != b"\x0d\x0a":
                    self._add_stat("tail_errors")
                    next_header = self._rx_buf.find(b"\x49\xaa", 2)
                    del self._rx_buf[:next_header if next_header >= 0 else 1]
                    continue
                frame = bytes(self._rx_buf[:self.FRAME_LEN])
                del self._rx_buf[:self.FRAME_LEN]
                return frame

    @staticmethod
    def _parse_frame(resp: bytes):
        values = [struct.unpack("<f", resp[offset:offset + 4])[0] * 9.8
                  for offset in range(2, 26, 4)]
        return [round(value, 2) for value in values]

    def _apply_zero(self, frame):
        with self._zero_lock:
            zero = tuple(self.zero_data)
        adjusted = [round(value - bias, 2)
                    for value, bias in zip(frame["data"], zero)]
        result = dict(frame)
        result["data"] = adjusted
        return result

    def _io_loop(self):
        try:
            while not self._stop_event.is_set():
                cycle_start = time.perf_counter()
                self._flush_output()
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

                deadline = time.perf_counter() + self._response_timeout_s
                response = None
                while (not self._stop_event.is_set()
                       and time.perf_counter() < deadline):
                    chunk = self._read_chunk(deadline - time.perf_counter())
                    if chunk:
                        self._append_rx(chunk)
                        response = self._try_pop_frame()
                        if response is not None:
                            break

                if response is not None:
                    # rx_t 必须位于完整帧确定之后、数值解析之前。
                    rx_t = time.perf_counter()
                    latency_s = rx_t - tx_t
                    values = self._parse_frame(response)
                    self._record_frame(rx_t, latency_s)
                    self._queue_frame({
                        "request_seq": request_seq,
                        "tx_t": tx_t,
                        "rx_t": rx_t,
                        "latency_s": latency_s,
                        "data": values,
                    })
                    # 每个请求只接受一个响应，粘包余量不能带入下一请求。
                    self._clear_parser()
                    self._publish_stats()
                elif not self._stop_event.is_set():
                    self._add_stat("response_timeouts")
                    # 超时后清除串口和解析缓存，避免晚到帧错配下一请求。
                    self._flush_input_output()

                elapsed = time.perf_counter() - cycle_start
                if elapsed < self._period_s:
                    self._stop_event.wait(self._period_s - elapsed)
                else:
                    self._add_stat("schedule_skips")
                self._publish_stats()
        except Exception as exc:
            self._error = exc
            self._stop_event.set()

    def read_frame(self, timeout_s=0.1):
        """返回带真实接收时间的力帧；data 在父进程扣除 zero_data。"""
        if timeout_s is not None and timeout_s < 0:
            raise ValueError("timeout_s 不能小于 0")
        if self._use_process:
            deadline = None if timeout_s is None else time.monotonic() + timeout_s
            while True:
                self._drain_process_status()
                if self._error is not None:
                    raise self._error
                if self._process is not None and not self._process.is_alive():
                    if self._stop_event.is_set():
                        return None
                    raise RuntimeError("六维力串口子进程已退出")
                wait_s = 0.1 if deadline is None else max(
                    0.0, min(0.1, deadline - time.monotonic())
                )
                try:
                    return self._apply_zero(self._ipc_frame_queue.get(timeout=wait_s))
                except queue.Empty:
                    if deadline is not None and time.monotonic() >= deadline:
                        return None
                    if self._stop_event.is_set():
                        return None
        while True:
            try:
                wait_s = 0.1 if timeout_s is None else timeout_s
                frame = self._frame_queue.get(timeout=wait_s)
                return self._apply_zero(frame)
            except queue.Empty:
                if self._error is not None:
                    raise RuntimeError(f"六维力 I/O 异常: {self._error}") from self._error
                if timeout_s is not None or self._stop_event.is_set():
                    return None

    def read(self):
        """兼容旧调用方：非阻塞读取一帧。"""
        return self.read_frame(timeout_s=0.0)

    def calibrate_zero(self, sample_count: int = 10, timeout_s: float = 1.0) -> bool:
        if sample_count <= 0 or timeout_s <= 0:
            raise ValueError("sample_count 和 timeout_s 必须大于 0")
        samples = []
        deadline = time.perf_counter() + timeout_s
        while len(samples) < sample_count and time.perf_counter() < deadline:
            frame = self.read_frame(timeout_s=min(0.1, max(
                0.0, deadline - time.perf_counter())))
            if frame is not None:
                samples.append(frame["data"])
            else:
                time.sleep(0.001)
        if len(samples) < sample_count:
            return False
        zero = [sum(values) / len(samples) for values in zip(*samples)]
        with self._zero_lock:
            self.zero_data = zero
        return True

    def add_zero_bias(self, fx: float, fy: float):
        with self._zero_lock:
            self.zero_data[0] += fx
            self.zero_data[1] += fy

    def get_timing_stats(self):
        self._drain_process_status()
        with self._stats_lock:
            result = dict(self._stats)
            result["tx_intervals_s"] = list(self._tx_intervals)
            result["rx_intervals_s"] = list(self._rx_intervals)
            result["latencies_s"] = list(self._latencies)
            return result

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._use_process:
            self._stop_process(join_timeout=1.0)
            self._drain_process_status()
            self._close_ipc_queues()
            return
        if self._io_thread is not None and self._io_thread.is_alive():
            self._io_thread.join(timeout=1.0)
        if self.ser is not None and getattr(self.ser, "is_open", True):
            self.ser.close()


def _force_process_main(port, period_s, response_timeout_s, queue_size,
                        frame_queue, status_queue, startup_queue, stop_event):
    """spawn 子进程入口；子进程内部仅由一个本地 I/O 线程访问串口。"""
    sensor = None
    try:
        sensor = SixAxisForceSensor(
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
                    status_queue.put_nowait(("error", "六维力 I/O 线程意外退出"))
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
