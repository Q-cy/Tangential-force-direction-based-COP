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

from ..config import ForceConfig

_DEFAULT_CONFIG = ForceConfig()
DATA_BAUDRATE_FORCE = _DEFAULT_CONFIG.baudrate
FORCE_SENSOR_PORT = _DEFAULT_CONFIG.port
FORCE_TARGET_HZ = _DEFAULT_CONFIG.target_hz
FORCE_PERIOD_S = _DEFAULT_CONFIG.period_s
FORCE_RESPONSE_TIMEOUT_S = _DEFAULT_CONFIG.response_timeout_s
FORCE_FRAME_QUEUE_SIZE = _DEFAULT_CONFIG.frame_queue_size


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

    def __init__(self, serial_instance=None, period_s=None,
                 response_timeout_s=None, queue_size=None, readiness_waiter=None,
                 port=None, baudrate=None, _use_process=None,
                 _mp_context=None,
                 _process_factory=None, _startup_timeout_s=None,
                 _status_sink=None, config: ForceConfig | None = None):
        """创建六维力采集器，并启动线程或独立进程。

        Args:
            serial_instance: 可选串口兼容对象；传入后不打开真实串口，主要用于
                测试。生产路径应由独立进程打开串口。
            period_s: 请求轮询周期，单位为秒；必须大于 0。默认与压力阵列
                共用 200 Hz/5 ms 配置。
            response_timeout_s: 单轮等待完整 28 字节力帧的最长时间，单位为秒；
                必须大于 0。
            queue_size: 本地或进程间帧队列容量，单位为帧数，必须大于 0。
            readiness_waiter: 测试 fake 的可读等待回调；生产路径使用 select。
            port: 六维力串口路径；``None`` 使用 ``/dev/ttyUSB1``。
            baudrate: 串口波特率，默认 460800；协议默认值由配置集中管理。
            _use_process: 是否使用独立进程；``None`` 时按是否注入串口自动选择。
            _mp_context: 可注入的 multiprocessing 上下文。
            _process_factory: 可注入的进程工厂。
            _startup_timeout_s: 等待子进程打开串口并报告 ready 的秒数。
            _status_sink: 可选的统计/错误消息队列。

        Raises:
            ValueError: 周期、超时或队列容量不为正数。
            Exception: 串口打开、子进程创建或启动握手失败。

        Side Effects:
            初始化六轴零点为 6 个 0.0，可能打开串口并启动后台线程或 spawn
            子进程。子进程只传递未扣零点的物理量。
        """
        defaults = (config or ForceConfig()).validate()
        period_s = defaults.period_s if period_s is None else period_s
        response_timeout_s = (
            defaults.response_timeout_s
            if response_timeout_s is None else response_timeout_s
        )
        queue_size = defaults.frame_queue_size if queue_size is None else queue_size
        port = defaults.port if port is None else port
        baudrate = defaults.baudrate if baudrate is None else baudrate
        if _startup_timeout_s is None:
            _startup_timeout_s = defaults.startup_timeout_s
        if period_s <= 0 or response_timeout_s <= 0 or queue_size <= 0:
            raise ValueError("六维力采集周期、响应超时和队列长度必须大于 0")

        self.ser = None
        self.port = port
        self._baudrate = int(baudrate)
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
        """启动六维力独立采集进程并等待串口初始化完成。

        Args:
            queue_size: 进程间力帧队列容量，单位为帧数。

        Returns:
            None。

        Raises:
            RuntimeError: 子进程启动失败或未返回 ``ready``。
            Exception: 进程创建、启动或 IPC 等待失败；已创建资源会尝试清理。

        Side Effects:
            创建 IPC 队列/停止事件并启动 `_force_process_main`；子进程退出前
            父进程仍负责应用软件零点。
        """
        context = self._mp_context or multiprocessing.get_context("spawn")
        self._ipc_frame_queue = context.Queue(maxsize=queue_size)
        self._ipc_status_queue = context.Queue(maxsize=8)
        self._ipc_startup_queue = context.Queue(maxsize=1)
        self._mp_stop_event = context.Event()
        process_factory = self._process_factory or context.Process
        process_args = (
            self.port, self._period_s, self._response_timeout_s, queue_size,
            self._ipc_frame_queue, self._ipc_status_queue,
            self._ipc_startup_queue, self._mp_stop_event,
        )
        if self._baudrate != DATA_BAUDRATE_FORCE:
            process_args += (self._baudrate,)
        self._process = process_factory(target=_force_process_main, args=process_args)
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
        """请求停止并等待六维力采集子进程，超时后终止本进程。

        Args:
            join_timeout: 单次等待子进程结束的秒数。

        Returns:
            None。

        Side Effects:
            设置 multiprocessing 停止事件；必要时只终止本实例创建的子进程。
        """
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
        """关闭六维力帧、状态和启动握手 IPC 队列。

        Returns:
            None；方法可重复调用。

        Side Effects:
            取消队列 feeder 等待、关闭队列，并清空实例中的队列引用。
        """
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
        """合并子进程发送的六维力统计快照。

        Args:
            snapshot: dict，包含累计计数及可选的三个秒级时序样本列表。

        Returns:
            None。

        Side Effects:
            在线程锁下更新 ``_stats``、发送/接收间隔和响应延迟样本。
        """
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
        """非阻塞读取子进程统计/错误消息并更新父进程状态。

        Returns:
            None。无状态队列时不做任何操作。

        Side Effects:
            可能更新统计快照或设置 ``_error``；不会等待新消息。
        """
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
        """生成线程安全的六维力累计统计和时序样本副本。

        Returns:
            dict：计数以及最近最多 1000 个 ``tx_intervals_s``、
            ``rx_intervals_s``、``latencies_s``，时间单位均为秒。
        """
        with self._stats_lock:
            result = dict(self._stats)
            result["tx_intervals_s"] = list(self._tx_intervals)
            result["rx_intervals_s"] = list(self._rx_intervals)
            result["latencies_s"] = list(self._latencies)
            return result

    def _publish_stats(self, force=False):
        """非阻塞发布六维力统计，避免状态队列拖慢串口轮询。

        Args:
            force: 是否绕过默认 0.2 秒发布节流。

        Returns:
            None；无 sink、节流期间或队列已满时直接返回。

        Side Effects:
            可能向状态队列写入 ``("stats", snapshot)``。
        """
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
        """以六维力协议参数打开串口并清理启动输入残留。

        Returns:
            None；串口对象保存到 ``self.ser``。

        Raises:
            serial.SerialException: 端口打开或配置失败。

        Side Effects:
            以 460800 baud、非阻塞读写打开 ``self.port``，等待设备稳定后尝试
            清空输入缓冲；清空失败只增加 ``serial_flush_errors``。
        """
        self.ser = serial.Serial(
            self.port, getattr(self, "_baudrate", DATA_BAUDRATE_FORCE),
            timeout=0, write_timeout=0
        )
        time.sleep(0.1)
        try:
            self.ser.reset_input_buffer()
        except Exception:
            self._add_stat("serial_flush_errors")

    def _add_stat(self, name, amount=1):
        """线程安全地增加一个六维力统计计数。

        Args:
            name: 统计字段名；未知字段也会按当前实现创建。
            amount: 增量，默认为 1。

        Returns:
            None；对象尚未完成统计初始化时直接返回。
        """
        if not hasattr(self, "_stats_lock") or not hasattr(self, "_stats"):
            return
        with self._stats_lock:
            self._stats[name] = self._stats.get(name, 0) + amount

    def _record_tx(self, tx_t):
        """记录力传感器请求发送时间及相邻发送间隔。

        Args:
            tx_t: 单调时钟发送时间，单位为秒。

        Returns:
            None。

        Side Effects:
            更新 ``requests``、上一发送时间和最近时序样本。
        """
        with self._stats_lock:
            if self._last_tx_t is not None:
                self._tx_intervals.append(tx_t - self._last_tx_t)
            self._last_tx_t = tx_t
            self._stats["requests"] += 1

    def _record_frame(self, rx_t, latency_s):
        """记录一帧完整合法力数据的接收时刻和响应延迟。

        Args:
            rx_t: 28 字节帧尾校验通过后的接收时间，单位为单调时钟秒。
            latency_s: 从请求发送至完整帧确认的时间，单位为秒。

        Returns:
            None。

        Side Effects:
            更新 ``frames``、上一接收时间及最近接收间隔/延迟样本。
        """
        with self._stats_lock:
            if self._last_rx_t is not None:
                self._rx_intervals.append(rx_t - self._last_rx_t)
            self._last_rx_t = rx_t
            self._latencies.append(latency_s)
            self._stats["frames"] += 1

    def _queue_frame(self, frame):
        """将原始六维力帧放入有界本地队列。

        Args:
            frame: dict，含 ``request_seq``、``tx_t``、``rx_t``、``latency_s``
                和未扣零点的 6 元素 ``data``。

        Returns:
            None。

        Side Effects:
            队列满时删除最旧帧、增加 ``queue_drops``，再尝试写入当前帧。
        """
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
        """在线程锁下清空六维力持久化解析缓存。

        Returns:
            None。

        Side Effects:
            丢弃当前残片和未消费粘包字节；用于轮次隔离或超时恢复。
        """
        with self._rx_lock:
            self._rx_buf.clear()

    def _flush_input_output(self):
        """清空六维力解析缓存以及串口输入/输出缓冲区。

        Returns:
            None。

        Side Effects:
            调用两个串口 reset 方法；异常增加 ``serial_flush_errors``，不抛出。
        """
        self._clear_parser()
        for reset in (self.ser.reset_input_buffer, self.ser.reset_output_buffer):
            try:
                reset()
            except Exception:
                self._add_stat("serial_flush_errors")

    def _flush_output(self):
        """仅清空六维力串口输出缓冲。

        Returns:
            None。

        Side Effects:
            调用 ``reset_output_buffer``；异常只增加清空错误统计。
        """
        try:
            self.ser.reset_output_buffer()
        except Exception:
            self._add_stat("serial_flush_errors")

    def _read_chunk(self, timeout_s):
        """等待并读取当前可用的六维力串口字节。

        Args:
            timeout_s: 本轮剩余等待时间，单位为秒；单次等待最多 10 ms。

        Returns:
            bytes：最多 1024 字节；无数据或读取异常时返回 ``b""``。

        Side Effects:
            使用 select/os.read 或测试 waiter；读取异常增加
            ``serial_read_errors`` 并短暂等待，避免断线时空转。
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

    def _append_rx(self, chunk):
        """追加六维力串口片段并限制持久化缓存大小。

        Args:
            chunk: bytes-like 串口片段；空片段忽略。

        Returns:
            None。

        Side Effects:
            将字节追加到 ``_rx_buf``；超过 4096 字节时丢弃最早字节并更新
            framing 统计，以便后续继续搜索帧头。
        """
        if not chunk:
            return
        with self._rx_lock:
            self._rx_buf.extend(chunk)
            if len(self._rx_buf) > self.MAX_RX_BUF:
                drop = len(self._rx_buf) - self.MAX_RX_BUF
                del self._rx_buf[:drop]
                self._add_stat("framing_bytes", drop)

    def _try_pop_frame(self):
        """从持久缓存取出一个合法的 28 字节六维力帧。

        Returns:
            bytes：完整帧（头 ``49 AA``、26 字节主体、尾 ``0D 0A``）。
            ``None``：当前缓存没有完整合法帧。

        Side Effects:
            消费前导噪声、已确认帧和坏帧头/尾，并更新 framing、tail 和字节
            统计；支持分包、粘包、噪声和错误帧尾恢复。
        """
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
        """把一帧 28 字节协议数据解析为六轴物理量。

        Args:
            resp: 已通过 ``49 AA`` 帧头和 ``0D 0A`` 帧尾校验的 28 字节帧；
                字节 2..25 包含 6 个 little-endian IEEE-754 float32。

        Returns:
            list[float]：六个按协议顺序排列的物理量，原始 float 乘以 9.8
            后四舍五入到 2 位；此处不扣软件零点。

        Raises:
            struct.error: 输入帧长度不足或字段无法按 float32 解包。
        """
        values = [struct.unpack("<f", resp[offset:offset + 4])[0] * 9.8
                  for offset in range(2, 26, 4)]
        return [round(value, 2) for value in values]

    def _apply_zero(self, frame):
        """对父进程取出的力帧应用当前软件零点快照。

        Args:
            frame: dict，``data`` 为六元素未扣零点力/力矩，其他时间戳字段
                原样保留。

        Returns:
            dict：浅复制的帧，``data`` 替换为逐轴扣除零点并四舍五入到 2 位的值。

        Side Effects:
            在 ``_zero_lock`` 下读取零点，保证与归零更新不会并发读写冲突；
            不修改输入 frame。
        """
        with self._zero_lock:
            zero = tuple(self.zero_data)
        adjusted = [round(value - bias, 2)
                    for value, bias in zip(frame["data"], zero)]
        result = dict(frame)
        result["data"] = adjusted
        return result

    def _io_loop(self):
        """执行六维力单请求在途轮询和持久化帧解析。

        Returns:
            None；未处理异常写入 ``_error`` 并设置停止事件。

        Side Effects:
            每轮发送 ``49 AA 0D 0A`` 请求、接收/解析一个 28 字节帧、记录
            ``request_seq``/``tx_t``/``rx_t``/``latency_s``，并在合法帧入队前
            保留原始物理量。超时会清空输入和解析缓存；周期不足时等待剩余
            时间，超周期时增加 ``schedule_skips``。
        """
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
        """读取下一帧六维力数据并在父进程应用软件零点。

        Args:
            timeout_s: 等待秒数；``None`` 持续等待，0 为非阻塞读取。

        Returns:
            dict：包含 ``request_seq``、``tx_t``、``rx_t``、``latency_s`` 和
            六元素 ``data``；``data`` 已扣除调用时的零点快照。
            ``None``：超时、停止或采集已结束且没有帧。

        Raises:
            ValueError: ``timeout_s`` 小于 0。
            RuntimeError: 子进程退出或 I/O 线程报告异常。
        """
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
        """以非阻塞方式读取一帧六维力数据。

        Returns:
            dict | None：等价于 ``read_frame(timeout_s=0.0)``；成功时 ``data``
            已应用软件零点。

        Raises:
            RuntimeError: 底层采集线程或进程报告错误。
        """
        return self.read_frame(timeout_s=0.0)

    def calibrate_zero(self, sample_count: int = 10, timeout_s: float = 1.0) -> bool:
        """使用普通六维力数据帧计算并设置六轴软件零点。

        Args:
            sample_count: 需要收集的有效帧数；必须大于 0。
            timeout_s: 收集窗口，单位为秒；必须大于 0。

        Returns:
            bool：收集到足够帧并成功写入零点返回 ``True``；超时且样本不足
            返回 ``False``，此时保留原零点不变。

        Raises:
            ValueError: ``sample_count`` 或 ``timeout_s`` 不为正数。
            RuntimeError: 读取帧时底层 I/O 报错。

        Side Effects:
            只从本采集器帧队列消费普通数据帧，不发送额外置零命令；成功时在
            ``_zero_lock`` 下将 ``zero_data`` 更新为六轴样本均值。
        """
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
        """在当前软件零点上叠加 Fx/Fy 两个轴的偏置。

        Args:
            fx: Fx 轴附加零偏，物理量单位与 ``data[0]`` 相同。
            fy: Fy 轴附加零偏，物理量单位与 ``data[1]`` 相同。

        Returns:
            None。

        Side Effects:
            在线程锁下修改 ``zero_data[0]`` 和 ``zero_data[1]``；其余四轴不变。
        """
        with self._zero_lock:
            self.zero_data[0] += fx
            self.zero_data[1] += fy

    def get_timing_stats(self):
        """返回六维力请求/响应的累计统计快照。

        Returns:
            dict：累计请求、帧、超时、帧头/帧尾、串口、队列和调度计数，以及
            最近最多 1000 个以秒为单位的 ``tx_intervals_s``、
            ``rx_intervals_s``、``latencies_s``。

        Side Effects:
            非阻塞消费一次子进程状态队列。
        """
        self._drain_process_status()
        with self._stats_lock:
            result = dict(self._stats)
            result["tx_intervals_s"] = list(self._tx_intervals)
            result["rx_intervals_s"] = list(self._rx_intervals)
            result["latencies_s"] = list(self._latencies)
            return result

    def close(self):
        """幂等停止采集并释放六维力线程/进程、串口和 IPC 队列。

        Returns:
            None。

        Side Effects:
            设置停止事件并等待后台 I/O；进程模式停止子进程并关闭 IPC，线程
            模式等待线程后关闭串口。重复调用不重复操作资源。
        """
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
                        frame_queue, status_queue, startup_queue, stop_event,
                        baudrate=DATA_BAUDRATE_FORCE):
    """运行六维力 spawn 子进程并转发原始力帧、统计和错误。

    Args:
        port: 六维力串口路径。
        period_s: 轮询周期，单位为秒。
        response_timeout_s: 单轮响应超时，单位为秒。
        queue_size: 进程间帧队列容量。
        frame_queue: 父子进程共享的力帧队列。
        status_queue: 父子进程共享的统计/错误队列。
        startup_queue: 发送 ready 或启动错误的握手队列。
        stop_event: 父进程控制的停止事件。

    Returns:
        None；退出前尝试发布最终统计并关闭子进程内传感器。

    Side Effects:
        打开六维力串口并启动唯一的本地 I/O 线程；转发的 data 保持未扣父进程
        ``zero_data`` 的原始物理量。
    """
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
            baudrate=baudrate,
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
