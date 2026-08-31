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

from ..config import ArrayConfig, PressureConfig

_DEFAULT_CONFIG = PressureConfig()
DATA_BAUDRATE_PRESS = _DEFAULT_CONFIG.baudrate
PRESSURE_SENSOR_PORT = _DEFAULT_CONFIG.port
PRESSURE_TARGET_HZ = _DEFAULT_CONFIG.target_hz
PRESSURE_PERIOD_S = _DEFAULT_CONFIG.period_s
PRESSURE_RESPONSE_TIMEOUT_S = _DEFAULT_CONFIG.response_timeout_s
PRESSURE_FRAME_QUEUE_SIZE = _DEFAULT_CONFIG.frame_queue_size


class PressureSensor:
    """可配置行列数的 PZT 压力阵列串口采集器。

    设备使用 921600 baud 的请求—响应协议。每轮只发送一个 14 字节请求，
    最多等待 ``response_timeout_s`` 秒，并将合法响应转换为
    ``2 * rows * cols`` 字节的 ``rows * cols`` 通道 ``uint16`` 小端 payload。生产模式使用 ``spawn`` 子进程承载
    串口 I/O；注入串口实例时使用本地 I/O 线程，便于协议测试。

    ``read_frame`` 返回的 ``rx_t`` 和 ``latency_s`` 使用
    :func:`time.perf_counter` 的单调时钟；帧队列、统计快照和进程资源均由
    本类负责关闭。
    """
    CMD_BYTES = bytes([0x55, 0xAA, 0x09, 0x00, 0x34, 0x00,
                       0xFB, 0x00, 0x1C, 0x00, 0x00, 0xA8, 0x00, 0x35])
    # 以下常量保留为默认 12x7 设备的兼容参考；实例运行时使用由 rows/cols
    # 计算出的 expected_* 字段，不依赖这些固定默认值。
    FRAME_LEN = 183              # 默认响应: 4B头 + 178B payload + 1B CRC
    EXPECTED_SENSOR_BYTES = 168  # 默认 12x7 的传感器字节数
    MIN_PAYLOAD_LEN = 10
    MAX_PAYLOAD_LEN = 0xFFFF
    MAX_RX_BUF = 8192
    RX_BUF_RETAIN = 4096
    READ_CHUNK_SIZE = 1024

    def __init__(self, serial_instance=None, period_s=None,
                 response_timeout_s=None, queue_size=None,
                 readiness_waiter=None, port=None, baudrate=None,
                 _use_process=None,
                 _mp_context=None, _process_factory=None, _startup_timeout_s=None,
                 _frame_sink=None, _status_sink=None,
                 config: PressureConfig | None = None,
                 array_config: ArrayConfig | None = None):
        """创建压力采集器，并启动线程或独立采集进程。

        Args:
            serial_instance: 可选的串口兼容对象。传入后不打开真实串口，
                通常用于测试；应提供 ``write``、``read``/``fileno`` 和清空
                缓冲区方法。
            period_s: 每轮轮询的目标周期，单位为秒；必须大于 0。设备响应
                超过该周期时不补发请求，实际频率自然下降。
            response_timeout_s: 单轮等待完整合法压力帧的最长时间，单位为秒；
                必须大于 0，超时只计数并进入下一轮。
            queue_size: 本地或进程间压力帧队列容量，必须大于 0。
            readiness_waiter: 测试用的可读等待回调，参数/返回值分别为等待
                秒数和 bool；生产串口路径使用 ``select``，通常保持 ``None``。
            port: 压力串口路径；``None`` 使用 ``/dev/ttyUSB0``。
            baudrate: 串口波特率，默认 921600；协议默认值由配置集中管理。
            _use_process: 是否强制使用独立进程；``None`` 时由是否注入串口
                或 frame sink 自动决定。以下下划线参数仅供测试/进程封装使用。
            _mp_context: 可注入的 multiprocessing 上下文。
            _process_factory: 可注入的进程工厂。
            _startup_timeout_s: 等待采集子进程报告 ready 的秒数。
            _frame_sink: 可选的测试帧输出队列；有值时帧直接写入该队列。
            _status_sink: 可选的统计/错误输出队列。
            array_config: 整个项目共用的阵列布局；省略时使用默认
                ``ArrayConfig()``。该对象决定请求长度、应答长度和解码通道数。

        Raises:
            ValueError: ``period_s``、``response_timeout_s`` 或 ``queue_size``
                不为正数。
            Exception: 串口打开失败、采集进程启动失败或启动握手超时。

        Side Effects:
            可能立即打开串口并启动后台线程，或创建并启动一个 spawn 子进程。
        """
        defaults = (config or PressureConfig()).validate()
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
            raise ValueError("压力采集周期、响应超时和队列长度必须大于 0")
        array_config = ArrayConfig() if array_config is None else array_config
        if not isinstance(array_config, ArrayConfig):
            raise TypeError("PressureSensor.array_config 必须是 ArrayConfig")
        array_config.validate()
        sensor_bytes = array_config.sensor_bytes

        self.ser = None
        self.port = port
        self.array_config = array_config
        # 这两个字段只是公共布局对象的派生快捷访问，不是独立配置来源。
        self.rows, self.cols = self.array_config.shape
        self.channel_count = self.array_config.channel_count
        self.expected_sensor_bytes = sensor_bytes
        self.expected_payload_len = sensor_bytes + self.MIN_PAYLOAD_LEN
        self.expected_frame_len = 4 + self.expected_payload_len + 1
        # 至少容纳一帧加下一次批量读入的数据；大阵列时随协议长度增长，
        # 不使用只适合默认 12x7 的固定接收缓存。
        self._max_rx_buf = max(
            self.MAX_RX_BUF,
            self.expected_frame_len + self.READ_CHUNK_SIZE,
        )
        self._rx_buf_retain = max(self.RX_BUF_RETAIN, self.expected_frame_len)
        self.cmd_bytes = self.build_read_command(sensor_bytes)
        self._baudrate = int(baudrate)
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
        """启动独立采集进程，并同步等待子进程完成串口初始化。

        Args:
            queue_size: 进程间压力帧队列容量，单位为帧数。

        Returns:
            None。成功时保存进程和 IPC 队列句柄到实例状态。

        Raises:
            RuntimeError: 子进程无法打开压力串口或未报告 ``ready``。
            Exception: 进程创建、启动或启动队列等待失败；失败时会清理已建资源。

        Side Effects:
            创建 frame/status/startup 三个 multiprocessing 队列和停止事件，
            并启动一个仅由本实例管理的采集子进程。
        """
        context = self._mp_context or multiprocessing.get_context("spawn")
        self._ipc_frame_queue = context.Queue(maxsize=queue_size)
        # 统计不是数据通道，使用小的有界队列即可；子进程不会因统计拥塞而阻塞采集。
        self._ipc_status_queue = context.Queue(maxsize=8)
        self._ipc_startup_queue = context.Queue(maxsize=1)
        self._mp_stop_event = context.Event()
        process_factory = self._process_factory or context.Process
        process_args = (
            self.port, self._period_s, self._response_timeout_s, queue_size,
            self._ipc_frame_queue, self._ipc_status_queue,
            self._ipc_startup_queue, self._mp_stop_event,
            self._baudrate, self.rows, self.cols,
        )
        self._process = process_factory(target=_pressure_process_main, args=process_args)
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
        """请求停止并等待压力采集子进程，必要时终止它。

        Args:
            join_timeout: 每次 ``join`` 最多等待的秒数。

        Returns:
            None。

        Side Effects:
            设置进程停止事件；若进程在等待期仍存活，只终止本实例创建的进程。
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
            # 只终止由本 PressureSensor 创建的子进程。
            try:
                self._process.terminate()
                self._process.join(timeout=join_timeout)
            except Exception:
                pass

    def _close_ipc_queues(self):
        """关闭压力采集使用的所有 multiprocessing IPC 队列。

        Returns:
            None。方法幂等；关闭失败的单个队列不会阻止其他队列释放。

        Side Effects:
            取消 feeder 线程等待、关闭队列，并将实例中的队列引用置为 ``None``。
        """
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
        """非阻塞消费子进程统计和错误消息。

        Returns:
            None。统计消息合并到本实例快照，错误消息保存到 ``_error``。

        Side Effects:
            更新累计计数、最近时序样本或后台错误状态；不会阻塞采集进程。
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
                self._error = RuntimeError(f"压力串口子进程异常: {payload}")

    def _apply_stats(self, snapshot):
        """将子进程发送的统计快照合并到父进程状态。

        Args:
            snapshot: dict，包含累计统计以及可选的 ``tx_intervals_s``、
                ``rx_intervals_s``、``latencies_s`` 秒级样本列表。

        Returns:
            None。

        Side Effects:
            在 ``_stats_lock`` 保护下覆盖对应计数和最近最多 1000 个时序样本。
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

    def _stats_snapshot(self):
        """生成当前压力采集统计的线程安全浅拷贝。

        Returns:
            dict：累计计数字段，以及 ``tx_intervals_s``、``rx_intervals_s``、
            ``latencies_s`` 三个最近样本列表；时间单位均为秒。
        """
        with self._stats_lock:
            result = dict(self._stats)
            result["tx_intervals_s"] = list(self._tx_intervals)
            result["rx_intervals_s"] = list(self._rx_intervals)
            result["latencies_s"] = list(self._latencies)
            return result

    def _publish_stats(self, force=False):
        """通过状态队列非阻塞发布压力采集统计。

        Args:
            force: 是否忽略 0.2 秒的发布节流；关闭前通常传 ``True``。

        Returns:
            None。无状态队列、队列已满或节流期间均静默返回。

        Side Effects:
            可能向 ``_status_sink`` 放入 ``("stats", snapshot)``；统计队列拥塞
            不会阻塞压力轮询。
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
        """打开压力串口并清理启动时的输入残留。

        Returns:
            None；打开后的 ``serial.Serial`` 对象保存到 ``self.ser``。

        Raises:
            serial.SerialException: 端口不存在、权限不足或无法配置时由
                pyserial 抛出。

        Side Effects:
            以 921600 baud、非阻塞读写打开 ``self.port``，等待 0.1 秒后清空
            输入缓冲区。该方法不启动 I/O 线程。
        """
        self.ser = serial.Serial(
            self.port,
            getattr(self, "_baudrate", DATA_BAUDRATE_PRESS),
            timeout=0,
            write_timeout=0,
        )
        time.sleep(0.1)
        self.ser.reset_input_buffer()

    @staticmethod
    def crc8_itu(data: bytes) -> int:
        """计算压力协议使用的 CRC-8-ITU 校验值。

        Args:
            data: 待校验的帧头和 payload 字节，不包含末尾 CRC 字节。

        Returns:
            int：0 到 255 的 CRC 值。多项式为 ``0x07``，初始值为 ``0x00``，
            最终异或值为 ``0x55``。
        """
        crc = 0x00
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x07) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc ^ 0x55

    @classmethod
    def build_read_command(cls, sensor_bytes: int) -> bytes:
        """按协议生成读请求，并把读取字节数写入 data[11:13]。"""
        if not 0 < sensor_bytes <= 0xFFFF:
            raise ValueError("读取数据长度必须在 1..65535 字节内")
        if sensor_bytes + cls.MIN_PAYLOAD_LEN > cls.MAX_PAYLOAD_LEN:
            raise ValueError(
                "响应 payload 长度溢出：传感器字节数加协议元数据必须不超过 65535"
            )
        command = bytearray([
            0x55, 0xAA, 0x09, 0x00, 0x34, 0x00, 0xFB,
            0x00, 0x1C, 0x00, 0x00,
            sensor_bytes & 0xFF, (sensor_bytes >> 8) & 0xFF,
        ])
        command.append(cls.crc8_itu(command))
        return bytes(command)

    def _add_stat(self, name, amount=1):
        """在线程安全地增加一个压力统计计数。

        Args:
            name: ``_stats`` 中的计数字段名。
            amount: 增量，默认为 1。

        Returns:
            None。对象尚未完成初始化时直接返回。

        Raises:
            KeyError: ``name`` 不存在于内部统计字典时由字典访问抛出。
        """
        if not hasattr(self, "_stats_lock"):
            return
        with self._stats_lock:
            self._stats[name] += amount

    def _append_rx(self, chunk):
        """把新收到的串口字节追加到持久化解析缓存。

        Args:
            chunk: ``bytes`` 或 bytes-like 串口片段；空片段不产生变化。

        Returns:
            None。

        Side Effects:
            在 ``_rx_lock`` 下追加到 ``_rx_buf``；超过动态接收上限时丢弃前部，
            至少保留完整的当前阵列响应，并增加 ``framing_bytes`` 统计。
        """
        if not chunk:
            return
        with self._rx_lock:
            self._rx_buf.extend(chunk)
            max_rx_buf = getattr(self, "_max_rx_buf", self.MAX_RX_BUF)
            rx_buf_retain = getattr(self, "_rx_buf_retain", self.RX_BUF_RETAIN)
            if len(self._rx_buf) > max_rx_buf:
                drop_count = len(self._rx_buf) - rx_buf_retain
                del self._rx_buf[:drop_count]
                self._add_stat("framing_bytes", drop_count)

    def _record_tx(self, tx_t):
        """记录一次请求发送时间并更新请求间隔统计。

        Args:
            tx_t: ``time.perf_counter()`` 返回的发送时间，单位为秒。

        Returns:
            None。

        Side Effects:
            更新上一发送时间、``requests`` 计数，并追加最近发送间隔样本。
        """
        with self._stats_lock:
            if self._last_tx_t is not None:
                self._tx_intervals.append(tx_t - self._last_tx_t)
            self._last_tx_t = tx_t
            self._stats["requests"] += 1

    def _record_frame(self, rx_t, latency_s):
        """记录一个合法压力帧的接收时间、延迟和接收间隔。

        Args:
            rx_t: 完整帧通过协议校验后的接收时间，单调时钟秒数。
            latency_s: 从本轮发送到完整帧确定的延迟，单位为秒。

        Returns:
            None。

        Side Effects:
            更新 ``frames``，上一接收时间及最近 1000 个接收间隔/延迟样本。
        """
        with self._stats_lock:
            if self._last_rx_t is not None:
                self._rx_intervals.append(rx_t - self._last_rx_t)
            self._last_rx_t = rx_t
            self._latencies.append(latency_s)
            self._stats["frames"] += 1

    def _queue_frame(self, frame):
        """将合法压力帧放入本地或注入的有界队列。

        Args:
            frame: dict，至少包含 ``request_seq``、``tx_t``、``rx_t``、
                ``latency_s`` 和 ``2 * rows * cols`` 字节 ``payload``。

        Returns:
            None。

        Side Effects:
            队列未满时入队；本地队列满时丢弃最旧帧再入队并增加
            ``queue_drops``，注入 sink 满时丢弃当前帧并计数。
        """
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
        """清空本轮开始前的串口输入、输出和解析缓存。

        Returns:
            None。

        Side Effects:
            清空 ``_rx_buf``，调用串口的 ``reset_input_buffer`` 和
            ``reset_output_buffer``；清空异常只增加 ``serial_flush_errors``，
            不直接终止轮询。
        """
        with self._rx_lock:
            self._rx_buf.clear()
        for reset in (self.ser.reset_input_buffer, self.ser.reset_output_buffer):
            try:
                reset()
            except Exception:
                self._add_stat("serial_flush_errors")

    def _read_chunk(self, timeout_s):
        """等待并批量读取当前可用字节，语义与 C++ SerialPort::read 一致。

        Args:
            timeout_s: 本轮剩余等待时间，单位为秒；内部单次最多等待 10 ms。

        Returns:
            bytes：当前可读的最多 1024 字节；无数据或读取异常时返回 ``b""``。

        Side Effects:
            生产串口通过 ``select`` 和 ``os.read`` 读取；异常增加
            ``serial_read_errors`` 并等待 1 ms。``readiness_waiter`` 仅供无文件
            描述符的测试 fake 使用。

        Raises:
            无：串口读取异常被转换为空字节并计数。
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
        """执行 C++ 式逐轮压力轮询，直到停止事件或发生未处理异常。

        Returns:
            None；后台线程退出时将异常写入 ``_error``，并设置停止事件。

        Side Effects:
            每轮清空缓冲、发送一次 14 字节 ``CMD_BYTES``，解析一个合法响应，
            写入带请求/接收时间的帧队列，并更新请求、帧、CRC、超时和调度统计。
            一轮不足目标周期时等待剩余时间，超周期时增加 ``schedule_skips``。
        """
        try:
            while not self._stop_event.is_set():
                cycle_start = time.perf_counter()
                self._clear_cycle_buffers()
                tx_t = time.perf_counter()
                request_seq = self._request_seq
                self._request_seq += 1
                self._record_tx(tx_t)
                try:
                    written = self.ser.write(self.cmd_bytes)
                    if written != len(self.cmd_bytes):
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
                        "payload": payload,
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
        """从持久化接收缓存解析一个合法的动态长度压力 payload。

        Returns:
            bytes：去除 4 字节帧头、10 字节协议元数据后的
            ``2 * rows * cols`` 字节传感器数据。
            ``None``：当前缓存不足以组成完整帧，或只包含噪声/残片。

        Side Effects:
            在 ``_rx_lock`` 下消费前导噪声、坏长度、CRC 错误、错误状态帧和
            成功帧；分别更新 framing、length、CRC、status 统计。支持分包、粘包、
            前导噪声和错误帧恢复。合法帧要求头为 ``AA 55``、payload 长度在
            协议范围内、返回字节数等于 ``2 * rows * cols``、状态字节为 0。
        """
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
                returned_sensor_len = self._rx_buf[11] | (self._rx_buf[12] << 8)
                expected_sensor_bytes = getattr(
                    self, "expected_sensor_bytes", self.EXPECTED_SENSOR_BYTES
                )
                if (sensor_len != expected_sensor_bytes
                        or returned_sensor_len != expected_sensor_bytes):
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
        """读取下一条合法压力帧及其单调时钟时间戳。

        Args:
            timeout_s: 等待队列数据的秒数；``None`` 表示持续等待，0 表示
                非阻塞读取。

        Returns:
            dict：包含 ``request_seq``（请求序号）、``tx_t``（发送时刻）、
            ``rx_t``（完整帧解析完成时刻）、``latency_s``（秒）和 ``payload``
            （长度为 ``2 * rows * cols`` 的传感器 payload）。
            ``None``：在超时、停止事件已设置或采集已结束时没有可用帧。

        Raises:
            ValueError: ``timeout_s`` 小于 0。
            RuntimeError: 子进程退出或 I/O 线程报告异常。
        """
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
        """返回压力请求/响应的线程安全统计快照。

        Returns:
            dict：累计 ``requests``、``frames``、超时、CRC/长度/状态、串口、
            队列和调度计数；另含最近最多 1000 个以秒为单位的
            ``tx_intervals_s``、``rx_intervals_s`` 和 ``latencies_s`` 列表。

        Side Effects:
            先非阻塞消费一次子进程状态队列。
        """
        self._drain_process_status()
        with self._stats_lock:
            result = dict(self._stats)
            result["tx_intervals_s"] = list(self._tx_intervals)
            result["rx_intervals_s"] = list(self._rx_intervals)
            result["latencies_s"] = list(self._latencies)
            return result

    def decode(self, payload):
        """将动态长度原始 payload 解码为 ``rows * cols`` 个 ADC 整数。

        Args:
            payload: 恰好 ``2 * rows * cols`` 字节的传感器 payload；按
                little-endian ``uint16`` 解析，顺序保持设备原始线序。

        Returns:
            list[int]：长度为 ``rows * cols`` 的 ADC 值，未进行翻转、标定或归一化。

        Raises:
            struct.error: 输入长度不足以完整解码时由 ``struct.unpack`` 抛出。
        """
        expected_sensor_bytes = getattr(
            self, "expected_sensor_bytes", self.EXPECTED_SENSOR_BYTES
        )
        channel_count = getattr(
            self, "channel_count", expected_sensor_bytes // 2
        )
        if len(payload) != expected_sensor_bytes:
            raise ValueError(
                f"压力 payload 必须为 {expected_sensor_bytes} 字节，"
                f"实际为 {len(payload)} 字节"
            )
        return list(struct.unpack(f"<{channel_count}H", payload))

    def close(self):
        """幂等停止压力采集并释放线程、进程、串口和 IPC 队列。

        Returns:
            None。

        Side Effects:
            设置停止事件，等待后台 I/O；进程模式停止子进程并关闭 IPC 队列，
            线程模式等待线程后关闭串口。重复调用不会重复释放资源。
        """
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
                           stop_event, baudrate, rows, cols):
    """运行压力采集 spawn 子进程，并转发帧、统计和错误消息。

    Args:
        port: 压力串口路径。
        period_s: 轮询周期，单位为秒。
        response_timeout_s: 单轮响应超时，单位为秒。
        queue_size: 进程间帧队列容量，单位为帧数。
        frame_queue: 父子进程共享的压力帧队列。
        status_queue: 父子进程共享的统计/错误队列。
        startup_queue: 用于发送 ``("ready", None)`` 或启动错误的队列。
        stop_event: 父进程控制的 multiprocessing 停止事件。
        baudrate: 已校验的串口波特率。
        rows: 已校验的阵列行数，由父进程的 ``ArrayConfig`` 传入。
        cols: 已校验的阵列列数，由父进程的 ``ArrayConfig`` 传入。

    Returns:
        None。子进程退出前总会尝试发布最终统计并关闭本地传感器。

    Side Effects:
        打开压力串口、启动本地 I/O 线程，并将合法帧和统计转发到父进程。
    """
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
            baudrate=baudrate,
            array_config=ArrayConfig(rows=rows, cols=cols),
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
