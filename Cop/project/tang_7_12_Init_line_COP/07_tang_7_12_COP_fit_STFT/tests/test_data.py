import struct
import statistics
import threading
import time
import unittest
from unittest import mock

import tangential.sensors.pressure as data_module
from tangential.acquisition.buffer import TimestampedBuffer, match_closest
from tangential.sensors import force as force_module
from tangential.sensors import pressure as pressure_module

from tangential.sensors.force import (
    FORCE_FRAME_QUEUE_SIZE,
    FORCE_PERIOD_S,
    FORCE_RESPONSE_TIMEOUT_S,
    FORCE_TARGET_HZ,
    FORCE_SENSOR_PORT,
    SixAxisForceSensor,
)
from tangential.sensors.pressure import (
    PRESSURE_PERIOD_S,
    PRESSURE_RESPONSE_TIMEOUT_S,
    PRESSURE_TARGET_HZ,
    PRESSURE_FRAME_QUEUE_SIZE,
    PressureSensor,
)


def make_pressure_frame(values):
    values = list(values)
    payload = b"".join(struct.pack("<H", value) for value in values)
    sensor_bytes = len(payload)
    payload_len = PressureSensor.MIN_PAYLOAD_LEN + sensor_bytes
    frame = bytearray(4 + payload_len + 1)
    frame[:2] = b"\xaa\x55"
    frame[2:4] = struct.pack("<H", payload_len)
    # 正式应答协议：data[11:13] 是本帧返回的传感器字节数 N，不能省略。
    frame[11:13] = struct.pack("<H", sensor_bytes)
    frame[13] = 0
    frame[14:14 + sensor_bytes] = payload
    frame[-1] = PressureSensor.crc8_itu(frame[:-1])
    return bytes(frame)


def make_force_frame(values=(1, 2, 3, 4, 5, 6)):
    return (
        b"\x49\xaa"
        + b"".join(struct.pack("<f", value) for value in values)
        + b"\x0d\x0a"
    )


def pressure_parser():
    sensor = PressureSensor.__new__(PressureSensor)
    sensor._rx_buf = bytearray()
    sensor._rx_lock = threading.Lock()
    sensor._stats_lock = threading.Lock()
    sensor._stats = {
        "crc_errors": 0,
        "length_errors": 0,
        "status_errors": 0,
        "framing_bytes": 0,
    }
    sensor.expected_sensor_bytes = PressureSensor.EXPECTED_SENSOR_BYTES
    sensor.channel_count = sensor.expected_sensor_bytes // 2
    return sensor


def force_parser():
    sensor = SixAxisForceSensor.__new__(SixAxisForceSensor)
    sensor._rx_buf = bytearray()
    sensor._rx_lock = threading.Lock()
    sensor._io_lock = threading.Lock()
    sensor._zero_lock = threading.Lock()
    sensor.zero_data = [0.0] * 6
    return sensor


class TimedPressureSerial:
    """在 write 后延迟返回一帧，用于验证单请求在途和调度行为。"""

    def __init__(self, response_delay_s=0.003, respond=True):
        self.response_delay_s = response_delay_s
        self.respond = respond
        self.is_open = True
        self.write_times = []
        self.overlapping_writes = 0
        self.input_resets = 0
        self.output_resets = 0
        self._pending = None
        self._ready_at = None
        self._outstanding = False
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            if self._outstanding:
                self.overlapping_writes += 1
            now = time.perf_counter()
            self.write_times.append(now)
            self._outstanding = True
            if self.respond:
                self._pending = make_pressure_frame(range(84))
                self._ready_at = now + self.response_delay_s
            return len(data)

    @property
    def in_waiting(self):
        with self._lock:
            if (
                self._pending is not None
                and time.perf_counter() >= self._ready_at
            ):
                return len(self._pending)
            return 0

    def read(self, size):
        with self._lock:
            if self._pending is None or time.perf_counter() < self._ready_at:
                return b""
            chunk = self._pending[:size]
            self._pending = self._pending[size:]
            if not self._pending:
                self._pending = None
                self._ready_at = None
                self._outstanding = False
            return chunk

    def wait_readable(self, timeout_s):
        """测试适配器：模拟 select 等待，不参与生产串口路径。"""
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                if (
                    self._pending is not None
                    and time.perf_counter() >= self._ready_at
                ):
                    return True
            time.sleep(min(0.0001, max(0.0, deadline - time.perf_counter())))
        with self._lock:
            return (
                self._pending is not None
                and time.perf_counter() >= self._ready_at
            )

    def reset_input_buffer(self):
        with self._lock:
            self.input_resets += 1
            self._pending = None
            self._ready_at = None
            self._outstanding = False

    def reset_output_buffer(self):
        with self._lock:
            self.output_resets += 1

    def close(self):
        self.is_open = False


class TimedForceSerial:
    """在力请求后延迟返回28B普通帧。"""

    def __init__(self, response_delay_s=0.003, respond=True):
        self.response_delay_s = response_delay_s
        self.respond = respond
        self.is_open = True
        self.write_times = []
        self.overlapping_writes = 0
        self.input_resets = 0
        self.output_resets = 0
        self._pending = b""
        self._ready_at = None
        self._outstanding = False
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            if self._outstanding:
                self.overlapping_writes += 1
            now = time.perf_counter()
            self.write_times.append(now)
            self._outstanding = True
            if self.respond:
                self._pending = make_force_frame()
                self._ready_at = now + self.response_delay_s
            return len(data)

    def wait_readable(self, timeout_s):
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            with self._lock:
                if self._pending and time.perf_counter() >= self._ready_at:
                    return True
            time.sleep(min(0.0001, max(0.0, deadline - time.perf_counter())))
        with self._lock:
            return bool(
                self._pending and time.perf_counter() >= self._ready_at
            )

    def read(self, size):
        with self._lock:
            if not self._pending or time.perf_counter() < self._ready_at:
                return b""
            chunk = self._pending[:size]
            self._pending = self._pending[size:]
            if not self._pending:
                self._ready_at = None
                self._outstanding = False
            return chunk

    def reset_input_buffer(self):
        with self._lock:
            self.input_resets += 1
            self._pending = b""
            self._ready_at = None
            self._outstanding = False

    def reset_output_buffer(self):
        self.output_resets += 1

    def close(self):
        self.is_open = False


class ThreadBackedProcess:
    """在单元测试中模拟 multiprocessing.Process 的生命周期。"""

    def __init__(self, target, args):
        self.target = target
        self.args = args
        self.daemon = False
        self.terminated = False
        self._thread = None

    def start(self):
        self._thread = threading.Thread(
            target=self.target, args=self.args, daemon=True
        )
        self._thread.start()

    def join(self, timeout=None):
        if self._thread is not None:
            self._thread.join(timeout)

    def is_alive(self):
        return self._thread is not None and self._thread.is_alive()

    def terminate(self):
        self.terminated = True


class FakeMultiprocessingContext:
    def __init__(self, real_context):
        self.real_context = real_context
        self.process = None

    def Queue(self, maxsize=0):
        return self.real_context.Queue(maxsize=maxsize)

    def Event(self):
        return self.real_context.Event()

    def Process(self, target, args):
        self.process = ThreadBackedProcess(target, args)
        return self.process


class ProtocolTests(unittest.TestCase):
    def test_pressure_fragment_and_sticky_frames(self):
        sensor = pressure_parser()
        frame1 = make_pressure_frame(range(84))
        frame2 = make_pressure_frame(range(84, 168))

        sensor._rx_buf.extend(frame1[:100])
        self.assertIsNone(sensor.read_data())
        sensor._rx_buf.extend(frame1[100:] + frame2)

        self.assertEqual(sensor.decode(sensor.read_data()), list(range(84)))
        self.assertEqual(sensor.decode(sensor.read_data()), list(range(84, 168)))
        self.assertEqual(len(sensor._rx_buf), 0)

    def test_pressure_noise_and_bad_crc_recover_to_next_frame(self):
        sensor = pressure_parser()
        bad = bytearray(make_pressure_frame(range(84)))
        bad[-1] ^= 0xFF
        good = make_pressure_frame(range(84))
        sensor._rx_buf.extend(b"noise" + bad + good)

        self.assertEqual(sensor.decode(sensor.read_data()), list(range(84)))
        self.assertEqual(len(sensor._rx_buf), 0)
        self.assertEqual(sensor._stats["crc_errors"], 1)
        self.assertGreaterEqual(sensor._stats["framing_bytes"], len(b"noise"))

    def test_pressure_length_status_and_resync(self):
        sensor = pressure_parser()
        bad_length = bytearray(make_pressure_frame(range(84)))
        bad_length[2:4] = struct.pack("<H", 9)
        bad_status = bytearray(make_pressure_frame(range(84)))
        bad_status[13] = 1
        bad_status[-1] = PressureSensor.crc8_itu(bad_status[:-1])
        good = make_pressure_frame(range(84))
        sensor._rx_buf.extend(bad_length + bad_status + good)

        self.assertEqual(sensor.decode(sensor.read_data()), list(range(84)))
        self.assertEqual(sensor._stats["length_errors"], 1)
        self.assertEqual(sensor._stats["status_errors"], 1)

    def test_force_fragment_bad_tail_and_sticky_frame(self):
        sensor = force_parser()
        good = (
            b"\x49\xaa"
            + b"".join(struct.pack("<f", value) for value in [1, 2, 3, 4, 5, 6])
            + b"\x0d\x0a"
        )
        bad = good[:-2] + b"\x00\x00"
        sensor._rx_buf.extend(bad[:12])
        self.assertIsNone(sensor._try_pop_frame())
        sensor._rx_buf.extend(bad[12:] + good)

        parsed = sensor._parse_frame(sensor._try_pop_frame())
        self.assertEqual(parsed, [9.8, 19.6, 29.4, 39.2, 49.0, 58.8])

    def test_force_zero_calibration_success_and_timeout(self):
        sensor = force_parser()
        readings = iter([[float(i)] * 6 for i in range(1, 11)])
        def next_frame(timeout_s=0.1):
            values = next(readings, None)
            return None if values is None else {"data": values}
        sensor.read_frame = next_frame
        self.assertTrue(sensor.calibrate_zero(sample_count=10, timeout_s=0.1))
        self.assertEqual(sensor.zero_data, [5.5] * 6)

        failed = force_parser()
        failed.read_frame = lambda timeout_s=0.1: None
        self.assertFalse(failed.calibrate_zero(sample_count=2, timeout_s=0.003))
        self.assertEqual(failed.zero_data, [0.0] * 6)


class PressureTimingTests(unittest.TestCase):
    @staticmethod
    def make_sensor(serial_port, **kwargs):
        return PressureSensor(
            serial_instance=serial_port,
            readiness_waiter=serial_port.wait_readable,
            **kwargs,
        )

    def test_fake_serial_keeps_thread_backend(self):
        serial_port = TimedPressureSerial(response_delay_s=0.001)
        with mock.patch.object(data_module.multiprocessing, "get_context") as get_context:
            sensor = self.make_sensor(serial_port)
            try:
                frame = sensor.read_frame(timeout_s=0.1)
                self.assertIsNotNone(frame)
                self.assertFalse(sensor._use_process)
                self.assertTrue(sensor._io_thread.is_alive())
            finally:
                sensor.close()
        get_context.assert_not_called()

    def test_production_backend_uses_spawn_and_forwards_frame_stats(self):
        real_context = data_module.multiprocessing.get_context("spawn")
        fake_context = FakeMultiprocessingContext(real_context)
        expected_frame = {
            "request_seq": 7,
            "tx_t": 10.0,
            "rx_t": 10.006,
            "latency_s": 0.006,
            "payload": b"raw",
        }

        def fake_process_entry(
            port, period_s, timeout_s, queue_size,
            frame_queue, status_queue, startup_queue, stop_event,
            baudrate, rows, cols,
        ):
            self.assertEqual(port, data_module.PRESSURE_SENSOR_PORT)
            self.assertEqual(period_s, data_module.PRESSURE_PERIOD_S)
            self.assertEqual(timeout_s, data_module.PRESSURE_RESPONSE_TIMEOUT_S)
            self.assertEqual(queue_size, data_module.PRESSURE_FRAME_QUEUE_SIZE)
            self.assertEqual(baudrate, data_module.DATA_BAUDRATE_PRESS)
            self.assertEqual((rows, cols), (12, 7))
            startup_queue.put(("ready", None))
            frame_queue.put(expected_frame)
            status_queue.put(("stats", {
                "frames": 1,
                "queue_drops": 3,
                "tx_intervals_s": [0.005],
                "rx_intervals_s": [0.006],
                "latencies_s": [0.006],
            }))
            stop_event.wait()

        with mock.patch.object(
            pressure_module.multiprocessing,
            "get_context",
            return_value=fake_context,
        ) as get_context, mock.patch.object(
            pressure_module,
            "_pressure_process_main",
            side_effect=fake_process_entry,
        ):
            sensor = PressureSensor(_startup_timeout_s=0.5)
            try:
                frame = sensor.read_frame(timeout_s=0.5)
                self.assertEqual(frame, expected_frame)
                deadline = time.monotonic() + 0.5
                stats = sensor.get_timing_stats()
                while stats["frames"] != 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                    stats = sensor.get_timing_stats()
                self.assertEqual(stats["frames"], 1)
                self.assertEqual(stats["queue_drops"], 3)
                self.assertEqual(stats["rx_intervals_s"], [0.006])
            finally:
                sensor.close()

        get_context.assert_called_once_with("spawn")
        self.assertIsNotNone(fake_context.process)
        self.assertFalse(fake_context.process.terminated)

    def test_production_startup_error_is_synchronous(self):
        real_context = data_module.multiprocessing.get_context("spawn")
        fake_context = FakeMultiprocessingContext(real_context)

        def failing_process_entry(*args):
            args[6].put(("error", "cannot open pressure port"))

        with mock.patch.object(
            pressure_module.multiprocessing,
            "get_context",
            return_value=fake_context,
        ), mock.patch.object(
            pressure_module,
            "_pressure_process_main",
            side_effect=failing_process_entry,
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot open pressure port"):
                PressureSensor(_startup_timeout_s=0.5)

    def test_production_runtime_error_is_raised_by_read_frame(self):
        real_context = data_module.multiprocessing.get_context("spawn")
        fake_context = FakeMultiprocessingContext(real_context)

        def failing_process_entry(*args):
            args[6].put(("ready", None))
            args[5].put(("error", "poll loop stopped"))
            args[7].wait()

        with mock.patch.object(
            pressure_module.multiprocessing,
            "get_context",
            return_value=fake_context,
        ), mock.patch.object(
            pressure_module,
            "_pressure_process_main",
            side_effect=failing_process_entry,
        ):
            sensor = PressureSensor(_startup_timeout_s=0.5)
            try:
                with self.assertRaisesRegex(RuntimeError, "poll loop stopped"):
                    sensor.read_frame(timeout_s=0.5)
            finally:
                sensor.close()

    def test_open_port_uses_nonblocking_serial(self):
        fake = mock.Mock()
        fake.is_open = True
        with mock.patch.object(data_module.serial, "Serial", return_value=fake) as serial_ctor:
            with mock.patch.object(data_module.time, "sleep"):
                sensor = PressureSensor.__new__(PressureSensor)
                sensor.port = "/dev/test"
                sensor.open_port()

        self.assertEqual(serial_ctor.call_args.kwargs["timeout"], 0)
        self.assertEqual(serial_ctor.call_args.kwargs["write_timeout"], 0)

    def test_select_timeout_does_not_read(self):
        sensor = PressureSensor.__new__(PressureSensor)
        sensor.ser = mock.Mock()
        sensor._readiness_waiter = None
        with mock.patch.object(data_module.select, "select", return_value=([], [], [])) as wait:
            with mock.patch.object(data_module.os, "read") as read:
                self.assertEqual(sensor._read_chunk(0.050), b"")

        wait.assert_called_once_with([sensor.ser.fileno()], [], [], 0.010)
        read.assert_not_called()

    def test_select_read_batches_up_to_1024_bytes(self):
        sensor = PressureSensor.__new__(PressureSensor)
        sensor.ser = mock.Mock()
        sensor._readiness_waiter = None
        sensor.ser.fileno.return_value = 17
        expected = b"x" * 1024
        with mock.patch.object(data_module.select, "select", return_value=([17], [], [])) as wait:
            with mock.patch.object(data_module.os, "read", return_value=expected) as read:
                self.assertEqual(sensor._read_chunk(0.050), expected)

        wait.assert_called_once_with([17], [], [], 0.010)
        read.assert_called_once_with(17, PressureSensor.READ_CHUNK_SIZE)

    def test_select_errors_are_counted_and_recover(self):
        sensor = PressureSensor.__new__(PressureSensor)
        sensor.ser = mock.Mock()
        sensor._readiness_waiter = None
        sensor._stop_event = threading.Event()
        sensor._stats_lock = threading.Lock()
        sensor._stats = {"serial_read_errors": 0}
        sensor.ser.fileno.return_value = 17
        with mock.patch.object(
            data_module.select,
            "select",
            side_effect=[OSError("temporary select error"), ([17], [], [])],
        ):
            with mock.patch.object(data_module.os, "read", return_value=b"ok"):
                self.assertEqual(sensor._read_chunk(0.010), b"")
                self.assertEqual(sensor._read_chunk(0.010), b"ok")

        self.assertEqual(sensor._stats["serial_read_errors"], 1)

    def test_default_pressure_rate_matches_cpp(self):
        self.assertEqual(PRESSURE_TARGET_HZ, 200)
        self.assertEqual(PRESSURE_PERIOD_S, 0.005)
        self.assertEqual(PRESSURE_RESPONSE_TIMEOUT_S, 0.050)
        self.assertEqual(PRESSURE_FRAME_QUEUE_SIZE, 256)

    def test_200hz_single_inflight_flush_and_receive_timestamp(self):
        serial_port = TimedPressureSerial(response_delay_s=0.003)
        sensor = self.make_sensor(
            serial_port,
            period_s=0.005,
            response_timeout_s=0.050,
        )
        try:
            frames = [sensor.read_frame(timeout_s=0.2) for _ in range(6)]
            first_rx_t = frames[0]["rx_t"]
            time.sleep(0.02)  # 消费延迟不能改变已记录的到帧时间
            self.assertEqual(frames[0]["rx_t"], first_rx_t)
            self.assertEqual(
                [frame["request_seq"] for frame in frames], list(range(6))
            )
            self.assertTrue(all(frame["latency_s"] >= 0.003 for frame in frames))
        finally:
            sensor.close()

        intervals = [
            b - a for a, b in zip(serial_port.write_times, serial_port.write_times[1:])
        ]
        self.assertEqual(serial_port.overlapping_writes, 0)
        self.assertGreaterEqual(statistics.median(intervals), 0.0045)
        self.assertLess(statistics.median(intervals), 0.008)
        self.assertGreaterEqual(serial_port.input_resets, len(frames))
        self.assertGreaterEqual(serial_port.output_resets, len(frames))

    def test_slow_response_skips_period_without_burst(self):
        for response_delay, minimum_interval in ((0.008, 0.007), (0.015, 0.014)):
            with self.subTest(response_delay=response_delay):
                serial_port = TimedPressureSerial(response_delay_s=response_delay)
                sensor = self.make_sensor(
                    serial_port,
                    period_s=0.005,
                    response_timeout_s=0.050,
                )
                try:
                    frames = [sensor.read_frame(timeout_s=0.2) for _ in range(4)]
                    stats = sensor.get_timing_stats()
                finally:
                    sensor.close()

                self.assertTrue(all(frame is not None for frame in frames))
                intervals = [
                    b - a
                    for a, b in zip(
                        serial_port.write_times, serial_port.write_times[1:]
                    )
                ]
                self.assertEqual(serial_port.overlapping_writes, 0)
                self.assertTrue(
                    all(interval >= minimum_interval for interval in intervals)
                )
                self.assertGreaterEqual(stats["schedule_skips"], 1)

    def test_timeout_clears_request_and_recovers_schedule(self):
        serial_port = TimedPressureSerial(respond=False)
        sensor = self.make_sensor(
            serial_port,
            period_s=0.010,
            response_timeout_s=0.020,
        )
        try:
            time.sleep(0.075)
            stats = sensor.get_timing_stats()
        finally:
            sensor.close()

        self.assertGreaterEqual(stats["response_timeouts"], 2)
        self.assertEqual(serial_port.overlapping_writes, 0)
        intervals = [
            b - a for a, b in zip(serial_port.write_times, serial_port.write_times[1:])
        ]
        self.assertTrue(all(interval >= 0.018 for interval in intervals))

    def test_50ms_response_deadline_recovers_on_next_cycle(self):
        class RecoverAfterTimeoutSerial(TimedPressureSerial):
            def __init__(self):
                super().__init__(response_delay_s=0.001)
                self.request_index = 0

            def write(self, data):
                if self.request_index == 0:
                    self.request_index += 1
                    with self._lock:
                        self.write_times.append(time.perf_counter())
                        self._outstanding = True
                    return len(data)
                self.request_index += 1
                return super().write(data)

        serial_port = RecoverAfterTimeoutSerial()
        sensor = self.make_sensor(
            serial_port,
            period_s=0.005,
            response_timeout_s=0.050,
        )
        try:
            frame = sensor.read_frame(timeout_s=0.2)
            stats = sensor.get_timing_stats()
        finally:
            sensor.close()

        self.assertIsNotNone(frame)
        self.assertGreaterEqual(stats["response_timeouts"], 1)
        self.assertGreaterEqual(serial_port.write_times[1] - serial_port.write_times[0], 0.045)
        self.assertEqual(serial_port.overlapping_writes, 0)

    def test_queue_overflow_drops_old_frame_and_continues(self):
        sensor = PressureSensor.__new__(PressureSensor)
        sensor._frame_queue = data_module.queue.Queue(maxsize=1)
        sensor._stats_lock = threading.Lock()
        sensor._stats = {"queue_drops": 0}
        first = {"request_seq": 0}
        second = {"request_seq": 1}

        sensor._queue_frame(first)
        sensor._queue_frame(second)

        self.assertEqual(sensor._frame_queue.get_nowait(), second)
        self.assertEqual(sensor._stats["queue_drops"], 1)

    def test_serial_write_errors_are_counted_and_recover(self):
        class RecoveringSerial(TimedPressureSerial):
            def __init__(self):
                super().__init__(response_delay_s=0.001)
                self.failed_writes = 2

            def write(self, data):
                if self.failed_writes > 0:
                    self.failed_writes -= 1
                    raise OSError("temporary disconnect")
                return super().write(data)

        serial_port = RecoveringSerial()
        sensor = self.make_sensor(
            serial_port,
            period_s=0.005,
            response_timeout_s=0.005,
        )
        try:
            frame = sensor.read_frame(timeout_s=0.1)
            stats = sensor.get_timing_stats()
        finally:
            sensor.close()

        self.assertIsNotNone(frame)
        self.assertGreaterEqual(stats["serial_write_errors"], 2)

    def test_read_errors_are_counted_and_recover(self):
        class RecoveringReadSerial(TimedPressureSerial):
            def __init__(self):
                super().__init__(response_delay_s=0.001)
                self.failed_reads = 3

            def read(self, size):
                if self.failed_reads > 0:
                    self.failed_reads -= 1
                    raise OSError("temporary read error")
                return super().read(size)

        serial_port = RecoveringReadSerial()
        sensor = self.make_sensor(
            serial_port,
            period_s=0.005,
            response_timeout_s=0.050,
        )
        try:
            frame = sensor.read_frame(timeout_s=0.1)
            stats = sensor.get_timing_stats()
        finally:
            sensor.close()

        self.assertIsNotNone(frame)
        self.assertGreaterEqual(stats["serial_read_errors"], 3)

    def test_flush_errors_are_counted_and_recover(self):
        class RecoveringFlushSerial(TimedPressureSerial):
            def __init__(self):
                super().__init__(response_delay_s=0.001)
                self.failed_flushes = 2

            def reset_input_buffer(self):
                if self.failed_flushes > 0:
                    self.failed_flushes -= 1
                    raise OSError("temporary flush error")
                super().reset_input_buffer()

        serial_port = RecoveringFlushSerial()
        sensor = self.make_sensor(
            serial_port,
            period_s=0.005,
            response_timeout_s=0.050,
        )
        try:
            frame = sensor.read_frame(timeout_s=0.1)
            stats = sensor.get_timing_stats()
        finally:
            sensor.close()

        self.assertIsNotNone(frame)
        self.assertGreaterEqual(stats["serial_flush_errors"], 1)

    def test_partial_frame_is_discarded_between_poll_cycles(self):
        class PartialThenGoodSerial(TimedPressureSerial):
            def __init__(self):
                super().__init__(response_delay_s=0.001)
                self.response_index = 0

            def write(self, data):
                with self._lock:
                    if self._outstanding:
                        self.overlapping_writes += 1
                    now = time.perf_counter()
                    self.write_times.append(now)
                    self._outstanding = True
                    frame = make_pressure_frame(range(84))
                    self._pending = frame[:100] if self.response_index == 0 else frame
                    self.response_index += 1
                    self._ready_at = now + self.response_delay_s
                    return len(data)

        serial_port = PartialThenGoodSerial()
        sensor = self.make_sensor(
            serial_port,
            period_s=0.005,
            response_timeout_s=0.006,
        )
        try:
            frame = sensor.read_frame(timeout_s=0.1)
            stats = sensor.get_timing_stats()
        finally:
            sensor.close()

        self.assertIsNotNone(frame)
        self.assertEqual(frame["request_seq"], 1)
        self.assertGreaterEqual(stats["response_timeouts"], 1)
        self.assertGreaterEqual(serial_port.input_resets, 2)


class ForceTimingTests(unittest.TestCase):
    @staticmethod
    def make_sensor(serial_port, **kwargs):
        return SixAxisForceSensor(
            serial_instance=serial_port,
            readiness_waiter=serial_port.wait_readable,
            **kwargs,
        )

    def test_default_force_rate_matches_pressure(self):
        self.assertEqual(FORCE_TARGET_HZ, PRESSURE_TARGET_HZ)
        self.assertEqual(FORCE_PERIOD_S, PRESSURE_PERIOD_S)
        self.assertEqual(FORCE_RESPONSE_TIMEOUT_S, 0.050)
        self.assertEqual(FORCE_FRAME_QUEUE_SIZE, 256)

    def test_single_inflight_uses_complete_frame_receive_timestamp(self):
        serial_port = TimedForceSerial(response_delay_s=0.003)
        sensor = self.make_sensor(serial_port)
        try:
            frames = [sensor.read_frame(timeout_s=0.2) for _ in range(6)]
            first_rx_t = frames[0]["rx_t"]
            time.sleep(0.02)
            self.assertEqual(frames[0]["rx_t"], first_rx_t)
            self.assertEqual(
                [frame["request_seq"] for frame in frames], list(range(6))
            )
            self.assertTrue(all(frame["latency_s"] >= 0.003 for frame in frames))
        finally:
            sensor.close()

        intervals = [
            b - a for a, b in zip(
                serial_port.write_times, serial_port.write_times[1:]
            )
        ]
        self.assertEqual(serial_port.overlapping_writes, 0)
        self.assertGreaterEqual(statistics.median(intervals), 0.0045)
        self.assertLess(statistics.median(intervals), 0.008)

    def test_slow_force_response_reduces_rate_without_burst(self):
        serial_port = TimedForceSerial(response_delay_s=0.008)
        sensor = self.make_sensor(serial_port)
        try:
            frames = [sensor.read_frame(timeout_s=0.2) for _ in range(4)]
            stats = sensor.get_timing_stats()
        finally:
            sensor.close()

        intervals = [
            b - a for a, b in zip(
                serial_port.write_times, serial_port.write_times[1:]
            )
        ]
        self.assertTrue(all(frame is not None for frame in frames))
        self.assertEqual(serial_port.overlapping_writes, 0)
        self.assertTrue(all(interval >= 0.007 for interval in intervals))
        self.assertGreaterEqual(stats["schedule_skips"], 1)

    def test_force_timeout_recovers_without_overlapping_requests(self):
        serial_port = TimedForceSerial(respond=False)
        sensor = self.make_sensor(
            serial_port, period_s=0.010, response_timeout_s=0.020
        )
        try:
            time.sleep(0.075)
            stats = sensor.get_timing_stats()
        finally:
            sensor.close()

        self.assertGreaterEqual(stats["response_timeouts"], 2)
        self.assertEqual(serial_port.overlapping_writes, 0)
        intervals = [
            b - a for a, b in zip(
                serial_port.write_times, serial_port.write_times[1:]
            )
        ]
        self.assertTrue(all(interval >= 0.018 for interval in intervals))

    def test_production_force_backend_forwards_frame_and_stats(self):
        real_context = data_module.multiprocessing.get_context("spawn")
        fake_context = FakeMultiprocessingContext(real_context)
        expected = {
            "request_seq": 3,
            "tx_t": 5.0,
            "rx_t": 5.006,
            "latency_s": 0.006,
            "data": [1.0] * 6,
        }

        def fake_process_entry(
            port, period_s, timeout_s, queue_size,
            frame_queue, status_queue, startup_queue, stop_event,
        ):
            self.assertEqual(port, FORCE_SENSOR_PORT)
            self.assertEqual(period_s, FORCE_PERIOD_S)
            self.assertEqual(timeout_s, FORCE_RESPONSE_TIMEOUT_S)
            self.assertEqual(queue_size, FORCE_FRAME_QUEUE_SIZE)
            startup_queue.put(("ready", None))
            frame_queue.put(expected)
            status_queue.put(("stats", {
                "frames": 1,
                "queue_drops": 0,
                "tx_intervals_s": [0.005],
                "rx_intervals_s": [0.006],
                "latencies_s": [0.006],
            }))
            stop_event.wait()

        with mock.patch.object(
            force_module.multiprocessing, "get_context", return_value=fake_context
        ), mock.patch.object(
            force_module, "_force_process_main", side_effect=fake_process_entry
        ):
            sensor = SixAxisForceSensor(_startup_timeout_s=0.5)
            try:
                frame = sensor.read_frame(timeout_s=0.5)
                self.assertEqual(frame, expected)
                deadline = time.monotonic() + 0.5
                stats = sensor.get_timing_stats()
                while stats["frames"] != 1 and time.monotonic() < deadline:
                    time.sleep(0.01)
                    stats = sensor.get_timing_stats()
                self.assertEqual(stats["rx_intervals_s"], [0.006])
            finally:
                sensor.close()

        self.assertFalse(fake_context.process.terminated)


class TimestampedBufferTests(unittest.TestCase):
    def test_seq_get_after_and_one_time_matching(self):
        buf = TimestampedBuffer(maxlen=10)
        self.assertEqual(buf.append({"t": 1.00, "data": "a"}), 0)
        self.assertEqual(buf.append({"t": 1.01, "data": "b"}), 1)
        self.assertEqual(buf.append({"t": 1.02, "data": "c"}), 2)

        self.assertEqual([item["seq"] for item in buf.get_after(0)], [1, 2])
        first = match_closest(buf, 1.009, 0.015, min_seq=-1)
        self.assertEqual(first["seq"], 1)
        second = match_closest(buf, 1.02, 0.015, min_seq=first["seq"])
        self.assertEqual(second["seq"], 2)
        self.assertIsNone(match_closest(buf, 1.02, 0.015, min_seq=second["seq"]))


if __name__ == "__main__":
    unittest.main()
