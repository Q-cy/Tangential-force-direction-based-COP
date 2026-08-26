import csv
import io
import os
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from tangential.acquisition.buffer import TimestampedBuffer
from tangential.config import (
    ConsistenceCalibrationConfig,
    ForceConfig,
    FullApplicationConfig,
    ProcessingConfig,
)
from tangential.runtime.session import (
    FullAcquisitionSession,
    PressureThread,
    acquisition_loop,
    g_main_stop_flag,
)
from tangential.storage.csv import TABLE_CSV_HEADER


class PlotStub:
    def __init__(self):
        self.frames = 0
        self.last_args = None
        self.last_kwargs = None

    def set_data(self, *args, **kwargs):
        self.frames += 1
        self.last_args = args
        self.last_kwargs = kwargs

    def append_full_data(self, *args, **kwargs):
        pass


class ExplodingPlot(PlotStub):
    def set_data(self, *args, **kwargs):
        raise RuntimeError("plot failed")


def disabled_processing():
    """测试用原始 ADC 模式，避免依赖尚未选定的内置 NPZ 资源。"""
    return ProcessingConfig(
        consistence=ConsistenceCalibrationConfig(enabled=False)
    )


class FakePressure:
    frames = 4
    initial_delay = 0.0
    instances = []

    def __init__(self):
        self.index = 0
        self.started = time.perf_counter()
        self.exhausted_at = None
        self.closed = False
        type(self).instances.append(self)

    def read_frame(self, timeout_s=0.1):
        now = time.perf_counter()
        if now - self.started < self.initial_delay:
            time.sleep(min(timeout_s, 0.001))
            return None
        if self.index < self.frames:
            value = self.index + 1
            request_seq = self.index
            self.index += 1
            tx_t = time.perf_counter()
            rx_t = time.perf_counter()
            return {
                "request_seq": request_seq,
                "tx_t": tx_t,
                "rx_t": rx_t,
                "latency_s": rx_t - tx_t,
                "payload": [float(value)] * 84,
            }
        if self.exhausted_at is None:
            self.exhausted_at = now
        if now - self.exhausted_at > 0.06:
            g_main_stop_flag.set()
        time.sleep(min(timeout_s, 0.001))
        return None

    def get_timing_stats(self):
        return {
            "requests": self.index,
            "frames": self.index,
            "response_timeouts": 0,
            "crc_errors": 0,
            "status_errors": 0,
            "framing_bytes": 0,
            "queue_drops": 0,
            "schedule_skips": 0,
            "tx_intervals_s": [],
            "rx_intervals_s": [],
            "latencies_s": [],
        }

    @staticmethod
    def decode(payload):
        return payload

    def close(self):
        self.closed = True


class FakeForce:
    calibrates = True
    initial_delay = 0.0
    instances = []

    def __init__(self):
        self.started = time.perf_counter()
        self.index = 0
        self.closed = False
        self.biases = []
        type(self).instances.append(self)

    def calibrate_zero(self, sample_count=10, timeout_s=1.0):
        return self.calibrates

    def read_frame(self, timeout_s=0.1):
        if time.perf_counter() - self.started < self.initial_delay:
            time.sleep(min(timeout_s, 0.001))
            return None
        # 模拟真实请求—响应等待并主动让出 GIL；若无等待，该 fake 会以数十万
        # 帧/秒覆盖有界缓存，使测试结果依赖线程调度而非 15 ms 匹配逻辑。
        time.sleep(min(timeout_s, 0.001))
        self.index += 1
        rx_t = time.perf_counter()
        return {
            "request_seq": self.index - 1,
            "tx_t": rx_t - 0.001,
            "rx_t": rx_t,
            "latency_s": 0.001,
            "data": [float(self.index), 2.0, 3.0, 0.0, 0.0, 0.0],
        }

    def get_timing_stats(self):
        return {
            "requests": self.index,
            "frames": self.index,
            "response_timeouts": 0,
            "framing_errors": 0,
            "tail_errors": 0,
            "serial_read_errors": 0,
            "serial_write_errors": 0,
            "serial_flush_errors": 0,
            "queue_drops": 0,
            "schedule_skips": 0,
            "tx_intervals_s": [],
            "rx_intervals_s": [],
            "latencies_s": [],
        }

    def add_zero_bias(self, fx, fy):
        self.biases.append((fx, fy))

    def close(self):
        self.closed = True


def read_csvs(directory):
    paths = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.endswith(".csv")
    ]
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as file_obj:
            rows.extend(csv.DictReader(file_obj))
    return paths, rows


class MainLoopIntegrationTests(unittest.TestCase):
    def setUp(self):
        g_main_stop_flag.clear()
        FakePressure.instances.clear()
        FakeForce.instances.clear()
        FakePressure.frames = 4
        FakePressure.initial_delay = 0.0
        FakeForce.calibrates = True
        FakeForce.initial_delay = 0.0

    def run_loop(self, directory, force_cls=FakeForce):
        config = FullApplicationConfig(
            save_dir=directory,
            model_path=os.path.join(directory, "missing.bin"),
            target_fps=1000,
            max_time_diff_s=0.015,
            processing=disabled_processing(),
        )
        acquisition_loop(
            PlotStub(),
            config=config,
            pressure_factory=FakePressure,
            force_factory=force_cls,
        )

    def test_pressure_required_failure_creates_no_csv(self):
        class MissingPressure:
            def __init__(self):
                raise OSError("missing")

        with tempfile.TemporaryDirectory() as directory:
            config = FullApplicationConfig(
                save_dir=directory,
                model_path=os.path.join(directory, "missing.bin"),
                processing=disabled_processing(),
            )
            with self.assertRaisesRegex(RuntimeError, "压力传感器未连接"):
                acquisition_loop(
                    PlotStub(),
                    config=config,
                    pressure_factory=MissingPressure,
                )
            self.assertEqual(os.listdir(directory), [])

    def test_force_calibration_failure_degrades_to_pressure_only(self):
        class FailedForce(FakeForce):
            calibrates = False
            instances = []

        with tempfile.TemporaryDirectory() as directory:
            self.run_loop(directory, force_cls=FailedForce)
            paths, rows = read_csvs(directory)

        self.assertEqual(len(paths), 1)
        self.assertEqual(len(rows), FakePressure.frames)
        self.assertEqual(len(rows[0]), 108)
        self.assertEqual(float(rows[0]["rel_ms"]), 0.0)
        self.assertEqual(float(rows[0]["delta_ms"]), 0.0)
        self.assertEqual(len({row["press_t"] for row in rows}), len(rows))
        self.assertTrue(all(row["force_t"].lower() == "nan" for row in rows))
        first_t = float(rows[0]["press_t"])
        for row in rows:
            expected_rel = max(0.0, round((float(row["press_t"]) - first_t) * 1000, 6))
            self.assertAlmostEqual(float(row["rel_ms"]), expected_rel, places=6)
        for previous, current in zip(rows, rows[1:]):
            expected_delta = max(
                0.0,
                round((float(current["press_t"]) - float(previous["press_t"])) * 1000, 6),
            )
            self.assertAlmostEqual(float(current["delta_ms"]), expected_delta, places=6)
        self.assertTrue(FakePressure.instances[-1].closed)
        self.assertTrue(FailedForce.instances[-1].closed)

    def test_disabled_force_does_not_construct_or_open_force_sensor(self):
        """ForceConfig.enabled=False 时完整会话不触碰默认力串口。"""
        with tempfile.TemporaryDirectory() as directory:
            config = FullApplicationConfig(
                save_dir=directory,
                model_path=os.path.join(directory, "missing.bin"),
                force=ForceConfig(enabled=False),
                target_fps=1000,
                processing=disabled_processing(),
            )
            acquisition_loop(
                PlotStub(),
                config=config,
                pressure_factory=FakePressure,
                force_factory=FakeForce,
            )
        self.assertEqual(FakeForce.instances, [])
        self.assertTrue(FakePressure.instances[-1].closed)

    def test_update_plot_forwards_angle_vector_magnitude(self):
        """完整会话必须把处理器的方向向量模长交给GUI。"""
        with tempfile.TemporaryDirectory() as directory:
            plot = PlotStub()
            config = FullApplicationConfig(
                save_dir=directory,
                model_path=os.path.join(directory, "missing.bin"),
                force=ForceConfig(enabled=False),
                target_fps=1000,
                plot_fps=1000,
                processing=disabled_processing(),
            )
            acquisition_loop(
                plot,
                config=config,
                pressure_factory=FakePressure,
                force_factory=FakeForce,
            )
        self.assertIsNotNone(plot.last_kwargs)
        self.assertIn("angle_vector_magnitude", plot.last_kwargs)
        self.assertIsInstance(plot.last_kwargs["angle_vector_magnitude"], float)

    def test_session_csv_and_gui_consume_sample_base_data(self):
        raw_data = np.arange(84, dtype=np.float64)
        base_data = raw_data + 1000.0
        sample = SimpleNamespace(
            raw_data=raw_data,
            consistence_data=base_data.copy(),
            base_data=base_data,
            rx_t=10.0,
            dx=1.0,
            dy=2.0,
            angle=30.0,
            calibrated_fx=3.0,
            calibrated_fy=4.0,
            calibrated_fz=5.0,
            calibrated_angle=45.0,
            state=1,
            adc_sum=float(np.sum(base_data)),
            cop_x=3.0,
            cop_y=4.0,
            origin_x=1.0,
            origin_y=2.0,
            gradient=np.zeros((12, 7, 2), dtype=np.float64),
            display_contact=True,
            refined=False,
            region_mask=np.zeros((12, 7), dtype=np.int32),
            regions=[],
            centroid=(3.0, 4.0),
            motion_state=1,
            is_slipping=False,
            slip_motion_distance=0.0,
            slip_confidence=0.0,
            angle_vector_magnitude=2.0,
            contact=False,
            rel_ms=0,
        )
        plot = PlotStub()
        config = FullApplicationConfig(
            force=ForceConfig(enabled=False),
            processing=disabled_processing(),
            plot_fps=1000,
        )
        session = FullAcquisitionSession(plot, config=config)
        csv_stream = io.StringIO()
        session.csv_writer = csv.writer(csv_stream)
        session.csv_file_obj = csv_stream

        session.write_snapshot(sample, None)
        row = next(csv.reader(io.StringIO(csv_stream.getvalue())))
        self.assertEqual(len(row), 108)
        row_by_name = dict(zip(TABLE_CSV_HEADER, row))
        csv_pressure = np.asarray(
            [float(row_by_name[f"ch{index}"]) for index in range(1, 85)]
        )
        np.testing.assert_array_equal(csv_pressure, base_data)
        self.assertFalse(np.array_equal(csv_pressure, raw_data))

        session.latest_sample = sample
        session.update_plot()
        self.assertIsNotNone(plot.last_args)
        np.testing.assert_array_equal(plot.last_args[2], base_data)
        self.assertFalse(np.array_equal(plot.last_args[2], raw_data))

    def test_force_first_and_one_to_one_matching(self):
        FakePressure.initial_delay = 0.02
        with tempfile.TemporaryDirectory() as directory:
            self.run_loop(directory)
            _, rows = read_csvs(directory)

        self.assertGreater(len(rows), 0)
        force_times = [row["force_t"] for row in rows]
        self.assertEqual(len(force_times), len(set(force_times)))
        self.assertTrue(all(float(row["dt"]) <= 0.015 for row in rows))
        self.assertTrue(FakePressure.instances[-1].closed)
        self.assertTrue(FakeForce.instances[-1].closed)

    def test_pressure_first_does_not_crash(self):
        FakeForce.initial_delay = 0.02
        with tempfile.TemporaryDirectory() as directory:
            self.run_loop(directory)
        self.assertTrue(FakePressure.instances[-1].closed)
        self.assertTrue(FakeForce.instances[-1].closed)

    def test_acquisition_thread_exposes_sensor_exception(self):
        class BrokenSensor:
            def read_frame(self, timeout_s=0.1):
                raise OSError("serial disconnected")

            @staticmethod
            def decode(payload):
                return payload

        g_main_stop_flag.clear()
        thread = PressureThread(
            BrokenSensor(), TimestampedBuffer(), g_main_stop_flag
        )
        thread.start()
        thread.join(timeout=1)
        self.assertIsInstance(thread.error, OSError)

    def test_runtime_exception_still_closes_resources_and_csv(self):
        class FailedForce(FakeForce):
            calibrates = False
            instances = []

        with tempfile.TemporaryDirectory() as directory:
            config = FullApplicationConfig(
                save_dir=directory,
                model_path=os.path.join(directory, "missing.bin"),
                target_fps=1000,
                processing=disabled_processing(),
            )
            with self.assertRaisesRegex(RuntimeError, "plot failed"):
                acquisition_loop(
                    ExplodingPlot(),
                    config=config,
                    pressure_factory=FakePressure,
                    force_factory=FailedForce,
                )
            paths, rows = read_csvs(directory)

        self.assertEqual(len(paths), 1)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(FakePressure.instances[-1].closed)
        self.assertTrue(FailedForce.instances[-1].closed)


if __name__ == "__main__":
    unittest.main()
