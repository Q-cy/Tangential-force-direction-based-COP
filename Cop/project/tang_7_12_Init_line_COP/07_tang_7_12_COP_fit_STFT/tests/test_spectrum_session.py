"""完整会话频谱接线和双路边界测试。"""

from __future__ import annotations

import csv
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from tangential.config import (
    ConsistenceCalibrationConfig,
    ForceConfig,
    FullApplicationConfig,
    ProcessingConfig,
    SpectrumConfig,
    SyncConfig,
)
from tangential.runtime.sensor import TangentialSample
from tangential.runtime.session import FullAcquisitionSession
from tangential.storage.csv import TABLE_CSV_HEADER


class _PlotStub:
    """只接收完整会话调用的非 Qt 绘图替身。"""

    def __init__(self):
        self.analysis_csv_path = None

    def set_data(self, *args, **kwargs):
        pass

    def append_full_data(self, *args, **kwargs):
        pass

    def set_analysis_csv_path(self, path):
        self.analysis_csv_path = path


class _SpectrumSink:
    def __init__(self):
        self.items = []
        self.progress = []
        self.lock = threading.Lock()

    def submit(self, snapshot):
        with self.lock:
            self.items.append(snapshot)

    def submit_progress(self, ready_samples, required_samples):
        with self.lock:
            self.progress.append((ready_samples, required_samples))


class _Pressure:
    count = 900

    def __init__(self):
        self.index = 0
        self.closed = False

    def read_frame(self, timeout_s=0.1):
        if self.index < type(self).count:
            index = self.index
            self.index += 1
            rx_t = 10.0 + index / 160.0
            return {
                "request_seq": index,
                "tx_t": rx_t - 0.001,
                "rx_t": rx_t,
                "latency_s": 0.001,
                "payload": np.full(84, index + 1, dtype=np.float64),
            }
        time.sleep(min(timeout_s, 0.001))
        return None

    @staticmethod
    def decode(payload):
        return payload

    def get_timing_stats(self):
        return {
            "requests": self.index,
            "frames": self.index,
            "response_timeouts": 0,
            "crc_errors": 0,
            "status_errors": 0,
            "framing_bytes": 0,
            "serial_read_errors": 0,
            "serial_write_errors": 0,
            "serial_flush_errors": 0,
            "queue_drops": 0,
            "schedule_skips": 0,
            "tx_intervals_s": [],
            "rx_intervals_s": [],
            "latencies_s": [],
        }

    def close(self):
        self.closed = True


class _Processor:
    def _process_sample(self, raw_data, frame):
        index = int(frame["request_seq"])
        timestamp = float(frame["rx_t"])
        values = np.asarray(raw_data, dtype=np.float64)
        cop_x = np.sin(2.0 * np.pi * 12.0 * timestamp)
        cop_y = 0.5 * np.sin(2.0 * np.pi * 21.0 * timestamp)
        return TangentialSample(
            raw_data=values.copy(),
            consistence_data=None,
            base_data=values.copy(),
            gradient=np.zeros((12, 7, 2), dtype=np.float64),
            adc_sum=float(values.sum()),
            cop_x=float(cop_x),
            cop_y=float(cop_y),
            angle=0.0,
            dx=0.0,
            dy=0.0,
            state=2,
            calibrated_fx=float("nan"),
            calibrated_fy=float("nan"),
            calibrated_fz=float("nan"),
            calibrated_angle=float("nan"),
            request_seq=index,
            tx_t=float(frame["tx_t"]),
            rx_t=timestamp,
            latency_s=float(frame["latency_s"]),
            origin_x=0.0,
            origin_y=0.0,
            contact=True,
            display_contact=True,
            refined=True,
            region_mask=np.zeros((12, 7), dtype=np.int32),
            regions=[],
            centroid=(0.0, 0.0),
        )


class SpectrumSessionTests(unittest.TestCase):
    def test_single_session_saves_the_same_snapshots_submitted_to_sink(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FullApplicationConfig(
                force=ForceConfig(enabled=False),
                processing=ProcessingConfig(
                    consistence=ConsistenceCalibrationConfig(enabled=False)
                ),
                sync=SyncConfig(buffer_size=2000, target_fps=1000.0),
                save_dir=directory,
                spectrum=SpectrumConfig(enabled=True),
            )
            sink = _SpectrumSink()
            session = FullAcquisitionSession(
                _PlotStub(),
                config=config,
                pressure_factory=_Pressure,
                sample_processor=_Processor(),
                spectrum_sink=sink,
            )
            session.start()
            registered_csv_path = Path(session.plot.analysis_csv_path)
            self.assertEqual(registered_csv_path, Path(session.csv_path))
            self.assertEqual(registered_csv_path.parent, Path(directory))
            deadline = time.perf_counter() + 3.0
            while (
                session.sensor_press.index < _Pressure.count
                and time.perf_counter() < deadline
            ):
                time.sleep(0.001)
            while session.process_new_pressure_frames():
                pass
            session.close()

            csv_paths = sorted(Path(directory).glob("*.csv"))
            self.assertEqual(len(csv_paths), 1)
            spectrum_path = csv_paths[0].with_name(
                csv_paths[0].stem + "_spectrum.npz"
            )
            self.assertTrue(spectrum_path.is_file())
            self.assertGreater(len(sink.items), 0)
            analyzer_snapshots = session.spectrum_analyzer.snapshots
            self.assertEqual(len(sink.items), len(analyzer_snapshots))
            for submitted, stored in zip(sink.items, analyzer_snapshots):
                np.testing.assert_array_equal(
                    submitted.velocity_amplitude_combined,
                    stored.velocity_amplitude_combined,
                )
                self.assertEqual(submitted.friction_state, stored.friction_state)
                self.assertEqual(
                    submitted.slip_band_power_ratio,
                    stored.slip_band_power_ratio,
                )
                self.assertEqual(submitted.threshold, stored.threshold)
                np.testing.assert_array_equal(
                    submitted.relative_power_db,
                    stored.relative_power_db,
                )
            with np.load(spectrum_path, allow_pickle=False) as archive:
                self.assertEqual(
                    archive["velocity_amplitude_combined"].shape[0],
                    len(sink.items),
                )
                self.assertEqual(
                    archive["velocity_amplitude_combined"].shape[1],
                    sink.items[0].frequency_hz.size,
                )
                np.testing.assert_allclose(
                    archive["slip_band_power_ratio"],
                    np.asarray([item.slip_band_power_ratio for item in sink.items]),
                )
                self.assertEqual(
                    archive["threshold"].item(),
                    config.spectrum.slip_band_power_ratio_threshold,
                )
                self.assertEqual(archive["friction_state"].dtype, np.int8)
                np.testing.assert_allclose(
                    archive["relative_power_db"][-1],
                    sink.items[-1].relative_power_db,
                )
                np.testing.assert_allclose(
                    archive["baseline_power"][-1],
                    sink.items[-1].baseline_power,
                )
            with csv_paths[0].open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), _Pressure.count)
            self.assertEqual(len(rows[0]), len(TABLE_CSV_HEADER))

    def test_spectrum_disabled_session_does_not_create_npz(self):
        with tempfile.TemporaryDirectory() as directory:
            config = FullApplicationConfig(
                force=ForceConfig(enabled=False),
                processing=ProcessingConfig(
                    consistence=ConsistenceCalibrationConfig(enabled=False)
                ),
                save_dir=directory,
                spectrum=SpectrumConfig(enabled=False),
            )
            session = FullAcquisitionSession(
                _PlotStub(),
                config=config,
                pressure_factory=_Pressure,
                sample_processor=_Processor(),
            )
            session.start()
            session.close()
            self.assertEqual(list(Path(directory).glob("*_spectrum.npz")), [])


if __name__ == "__main__":
    unittest.main()
