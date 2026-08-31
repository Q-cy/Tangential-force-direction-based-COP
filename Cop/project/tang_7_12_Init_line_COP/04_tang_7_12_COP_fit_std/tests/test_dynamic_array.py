"""动态压力阵列尺寸的协议、处理、CSV 和 GUI 回归测试。"""

from __future__ import annotations

import csv
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pyqtgraph.Qt import QtWidgets

from tangential.config import (
    ArrayConfig, CopConfig, ConsistenceCalibrationConfig, ForceConfig,
    FullApplicationConfig, ProcessingConfig,
)
from tangential.gui.realtime import RealTimePlot
from tangential.processing.calconsistence import ConsistenceCalibrator
from tangential.runtime.sensor import TangentialFrameProcessor, TangentialSensorAPI
from tangential.runtime.session import FullAcquisitionSession
from tangential.sensors.pressure import PressureSensor
from tangential.storage.csv import build_csv_header, build_csv_row, init_csv_file


def _make_pressure_frame(values) -> bytes:
    """按正式应答协议构造任意通道数的压力帧。"""
    values = list(values)
    payload = b"".join(struct.pack("<H", value) for value in values)
    sensor_bytes = len(payload)
    payload_length = PressureSensor.MIN_PAYLOAD_LEN + sensor_bytes
    frame = bytearray(4 + payload_length + 1)
    frame[:2] = b"\xaa\x55"
    frame[2:4] = struct.pack("<H", payload_length)
    frame[11:13] = struct.pack("<H", sensor_bytes)
    frame[13] = 0
    frame[14:14 + sensor_bytes] = payload
    frame[-1] = PressureSensor.crc8_itu(frame[:-1])
    return bytes(frame)


def _pressure_parser(rows: int, cols: int) -> PressureSensor:
    """创建不启动线程的动态压力解析器测试替身。"""
    sensor = PressureSensor.__new__(PressureSensor)
    sensor.rows = rows
    sensor.cols = cols
    sensor.channel_count = rows * cols
    sensor.expected_sensor_bytes = sensor.channel_count * 2
    sensor.expected_payload_len = sensor.expected_sensor_bytes + sensor.MIN_PAYLOAD_LEN
    sensor.expected_frame_len = 4 + sensor.expected_payload_len + 1
    sensor._max_rx_buf = max(
        sensor.MAX_RX_BUF, sensor.expected_frame_len + sensor.READ_CHUNK_SIZE
    )
    sensor._rx_buf_retain = max(sensor.RX_BUF_RETAIN, sensor.expected_frame_len)
    sensor._rx_buf = bytearray()
    import threading

    sensor._rx_lock = threading.Lock()
    sensor._stats_lock = threading.Lock()
    sensor._stats = {
        "crc_errors": 0,
        "length_errors": 0,
        "status_errors": 0,
        "framing_bytes": 0,
    }
    return sensor


class DynamicPressureTests(unittest.TestCase):
    def test_three_by_five_protocol_and_decode(self):
        sensor = _pressure_parser(3, 5)
        self.assertEqual(PressureSensor.build_read_command(168), PressureSensor.CMD_BYTES)
        command = PressureSensor.build_read_command(30)
        self.assertEqual(command[11:13], struct.pack("<H", 30))

        expected = list(range(15))
        sensor._rx_buf.extend(_make_pressure_frame(expected))
        payload = sensor.read_data()
        self.assertEqual(len(payload), 30)
        self.assertEqual(sensor.decode(payload), expected)

    def test_large_fragmented_frame_exceeds_legacy_receive_buffer(self):
        sensor = _pressure_parser(64, 64)
        expected = [index % 65536 for index in range(64 * 64)]
        frame = _make_pressure_frame(expected)
        self.assertGreater(len(frame), PressureSensor.MAX_RX_BUF)
        for start in range(0, len(frame), 317):
            sensor._append_rx(frame[start:start + 317])
        payload = sensor.read_data()
        self.assertIsNotNone(payload)
        self.assertEqual(sensor.decode(payload), expected)

    def test_three_by_five_processing_and_csv(self):
        array_config = ArrayConfig(rows=3, cols=5)
        config = ProcessingConfig(
            cop=CopConfig(),
            consistence=ConsistenceCalibrationConfig(enabled=False),
        )
        frame = TangentialFrameProcessor(
            array_config=array_config,
            processing_config=config, calibration=None
        ).process_frame(np.arange(15, dtype=np.float64))
        self.assertEqual(frame.base_data.shape, (15,))
        self.assertEqual(frame.adc_sum, 105.0)

        header = build_csv_header(array_config)
        self.assertEqual(len(header), 39)
        row = build_csv_row(
            press_timestamp=1.0,
            rel_ms=0.0,
            delta_ms=0.0,
            ch_data=np.arange(15),
            force_data=[float("nan")] * 6,
            force_timestamp=float("nan"),
            delta_cop_x=0.0,
            delta_cop_y=0.0,
            delta_force_x=float("nan"),
            delta_force_y=float("nan"),
            delta_force_z=float("nan"),
            adc_angle=0.0,
            force_angle=float("nan"),
            adc_sum=105.0,
            array_config=array_config,
        )
        self.assertEqual(len(row), len(header))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dynamic.csv"
            writer, stream = init_csv_file(path, array_config=array_config)
            writer.writerow(row)
            stream.close()
            with path.open(newline="") as input_stream:
                self.assertEqual(len(next(csv.reader(input_stream))), 39)
                self.assertEqual(len(next(csv.reader(input_stream))), 39)

    def test_fourteen_by_five_gui_displays_all_channels_and_xy_curves(self):
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        del app
        array_config = ArrayConfig(rows=14, cols=5)
        plot = RealTimePlot(array_config=array_config)
        session = FullAcquisitionSession(
            plot,
            config=FullApplicationConfig(
                array=array_config,
                force=ForceConfig(enabled=False),
                processing=ProcessingConfig(
                    consistence=ConsistenceCalibrationConfig(enabled=False),
                ),
                plot_fps=1000.0,
            ),
        )
        try:
            session.force_angle_deg = 0.0
            session.force_fx_filt = 2.5
            session.force_fy_filt = -1.5
            session.force_fz_filt = 4.0
            session.latest_sample = SimpleNamespace(
                angle=0.0,
                base_data=np.arange(70, dtype=float),
                adc_sum=2415.0,
                cop_x=2.0,
                cop_y=1.0,
                origin_x=1.0,
                origin_y=1.0,
                dx=1.25,
                dy=-0.75,
                calibrated_fx=0.0,
                calibrated_fy=0.0,
                calibrated_fz=0.0,
                calibrated_angle=0.0,
                state=0,
                gradient=np.zeros((14, 5, 2), dtype=float),
                region_mask=np.zeros((14, 5), dtype=int),
                display_contact=False,
                refined=False,
                regions=[],
                centroid=(2.0, 1.0),
                motion_state=0,
                is_slipping=False,
                slip_motion_distance=0.0,
                slip_confidence=0.0,
                angle_vector_magnitude=0.0,
                contact=False,
                rel_ms=0,
            )
            session.update_plot()
            plot.update_all()
            self.assertEqual(plot.rows, 14)
            self.assertEqual(plot.cols, 5)
            self.assertEqual(plot._press_table_arr.shape, (14, 5))
            self.assertEqual(plot._gradient_arr.shape, (14, 5, 2))
            self.assertEqual(len(plot._cell_txts), 14)
            self.assertEqual(len(plot._cell_txts[0]), 5)
            self.assertEqual(len(plot._g_lines), 70)
            self.assertEqual(len(plot._g_txts), 70)
            np.testing.assert_array_equal(
                plot._cell_grid.data, np.arange(70, dtype=float).reshape(14, 5)
            )
            self.assertEqual(plot._c_pzt_fx.getData()[1][-1], 1.25)
            self.assertEqual(plot._c_pzt_fy.getData()[1][-1], -0.75)
            self.assertEqual(plot._c_frc_fx.getData()[1][-1], 2.5)
            self.assertEqual(plot._c_frc_fy.getData()[1][-1], -1.5)
        finally:
            plot.win.close()

    def test_injected_components_reject_mismatched_array_layout(self):
        expected = ArrayConfig(rows=14, cols=5)
        processor = TangentialFrameProcessor(
            array_config=ArrayConfig(rows=12, cols=7),
            processing_config=ProcessingConfig(
                consistence=ConsistenceCalibrationConfig(enabled=False),
            ),
            calibration=None,
        )
        sensor = SimpleNamespace(
            array_config=expected,
            close=lambda: None,
        )
        with self.assertRaisesRegex(ValueError, "processor 阵列尺寸"):
            TangentialSensorAPI(
                sensor=sensor,
                processor=processor,
                array_config=expected,
            )

        sample_processor = processor._sample_processor
        with self.assertRaisesRegex(ValueError, "sample_processor 阵列尺寸"):
            FullAcquisitionSession(
                object(),
                config=FullApplicationConfig(
                    array=expected,
                    force=ForceConfig(enabled=False),
                ),
                sample_processor=sample_processor,
            )


class DynamicConfigAndConsistencyTests(unittest.TestCase):
    def test_array_dimensions_reject_bool_float_nonpositive_and_protocol_overflow(self):
        for rows, cols in ((True, 7), (12.0, 7), (0, 7), (12, -1)):
            with self.subTest(rows=rows, cols=cols):
                with self.assertRaises(ValueError):
                    ArrayConfig(rows=rows, cols=cols)
        with self.assertRaises(ValueError):
            ArrayConfig(rows=1, cols=32763)

    def test_fifteen_channel_consistency_coefficients_apply(self):
        calibrator = ConsistenceCalibrator(np.ones(15), np.zeros(15))
        corrected = calibrator.apply(np.arange(15, dtype=float))
        np.testing.assert_array_equal(corrected, np.arange(15, dtype=float))
        with self.assertRaises(ValueError):
            calibrator.apply(np.arange(84, dtype=float))


if __name__ == "__main__":
    unittest.main()
