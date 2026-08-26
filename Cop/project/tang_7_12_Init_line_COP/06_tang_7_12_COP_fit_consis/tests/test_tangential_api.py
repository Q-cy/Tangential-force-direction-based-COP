import io
import inspect
import os
import pathlib
import subprocess
import sys
import unittest
from dataclasses import fields
from importlib import resources
from unittest import mock

import numpy as np

import tangential as package
from tangential.config import ConsistenceCalibrationConfig, ProcessingConfig
from tangential.processing.calibration import (
    FitCalibrationModel,
    apply_fit_predict_multi,
    load_fit_coefs,
)
from tangential.runtime import sensor as runtime_sensor


class FakePressureSensor:
    """可注入的压力传感器：payload 由 decode 解码为 raw_data。"""

    def __init__(self, frames):
        self.frames = list(frames)
        self.closed = False
        self.read_count = 0

    def read_frame(self, timeout_s=0.1):
        del timeout_s
        self.read_count += 1
        if not self.frames:
            return None
        values = self.frames.pop(0)
        return {
            "request_seq": self.read_count - 1,
            "tx_t": float(self.read_count),
            "rx_t": float(self.read_count) + 0.001,
            "latency_s": 0.001,
            "payload": values,
        }

    @staticmethod
    def decode(payload):
        return list(payload)

    def close(self):
        self.closed = True


class FakeCopProcessor:
    """记录 PRSensorAngle 兼容调用，避免测试重复实现 CoP 算法。"""

    rows = 12
    cols = 7

    def __init__(self):
        self.threshold_frames = []
        self.all_frames = []
        self.gradient_frames = []

    def dynamic_threshold(self, frame):
        self.threshold_frames.append(np.array(frame, copy=True))

    def get_all(self, values):
        self.all_frames.append(np.array(values, copy=True))
        return 12.5, 1.25, -2.5, 3.5, 4.5

    def get_gradient(self, values):
        self.gradient_frames.append(np.array(values, copy=True))
        return np.ones((12, 7, 2), dtype=np.float32)

    def get_origin(self):
        return 3.0, 4.0

    def get_state(self):
        return 1

    @staticmethod
    def _compute_centroid(frame):
        del frame
        return 2.0, 3.0


class FakeCalibrator:
    def __init__(self):
        self.calls = []

    def predict(self, values):
        self.calls.append(values)
        return [1.0, 2.0, 3.0]


def _read_sample(api):
    reader = getattr(api, "read", None)
    if reader is None:
        raise AssertionError("TangentialSensorAPI 必须提供 read()")
    return reader()


class TangentialApiTests(unittest.TestCase):
    def setUp(self):
        missing = [
            name
            for name in (
                "TangentialFrame",
                "TangentialFrameProcessor",
                "TangentialSensorAPI",
                "FixedTerminalRenderer",
            )
            if not hasattr(package, name)
        ]
        if missing:
            self.fail(
                "正式 tangential API 缺少计划中的 API: "
                + ", ".join(missing)
            )
        self.assertFalse(hasattr(package, "TangentialSample"))

    def make_processor(self):
        cop = FakeCopProcessor()
        calibrator = FakeCalibrator()
        processor = package.TangentialFrameProcessor(
            cop_sensor=cop,
            calibration=calibrator,
            processing_config=ProcessingConfig(
                consistence=ConsistenceCalibrationConfig(enabled=False),
            ),
        )
        return processor, cop, calibrator

    def test_frame_has_exact_public_fields_and_process_frame_delegates_once(self):
        processor, cop, calibrator = self.make_processor()
        sample_processor = processor._sample_processor
        values = np.arange(84, dtype=np.float64)

        self.assertTrue(hasattr(processor, "process_frame"))
        self.assertFalse(hasattr(processor, "process"))
        self.assertFalse(hasattr(processor, "_process_sample"))
        self.assertFalse(hasattr(processor, "sample_processor"))
        with mock.patch.object(
            sample_processor, "_process_sample", wraps=sample_processor._process_sample
        ) as process_sample:
            frame = processor.process_frame(values, frame={"rx_t": 12.5})
        process_sample.assert_called_once()

        self.assertIsInstance(frame, package.TangentialFrame)
        self.assertEqual(
            [item.name for item in fields(package.TangentialFrame)],
            ["base_data", "adc_sum", "cop_x", "cop_y", "angle", "dx", "dy", "motion_state"],
        )
        self.assertEqual(np.asarray(frame.base_data).shape, (84,))
        self.assertEqual(float(frame.adc_sum), 3486.0)
        self.assertEqual(frame.cop_x, 3.5)
        self.assertEqual(frame.cop_y, 4.5)
        self.assertEqual(frame.angle, 12.5)
        for name in (
            "total", "sum", "minimum", "maximum", "mean", "matrix",
            "raw_2d", "copX", "copY", "gradient", "calibrated_fx",
        ):
            self.assertFalse(hasattr(frame, name), name)
        self.assertEqual(len(cop.threshold_frames), 1)
        self.assertEqual(len(cop.all_frames), 1)
        self.assertEqual(len(cop.gradient_frames), 1)
        self.assertEqual(len(calibrator.calls), 1)

    def test_frame_processors_own_distinct_sample_processors(self):
        first, _, _ = self.make_processor()
        second, _, _ = self.make_processor()

        self.assertIsNot(first._sample_processor, second._sample_processor)
        self.assertIsNot(
            first._sample_processor.slip_detector,
            second._sample_processor.slip_detector,
        )

    def test_sensor_api_reads_decodes_and_closes_in_context(self):
        processor, _, _ = self.make_processor()
        sensor = FakePressureSensor([np.arange(84, dtype=np.uint16)])

        with package.TangentialSensorAPI(sensor=sensor, processor=processor) as api:
            frame = _read_sample(api)

        self.assertIsInstance(frame, package.TangentialFrame)
        self.assertEqual(np.asarray(frame.base_data).shape, (84,))
        self.assertEqual(frame.adc_sum, float(np.sum(np.arange(84))))
        self.assertEqual(sensor.read_count, 1)
        self.assertTrue(sensor.closed)

    def test_sensor_api_rejects_internal_sample_processor_injection(self):
        processor, _, _ = self.make_processor()
        with self.assertRaises(TypeError):
            package.TangentialSensorAPI(
                sensor=FakePressureSensor([]),
                processor=processor._sample_processor,
            )

    def test_internal_sample_has_only_canonical_detailed_fields(self):
        processor, _, _ = self.make_processor()
        sample_processor = processor._sample_processor
        internal = sample_processor._process_sample(np.arange(84, dtype=np.float64))

        self.assertIsInstance(internal, runtime_sensor.TangentialSample)
        self.assertEqual(
            [item.name for item in fields(runtime_sensor.TangentialSample)],
            [
                "raw_data", "consistence_data", "base_data", "gradient", "adc_sum", "cop_x", "cop_y", "angle",
                "dx", "dy", "state", "calibrated_fx", "calibrated_fy",
                "calibrated_fz", "calibrated_angle", "request_seq", "tx_t",
                "rx_t", "latency_s", "origin_x", "origin_y", "contact",
                "display_contact", "refined", "region_mask", "regions",
                "centroid", "rel_ms", "motion_state", "is_slipping",
                "slip_motion_distance", "slip_confidence",
                "angle_vector_magnitude",
            ],
        )
        self.assertEqual(internal.adc_sum, 3486.0)
        for name in (
            "total", "sum", "minimum", "maximum", "mean", "matrix",
            "raw_2d", "copX", "copY",
        ):
            self.assertFalse(hasattr(internal, name), name)

        algorithm_call_counts = (
            len(sample_processor.cop_sensor.threshold_frames),
            len(sample_processor.cop_sensor.all_frames),
            len(sample_processor.cop_sensor.gradient_frames),
        )
        self.assertIsInstance(
            inspect.getattr_static(
                package.TangentialFrameProcessor, "_to_tangential_frame"
            ),
            staticmethod,
        )
        legacy_projection_name = "to_" + "tangential_" + "frame"
        self.assertFalse(hasattr(runtime_sensor, legacy_projection_name))
        frame = processor._to_tangential_frame(internal)
        self.assertIsInstance(frame, package.TangentialFrame)
        self.assertEqual(
            [item.name for item in fields(frame)],
            ["base_data", "adc_sum", "cop_x", "cop_y", "angle", "dx", "dy", "motion_state"],
        )
        np.testing.assert_array_equal(frame.base_data, internal.base_data)
        self.assertIsNot(frame.base_data, internal.base_data)
        self.assertEqual(
            algorithm_call_counts,
            (
                len(sample_processor.cop_sensor.threshold_frames),
                len(sample_processor.cop_sensor.all_frames),
                len(sample_processor.cop_sensor.gradient_frames),
            ),
        )

    def test_sensor_api_passes_pressure_port_to_factory(self):
        calls = []

        def factory(*, port):
            calls.append(port)
            return FakePressureSensor([])

        processor, _, _ = self.make_processor()
        api = package.TangentialSensorAPI(
            processor=processor,
            sensor_factory=factory,
            pressure_port="/dev/test-pressure",
        )
        api.close()
        self.assertEqual(calls, ["/dev/test-pressure"])

    def test_sensor_api_uses_bundled_model_when_model_path_is_none(self):
        def factory(*, port):
            self.assertEqual(port, "/dev/bundled-model-test")
            return FakePressureSensor([])

        api = package.TangentialSensorAPI(
            sensor_factory=factory,
            pressure_port="/dev/bundled-model-test",
            model_path=None,
            processing_config=ProcessingConfig(
                consistence=ConsistenceCalibrationConfig(enabled=False),
            ),
        )
        self.assertTrue(api.processor._sample_processor.calibration.available)
        api.close()

    def test_sensor_api_close_is_idempotent(self):
        processor, _, _ = self.make_processor()
        sensor = FakePressureSensor([])
        api = package.TangentialSensorAPI(sensor=sensor, processor=processor)

        api.close()
        api.close()

        self.assertTrue(sensor.closed)

    def test_fixed_terminal_renderer_writes_and_flushes_once(self):
        processor, _, _ = self.make_processor()
        values = np.arange(84, dtype=np.float64)
        sample = processor.process_frame(values, frame={"rx_t": 1.0})
        stream = mock.Mock(spec=io.StringIO)

        renderer = package.FixedTerminalRenderer(stream=stream)
        rendered = renderer.render(sample)

        stream.write.assert_called_once()
        stream.flush.assert_called_once_with()
        written = stream.write.call_args.args[0]
        self.assertIn(rendered, written)
        self.assertIn("adc_sum=", rendered)
        self.assertIn("motion_state=", rendered)
        matrix_lines = rendered.splitlines()[:12]
        self.assertEqual(len(matrix_lines), 12)
        widths = {len(line) for line in matrix_lines}
        self.assertEqual(len(widths), 1)
        for line in matrix_lines:
            fields = line.split()
            self.assertEqual(len(fields), 7)
            self.assertTrue(all(field.isdigit() for field in fields))

    def test_existing_fit_model_prediction_is_unchanged(self):
        resource = resources.files("tangential.resources").joinpath(
            "fit_coefs.bin"
        )
        fit_type, _, params, split_sign = load_fit_coefs(resource.read_bytes())
        inputs = [0.25, -0.5, 1200.0]
        expected = apply_fit_predict_multi(inputs, params, fit_type, split_sign)

        model = package.FitCalibrationModel.from_default()
        actual = model.predict(*inputs)

        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


class PublicApiStructureTests(unittest.TestCase):
    def test_public_exports_and_no_gui_import(self):
        required = {
            "TangentialSensorAPI", "angle_difference", "TrainingConfig",
            "TrainingResult", "train_model", "PlotConfig", "PlotResult",
            "plot_csv", "plot_full_analysis",
        }
        self.assertTrue(required.issubset(set(package.__all__)))
        self.assertFalse(hasattr(package, "TangentialSensor"))
        env = os.environ.copy()
        env["PYTHONPATH"] = str(pathlib.Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import tangential; assert 'pyqtgraph' not in sys.modules"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)


if __name__ == "__main__":
    unittest.main()
