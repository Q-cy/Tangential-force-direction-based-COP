import io
import os
import pathlib
import subprocess
import sys
import unittest
from importlib import resources
from unittest import mock

import numpy as np

import tangential as package
from tangential.processing.calibration import apply_fit_predict_multi, load_fit_coefs


class FakePressureSensor:
    """可注入的压力传感器：raw 由 decode 解码为 84 个通道。"""

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
            "raw": values,
        }

    @staticmethod
    def decode(raw):
        return list(raw)

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


def _sample_field(sample, name):
    """读取约定的 TangentialSample 字段，并给出清晰的契约错误。"""
    if not hasattr(sample, name):
        raise AssertionError(f"TangentialSample 缺少公共字段 {name!r}")
    return getattr(sample, name)


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
                "TangentialSample",
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

    def make_processor(self):
        cop = FakeCopProcessor()
        calibrator = FakeCalibrator()
        processor = package.TangentialFrameProcessor(
            cop_sensor=cop,
            calibration=calibrator,
        )
        return processor, cop, calibrator

    def test_sample_and_processor_preserve_shape_statistics_and_delegation(self):
        processor, cop, calibrator = self.make_processor()
        values = np.arange(84, dtype=np.float64)

        sample = processor.process(values, frame={"rx_t": 12.5})

        self.assertIsInstance(sample, package.TangentialSample)
        matrix = np.asarray(_sample_field(sample, "matrix"))
        self.assertEqual(matrix.shape, (12, 7))
        self.assertEqual(float(_sample_field(sample, "minimum")), 0.0)
        self.assertEqual(float(_sample_field(sample, "maximum")), 83.0)
        self.assertEqual(float(_sample_field(sample, "total")), 3486.0)
        self.assertAlmostEqual(float(_sample_field(sample, "mean")), 41.5)
        self.assertEqual(sample.min, sample.minimum)
        self.assertEqual(sample.max, sample.maximum)
        self.assertEqual(sample.sum, sample.total)
        self.assertEqual(_sample_field(sample, "rx_t"), 12.5)

        self.assertEqual(_sample_field(sample, "cop_x"), 3.5)
        self.assertEqual(_sample_field(sample, "cop_y"), 4.5)
        self.assertEqual(_sample_field(sample, "angle"), 12.5)
        self.assertEqual(_sample_field(sample, "gradient").shape, (12, 7, 2))
        self.assertEqual(_sample_field(sample, "calibrated_fx"), 1.0)
        self.assertEqual(_sample_field(sample, "calibrated_fy"), 2.0)
        self.assertEqual(_sample_field(sample, "calibrated_fz"), 3.0)
        self.assertAlmostEqual(
            _sample_field(sample, "calibrated_angle"),
            63.434948,
            places=5,
        )
        self.assertEqual(len(cop.threshold_frames), 1)
        self.assertEqual(len(cop.all_frames), 1)
        self.assertEqual(len(cop.gradient_frames), 1)
        self.assertEqual(len(calibrator.calls), 1)

    def test_sensor_api_reads_decodes_and_closes_in_context(self):
        processor, _, _ = self.make_processor()
        sensor = FakePressureSensor([np.arange(84, dtype=np.uint16)])

        with package.TangentialSensorAPI(sensor=sensor, processor=processor) as api:
            sample = _read_sample(api)

        self.assertIsInstance(sample, package.TangentialSample)
        self.assertEqual(np.asarray(_sample_field(sample, "matrix")).shape, (12, 7))
        self.assertEqual(sensor.read_count, 1)
        self.assertTrue(sensor.closed)

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
        )
        self.assertTrue(api.processor.calibration.available)
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
        sample = processor.process(values, frame={"rx_t": 1.0})
        stream = mock.Mock(spec=io.StringIO)

        renderer = package.FixedTerminalRenderer(stream=stream)
        rendered = renderer.render(sample)

        stream.write.assert_called_once()
        stream.flush.assert_called_once_with()
        written = stream.write.call_args.args[0]
        self.assertIn(rendered, written)
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
            "TangentialSensor", "angle_difference", "TrainingConfig",
            "TrainingResult", "train_model", "PlotConfig", "PlotResult",
            "plot_csv", "plot_full_analysis",
        }
        self.assertTrue(required.issubset(set(package.__all__)))
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
