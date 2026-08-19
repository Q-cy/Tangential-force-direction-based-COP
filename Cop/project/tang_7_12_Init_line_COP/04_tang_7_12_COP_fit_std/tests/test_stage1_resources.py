import os
import unittest
from pathlib import Path
from unittest import mock

from tangential.config import (
    FullApplicationConfig,
    default_model_path,
    default_save_dir,
)
from tangential.full import FullAcquisitionSession
from tangential.processing.calibration import FitCalibrationModel


class Stage1ResourceTests(unittest.TestCase):
    def test_default_model_comes_from_package_resource(self):
        model = FitCalibrationModel.from_default()
        self.assertTrue(model.available)
        self.assertEqual(model.path, "tangential.resources/fit_coefs.bin")
        self.assertAlmostEqual(
            model.predict(0.1, 0.1, 100000)[0],
            1.4477653909084447,
            places=12,
        )

    def test_model_environment_override_remains_external_path(self):
        with mock.patch.dict(
            os.environ, {"TANGENTIAL_MODEL_PATH": "/tmp/external-model.bin"}
        ):
            self.assertEqual(default_model_path(), "/tmp/external-model.bin")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(default_model_path())

    def test_default_save_dir_uses_current_working_directory(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                default_save_dir(),
                str((Path.cwd() / "data").resolve()),
            )
        with mock.patch.dict(
            os.environ, {"TANGENTIAL_DATA_DIR": "/tmp/tangential-data"}, clear=True
        ):
            self.assertEqual(default_save_dir(), "/tmp/tangential-data")

    def test_full_config_has_explicit_device_ports_and_builtin_model_default(self):
        config = FullApplicationConfig()
        self.assertEqual(config.pressure_port, "/dev/ttyUSB0")
        self.assertEqual(config.force_port, "/dev/ttyUSB1")
        self.assertIsNone(config.model_path)

    def test_full_session_passes_ports_to_injected_factories(self):
        plot = mock.Mock()
        pressure_calls = []
        force_calls = []

        def pressure_factory(*, port):
            pressure_calls.append(port)
            raise OSError("stop test before opening hardware")

        def force_factory(*, port):
            force_calls.append(port)
            raise AssertionError("force factory must not run")

        config = FullApplicationConfig(
            pressure_port="pressure-test",
            force_port="force-test",
        )
        session = FullAcquisitionSession(
            plot,
            config=config,
            pressure_factory=pressure_factory,
            force_factory=force_factory,
        )
        with self.assertRaisesRegex(RuntimeError, "压力传感器未连接"):
            session.start()
        self.assertEqual(pressure_calls, ["pressure-test"])
        self.assertEqual(force_calls, [])


if __name__ == "__main__":
    unittest.main()
