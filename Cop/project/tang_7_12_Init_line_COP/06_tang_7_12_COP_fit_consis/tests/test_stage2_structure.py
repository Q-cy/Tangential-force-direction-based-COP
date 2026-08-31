"""结构、配置和公开 API 阶段的回归测试。"""

from __future__ import annotations

import os
import inspect
import subprocess
import sys
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

import tangential
from tangential import (
    CalibrationConfig,
    CopConfig,
    ForceConfig,
    FullApplicationConfig,
    GuiConfig,
    OutputConfig,
    PlotConfig,
    PressureConfig,
    ProcessingConfig,
    SlipConfig,
    SyncConfig,
    TrainingConfig,
)
from tangential.config import ConsistenceCalibrationConfig
from tangential.processing.calconsistence import (
    ConsistenceCalibrator,
    fit_consistence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "src" / "tangential"


class StructureAndConfigTests(unittest.TestCase):
    def test_required_structure_exists_and_old_implementations_are_moved(self):
        for relative in (
            "runtime/sensor.py",
            "runtime/session.py",
            "runtime/synchronization.py",
            "examples/minimal.py",
            "examples/full.py",
            "examples/dual_sensor.py",
            "tools/training.py",
            "tools/plotting.py",
            "application.py",
        ):
            self.assertTrue((PACKAGE_ROOT / relative).is_file(), relative)
        for relative in ("full.py", "training.py", "plotting.py"):
            self.assertFalse((PACKAGE_ROOT / relative).exists(), relative)

    def test_environment_defaults_and_explicit_nested_config_override(self):
        with mock.patch.dict(
            os.environ,
            {
                "TANGENTIAL_PRESSURE_PORT": "/dev/env-pressure",
                "TANGENTIAL_FORCE_PORT": "/dev/env-force",
                "TANGENTIAL_MAX_TIME_DIFF_S": "0.021",
                "TANGENTIAL_DATA_DIR": "/tmp/env-data",
            },
            clear=False,
        ):
            config = FullApplicationConfig()
        self.assertEqual(config.pressure.port, "/dev/env-pressure")
        self.assertEqual(config.force.port, "/dev/env-force")
        self.assertEqual(config.sync.max_time_diff_s, 0.021)
        self.assertEqual(config.output.save_dir, "/tmp/env-data")

        explicit = FullApplicationConfig(
            pressure=PressureConfig(port="explicit-pressure"),
            force=ForceConfig(port="explicit-force"),
            processing=ProcessingConfig(cop=CopConfig(refine_cnt=0)),
            calibration=CalibrationConfig(model_path=None),
            sync=SyncConfig(max_time_diff_s=0.012),
            output=OutputConfig(save_dir="explicit-data"),
            gui=GuiConfig(history_size=12),
        )
        self.assertEqual(explicit.pressure_port, "explicit-pressure")
        self.assertEqual(explicit.force_port, "explicit-force")
        self.assertEqual(explicit.max_time_diff_s, 0.012)
        self.assertEqual(explicit.save_dir, "explicit-data")
        self.assertEqual(explicit.gui.history_size, 12)

    def test_public_api_exports_config_groups_and_application(self):
        import tangential.api
        import tangential.config
        import tangential.runtime

        expected = {
            "TangentialSensorAPI", "TangentialFrame",
            "TangentialFrameProcessor", "FixedTerminalRenderer",
            "FitCalibrationModel", "FullApplicationConfig", "PRSensorAngle",
            "PressureSensor", "compute_vector_angle", "angle_difference",
            "format_terminal_sample", "TrainingConfig", "TrainingResult",
            "train_model", "PlotConfig", "PlotResult", "plot_csv",
            "plot_full_analysis", "PressureConfig", "ForceConfig", "CopConfig",
            "ProcessingConfig", "CalibrationConfig", "SyncConfig", "OutputConfig",
            "GuiConfig", "SlipConfig", "TangentialMotionState", "SlipResult",
            "SlipDetector", "run_application", "run_dual_application",
        }
        self.assertEqual(set(tangential.__all__), expected)
        self.assertEqual(len(tangential.__all__), 32)
        for name in (
            "ConsistenceCalibrationConfig",
            "ConsistenceCalibrator",
            "fit_consistence",
        ):
            self.assertNotIn(name, tangential.__all__)
            self.assertFalse(hasattr(tangential, name))
            self.assertNotIn(name, tangential.api.__all__)
            self.assertFalse(hasattr(tangential.api, name))
        self.assertIs(
            tangential.config.ConsistenceCalibrationConfig,
            ConsistenceCalibrationConfig,
        )
        self.assertTrue(callable(ConsistenceCalibrator.from_config))
        self.assertTrue(callable(fit_consistence))
        self.assertIn("TangentialSensorAPI", tangential.__all__)
        self.assertFalse(hasattr(tangential, "TangentialSensor"))
        self.assertNotIn("TangentialSensor", tangential.api.__all__)
        self.assertFalse(hasattr(tangential.api, "TangentialSensor"))
        self.assertNotIn("TangentialSensor", tangential.runtime.__all__)
        self.assertFalse(hasattr(tangential.runtime, "TangentialSensor"))
        self.assertNotIn("TangentialSample", tangential.__all__)
        self.assertFalse(hasattr(tangential, "TangentialSample"))
        self.assertNotIn("TangentialSampleProcessor", tangential.__all__)
        self.assertFalse(hasattr(tangential, "TangentialSampleProcessor"))
        legacy_projection_name = "to_" + "tangential_" + "frame"
        self.assertNotIn(legacy_projection_name, tangential.__all__)
        self.assertFalse(hasattr(tangential, legacy_projection_name))
        self.assertNotIn("TangentialSample", tangential.api.__all__)
        self.assertFalse(hasattr(tangential.api, "TangentialSample"))
        self.assertNotIn("TangentialSampleProcessor", tangential.api.__all__)
        self.assertFalse(hasattr(tangential.api, "TangentialSampleProcessor"))
        self.assertFalse(hasattr(tangential.api, legacy_projection_name))
        self.assertNotIn("TangentialSample", tangential.runtime.__all__)
        self.assertFalse(hasattr(tangential.runtime, "TangentialSample"))
        self.assertNotIn("TangentialSampleProcessor", tangential.runtime.__all__)
        self.assertFalse(hasattr(tangential.runtime, "TangentialSampleProcessor"))
        self.assertNotIn(
            "sample_processor",
            inspect.signature(tangential.TangentialFrameProcessor).parameters,
        )
        self.assertFalse(hasattr(tangential.runtime, legacy_projection_name))
        sensor_stub = (PACKAGE_ROOT / "runtime" / "sensor.pyi").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("TangentialSample", sensor_stub)
        self.assertNotIn("TangentialSampleProcessor", sensor_stub)
        self.assertNotIn("sample_processor", sensor_stub)
        self.assertNotIn("_process_sample", sensor_stub)
        self.assertIn("def process_frame(", sensor_stub)
        self.assertNotIn("def process(", sensor_stub)
        self.assertIs(tangential.TangentialFrame, tangential.api.TangentialFrame)

    def test_unified_consistence_defaults_resolve_from_source_project(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = ConsistenceCalibrationConfig()
            processing_config = ProcessingConfig()
        csv_directory = Path(config.csv_directory)
        self.assertEqual(
            csv_directory,
            PROJECT_ROOT / "src" / "tangential" / "resources" / "consistence",
        )
        self.assertEqual(config.csv_pattern, "*.csv")
        self.assertEqual(config.tail_rows, 10)
        self.assertEqual(config.minimum_breakpoint_step, 1.0)
        self.assertEqual(config.max_segment_scale, 100.0)
        consistence_fields = {item.name for item in fields(config)}
        for removed_field in (
            "csv_path",
            "state_column",
            "baseline_state",
            "loaded_state",
            "target_min",
            "target_max",
        ):
            self.assertNotIn(removed_field, consistence_fields)
        self.assertEqual(
            Path(config.output_path),
            PROJECT_ROOT
            / "src"
            / "tangential"
            / "resources"
            / "consistence_coeffs.npz",
        )
        self.assertIsNone(config.coefficients_path)
        self.assertTrue(config.force)
        self.assertIsInstance(processing_config.consistence, ConsistenceCalibrationConfig)
        self.assertEqual(
            Path(processing_config.consistence.csv_directory), csv_directory
        )

        with mock.patch.dict(
            os.environ, {"TANGENTIAL_CONSISTENCE_ENABLED": "false"}, clear=False
        ):
            self.assertFalse(ConsistenceCalibrationConfig().enabled)

    def test_base_import_does_not_load_optional_gui_or_plotting(self):
        source = "import sys; import tangential; assert 'pyqtgraph' not in sys.modules; assert 'matplotlib' not in sys.modules"
        result = subprocess.run(
            [sys.executable, "-c", source],
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_user_and_developer_documentation_boundaries(self):
        user_guide = (PROJECT_ROOT / "readme.md").read_text(encoding="utf-8")
        developer_guide = (PROJECT_ROOT / "readme_developer.md").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("TangentialSample", user_guide)
        self.assertNotIn("PYTHONPATH=src", user_guide)
        self.assertNotIn("requirements.txt", user_guide)
        self.assertNotIn("python -m tangential.examples", user_guide)
        self.assertNotIn("Cython", user_guide)
        for sensitive_term in (
            "consistence",
            "consistency",
            "calconsistence",
            "coeffs",
            "一致性标定",
            "一致性系数",
        ):
            self.assertNotIn(sensitive_term.lower(), user_guide.lower())
        for name in (
            "FitCalibrationModel", "PRSensorAngle", "PressureSensor",
            "SlipDetector", "SlipResult", "compute_vector_angle",
            "angle_difference", "FixedTerminalRenderer",
            "format_terminal_sample",
        ):
            self.assertNotIn(name, user_guide)
            self.assertIn(name, developer_guide)

        self.assertTrue(
            developer_guide.startswith(
                "# Tangential Sensor SDK 0.5.0 开发者维护指南"
            )
        )
        self.assertIn("TangentialSample", developer_guide)
        self.assertIn("_process_sample", developer_guide)
        self.assertIn("PYTHONPATH=src", developer_guide)
        self.assertIn("requirements.txt", developer_guide)
        for maintainer_detail in (
            "ConsistenceCalibrationConfig",
            "enabled",
            "csv_directory",
            "csv_pattern",
            "tail_rows",
            "minimum_breakpoint_step",
            "max_segment_scale",
            "output_path",
            "coefficients_path",
            "clip_min",
            "clip_max",
            "force",
            "PYTHONPATH=src python -m tangential.processing.calconsistence",
            "TANGENTIAL_CONSISTENCE_ENABLED",
            "run_dual_application(config_a, config_b)",
        ):
            self.assertIn(maintainer_detail, developer_guide)
        self.assertIn("## 1. 开发目标与不可破坏边界", developer_guide)
        self.assertIn("## 23. 修改完成的定义", developer_guide)
        self.assertNotIn("第一部分：", developer_guide)
        self.assertNotIn("第二部分：", developer_guide)
        self.assertNotIn("严格内容超集", developer_guide)

        for document in (user_guide, developer_guide):
            self.assertNotIn("Rust", document)
            self.assertNotIn("rust", document)
            self.assertNotIn("<td>", document)
            self.assertEqual(
                document.count("<td"),
                document.count('<td style="white-space:normal">'),
            )
            for table in document.split("<table>")[1:]:
                self.assertIn('<th style="min-width:180px">', table.split("</table>", 1)[0])

    def test_example_modules_are_importable_without_running_hardware(self):
        from tangential.examples import dual_sensor, full, minimal

        self.assertTrue(callable(minimal.run))
        self.assertTrue(callable(full.main))
        self.assertTrue(callable(dual_sensor.run))
        self.assertTrue(callable(tangential.run_application))
        self.assertTrue(callable(tangential.run_dual_application))

    def test_low_level_defaults_come_from_grouped_configs(self):
        from tangential.gui.realtime import RealTimePlot
        from tangential.processing.cop import PRSensorAngle
        from tangential.sensors.force import SixAxisForceSensor
        from tangential.sensors.pressure import PressureSensor

        pressure = inspect.signature(PressureSensor).parameters
        force = inspect.signature(SixAxisForceSensor).parameters
        cop = inspect.signature(PRSensorAngle).parameters
        self.assertIsNone(pressure["period_s"].default)
        self.assertIsNone(pressure["baudrate"].default)
        self.assertIsNone(force["period_s"].default)
        self.assertIsNone(force["baudrate"].default)
        self.assertIsNone(cop["total_threshold_factor"].default)
        self.assertIn("config", inspect.signature(RealTimePlot).parameters)

    def test_invalid_environment_configuration_is_not_silently_ignored(self):
        with mock.patch.dict(
            os.environ,
            {"TANGENTIAL_PRESSURE_HZ": "not-a-number"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "TANGENTIAL_PRESSURE_HZ"):
                PressureConfig()


if __name__ == "__main__":
    unittest.main()
