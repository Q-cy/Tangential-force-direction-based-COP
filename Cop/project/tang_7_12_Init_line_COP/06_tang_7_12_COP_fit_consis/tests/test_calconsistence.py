"""一致性标定的临时数据、运行时数据流和维护者源码入口测试。"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tangential import TangentialFrame, TangentialFrameProcessor
from tangential.config import (
    ConsistenceCalibrationConfig,
    ProcessingConfig,
)
from tangential.processing import calconsistence
from tangential.processing.calconsistence import ConsistenceCalibrator
from tangential.processing.slip import SlipResult, TangentialMotionState


def write_calibration_segment(
    path: Path,
    *,
    endpoint_base: float,
    rows: int = 12,
    omit_channel: int | None = None,
    nan_channel: int | None = None,
    nonpositive_channel: int | None = None,
    early_malformed_row: bool = False,
) -> None:
    """写入带84通道和额外totalSum列的临时量程CSV。"""
    channels = [channel for channel in range(1, 85) if channel != omit_channel]
    headers = [*(f" channel{channel} " for channel in channels), " totalSum "]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        if early_malformed_row:
            writer.writerow(["not-used-by-tail"])
        for row_index in range(rows):
            tail_offset = row_index - max(rows - 10, 0) - 4.5
            values = [endpoint_base + channel + tail_offset for channel in channels]
            if nan_channel in channels and row_index == rows - 1:
                values[channels.index(nan_channel)] = float("nan")
            if nonpositive_channel in channels and row_index >= max(rows - 10, 0):
                values[channels.index(nonpositive_channel)] = 0.0
            writer.writerow([*values, sum(float(value) for value in values)])
        writer.writerow([])


def write_runtime_coefficients(
    path: Path,
    *,
    scale: float,
    offset: float,
) -> None:
    """写入测试运行时使用的最小分段v2系数。"""
    np.savez(
        path,
        format_version=np.array(2, dtype=np.int64),
        input_breakpoints=np.vstack([np.zeros(84), np.full(84, 100.0)]),
        target_breakpoints=np.vstack([np.zeros(84), np.full(84, 200.0)]),
        segment_scale=np.full((1, 84), scale),
        segment_offset=np.full((1, 84), offset),
        segment_values=np.array([100.0]),
    )


class ConsistenceCalibrationTests(unittest.TestCase):
    def test_fit_piecewise_tail_means_metadata_and_no_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "coefficients.npz"
            write_calibration_segment(root / "sensor-100G.csv", endpoint_base=100.0)
            write_calibration_segment(root / "sensor-200G.csv", endpoint_base=200.0)
            write_calibration_segment(root / "sensor-1100G.csv", endpoint_base=400.0)
            config = ConsistenceCalibrationConfig(
                csv_directory=root,
                output_path=output_path,
            )

            calibrator = ConsistenceCalibrator.fit_from_directory(config)
            np.testing.assert_array_equal(calibrator.segment_values, [100, 200, 1100])
            self.assertEqual(calibrator.input_breakpoints.shape, (4, 84))
            self.assertEqual(calibrator.segment_scale.shape, (3, 84))
            expected_raw_endpoints = np.vstack(
                [base + np.arange(1, 85, dtype=np.float64) for base in (100, 200, 400)]
            )
            np.testing.assert_allclose(
                calibrator.metadata["raw_segment_endpoints"], expected_raw_endpoints
            )
            expected_reference_endpoints = np.mean(expected_raw_endpoints, axis=1)
            np.testing.assert_allclose(
                calibrator.metadata["reference_endpoints"],
                expected_reference_endpoints,
            )
            np.testing.assert_allclose(
                calibrator.metadata["fitted_at_source_endpoints"],
                np.broadcast_to(
                    expected_reference_endpoints[:, None], expected_raw_endpoints.shape
                ),
            )
            for index, target in enumerate(calibrator.target_breakpoints[1:], start=1):
                corrected_endpoint = calibrator.apply(
                    calibrator.input_breakpoints[index]
                )
                np.testing.assert_allclose(corrected_endpoint, target)
            raw = np.arange(84, dtype=np.float64)
            before = raw.copy()
            corrected = calibrator.apply(raw)
            np.testing.assert_array_equal(raw, before)
            self.assertEqual(corrected.shape, (84,))

            saved = calibrator.save()
            self.assertEqual(saved, output_path)
            with np.load(saved, allow_pickle=False) as archive:
                self.assertTrue(
                    {
                        "format_version", "input_breakpoints", "target_breakpoints",
                        "segment_scale", "segment_offset", "segment_values",
                        "raw_segment_endpoints", "reference_endpoints",
                        "source_file_names", "source_file_sha256",
                    }.issubset(archive.files)
                )
                self.assertEqual(archive["format_version"].item(), 2)
                self.assertEqual(archive["segment_scale"].shape, (3, 84))
                self.assertEqual(archive["source_file_sha256"].shape, (3,))

            loaded = ConsistenceCalibrator.from_path(
                saved, clip_min=None, clip_max=None
            )
            np.testing.assert_allclose(loaded.apply(raw), corrected)
            with self.assertRaises(FileExistsError):
                calibrator.save()
            calibrator.save(force=True)

            unsafe = ConsistenceCalibrator(
                np.vstack([np.zeros(84), np.ones(84)]),
                np.vstack([np.zeros(84), np.ones(84)]),
                np.ones((1, 84)),
                np.zeros((1, 84)),
                metadata={"unsafe": object()},
            )
            with self.assertRaises(ValueError):
                unsafe.save(root / "unsafe.npz")

    def test_tail_rows_only_and_piecewise_fit_is_monotonic_and_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_calibration_segment(
                root / "sensor-100G.csv",
                endpoint_base=100.0,
                early_malformed_row=True,
            )
            write_calibration_segment(root / "sensor-200G.csv", endpoint_base=200.0)
            # channel1 的高量程端点低于低量程；保序回归后还必须留出最小间距。
            path = root / "sensor-300G.csv"
            write_calibration_segment(path, endpoint_base=300.0)
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            for row in rows[-11:-1]:
                row[0] = "50"
            with path.open("w", encoding="utf-8", newline="") as stream:
                csv.writer(stream).writerows(rows)

            calibrator = ConsistenceCalibrator.fit_from_directory(
                ConsistenceCalibrationConfig(
                    csv_directory=root,
                    output_path=root / "output.npz",
                    minimum_breakpoint_step=1.0,
                    max_segment_scale=100.0,
                )
            )
            np.testing.assert_allclose(
                calibrator.metadata["raw_segment_endpoints"][0],
                100.0 + np.arange(1, 85, dtype=np.float64),
            )
            self.assertTrue(np.all(np.diff(calibrator.input_breakpoints, axis=0) >= 1.0))
            self.assertTrue(
                np.all(np.diff(calibrator.target_breakpoints, axis=0) >= 0.0)
            )
            self.assertLessEqual(float(np.max(calibrator.segment_scale)), 100.0 + 1e-9)
            self.assertGreater(calibrator.metadata["fit_residual_max_abs"][0], 0.0)

            with self.assertRaisesRegex(ValueError, "不能大于100"):
                ConsistenceCalibrationConfig(max_segment_scale=100.1).validate()

            with self.assertRaisesRegex(ValueError, "segment_scale不能大于100"):
                ConsistenceCalibrator(
                    np.vstack([np.zeros(84), np.ones(84)]),
                    np.vstack([np.zeros(84), np.ones(84)]),
                    np.full((1, 84), 100.1),
                    np.zeros((1, 84)),
                )

    def test_project_multisegment_data_reduces_point_to_point_variation(self):
        config = ConsistenceCalibrationConfig()
        calibrator = ConsistenceCalibrator.fit_from_directory(config)
        raw_endpoints = np.asarray(
            calibrator.metadata["raw_segment_endpoints"], dtype=np.float64
        )
        corrected_endpoints = np.stack(
            [calibrator.apply(endpoint) for endpoint in raw_endpoints]
        )
        raw_cv = np.std(raw_endpoints, axis=1) / np.mean(raw_endpoints, axis=1)
        corrected_cv = (
            np.std(corrected_endpoints, axis=1)
            / np.mean(corrected_endpoints, axis=1)
        )

        self.assertEqual(raw_endpoints.shape, (8, 84))
        self.assertTrue(np.all(np.isfinite(corrected_endpoints)))
        self.assertTrue(np.all(corrected_cv < raw_cv))
        self.assertLessEqual(
            float(np.max(calibrator.segment_scale)),
            config.max_segment_scale + 1e-9,
        )

    def test_validation_rejects_bad_directory_and_tail_data(self):
        cases = (
            ("missing channel", {"omit_channel": 84}, "缺少列"),
            ("too few rows", {"rows": 9}, "至少需要10"),
            ("nan", {"nan_channel": 1}, "有限数字"),
            ("nonpositive", {"nonpositive_channel": 1}, "必须大于0"),
        )
        for name, options, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_calibration_segment(
                    root / "sensor-100G.csv", endpoint_base=100.0, **options
                )
                write_calibration_segment(
                    root / "sensor-200G.csv", endpoint_base=200.0
                )
                config = ConsistenceCalibrationConfig(
                    csv_directory=root,
                    output_path=root / "output.npz",
                )
                with self.assertRaisesRegex(ValueError, expected):
                    ConsistenceCalibrator.fit_from_directory(config)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "没有匹配"):
                ConsistenceCalibrator.fit_from_directory(
                    ConsistenceCalibrationConfig(
                        csv_directory=root,
                        output_path=root / "output.npz",
                    )
                )
            bad_name = root / "segment.csv"
            write_calibration_segment(bad_name, endpoint_base=100.0)
            with self.assertRaisesRegex(ValueError, "文件名必须"):
                ConsistenceCalibrator.fit_from_directory(
                    ConsistenceCalibrationConfig(
                        csv_directory=root,
                        output_path=root / "output.npz",
                    )
                )

    def test_legacy_and_corrupt_npz_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "corrupt.npz"
            corrupt.write_bytes(b"not an npz archive")
            with self.assertRaises(ValueError):
                ConsistenceCalibrator.from_path(corrupt)
            legacy = root / "legacy.npz"
            np.savez(legacy, scale=np.ones(84), offset=np.zeros(84))
            with self.assertRaisesRegex(ValueError, "旧单段"):
                ConsistenceCalibrator.from_path(legacy)

    def test_source_main_uses_config_without_argparse_or_hardware(self):
        config = SimpleNamespace(
            csv_directory=Path("/maintainer/segments"),
            output_path=Path("/maintainer/output.npz"),
        )
        calibrator = SimpleNamespace(
            segment_values=np.array([100.0, 200.0]),
            segment_scale=np.ones((2, 84)),
        )
        self.assertNotIn("argparse", vars(calconsistence))

        with mock.patch.object(
            calconsistence,
            "ConsistenceCalibrationConfig",
            return_value=config,
        ) as config_factory, mock.patch.object(
            calconsistence,
            "fit_consistence",
            return_value=calibrator,
        ) as fit, mock.patch(
            "tangential.sensors.pressure.PressureSensor",
            side_effect=AssertionError("维护者标定入口不应打开硬件"),
        ), mock.patch("builtins.print") as output:
            result = calconsistence.main()

        self.assertEqual(result, 0)
        config_factory.assert_called_once_with()
        fit.assert_called_once_with(config)
        printed = "\n".join(str(call.args[0]) for call in output.call_args_list)
        self.assertIn(str(config.csv_directory), printed)
        self.assertIn(str(config.output_path), printed)

    def test_source_main_overwrites_existing_default_output_and_updates_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "coefficients.npz"
            segment = root / "sensor-100G.csv"
            write_calibration_segment(segment, endpoint_base=110.0)
            write_calibration_segment(root / "sensor-200G.csv", endpoint_base=210.0)
            config = ConsistenceCalibrationConfig(
                csv_directory=root,
                output_path=output_path,
            )
            self.assertTrue(config.force)

            with mock.patch.object(
                calconsistence,
                "ConsistenceCalibrationConfig",
                return_value=config,
            ), mock.patch("builtins.print"):
                self.assertEqual(calconsistence.main(), 0)
                with np.load(output_path, allow_pickle=False) as archive:
                    first_scale = archive["segment_scale"].copy()
                    first_source_hash = str(archive["source_combined_sha256"].item())

                write_calibration_segment(segment, endpoint_base=150.0)
                self.assertEqual(calconsistence.main(), 0)
                with np.load(output_path, allow_pickle=False) as archive:
                    second_scale = archive["segment_scale"].copy()
                    second_source_hash = str(archive["source_combined_sha256"].item())

            self.assertFalse(np.array_equal(first_scale, second_scale))
            self.assertNotEqual(first_source_hash, second_source_hash)

    def test_fit_consistence_explicit_force_false_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "coefficients.npz"
            write_calibration_segment(root / "sensor-100G.csv", endpoint_base=100.0)
            write_calibration_segment(root / "sensor-200G.csv", endpoint_base=200.0)
            config = ConsistenceCalibrationConfig(
                csv_directory=root,
                output_path=output_path,
                force=False,
            )

            calconsistence.fit_consistence(config)
            with self.assertRaises(FileExistsError):
                calconsistence.fit_consistence(config)

    def test_user_cli_does_not_expose_maintainer_calibration(self):
        from tangential import cli

        parser = cli._build_parser()
        commands = next(
            action.choices
            for action in parser._actions
            if hasattr(action, "choices") and isinstance(action.choices, dict)
        )
        self.assertNotIn("calconsistence", commands)
        with mock.patch("sys.stderr"):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["calconsistence"])
        self.assertEqual(raised.exception.code, 2)

    def test_fit_function_remains_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_path = root / "output.npz"
            write_calibration_segment(root / "sensor-100G.csv", endpoint_base=100.0)
            write_calibration_segment(root / "sensor-200G.csv", endpoint_base=200.0)
            with mock.patch(
                "tangential.sensors.pressure.PressureSensor",
                side_effect=AssertionError("离线标定不应打开硬件"),
            ):
                calibrator = calconsistence.fit_consistence(
                    ConsistenceCalibrationConfig(
                        csv_directory=root,
                        output_path=output_path,
                    )
                )
            self.assertEqual(calibrator.output_path, output_path)
            self.assertTrue(output_path.is_file())

    def test_runtime_enabled_and_disabled_base_data_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            coefficients = Path(directory) / "runtime.npz"
            write_runtime_coefficients(coefficients, scale=2.0, offset=1.0)
            raw = np.arange(84, dtype=np.float64)
            enabled = ProcessingConfig(
                consistence=ConsistenceCalibrationConfig(
                    enabled=True,
                    coefficients_path=coefficients,
                    clip_min=None,
                    clip_max=None,
                )
            )
            enabled_processor = TangentialFrameProcessor(
                processing_config=enabled,
            )
            internal = enabled_processor._sample_processor._process_sample(raw)
            np.testing.assert_array_equal(internal.raw_data, raw)
            np.testing.assert_array_equal(internal.consistence_data, raw * 2 + 1)
            np.testing.assert_array_equal(internal.base_data, raw * 2 + 1)
            frame = TangentialFrameProcessor(processing_config=enabled).process_frame(raw)
            self.assertEqual(
                [field.name for field in TangentialFrame.__dataclass_fields__.values()],
                ["base_data", "adc_sum", "cop_x", "cop_y", "angle", "dx", "dy", "motion_state"],
            )
            self.assertFalse(hasattr(frame, "raw"))
            np.testing.assert_array_equal(frame.base_data, raw * 2 + 1)

            disabled = ProcessingConfig(
                consistence=ConsistenceCalibrationConfig(enabled=False),
            )
            disabled_processor = TangentialFrameProcessor(
                processing_config=disabled,
            )
            disabled_internal = disabled_processor._sample_processor._process_sample(raw)
            self.assertIsNone(disabled_internal.consistence_data)
            np.testing.assert_array_equal(disabled_internal.base_data, raw)
            np.testing.assert_array_equal(
                TangentialFrameProcessor(processing_config=disabled)
                .process_frame(raw)
                .base_data,
                raw,
            )

    def test_corrected_base_data_drives_algorithms_slip_and_model(self):
        class CopSpy:
            rows = 12
            cols = 7

            def __init__(self):
                self.threshold_input = None
                self.all_input = None
                self.gradient_input = None

            def dynamic_threshold(self, matrix):
                self.threshold_input = np.array(matrix, copy=True)

            def get_all(self, values):
                self.all_input = np.array(values, copy=True)
                return 30.0, 1.5, -2.5, 3.0, 4.0

            def get_origin(self):
                return 1.0, 2.0

            def get_state(self):
                return 1

            def get_gradient(self, values):
                self.gradient_input = np.array(values, copy=True)
                return np.zeros((12, 7, 2), dtype=np.float64)

            @staticmethod
            def _compute_centroid(matrix):
                del matrix
                return 3.0, 4.0

            @staticmethod
            def is_motion_ready():
                return True

        class SlipSpy:
            def __init__(self):
                self.matrix_input = None

            def update(self, matrix, cop_x, cop_y, *, contact, ready):
                del cop_x, cop_y, contact, ready
                self.matrix_input = np.array(matrix, copy=True)
                return SlipResult(motion_state=TangentialMotionState.STICK)

        class CalibrationSpy:
            def __init__(self):
                self.inputs = None

            def predict(self, values):
                self.inputs = np.asarray(values, dtype=np.float64)
                return [1.0, 2.0, 3.0]

        with tempfile.TemporaryDirectory() as directory:
            coefficients = Path(directory) / "runtime.npz"
            write_runtime_coefficients(coefficients, scale=2.0, offset=5.0)
            raw_data = np.arange(84, dtype=np.float64)
            expected = raw_data * 2.0 + 5.0
            cop = CopSpy()
            slip = SlipSpy()
            calibration = CalibrationSpy()
            processor = TangentialFrameProcessor(
                cop_sensor=cop,
                slip_detector=slip,
                calibration=calibration,
                processing_config=ProcessingConfig(
                    consistence=ConsistenceCalibrationConfig(
                        enabled=True,
                        coefficients_path=coefficients,
                        clip_min=None,
                        clip_max=None,
                    )
                ),
            )

            frame = processor.process_frame(raw_data)

            np.testing.assert_array_equal(
                cop.threshold_input, expected.reshape(12, 7)
            )
            np.testing.assert_array_equal(cop.all_input, expected)
            np.testing.assert_array_equal(cop.gradient_input, expected)
            np.testing.assert_array_equal(
                slip.matrix_input, expected.reshape(12, 7)
            )
            self.assertAlmostEqual(calibration.inputs[2], float(np.sum(expected)))
            self.assertAlmostEqual(frame.adc_sum, float(np.sum(expected)))
            np.testing.assert_array_equal(frame.base_data, expected)


if __name__ == "__main__":
    unittest.main()
