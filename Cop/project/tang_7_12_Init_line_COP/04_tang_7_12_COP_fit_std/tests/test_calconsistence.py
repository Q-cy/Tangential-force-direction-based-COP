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


def write_calibration_csv(
    path: Path,
    *,
    channel_count: int = 84,
    baseline_rows: int = 2,
    loaded_rows: int = 2,
    omit_channel: int | None = None,
    include_loaded: bool = True,
    nan_channel: int | None = None,
    nonpositive_channel: int | None = None,
    unrelated_nan_row: bool = False,
    loaded_base: float = 110.0,
) -> None:
    """写入只存在于临时目录的最小两状态标定 CSV。"""
    channels = [
        channel for channel in range(1, channel_count + 1)
        if channel != omit_channel
    ]
    headers = ["  CoP_state  ", *(f" ch{channel} " for channel in channels)]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        for row_index in range(baseline_rows):
            values = [10.0 + channel + row_index for channel in channels]
            if nan_channel in channels and row_index == 0:
                values[channels.index(nan_channel)] = float("nan")
            writer.writerow([0, *values])
        if unrelated_nan_row:
            writer.writerow([1, *([float("nan")] * len(channels))])
        if include_loaded:
            for row_index in range(loaded_rows):
                values = [loaded_base + channel + row_index for channel in channels]
                if nonpositive_channel in channels:
                    index = channels.index(nonpositive_channel)
                    values[index] = 10.0 + nonpositive_channel + row_index
                writer.writerow([2, *values])


class ConsistenceCalibrationTests(unittest.TestCase):
    def test_fit_affine_mapping_metadata_and_no_input_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "temporary-calibration.csv"
            output_path = root / "coefficients.npz"
            write_calibration_csv(csv_path)
            config = ConsistenceCalibrationConfig(
                csv_path=csv_path,
                output_path=output_path,
                target_min=100.0,
                target_max=200.0,
            )

            calibrator = ConsistenceCalibrator.fit_from_csv(config)
            self.assertTrue(np.allclose(calibrator.scale, 1.0))
            self.assertAlmostEqual(calibrator.offset[0], 88.5)
            raw = np.arange(84, dtype=np.float64)
            before = raw.copy()
            corrected = calibrator.apply(raw)
            np.testing.assert_array_equal(raw, before)
            np.testing.assert_allclose(corrected, raw + calibrator.offset)

            saved = calibrator.save()
            self.assertEqual(saved, output_path)
            with np.load(saved, allow_pickle=False) as archive:
                self.assertTrue(
                    {
                        "scale", "offset", "states", "targets", "sample_counts",
                        "source_sha256", "state_column", "channel_count",
                    }.issubset(archive.files)
                )
                self.assertEqual(archive["scale"].shape, (84,))
                self.assertEqual(archive["offset"].shape, (84,))
                np.testing.assert_array_equal(archive["states"], [0, 2])
                np.testing.assert_array_equal(archive["targets"], [100.0, 200.0])
                np.testing.assert_array_equal(archive["sample_counts"], [2, 2])
                self.assertEqual(archive["state_column"].item(), "CoP_state")
                self.assertEqual(len(str(archive["source_sha256"].item())), 64)

            loaded = ConsistenceCalibrator.from_path(
                saved, clip_min=None, clip_max=None
            )
            np.testing.assert_allclose(loaded.apply(raw), corrected)
            with self.assertRaises(FileExistsError):
                calibrator.save()
            calibrator.save(force=True)

            unsafe = ConsistenceCalibrator(
                np.ones(84),
                np.zeros(84),
                metadata={"unsafe": object()},
            )
            with self.assertRaises(ValueError):
                unsafe.save(root / "unsafe.npz")

    def test_validation_rejects_missing_columns_states_nan_and_nonpositive_span(self):
        cases = (
            # 动态发现以文件中最大的连续通道号为 N；因此缺少末尾 ch84
            # 不再是“固定84通道”错误，缺少中间编号才明确构成非连续表头。
            ("non-contiguous channel", {"omit_channel": 2}, "连续"),
            ("missing loaded state", {"include_loaded": False}, "state=2"),
            ("nan", {"nan_channel": 1}, "有限数字"),
            ("nonpositive span", {"nonpositive_channel": 1}, "严格大于"),
        )
        for name, options, expected in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                csv_path = Path(directory) / "input.csv"
                write_calibration_csv(csv_path, **options)
                config = ConsistenceCalibrationConfig(
                    csv_path=csv_path,
                    output_path=Path(directory) / "output.npz",
                )
                with self.assertRaisesRegex(ValueError, expected):
                    ConsistenceCalibrator.fit_from_csv(config)

    def test_dynamic_15_channel_fit_save_and_load(self):
        """一致性标定应从 ch1...ch15 自动得到 15 通道系数。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "input-15.csv"
            output_path = root / "coefficients-15.npz"
            write_calibration_csv(csv_path, channel_count=15)
            config = ConsistenceCalibrationConfig(
                csv_path=csv_path,
                output_path=output_path,
                target_min=100.0,
                target_max=200.0,
            )

            calibrator = ConsistenceCalibrator.fit_from_csv(config)
            self.assertEqual(calibrator.channel_count, 15)
            self.assertEqual(calibrator.scale.shape, (15,))
            self.assertEqual(calibrator.offset.shape, (15,))
            raw = np.arange(15, dtype=np.float64)
            corrected = calibrator.apply(raw)
            saved = calibrator.save(force=False)
            with np.load(saved, allow_pickle=False) as archive:
                self.assertEqual(int(archive["channel_count"]), 15)
                self.assertEqual(archive["scale"].shape, (15,))
            loaded = ConsistenceCalibrator.from_path(
                saved, clip_min=None, clip_max=None
            )
            self.assertEqual(loaded.channel_count, 15)
            np.testing.assert_allclose(loaded.apply(raw), corrected)

    def test_non_contiguous_channel_headers_are_rejected(self):
        """ch1、ch2、ch4 这类跳号表头不能被静默解释为三通道。"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "non-contiguous.csv"
            write_calibration_csv(path, channel_count=4, omit_channel=3)
            with self.assertRaisesRegex(ValueError, "连续"):
                ConsistenceCalibrator.fit_from_csv(
                    ConsistenceCalibrationConfig(
                        csv_path=path,
                        output_path=Path(directory) / "output.npz",
                    )
                )

    def test_unrelated_state_nan_is_ignored_but_selected_state_nan_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ignored_path = root / "ignored-state.csv"
            write_calibration_csv(ignored_path, unrelated_nan_row=True)
            calibrator = ConsistenceCalibrator.fit_from_csv(
                ConsistenceCalibrationConfig(
                    csv_path=ignored_path,
                    output_path=root / "ignored-state.npz",
                )
            )
            self.assertEqual(calibrator.metadata["baseline_count"], 2)
            self.assertEqual(calibrator.metadata["loaded_count"], 2)

            selected_path = root / "selected-state.csv"
            write_calibration_csv(selected_path, nan_channel=1)
            with self.assertRaisesRegex(ValueError, "有限数字"):
                ConsistenceCalibrator.fit_from_csv(
                    ConsistenceCalibrationConfig(
                        csv_path=selected_path,
                        output_path=root / "selected-state.npz",
                    )
                )

    def test_corrupt_npz_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "corrupt.npz"
            corrupt.write_bytes(b"not an npz archive")
            with self.assertRaises(ValueError):
                ConsistenceCalibrator.from_path(corrupt)

    def test_source_main_uses_config_without_argparse_or_hardware(self):
        config = SimpleNamespace(
            csv_path=Path("/maintainer/input.csv"),
            output_path=Path("/maintainer/output.npz"),
        )
        calibrator = SimpleNamespace(scale=np.ones(84))
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
        self.assertIn(str(config.csv_path), printed)
        self.assertIn(str(config.output_path), printed)

    def test_source_main_overwrites_existing_default_output_and_updates_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "input.csv"
            output_path = root / "coefficients.npz"
            write_calibration_csv(csv_path, loaded_base=110.0)
            config = ConsistenceCalibrationConfig(
                csv_path=csv_path,
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
                    first_scale = archive["scale"].copy()
                    first_source_hash = str(archive["source_sha256"].item())

                write_calibration_csv(csv_path, loaded_base=210.0)
                self.assertEqual(calconsistence.main(), 0)
                with np.load(output_path, allow_pickle=False) as archive:
                    second_scale = archive["scale"].copy()
                    second_source_hash = str(archive["source_sha256"].item())

            self.assertFalse(np.array_equal(first_scale, second_scale))
            self.assertNotEqual(first_source_hash, second_source_hash)

    def test_fit_consistence_explicit_force_false_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "input.csv"
            output_path = root / "coefficients.npz"
            write_calibration_csv(csv_path)
            config = ConsistenceCalibrationConfig(
                csv_path=csv_path,
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
            csv_path = root / "input.csv"
            output_path = root / "output.npz"
            write_calibration_csv(csv_path)
            with mock.patch(
                "tangential.sensors.pressure.PressureSensor",
                side_effect=AssertionError("离线标定不应打开硬件"),
            ):
                calibrator = calconsistence.fit_consistence(
                    ConsistenceCalibrationConfig(
                        csv_path=csv_path,
                        output_path=output_path,
                    )
                )
            self.assertEqual(calibrator.output_path, output_path)
            self.assertTrue(output_path.is_file())

    def test_runtime_enabled_and_disabled_base_data_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            coefficients = Path(directory) / "runtime.npz"
            np.savez(
                coefficients,
                scale=np.full(84, 2.0),
                offset=np.full(84, 1.0),
            )
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
            np.savez(
                coefficients,
                scale=np.full(84, 2.0),
                offset=np.full(84, 5.0),
            )
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
