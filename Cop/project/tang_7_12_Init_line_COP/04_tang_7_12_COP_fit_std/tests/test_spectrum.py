"""精简 CoP 速度 STFT、功率占比、GUI 和 NPZ 回归测试。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from pyqtgraph.Qt import QtWidgets

from tangential.config import SpectrumConfig
from tangential.gui.spectrum import SpectrumWindow
from tangential.processing.spectrum import CopSpectrumAnalyzer, SpectralFrictionState


def _feed(analyzer, count, signal=None, start_t=0.0, state=2):
    snapshots = []
    for index in range(count):
        timestamp = start_t + index / analyzer.sample_rate_hz
        x_value, y_value = (0.0, 0.0) if signal is None else signal(timestamp)
        snapshot = analyzer.process(timestamp, x_value, y_value, state)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def _target_signal(timestamp):
    return np.sin(2.0 * np.pi * 26.0 * timestamp), 0.0


def _outside_signal(timestamp):
    return np.sin(2.0 * np.pi * 12.0 * timestamp), 0.0


class SpectrumAnalyzerTests(unittest.TestCase):
    def test_defaults_axis_and_removed_runtime_configuration(self):
        with mock.patch.dict(os.environ, {
            "TANGENTIAL_SPECTRUM_SLIP_BAND_POWER_RATIO_THRESHOLD": "0.3",
            "TANGENTIAL_SPECTRUM_ENTER_WINDOWS": "3",
            "TANGENTIAL_SPECTRUM_EXIT_WINDOWS": "5",
        }, clear=False):
            config = SpectrumConfig().validate()
        self.assertEqual(config.window_duration_s, 0.5)
        self.assertEqual(config.window_samples, 80)
        self.assertEqual(config.required_samples, 81)
        self.assertEqual(config.slip_band_hz, (24.0, 28.0))
        self.assertEqual(config.slip_band_power_ratio_threshold, 0.3)
        self.assertEqual(config.enter_windows, 3)
        self.assertEqual(config.exit_windows, 5)
        self.assertEqual(config.baseline_duration_s, 1.0)
        self.assertEqual(config.baseline_power_floor, 1e-6)
        analyzer = CopSpectrumAnalyzer(config)
        np.testing.assert_allclose(analyzer.frequencies_hz, np.arange(2.0, 72.0, 2.0))
        removed = (
            "ignored_frequency_bands_hz",
            "score2_target_band_hz", "score2_fraction_threshold", "score2_enter_windows",
            "score2_exit_windows", "slip_enter_score", "slip_exit_score",
            "auxiliary_high_frequency_min_hz", "auxiliary_high_frequency_max_hz",
            "activity_start_db", "active_fraction_reference", "entropy_reference",
            "flatness_reference", "flux_reference_db", "peak_concentration_reference",
        )
        for name in removed:
            self.assertFalse(hasattr(config, name), name)

    def test_snapshot_contains_only_simplified_runtime_fields(self):
        snapshot = _feed(CopSpectrumAnalyzer(), 120, _target_signal)[-1]
        for name in (
            "slip_score", "slip_score2", "target_band_power_fraction",
            "target_band_peak_prominence_db", "speed_cv", "spectral_centroid_hz",
            "high_frequency_power_fraction",
            "valid_frequency_mask", "motion_level_db", "spectral_entropy",
        ):
            self.assertFalse(hasattr(snapshot, name), name)
        np.testing.assert_allclose(
            snapshot.velocity_amplitude_combined,
            np.hypot(snapshot.velocity_amplitude_x, snapshot.velocity_amplitude_y),
            rtol=1e-6,
        )
        self.assertEqual(snapshot.relative_power_db.shape, snapshot.frequency_hz.shape)
        self.assertEqual(snapshot.baseline_power.shape, snapshot.frequency_hz.shape)

    def test_ratio_is_target_power_over_complete_analysis_power(self):
        analyzer = CopSpectrumAnalyzer()
        amplitude_x = np.arange(1, analyzer.frequencies_hz.size + 1, dtype=float)
        amplitude_y = amplitude_x[::-1] * 0.5
        power = amplitude_x ** 2 + amplitude_y ** 2
        low, high = analyzer.config.slip_band_hz
        target = (analyzer.frequencies_hz >= low) & (analyzer.frequencies_hz <= high)
        expected = float(np.sum(power[target]) / np.sum(power))
        self.assertAlmostEqual(analyzer._power_ratio(amplitude_x, amplitude_y), expected)

    def test_first_complete_window_is_stick_without_baseline_wait(self):
        analyzer = CopSpectrumAnalyzer(SpectrumConfig(enter_windows=1))
        snapshots = _feed(analyzer, 81, _target_signal)
        self.assertGreater(len(snapshots), 0)
        self.assertEqual(snapshots[0].friction_state, SpectralFrictionState.STICK)
        self.assertGreaterEqual(snapshots[0].slip_band_power_ratio, snapshots[0].threshold)
        self.assertNotEqual(snapshots[0].state_name, "WAITING")
        self.assertFalse(snapshots[0].baseline_established)
        self.assertTrue(np.all(np.isnan(snapshots[0].relative_power_db)))
        self.assertEqual(analyzer._enter_count, 1)

    def test_baseline_is_side_channel_and_freezes_after_duration(self):
        analyzer = CopSpectrumAnalyzer(
            SpectrumConfig(baseline_duration_s=0.1).validate()
        )
        snapshots = _feed(analyzer, 130, _target_signal)
        self.assertEqual(snapshots[0].friction_state, SpectralFrictionState.STICK)
        self.assertFalse(snapshots[0].baseline_established)
        established = [item for item in snapshots if item.baseline_established]
        self.assertGreater(len(established), 0)
        frozen = analyzer.baseline_power
        self.assertIsNotNone(frozen)
        self.assertTrue(np.all(np.isfinite(established[-1].relative_power_db)))
        _feed(analyzer, 80, _outside_signal, start_t=1.0)
        np.testing.assert_allclose(analyzer.baseline_power, frozen)

    def test_gap_preserves_frozen_baseline_but_discards_partial_baseline(self):
        frozen_analyzer = CopSpectrumAnalyzer(
            SpectrumConfig(baseline_duration_s=0.1).validate()
        )
        _feed(frozen_analyzer, 130, _target_signal)
        frozen = frozen_analyzer.baseline_power
        frozen_analyzer.process(2.0, 0.0, 0.0, 2)
        np.testing.assert_allclose(frozen_analyzer.baseline_power, frozen)
        self.assertEqual(frozen_analyzer.friction_state, SpectralFrictionState.WAITING)

        partial = CopSpectrumAnalyzer()
        _feed(partial, 100, _target_signal)
        self.assertIsNone(partial.baseline_power)
        self.assertGreater(len(partial._baseline_power_samples), 0)
        partial.process(2.0, 0.0, 0.0, 2)
        self.assertIsNone(partial.baseline_power)
        self.assertEqual(partial._baseline_power_samples, [])
        self.assertIsNone(partial._baseline_start_t)

    def test_contact_end_clears_frozen_baseline(self):
        analyzer = CopSpectrumAnalyzer(
            SpectrumConfig(baseline_duration_s=0.1).validate()
        )
        _feed(analyzer, 130, _target_signal)
        self.assertIsNotNone(analyzer.baseline_power)
        analyzer.process(1.0, 0.0, 0.0, 0)
        self.assertIsNone(analyzer.baseline_power)

    def test_same_threshold_boundary_and_time_hysteresis(self):
        config = SpectrumConfig(enter_windows=2, exit_windows=2).validate()
        analyzer = CopSpectrumAnalyzer(config)
        analyzer._friction_state = SpectralFrictionState.STICK
        threshold = config.slip_band_power_ratio_threshold
        analyzer._update_state(threshold)
        self.assertEqual(analyzer.friction_state, SpectralFrictionState.STICK)
        analyzer._update_state(threshold)
        self.assertEqual(analyzer.friction_state, SpectralFrictionState.SLIP)
        analyzer._update_state(threshold)
        self.assertEqual(analyzer.friction_state, SpectralFrictionState.SLIP)
        analyzer._update_state(np.nextafter(threshold, 0.0))
        self.assertEqual(analyzer.friction_state, SpectralFrictionState.SLIP)
        analyzer._update_state(np.nextafter(threshold, 0.0))
        self.assertEqual(analyzer.friction_state, SpectralFrictionState.STICK)

    def test_invalid_state_and_large_gap_require_a_new_full_window(self):
        analyzer = CopSpectrumAnalyzer()
        _feed(analyzer, 100, _outside_signal)
        analyzer.process(1.0, 0.0, 0.0, 0)
        self.assertEqual(analyzer.friction_state, SpectralFrictionState.WAITING)
        self.assertEqual(analyzer.ready_samples, 0)
        analyzer.process(2.0, 0.0, 0.0, 2)
        analyzer.process(2.2, 0.0, 0.0, 2)
        self.assertEqual(analyzer.friction_state, SpectralFrictionState.WAITING)
        self.assertEqual(analyzer.ready_samples, 1)

    def test_npz_uses_only_simplified_schema(self):
        analyzer = CopSpectrumAnalyzer()
        snapshots = _feed(analyzer, 300, _target_signal)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new_session_spectrum.npz"
            self.assertTrue(analyzer.save_npz(path, "new_session.csv"))
            with np.load(path, allow_pickle=False) as archive:
                expected = {
                    "frequency_hz", "spectrum_time_s", "velocity_amplitude_x",
                    "velocity_amplitude_y", "velocity_amplitude_combined",
                    "baseline_power", "relative_power_db", "baseline_established",
                    "slip_band_power_ratio", "friction_state", "threshold",
                    "sample_rate_hz", "window_duration_s", "update_interval_s",
                    "analysis_frequency_hz", "slip_band_hz", "enter_windows",
                    "exit_windows", "baseline_duration_s", "baseline_power_floor",
                    "max_gap_s", "required_cop_state",
                    "window_name", "csv_file_name",
                }
                self.assertEqual(set(archive.files), expected)
                np.testing.assert_allclose(
                    archive["slip_band_power_ratio"],
                    [snapshot.slip_band_power_ratio for snapshot in snapshots],
                )
                self.assertEqual(archive["friction_state"].dtype, np.int8)
                self.assertEqual(archive["baseline_established"].dtype, np.bool_)
                self.assertTrue(np.all(np.isnan(archive["relative_power_db"][0])))
                self.assertTrue(archive["baseline_established"][-1])
                self.assertTrue(np.all(np.isfinite(archive["relative_power_db"][-1])))

    def test_config_validation_and_environment(self):
        invalid = (
            {"slip_band_power_ratio_threshold": 0.0},
            {"slip_band_power_ratio_threshold": 1.0},
            {"slip_band_hz": (24.0, 24.0)},
            {"analysis_max_frequency_hz": 81.0},
            {"window_duration_s": 0.333},
            {"baseline_duration_s": 0.0},
            {"baseline_power_floor": 0.0},
            {"enter_windows": 0},
            {"exit_windows": 0},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                SpectrumConfig(**values).validate()
        with mock.patch.dict(os.environ, {
            "TANGENTIAL_SPECTRUM_SLIP_BAND_HZ": "22:30",
            "TANGENTIAL_SPECTRUM_SLIP_BAND_POWER_RATIO_THRESHOLD": "0.4",
        }, clear=False):
            config = SpectrumConfig().validate()
        self.assertEqual(config.slip_band_hz, (22.0, 30.0))
        self.assertEqual(config.slip_band_power_ratio_threshold, 0.4)


class SpectrumGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_gui_keeps_relative_plot_and_relative_db_waterfall(self):
        analyzer = CopSpectrumAnalyzer()
        latest = _feed(analyzer, 300, _target_signal)[-1]
        window = SpectrumWindow(analyzer.config)
        window.submit(latest)
        window.refresh()
        self.assertIn("slip_band_power_ratio=", window.friction_label.text())
        self.assertIn("threshold=", window.friction_label.text())
        self.assertIn(
            f"enter={analyzer.config.enter_windows}, "
            f"exit={analyzer.config.exit_windows}",
            window.friction_label.text(),
        )
        self.assertTrue(hasattr(window, "relative_plot"))
        self.assertFalse(hasattr(window, "feature_label"))
        self.assertFalse(hasattr(window, "target_band_label"))
        np.testing.assert_allclose(window._curve_combined.yData, latest.velocity_amplitude_combined)
        np.testing.assert_allclose(
            window._relative_curve.yData,
            latest.relative_power_db,
            equal_nan=True,
        )
        self.assertIn("相对基线已冻结", window.status_label.text())
        self.assertIsNotNone(window._waterfall_image.image)
        window.close()


if __name__ == "__main__":
    unittest.main()
