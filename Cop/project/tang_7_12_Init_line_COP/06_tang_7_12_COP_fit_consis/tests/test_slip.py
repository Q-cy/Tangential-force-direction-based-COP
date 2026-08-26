"""滑移检测的 C++ 语义等价测试。"""

from __future__ import annotations

import unittest

import numpy as np

from tangential import (
    CopConfig,
    ProcessingConfig,
    SlipConfig,
    SlipDetector,
    SlipResult,
    TangentialFrameProcessor,
    TangentialMotionState,
    TangentialSensorAPI,
)
from tangential.config import ConsistenceCalibrationConfig
from tangential.processing.cop import PRSensorAngle
from tangential.runtime.sensor import TangentialSampleProcessor


def point_frame(row: int, col: int, rows: int = 5, cols: int = 5) -> np.ndarray:
    frame = np.zeros((rows, cols), dtype=np.float64)
    frame[row, col] = 100.0
    return frame


class SlipDetectorTests(unittest.TestCase):
    def make_detector(self, **overrides):
        values = dict(
            window_frames=3,
            enter_distance=0.3,
            exit_distance=0.05,
            reanchor_distance=0.8,
            enter_frames=2,
            exit_frames=3,
            direction_smoothing=1.0,
            patch_search_radius=2,
            patch_min_correlation=0.8,
            patch_min_improvement=0.1,
        )
        values.update(overrides)
        return SlipDetector(SlipConfig(**values), rows=5, cols=5)

    def test_static_patch_stays_stick_and_no_contact_resets(self):
        detector = self.make_detector()
        frame = point_frame(2, 2)
        results = [detector.update(frame, 2.0, 2.0, True) for _ in range(20)]
        self.assertTrue(all(result.motion_state is TangentialMotionState.STICK for result in results))
        self.assertTrue(all(not result.is_slipping for result in results))
        reset = detector.update(np.zeros_like(frame), 0.0, 0.0, False)
        self.assertEqual(reset.motion_state, TangentialMotionState.NO_CONTACT)
        self.assertIsNone(detector.anchor[0])

    def test_contact_before_refinement_holds_stick_without_history(self):
        detector = self.make_detector()
        frame = point_frame(2, 2)
        for _ in range(10):
            result = detector.update(frame, 2.0, 2.0, True, ready=False)
        self.assertEqual(result.motion_state, TangentialMotionState.STICK)
        self.assertFalse(result.is_slipping)
        self.assertEqual(len(detector._history), 0)
        self.assertEqual(detector.anchor, (2.0, 2.0))

    def test_downward_patch_enters_slip_and_exit_reanchors(self):
        detector = self.make_detector()
        detector.update(point_frame(1, 2), 2.0, 1.0, True)
        detector.update(point_frame(1, 2), 2.0, 1.0, True)
        detector.update(point_frame(2, 2), 2.0, 2.0, True)
        result = detector.update(point_frame(3, 2), 2.0, 3.0, True)
        self.assertEqual(result.motion_state, TangentialMotionState.SLIP)
        self.assertGreaterEqual(result.confidence, 0.8)
        self.assertGreater(result.motion_distance, 0.3)
        self.assertEqual((result.patch_row_shift, result.patch_col_shift), (2, 0))
        # 当前项目角度约定：+row/+y 经过 _compute_cop_angle 为 270°。
        angle = PRSensorAngle._compute_cop_angle(result.direction_x, result.direction_y)
        self.assertAlmostEqual(angle, 270.0, places=5)

        stopped = result
        reanchored_results = []
        for _ in range(4):
            stopped = detector.update(point_frame(3, 2), 2.0, 3.0, True)
            reanchored_results.append(stopped)
        self.assertEqual(stopped.motion_state, TangentialMotionState.STICK)
        self.assertTrue(any(item.reanchored for item in reanchored_results))
        self.assertEqual(stopped.motion_distance, 0.0)
        self.assertEqual(detector.anchor, (2.0, 3.0))

        after = detector.update(point_frame(3, 3), 3.0, 3.0, True)
        self.assertEqual(after.motion_state, TangentialMotionState.STICK)
        self.assertAlmostEqual(detector.anchor[0], 2.0)

    def test_cop_only_motion_needs_patch_or_large_fallback(self):
        detector = self.make_detector()
        frame = point_frame(2, 2)
        detector.update(frame, 2.0, 2.0, True)
        for cop_x in (2.2, 2.4, 2.6, 2.8):
            result = detector.update(frame, cop_x, 2.0, True)
        self.assertFalse(result.is_slipping)

        fallback = self.make_detector(enter_frames=1, reanchor_distance=1.0)
        fallback.update(frame, 2.0, 2.0, True)
        fallback.update(frame, 2.0, 2.0, True)
        result = fallback.update(frame, 3.2, 2.0, True)
        self.assertTrue(result.is_slipping)

    def test_direction_uses_ema_after_entering_slip(self):
        detector = self.make_detector(
            enter_frames=1, direction_smoothing=0.5,
            reanchor_distance=5.0,
        )
        detector.update(point_frame(1, 2), 2.0, 1.0, True)
        detector.update(point_frame(1, 2), 2.0, 1.0, True)
        detector.update(point_frame(2, 2), 2.0, 2.0, True)
        entered = detector.update(point_frame(3, 2), 2.0, 3.0, True)
        self.assertTrue(entered.is_slipping)
        detector.update(point_frame(3, 3), 3.0, 3.0, True)
        updated = detector.update(point_frame(3, 3), 3.0, 3.0, True)
        self.assertTrue(updated.is_slipping)
        self.assertGreater(updated.direction_y, 0.0)
        self.assertGreater(updated.direction_x, 0.0)
        self.assertLess(updated.direction_x, 1.0)

    def test_result_is_immutable(self):
        with self.assertRaises((AttributeError, TypeError)):
            SlipResult().confidence = 1.0

    def test_processors_have_independent_detectors(self):
        config = ProcessingConfig(cop=CopConfig(
            collect_frames=1, refine_cnt=0
        ), slip=self.make_detector().config,
            consistence=ConsistenceCalibrationConfig(enabled=False),
        )
        first = TangentialFrameProcessor(processing_config=config, calibration=None)
        second = TangentialFrameProcessor(processing_config=config, calibration=None)
        self.assertIsNot(
            first._sample_processor.slip_detector,
            second._sample_processor.slip_detector,
        )
        self.assertIsNot(
            first._sample_processor.slip_detector._history,
            second._sample_processor.slip_detector._history,
        )

    def test_sensor_api_passes_custom_slip_config_to_processor(self):
        processing = ProcessingConfig(
            cop=CopConfig(rows=5, cols=5, collect_frames=1, refine_cnt=0),
            slip=SlipConfig(window_frames=7, angle_deadband=0.42),
            consistence=ConsistenceCalibrationConfig(enabled=False),
        )
        api = TangentialSensorAPI(
            sensor=object(),
            processing_config=processing,
        )
        self.assertIsInstance(api.processor._sample_processor, TangentialSampleProcessor)
        self.assertEqual(api.processor._sample_processor.slip_config.window_frames, 7)
        self.assertEqual(api.processor._sample_processor.slip_config.angle_deadband, 0.42)
        self.assertIs(api.processor._sample_processor.slip_config, processing.slip)

    def test_reanchor_preserves_cop_refinement_state(self):
        stick = PRSensorAngle(config=CopConfig(
            rows=5, cols=5, collect_frames=1, refine_cnt=0
        ))
        for frame in (np.zeros((5, 5)), point_frame(2, 2)):
            stick.dynamic_threshold(frame)
            stick.get_all(frame.reshape(-1))
        stick.reanchor_origin(2.0, 2.0)
        self.assertEqual(stick.get_state(), 1)

        refined = PRSensorAngle(
            config=CopConfig(
                rows=5, cols=5, collect_frames=1,
                refine_cnt=2, refine_distance=0.1,
            )
        )
        frame = point_frame(2, 2).reshape(-1)
        for value in (np.zeros((5, 5)), point_frame(2, 2), point_frame(2, 2)):
            refined.dynamic_threshold(value)
            refined.get_all(value.reshape(-1))
        self.assertEqual(refined.get_state(), 2)
        refined.reanchor_origin(2.0, 2.0)
        self.assertEqual(refined.get_state(), 2)

    def test_frame_exposes_motion_state_and_applies_angle_deadband(self):
        processing = ProcessingConfig(
            cop=CopConfig(rows=5, cols=5, collect_frames=1, refine_cnt=0),
            slip=SlipConfig(angle_deadband=0.1),
            consistence=ConsistenceCalibrationConfig(enabled=False),
        )
        processor = TangentialFrameProcessor(
            processing_config=processing, calibration=None
        )
        empty = np.zeros((5, 5))
        center = point_frame(2, 2)
        slight = center.copy()
        slight[2, 3] = 5.0
        processor.process_frame(empty.reshape(-1))
        first = processor.process_frame(center.reshape(-1))
        second = processor.process_frame(slight.reshape(-1))
        self.assertEqual(first.motion_state, TangentialMotionState.STICK)
        self.assertEqual(second.motion_state, TangentialMotionState.STICK)
        self.assertEqual(second.angle, 0.0)


if __name__ == "__main__":
    unittest.main()
