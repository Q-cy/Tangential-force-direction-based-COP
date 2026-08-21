from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import OptimizeWarning

from tangential.processing.calibration import FitCalibrationModel
from tangential.tools import training as training_module
from tangential.config import TrainingConfig
from tangential.tools.training import train_model


class TrainingModuleTests(unittest.TestCase):
    def _train(self, config: TrainingConfig):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            warnings.simplefilter("ignore", RuntimeWarning)
            return train_model(config)

    def test_import_does_not_load_matplotlib(self):
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import tangential.tools.training; assert 'matplotlib' not in sys.modules"],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def _write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _sources(self, directory: Path) -> tuple[Path, Path]:
        xy_rows = []
        z_rows = []
        values = np.linspace(-2.0, 2.0, 9)
        for index, value in enumerate(values):
            valid = 0 if index == 0 else 1
            fx = 2.0 * np.sign(value) * np.log1p(0.5 * abs(value))
            fy = 1.5 * np.sign(value) * np.log1p(0.4 * abs(value))
            adc_sum = 100.0 + index * 3.0
            xy_rows.append({
                "delta_CoP_X": value,
                "delta_CoP_Y": -value,
                "adc_sum": adc_sum,
                "delta_Force_X": fx,
                "delta_Force_Y": fy,
                "valid": valid,
                "CoP_state": valid,
            })
            z_rows.append({
                "adc_sum": adc_sum,
                "delta_Force_Z": -(2.0 * np.exp(0.1 * (index - 4)) + 1.0),
                "valid": valid,
                "CoP_state": valid,
            })
        xy_path = directory / "xy.csv"
        z_path = directory / "z.csv"
        self._write_csv(xy_path, xy_rows)
        self._write_csv(z_path, z_rows)
        return xy_path, z_path

    def _multidim_source(self, directory: Path) -> Path:
        rows = []
        for index in range(-4, 5):
            x = float(index)
            y = float(index * 2)
            adc_sum = float(100 + index * 4)
            rows.append({
                "delta_CoP_X": x,
                "delta_CoP_Y": y,
                "adc_sum": adc_sum,
                "delta_Force_X": 2.0 * x + 0.5 * y + 0.01 * adc_sum,
                "delta_Force_Y": -x + 3.0 * y - 0.02 * adc_sum,
                "delta_Force_Z": 0.25 * x - 0.5 * y + 0.03 * adc_sum,
                "valid": 1,
                "CoP_state": 1,
            })
        path = directory / "multidim.csv"
        self._write_csv(path, rows)
        return path

    def test_config_defaults_are_stable(self):
        config = TrainingConfig("xy.csv", "z.csv")
        self.assertEqual(config.output_model, "fit_coefs.bin")
        self.assertEqual(config.output_plot, "fit_report.png")
        self.assertEqual(config.dim, 1)
        self.assertEqual(config.poly_order, 3)
        self.assertEqual((config.fx, config.fy, config.fz), ("sym_log", "sym_log", "exp"))
        self.assertTrue(config.valid_only)
        self.assertTrue(config.split_sign)
        self.assertTrue(config.one_on_one)
        self.assertIsNone(config.write_back)
        self.assertFalse(config.force)

    def test_train_model_filters_valid_rows_and_loads_runtime_model(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            xy_path, z_path = self._sources(directory)
            original_xy = xy_path.read_bytes()
            original_z = z_path.read_bytes()
            model_path = directory / "model.bin"
            plot_path = directory / "report.png"
            result = self._train(TrainingConfig(
                xy_path,
                z_path,
                output_model=model_path,
                output_plot=plot_path,
            ))

            self.assertEqual(result.sample_counts["delta_Force_X"], 8)
            self.assertEqual(result.sample_counts["delta_Force_Y"], 8)
            self.assertEqual(result.sample_counts["delta_Force_Z"], 8)
            self.assertTrue(model_path.is_file())
            self.assertTrue(plot_path.is_file())
            self.assertEqual(xy_path.read_bytes(), original_xy)
            self.assertEqual(z_path.read_bytes(), original_z)

            model = FitCalibrationModel.from_path(str(model_path))
            self.assertTrue(model.available, model.error)
            prediction = model.predict(1.0, -1.0, 112.0)
            self.assertEqual(len(prediction), 3)
            self.assertTrue(all(np.isfinite(value) for value in prediction))

            grid = np.linspace(-1.0, 1.0, 17)
            grid_matrix = grid.reshape(-1, 1)
            batch = training_module._predict_result_batch(
                result.fit_results[0], grid_matrix
            )
            self.assertEqual(batch.shape, (len(grid), 1))

    def test_dim_2_and_dim_3_fit_multivariate_models(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            source = self._multidim_source(directory)

            dim2_model = directory / "dim2.bin"
            dim2 = self._train(TrainingConfig(
                source,
                source,
                output_model=dim2_model,
                output_plot=None,
                dim=2,
                fx="poly",
                fy="poly",
                split_sign=False,
            ))
            self.assertEqual(len(dim2.fit_results), 1)
            model2 = FitCalibrationModel.from_path(str(dim2_model))
            self.assertTrue(model2.available, model2.error)
            self.assertEqual(len(model2.params_list), 2)
            self.assertEqual(len(model2.predict(1.0, 2.0, 105.0)), 3)
            batch2 = training_module._predict_result_batch(
                dim2.fit_results[0], np.zeros((13, 2))
            )
            self.assertEqual(batch2.shape, (13, 2))

            sigmoid_model = directory / "dim2_sigmoid.bin"
            sigmoid = self._train(TrainingConfig(
                source,
                source,
                output_model=sigmoid_model,
                output_plot=None,
                dim=2,
                fx="sigmoid",
                fy="sigmoid",
                split_sign=True,
            ))
            self.assertEqual(sigmoid.fit_results[0][3], "sigmoid")
            loaded_sigmoid = FitCalibrationModel.from_path(str(sigmoid_model))
            self.assertTrue(loaded_sigmoid.available, loaded_sigmoid.error)
            self.assertEqual(len(loaded_sigmoid.params_list), 2)

            separate_xy, separate_z = self._sources(directory)
            with self.assertRaisesRegex(ValueError, "dim=3 requires"):
                self._train(TrainingConfig(
                    separate_xy,
                    separate_z,
                    output_model=directory / "rejected-dim3.bin",
                    output_plot=None,
                    dim=3,
                    fx="poly",
                    fy="poly",
                    fz="poly",
                    split_sign=False,
                ))

            dim3_model = directory / "dim3.bin"
            dim3 = self._train(TrainingConfig(
                source,
                source,
                output_model=dim3_model,
                output_plot=None,
                dim=3,
                fx="poly",
                fy="poly",
                fz="poly",
                split_sign=False,
            ))
            self.assertEqual(len(dim3.fit_results), 1)
            model3 = FitCalibrationModel.from_path(str(dim3_model))
            self.assertTrue(model3.available, model3.error)
            self.assertEqual(len(model3.predict(1.0, 2.0, 105.0)), 3)
            batch3 = training_module._predict_result_batch(
                dim3.fit_results[0], np.zeros((11, 3))
            )
            self.assertEqual(batch3.shape, (11, 3))

    def test_write_back_requires_force_for_existing_target_and_same_input(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = Path(directory_name)
            xy_path, z_path = self._sources(directory)
            model_path = directory / "model.bin"
            config = TrainingConfig(xy_path, z_path, output_model=model_path, output_plot=None)
            self._train(config)

            target = directory / "calibrated.csv"
            write_config = TrainingConfig(
                xy_path,
                z_path,
                output_model=directory / "model2.bin",
                output_plot=None,
                write_back=target,
            )
            self._train(write_config)
            self.assertTrue(target.is_file())
            with target.open(newline="", encoding="utf-8") as stream:
                header = next(csv.reader(stream))
            self.assertIn("Fx_cal", header)
            self.assertIn("Fy_cal", header)
            self.assertIn("Force_cal_angle", header)

            blocked_model = directory / "blocked.bin"
            blocked_plot = directory / "blocked.png"
            blocked_config = TrainingConfig(
                xy_path,
                z_path,
                output_model=blocked_model,
                output_plot=blocked_plot,
                write_back=target,
            )
            with self.assertRaises(FileExistsError):
                self._train(blocked_config)
            self.assertFalse(blocked_model.exists())
            self.assertFalse(blocked_plot.exists())

            same_input = TrainingConfig(
                xy_path,
                z_path,
                output_model=directory / "model3.bin",
                output_plot=None,
                write_back=xy_path,
            )
            with self.assertRaises(FileExistsError):
                self._train(same_input)

            same_input.force = True
            self._train(same_input)
            self.assertTrue(xy_path.is_file())


if __name__ == "__main__":
    unittest.main()
