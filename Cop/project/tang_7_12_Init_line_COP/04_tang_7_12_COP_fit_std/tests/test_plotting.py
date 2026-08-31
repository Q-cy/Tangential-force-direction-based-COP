import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tangential.config import PlotConfig
from tangential.tools.plotting import (
    compute_errors,
    list_files,
    load_csv,
    plot_csv,
    plot_full_analysis,
    resolve_csvs,
)
from tangential.storage.csv import TABLE_CSV_HEADER, full_analysis_png_path


class PlottingApiTests(unittest.TestCase):
    def _write_csv(self, path, header, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)

    def test_import_does_not_load_matplotlib(self):
        env = os.environ.copy()
        project_root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = str(project_root / "src")
        code = "import sys; import tangential.tools.plotting; assert 'matplotlib' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True, env=env)

    def test_actual_header_resolution_and_directory_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "sample.csv"
            self._write_csv(path, ["timestamp", "measured", "estimate"], [[0, 1, 2], [1, 2, 3]])
            header, data = load_csv(path)
            self.assertEqual(header, ["timestamp", "measured", "estimate"])
            self.assertEqual(data.shape, (2, 3))
            self.assertEqual(resolve_csvs("sample.csv", directory), [path.resolve()])
            infos = list_files(directory)
            self.assertEqual(infos[0].path, path)
            self.assertEqual(infos[0].row_count, 2)

    def test_full_analysis_png_path_uses_csv_stem(self):
        with tempfile.TemporaryDirectory() as temp:
            csv_path = Path(temp) / "foo.csv"
            self.assertEqual(
                full_analysis_png_path(csv_path),
                Path(temp) / "foo.png",
            )

    def test_empty_csv_and_bad_rows_have_explicit_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            empty = directory / "empty.csv"
            empty.touch()
            with self.assertRaisesRegex(ValueError, "为空且没有表头"):
                load_csv(empty)

            bad = directory / "bad.csv"
            self._write_csv(bad, ["x", "y"], [[1]])
            with self.assertRaisesRegex(ValueError, "列数与表头不一致"):
                load_csv(bad)

    def test_plot_mode_uses_row_range_and_generates_png(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "angles.csv"
            self._write_csv(
                path,
                ["rel_ms", "measured", "estimate", "valid"],
                [[0, 359, 1, 1], [1, 0, 2, 1], [2, 1, 3, 0], [3, 2, 4, 1]],
            )
            output = directory / "plot.png"
            result = plot_csv(
                PlotConfig(
                    files=[path],
                    columns=["estimate"],
                    rows=(1, 3),
                    x_column="rel_ms",
                    save_path=output,
                    error_ref="measured",
                    force_min=0,
                )
            )
            self.assertEqual(result.save_path, output)
            self.assertTrue(output.is_file())
            self.assertIsNotNone(result.error_path)
            self.assertTrue(result.error_path.is_file())

    def test_full_analysis_generates_png_from_108_column_header(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            path = directory / "full.csv"
            rows = []
            for index in range(4):
                values = [0.0] * len(TABLE_CSV_HEADER)
                values[TABLE_CSV_HEADER.index("rel_ms")] = index
                values[TABLE_CSV_HEADER.index("adc_sum")] = 100 + index
                values[TABLE_CSV_HEADER.index("ADC_angle")] = index
                values[TABLE_CSV_HEADER.index("Force_angle")] = index + 1
                values[TABLE_CSV_HEADER.index("Force_cal_angle")] = index + 2
                values[TABLE_CSV_HEADER.index("delta_Force_X")] = index + 1
                values[TABLE_CSV_HEADER.index("delta_Force_Y")] = index + 2
                values[TABLE_CSV_HEADER.index("delta_Force_Z")] = index + 3
                values[TABLE_CSV_HEADER.index("Fx_cal")] = index + 1.5
                values[TABLE_CSV_HEADER.index("Fy_cal")] = index + 2.5
                values[TABLE_CSV_HEADER.index("valid")] = 1
                rows.append(values)
            self._write_csv(path, TABLE_CSV_HEADER, rows)
            result = plot_full_analysis(PlotConfig(files=path, rows="0:4"))
            output = directory / "full.png"
            self.assertEqual(result.save_path, output)
            self.assertTrue(output.is_file())
            self.assertEqual(len(result.files), 1)

    def test_angle_error_wraps_at_zero(self):
        result = compute_errors([359.0, 1.0], [1.0, 359.0], is_angle=True)
        self.assertAlmostEqual(result["MAE"], 2.0)
        self.assertAlmostEqual(result["Max_Error"], 2.0)


if __name__ == "__main__":
    unittest.main()
