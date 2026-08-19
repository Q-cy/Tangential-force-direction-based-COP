import csv
import os
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/pzt-test-mplconfig")

import numpy as np
from pyqtgraph.Qt import QtWidgets

from tangential.plotting import load_csv, resolve_column
from tangential.gui.realtime import RealTimePlot


class StaticPlotTests(unittest.TestCase):
    def test_resolve_column_uses_new_header_positions(self):
        header = ["rel_ms", "delta_ms", "adc_sum", "valid"]
        self.assertEqual(resolve_column("rel_ms", header), 0)
        self.assertEqual(resolve_column("delta_ms", header), 1)

    def test_resolve_column_uses_actual_legacy_header(self):
        header = ["timestamp", "ADC_angle", "ADC_mag", "Force_angle", "valid"]
        self.assertEqual(resolve_column("Force_angle", header), 3)
        self.assertEqual(resolve_column("3", header), 3)
        with self.assertRaisesRegex(ValueError, "未知列名"):
            resolve_column("missing", header)
        with self.assertRaisesRegex(ValueError, "列号越界"):
            resolve_column(9, header)

    def test_empty_csv_with_header_has_stable_shape(self):
        with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as file_obj:
            path = file_obj.name
            csv.writer(file_obj).writerow(["a", "b", "c"])
        try:
            header, data = load_csv(path)
            self.assertEqual(header, ["a", "b", "c"])
            self.assertEqual(data.shape, (0, 3))
        finally:
            os.remove(path)

class RealtimePlotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_removed_region_arrow_is_cleared(self):
        plot = RealTimePlot()
        common = dict(
            pzt_angle_deg=0.0,
            force_angle_deg=0.0,
            press_table_arr=np.ones(84),
            total_press_val=84.0,
            cop_curr_x=float("nan"),
            cop_curr_y=float("nan"),
            cop_base_x=None,
            cop_base_y=None,
            cop_delta_x=0.0,
            cop_delta_y=0.0,
            force_fx_val=0.0,
            force_fy_val=0.0,
            force_fz_val=0.0,
            contact_init=True,
            region_mask=np.ones((12, 7), dtype=int),
        )
        regions = [
            {"id": 1, "cop": (2.0, 2.0), "delta": (0.5, 0.0)},
            {"id": 2, "cop": (4.0, 7.0), "delta": (0.5, 0.0)},
        ]
        plot.set_data(**common, regions=regions)
        plot.update_all()
        self.assertEqual(len(plot._region_arrows[1][0].xData), 2)

        plot.set_data(**common, regions=regions[:1])
        plot.update_all()
        cleared = plot._region_arrows[1][0].xData
        self.assertTrue(cleared is None or len(cleared) == 0)
        plot.win.close()


if __name__ == "__main__":
    unittest.main()
