import unittest

from fit import apply_predict_multi, load_coefs
from table import TABLE_CSV_HEADER, build_csv_row


class RegressionTests(unittest.TestCase):
    def test_existing_model_predictions_are_unchanged(self):
        fit_type, n_inputs, params, split = load_coefs("fit_coefs.bin")
        self.assertEqual(n_inputs, 1)
        self.assertEqual([entry[1] for entry in params], ["sym_log", "sym_log", "exp"])
        result = apply_predict_multi([0.1, 0.1, 100000], params, fit_type, split)
        expected = [1.4477653909084447, 0.6436570586070975, -3.5036069423285605]
        for actual, wanted in zip(result, expected):
            self.assertAlmostEqual(actual, wanted, places=12)

    def test_csv_schema_remains_108_columns(self):
        row = build_csv_row(
            press_timestamp=1.0,
            rel_ms=10.5,
            delta_ms=2.5,
            ch_data=list(range(84)),
            force_data=[1, 2, 3, 4, 5, 6],
            force_timestamp=1.005,
            delta_cop_x=0.1,
            delta_cop_y=0.2,
            delta_force_x=1,
            delta_force_y=2,
            delta_force_z=3,
            adc_angle=10,
            force_angle=20,
        )
        self.assertEqual(len(TABLE_CSV_HEADER), 108)
        self.assertEqual(len(row), 108)
        self.assertEqual(TABLE_CSV_HEADER[:2], ["rel_ms", "delta_ms"])
        self.assertEqual(row[:2], [10.5, 2.5])

if __name__ == "__main__":
    unittest.main()
