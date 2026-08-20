import unittest

import numpy as np

from tangential.processing.calibration import FitCalibrationModel


class MultiInputCalibrationTests(unittest.TestCase):
    def test_multivariable_poly_uses_training_basis_order(self):
        # order 2, inputs [dx, dy, total]:
        # 1, x0, x1, x2, x0*x0, x0*x1, x0*x2, x1*x1, x1*x2, x2*x2
        coefficients = np.arange(1.0, 11.0)
        model = FitCalibrationModel(
            fit_type="poly", params_list=[(coefficients, "poly", False)],
            n_inputs=3,
        )
        values = (2.0, 3.0, 4.0)
        basis = np.array([1, 2, 3, 4, 4, 6, 8, 9, 12, 16], dtype=float)
        expected = float(np.dot(coefficients, basis))
        self.assertEqual(model.n_inputs, 3)
        self.assertAlmostEqual(model.predict(*values)[0], expected)

    def test_multivariable_split_poly_selects_by_first_input(self):
        positive = np.array([1, 2, 0], dtype=float)
        negative = np.array([10, 20, 0], dtype=float)
        model = FitCalibrationModel(
            fit_type="poly",
            params_list=[((positive, negative), "poly", True)],
            split_sign=True,
            n_inputs=2,
        )
        self.assertAlmostEqual(model.predict(2.0, 8.0, 0.0)[0], 1 + 4)
        self.assertAlmostEqual(model.predict(-2.0, 8.0, 0.0)[0], 10 - 40)

    def test_mixed_entries_use_existing_scalar_paths(self):
        model = FitCalibrationModel(
            fit_type="poly",
            params_list=[
                (np.array([1.0, 2.0]), "poly", False),
                (np.array([1.0, 0.0, 0.0, 0.0, 1.0]), "exp", False),
            ],
            n_inputs=2,
        )
        prediction = model.predict(2.0, 3.0, 0.0)
        self.assertAlmostEqual(prediction[0], 5.0)
        self.assertAlmostEqual(prediction[1], -1.0)


if __name__ == "__main__":
    unittest.main()
