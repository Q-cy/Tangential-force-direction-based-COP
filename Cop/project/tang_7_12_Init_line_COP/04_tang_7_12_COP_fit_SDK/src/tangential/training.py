"""Offline calibration model training.

This module is the reusable replacement for the training portion of the old
``fit.py`` script.  It deliberately does not import the script, GUI modules,
or matplotlib at import time.  The model writer follows the binary format
consumed by :mod:`tangential.processing.calibration`.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

from .processing.calibration import apply_fit_predict_multi


INPUT_COLS = ("delta_CoP_X", "delta_CoP_Y", "adc_sum")
OUTPUT_COLS = ("delta_Force_X", "delta_Force_Y", "delta_Force_Z")


@dataclass
class TrainingConfig:
    """Configuration for offline model fitting.

    Attributes:
        xy_csv: CSV containing the ``delta_CoP_X``, ``delta_CoP_Y`` and
            ``delta_Force_X/Y`` columns, normally used for Fx/Fy.
        z_csv: CSV containing the ``adc_sum`` and ``delta_Force_Z`` columns,
            normally used for Fz.
        output_model: Destination path for the binary ``fit_coefs.bin``
            compatible with :class:`FitCalibrationModel`.
        output_plot: Optional PNG diagnostic report path; ``None`` disables it.
        dim: Number of jointly fitted input/output dimensions, 1, 2 or 3.
        poly_order: Total-degree polynomial order, 1, 2 or 3.
        fx: Fit type for ``delta_Force_X``.
        fy: Fit type for ``delta_Force_Y``.
        fz: Fit type for ``delta_Force_Z``.
        valid_only: If true, retain rows whose ``valid`` value is nonzero,
            falling back to nonzero ``CoP_state`` when ``valid`` is absent.
        split_sign: Whether supported scalar fits use separate positive and
            negative branches.
        one_on_one: Whether symmetric-log median grouping uses half-unit force
            bins; false uses the legacy 0.2-unit grouping.
        write_back: Optional target CSV to which calibration columns are added.
        force: Allow an existing write-back target or source/target collision.

    Side Effects:
        No file is written when this dataclass is constructed. Input CSVs are
        never modified unless :func:`train_model` receives ``write_back``.
    """

    xy_csv: str | os.PathLike[str]
    z_csv: str | os.PathLike[str]
    output_model: str | os.PathLike[str] = "fit_coefs.bin"
    output_plot: str | os.PathLike[str] | None = "fit_report.png"
    dim: int = 1
    poly_order: int = 3
    fx: str = "sym_log"
    fy: str = "sym_log"
    fz: str = "exp"
    valid_only: bool = True
    split_sign: bool = True
    one_on_one: bool = True
    write_back: str | os.PathLike[str] | None = None
    force: bool = False


@dataclass
class TrainingResult:
    """Artifacts and diagnostics produced by :func:`train_model`.

    Attributes:
        model_path: Path of the binary model written by training.
        plot_path: Diagnostic PNG path, or ``None`` when disabled.
        fit_results: Tuples ``(input_columns, output_columns, parameters,
            fit_type, split)`` in runtime-predictor format.
        sample_counts: Number of accepted rows recorded per output column.
        written_path: Write-back CSV path, or ``None`` when write-back was not
            requested.
    """

    model_path: Path
    plot_path: Path | None
    fit_results: list[tuple[list[str], list[str], Any, str, bool]]
    sample_counts: dict[str, int]
    written_path: Path | None = None


def _row_valid(row: dict[str, str], fieldnames: Sequence[str]) -> bool:
    """Determine whether one CSV row is eligible for fitting.

    Args:
        row: Mapping of stripped CSV column names to their raw string values.
        fieldnames: Complete header sequence used to decide which validity
            column exists.

    Returns:
        ``True`` when ``valid`` is nonzero, or, if absent, when
        ``CoP_state`` is nonzero. If neither column exists, returns ``True``;
        malformed validity values return ``False``.

    Raises:
        No exception is intentionally raised for malformed cell values;
        ``TypeError`` and ``ValueError`` are converted to ``False``.
    """

    field_set = set(fieldnames)
    try:
        if "valid" in field_set:
            return float(row.get("valid", 0)) != 0
        if "CoP_state" in field_set:
            return float(row.get("CoP_state", 0)) != 0
    except (TypeError, ValueError):
        return False
    return True


def load_csv(
    csv_path: str | os.PathLike[str],
    input_cols: Sequence[str],
    output_cols: Sequence[str],
    valid_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Load named numeric training columns from one CSV file.

    Args:
        csv_path: Input CSV path. The file is opened as UTF-8 text.
        input_cols: Ordered feature column names; the result has shape
            ``(n_rows, len(input_cols))``.
        output_cols: Ordered target column names; the result has shape
            ``(n_rows, len(output_cols))``.
        valid_only: If true, apply :func:`_row_valid` before conversion.

    Returns:
        A tuple ``(x, y)`` of ``float64`` arrays. Rows with missing or
        non-numeric feature/target cells are silently skipped, so either array
        can have zero rows.

    Raises:
        FileNotFoundError/OSError: Propagated when ``csv_path`` cannot be read.
        ValueError: If the file has no header or lacks a requested column.
    """

    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        reader.fieldnames = fieldnames
        missing = [name for name in (*input_cols, *output_cols) if name not in fieldnames]
        if missing:
            raise ValueError(f"CSV {csv_path} is missing columns: {', '.join(missing)}")
        for row in reader:
            if valid_only and not _row_valid(row, fieldnames):
                continue
            try:
                x_rows.append([float(row[name]) for name in input_cols])
                y_rows.append([float(row[name]) for name in output_cols])
            except (KeyError, TypeError, ValueError):
                continue
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.float64)


def log_func(x: Any, a: float, b: float, c: float) -> Any:
    """Evaluate ``a * ln(b*x + 1) + c`` elementwise.

    Args:
        x: Scalar or array-like input values.
        a: Logarithmic amplitude.
        b: Input scale; the log domain must remain valid.
        c: Additive offset.
    Returns:
        Scalar/array-like NumPy result with the broadcast shape of ``x``.
    Raises:
        FloatingPointError: Only if NumPy floating-point errors are configured
            to raise for an invalid logarithm.
    """

    return a * np.log(b * x + 1) + c


def log_func_o(x: Any, a: float, b: float) -> Any:
    """Evaluate the origin-passing curve ``a * ln(b*x + 1)``.

    Args:
        x: Scalar or array-like input values.
        a: Logarithmic amplitude.
        b: Input scale; the log domain must remain valid.
    Returns:
        NumPy scalar or array with the broadcast shape of ``x``.
    Raises:
        FloatingPointError: Possible for invalid log-domain values when NumPy
            is configured to raise floating-point errors.
    """

    return a * np.log(b * x + 1)


def exp_func(x: Any, a: float, b: float, c: float) -> Any:
    """Evaluate ``a * exp(b*x) + c`` elementwise.

    Args:
        x: Scalar or array-like input values.
        a: Exponential amplitude.
        b: Exponential rate.
        c: Additive offset.
    Returns:
        NumPy scalar or array with the broadcast shape of ``x``.
    """

    return a * np.exp(b * x) + c


def sigmoid(x: Any, level: float, slope: float, center: float, bias: float) -> Any:
    """Evaluate a four-parameter logistic curve.

    Args:
        x: Scalar or array-like input values.
        level: Logistic amplitude.
        slope: Steepness and direction.
        center: Input value at the logistic midpoint.
        bias: Additive output offset.
    Returns:
        NumPy scalar or array with the broadcast shape of ``x``.
    """
    return level / (1 + np.exp(-slope * (x - center))) + bias


def build_design_matrix(x: np.ndarray, order: int) -> np.ndarray:
    """Build the total-degree polynomial design matrix.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``.
        order: Maximum total degree; only 1, 2 and 3 are supported.
    Returns:
        ``float`` matrix of shape ``(n_rows, n_terms)`` containing an intercept,
        linear terms, and ordered interaction terms through ``order``.
    Raises:
        ValueError: If ``order`` is not 1, 2 or 3, or if ``x`` is not a
            two-dimensional matrix.
    """

    if order not in (1, 2, 3):
        raise ValueError(f"Unsupported polynomial order: {order}")
    n_vars = x.shape[1]
    columns: list[np.ndarray] = [np.ones(len(x))]
    if order >= 1:
        columns.extend(x[:, index] for index in range(n_vars))
    if order >= 2:
        columns.extend(
            x[:, i] * x[:, j]
            for i in range(n_vars)
            for j in range(i, n_vars)
        )
    if order >= 3:
        columns.extend(
            x[:, i] * x[:, j] * x[:, k]
            for i in range(n_vars)
            for j in range(i, n_vars)
            for k in range(j, n_vars)
        )
    return np.column_stack(columns)


def get_term_labels(input_cols: Sequence[str], order: int) -> list[str]:
    """Return labels in the same order as :func:`build_design_matrix`.

    Args:
        input_cols: Ordered feature names, for example ``delta_CoP_X``.
        order: Maximum total degree, expected to be 1, 2 or 3.
    Returns:
        List beginning with ``"1"`` followed by compact linear and interaction
        labels. No file is read or written.
    """
    short = [column.replace("delta_", "").replace("_", "") for column in input_cols]
    labels = ["1"]
    if order >= 1:
        labels.extend(short)
    if order >= 2:
        labels.extend(
            f"{short[i]}*{short[j]}"
            for i in range(len(short))
            for j in range(i, len(short))
        )
    if order >= 3:
        labels.extend(
            f"{short[i]}*{short[j]}*{short[k]}"
            for i in range(len(short))
            for j in range(i, len(short))
            for k in range(j, len(short))
        )
    return labels


def fit_polynomial(x: np.ndarray, y: np.ndarray, order: int) -> np.ndarray:
    """Fit least-squares total-degree polynomial coefficients.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``.
        y: Target matrix of shape ``(n_rows, n_outputs)``.
        order: Polynomial degree 1, 2 or 3.
    Returns:
        Coefficient matrix of shape ``(n_terms, n_outputs)``.
    Raises:
        ValueError: Propagated from :func:`build_design_matrix` for an invalid
            order or feature matrix.
    """
    return np.linalg.lstsq(build_design_matrix(x, order), y, rcond=None)[0]


def predict(x: np.ndarray, coefs: np.ndarray, order: int) -> np.ndarray:
    """Evaluate polynomial coefficients on a feature matrix.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``.
        coefs: Coefficients of shape ``(n_terms, n_outputs)``.
        order: Degree used to create ``coefs``.
    Returns:
        Prediction matrix of shape ``(n_rows, n_outputs)``.
    Raises:
        ValueError: If ``order`` is unsupported or dimensions are incompatible.
    """
    return build_design_matrix(x, order) @ coefs


def _fit_exp_side(x_data: np.ndarray, y_data: np.ndarray) -> np.ndarray:
    """Fit one nonnegative branch of an exponential curve.

    Args:
        x_data: One-dimensional branch inputs, normally nonnegative.
        y_data: One-dimensional target values with the same length.
    Returns:
        Three parameters ``(a, b, c)`` for :func:`exp_func`; a default
        ``[1, 1, 0]`` is returned for too few samples or optimizer failure.
    Side Effects:
        Runs SciPy nonlinear least squares; no files are written.
    """
    if len(x_data) < 3:
        return np.array([1.0, 1.0, 0.0])
    x_max = np.max(np.abs(x_data))
    if x_max > 10:
        scale = 1.0 / x_max
        x_scaled = x_data * scale
        b0 = 1.0
    else:
        scale = 1.0
        x_scaled = x_data
        b0 = 5.0
    try:
        params, _ = curve_fit(
            exp_func,
            x_scaled,
            y_data,
            p0=[1.0, b0, np.min(y_data)],
            maxfev=10000,
        )
        params[1] *= scale
        return params
    except Exception:
        return np.array([1.0, 1.0, 0.0])


def fit_sym_exp(x: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit separate negative and positive exponential branches.

    Args:
        x: Feature matrix whose first column has shape ``(n_rows,)`` when
            selected as ``x[:, 0]``.
        y: Target matrix of shape ``(n_rows, n_outputs)``.
    Returns:
        One ``(negative_params, positive_params)`` pair per output; each
        parameter array has three values ``(a, b, c)``.
    """
    results = []
    values = x[:, 0]
    for output in range(y.shape[1]):
        positive = values >= 0
        negative = values < 0
        p_pos = _fit_exp_side(values[positive], y[positive, output]) if np.sum(positive) > 3 else np.array([1.0, 1.0, 0.0])
        p_neg = _fit_exp_side(-values[negative], -y[negative, output]) if np.sum(negative) > 3 else p_pos.copy()
        results.append((p_neg, p_pos))
    return results


def predict_sym_exp(x: np.ndarray, params: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Evaluate symmetric exponential branch parameters.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``; only column 0 is
            used.
        params: One negative/positive three-parameter pair per output.
    Returns:
        Prediction matrix of shape ``(n_rows, len(params))``.
    """
    result = np.zeros((len(x), len(params)))
    values = x[:, 0]
    for index, (p_neg, p_pos) in enumerate(params):
        negative = values < 0
        positive = ~negative
        result[positive, index] = exp_func(values[positive], *p_pos)
        result[negative, index] = -exp_func(-values[negative], *p_neg)
    return result


def fit_sym_log(x: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit separate origin-passing logarithmic branches by output.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``; column 0 is the
            signed scalar input and may be standardized when its magnitude is
            large.
        y: Target matrix of shape ``(n_rows, n_outputs)``.
    Returns:
        One ``(negative_params, positive_params)`` pair per output, each with
            three values ``(a, b, 0)`` compatible with :func:`log_func`.
    Side Effects:
        Runs SciPy curve fitting; no files are written.
    """
    raw = x[:, 0]
    values = raw
    if np.max(np.abs(raw)) > 100:
        values = (raw - float(np.mean(raw))) / (float(np.std(raw)) or 1.0)
    results = []
    for output in range(y.shape[1]):
        positive = values >= 0
        negative = values < 0
        if np.sum(positive) >= 2:
            try:
                ab, _ = curve_fit(log_func_o, values[positive], y[positive, output], p0=[1.0, 1.0], maxfev=10000)
                p_pos = np.array([ab[0], ab[1], 0.0])
            except Exception:
                p_pos = np.array([1.0, 1.0, 0.0])
        else:
            p_pos = np.array([1.0, 1.0, 0.0])
        if np.sum(negative) >= 2:
            try:
                ab, _ = curve_fit(log_func_o, -values[negative], -y[negative, output], p0=[1.0, 1.0], maxfev=10000)
                p_neg = np.array([ab[0], ab[1], 0.0])
            except Exception:
                p_neg = np.array([1.0, 1.0, 0.0])
        else:
            p_neg = p_pos.copy()
        results.append((p_neg, p_pos))
    return results


def predict_sym_log(x: np.ndarray, params: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Evaluate symmetric logarithmic branch parameters.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``; only column 0 is
            used with the same large-magnitude standardization as fitting.
        params: One negative/positive parameter pair per output.
    Returns:
        Prediction matrix of shape ``(n_rows, len(params))``.
    """
    raw = x[:, 0]
    values = raw
    if np.max(np.abs(raw)) > 100:
        values = (raw - float(np.mean(raw))) / (float(np.std(raw)) or 1.0)
    result = np.zeros((len(x), len(params)))
    for index, (p_neg, p_pos) in enumerate(params):
        positive = values >= 0
        negative = ~positive
        result[positive, index] = log_func(values[positive], *p_pos)
        result[negative, index] = -log_func(-values[negative], *p_neg)
    return result


def fit_exp_log(x: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit exponential behavior for negative inputs and logarithmic behavior for positive inputs.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``; column 0 selects
            the signed scalar input.
        y: Target matrix of shape ``(n_rows, n_outputs)``.
    Returns:
        One ``(negative_params, positive_params)`` pair per output, with three
            parameters for each branch.
    Side Effects:
        Runs SciPy curve fitting; optimizer failures use default parameters.
    """
    results = []
    values = x[:, 0]
    for output in range(y.shape[1]):
        negative = values < 0
        positive = ~negative
        if np.sum(negative) > 3:
            try:
                p_neg, _ = curve_fit(exp_func, values[negative], y[negative, output], p0=[1.0, 1.0, np.min(y[negative, output])], maxfev=10000)
            except Exception:
                p_neg = np.array([1.0, 1.0, 0.0])
        else:
            p_neg = np.array([1.0, 1.0, 0.0])
        if np.sum(positive) > 3:
            try:
                p_pos, _ = curve_fit(log_func, values[positive], y[positive, output], p0=[1.0, 1.0, np.min(y[positive, output])], maxfev=10000)
            except Exception:
                p_pos = np.array([1.0, 1.0, 0.0])
        else:
            p_pos = np.array([1.0, 1.0, 0.0])
        results.append((p_neg, p_pos))
    return results


def predict_exp_log(x: np.ndarray, params: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Evaluate the piecewise exponential/logarithmic fit.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``; column 0 is used.
        params: One three-parameter negative/positive pair per output.
    Returns:
        Prediction matrix of shape ``(n_rows, len(params))``.
    """
    result = np.zeros((len(x), len(params)))
    values = x[:, 0]
    negative = values < 0
    positive = ~negative
    for index, (p_neg, p_pos) in enumerate(params):
        result[negative, index] = exp_func(values[negative], *p_neg)
        result[positive, index] = log_func(values[positive], *p_pos)
    return result


def fit_sigmoid(x: np.ndarray, y: np.ndarray) -> list[np.ndarray]:
    """Fit one four-parameter sigmoid independently for each output.

    Args:
        x: Feature matrix; only column 0 is fitted as the scalar input.
        y: Target matrix of shape ``(n_rows, n_outputs)``.
    Returns:
        List of parameter arrays ``[level, slope, center, bias]``, one per
            output.
    Side Effects:
        Runs SciPy curve fitting; failed fits retain their deterministic
        initial parameters and do not raise the optimizer exception.
    """
    results = []
    for output in range(y.shape[1]):
        values = y[:, output]
        y_min, y_max = np.min(values), np.max(values)
        initial = [y_max - y_min, 10.0, np.median(x[:, 0]), y_min]
        try:
            params, _ = curve_fit(sigmoid, x[:, 0], values, p0=initial, maxfev=10000)
        except Exception:
            params = np.asarray(initial, dtype=np.float64)
        results.append(params)
    return results


def predict_sigmoid(x: np.ndarray, params: Sequence[np.ndarray]) -> np.ndarray:
    """Evaluate sigmoid parameters for every output.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``; column 0 is used.
        params: Sequence of four-parameter arrays, one per output.
    Returns:
        Prediction matrix of shape ``(n_rows, len(params))``.
    """
    return np.column_stack([sigmoid(x[:, 0], *item) for item in params])


def fit_pchip(x: np.ndarray, y: np.ndarray) -> list[PchipInterpolator]:
    """Fit monotone piecewise-cubic interpolators per output.

    Args:
        x: Feature matrix; column 0 is sorted and duplicate input values are
            removed before interpolation.
        y: Target matrix of shape ``(n_rows, n_outputs)``.
    Returns:
        List of :class:`scipy.interpolate.PchipInterpolator` objects, one per
            output.
    Raises:
        ValueError: If an output has insufficient unique finite input points
            for PCHIP construction.
    """
    results = []
    for output in range(y.shape[1]):
        order = np.argsort(x[:, 0])
        x_sorted, y_sorted = x[order, 0], y[order, output]
        keep = np.diff(x_sorted, prepend=x_sorted[0] - 1) != 0
        results.append(PchipInterpolator(x_sorted[keep], y_sorted[keep]))
    return results


def predict_pchip(x: np.ndarray, params: Sequence[PchipInterpolator]) -> np.ndarray:
    """Evaluate fitted PCHIP interpolators.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``; column 0 is used.
        params: PCHIP interpolators, one per output.
    Returns:
        Prediction matrix of shape ``(n_rows, len(params))``.
    Raises:
        ValueError: If an input is outside an interpolator's configured domain.
    """
    return np.column_stack([interp(x[:, 0]) for interp in params])


def fit_exp(x: np.ndarray, y: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """Fit the legacy Fz exponential model with optional input normalization.

    Args:
        x: Feature matrix; column 0 is the scalar input.
        y: Target matrix; only column 0 is used and its sign is inverted to
            match the existing Fz model convention.
    Returns:
        A one-item list ``[(a, b, c, mean, scale)]`` consumed by
            :func:`predict_exp`.
    Side Effects:
        Runs SciPy curve fitting; failure returns a deterministic fallback.
    """
    raw = x[:, 0]
    mean, scale = 0.0, 1.0
    values = raw
    if np.max(np.abs(raw)) > 100:
        mean, scale = float(np.mean(raw)), float(np.std(raw)) or 1.0
        values = (raw - mean) / scale
    target = -y[:, 0]
    try:
        params, _ = curve_fit(exp_func, values, target, p0=[1.0, 1.0, np.min(target)], maxfev=10000)
    except Exception:
        params = np.array([1.0, 1.0, np.min(target)])
    return [(float(params[0]), float(params[1]), float(params[2]), mean, scale)]


def predict_exp(x: np.ndarray, params: Sequence[Sequence[float]]) -> np.ndarray:
    """Evaluate the legacy normalized exponential Fz model.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``; column 0 is used.
        params: Sequence whose first item contains ``(a, b, c, mean, scale)``.
    Returns:
        Prediction matrix of shape ``(n_rows, 1)`` with the legacy sign.
    Raises:
        IndexError/ValueError: If the parameter sequence or feature matrix has
            an incompatible structure.
    """
    a, b, c, mean, scale = params[0]
    return -np.asarray([exp_func((x[:, 0] - mean) / scale, a, b, c)]).T


def get_medians(
    x: np.ndarray,
    y: np.ndarray,
    one_on_one: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return median samples grouped by the first target-force column.

    Args:
        x: Feature matrix of shape ``(n_rows, n_features)``.
        y: Target matrix of shape ``(n_rows, n_outputs)``; column 0 defines
            the force groups.
        one_on_one: If true, group by force rounded to 0.5 units; otherwise
            sort and group adjacent values within the legacy 0.2-unit bin.
    Returns:
        Tuple ``(x_median, y_median)`` with one row per force group and the
        same feature/target column counts as the inputs. Empty inputs are
        returned unchanged.
    """

    if len(x) == 0:
        return x, y
    if one_on_one:
        rounded = np.round(y[:, 0] * 2) / 2
        return (
            np.asarray([np.median(x[rounded == value], axis=0) for value in np.sort(np.unique(rounded))]),
            np.asarray([np.median(y[rounded == value], axis=0) for value in np.sort(np.unique(rounded))]),
        )
    force_bin = 0.2
    order = np.argsort(y[:, 0])
    x_sorted, y_sorted = x[order], y[order]
    x_median: list[np.ndarray] = []
    y_median: list[np.ndarray] = []
    start = 0
    while start < len(y_sorted):
        end = start
        while end + 1 < len(y_sorted) and abs(y_sorted[end + 1, 0] - y_sorted[start, 0]) <= force_bin:
            end += 1
        x_median.append(np.median(x_sorted[start:end + 1], axis=0))
        y_median.append(np.median(y_sorted[start:end + 1], axis=0))
        start = end + 1
    return np.asarray(x_median), np.asarray(y_median)


def _fit_one(
    x: np.ndarray,
    y: np.ndarray,
    fit_type: str,
    poly_order: int,
    split_sign: bool,
    one_on_one: bool,
    allow_split: bool,
    feature_index: int = 0,
) -> tuple[Any, str, bool, np.ndarray]:
    """Fit one scalar input/output group and evaluate its training prediction.

    Args:
        x: Feature matrix ``(n_rows, n_features)``; ``feature_index`` selects
            the scalar curve input while polynomial paths retain all features.
        y: Target matrix ``(n_rows, n_outputs)``.
        fit_type: One of ``poly``, ``sigmoid``, ``exp_log``, ``pchip``,
            ``sym_exp``, ``sym_log`` or ``exp``.
        poly_order: Polynomial degree when the effective fit is polynomial.
        split_sign: Enable positive/negative branches where supported.
        one_on_one: Median grouping option for ``sym_log``.
        allow_split: Whether this output is allowed to use sign splitting;
            Fz normally disables it.
        feature_index: Zero-based scalar feature selected for non-polynomial
            curves.
    Returns:
        Tuple ``(parameters, normalized_fit_type, split, prediction)`` where
        ``prediction`` has shape ``y.shape`` and ``split`` describes the
        runtime parameter layout.
    Raises:
        ValueError: For empty data, unknown fit type, invalid feature index, or
            sign splitting without both signs.
    """

    if len(x) == 0:
        raise ValueError("no valid training rows")
    fit_type = fit_type.lower()
    supported = {"poly", "sigmoid", "exp_log", "pchip", "sym_exp", "sym_log", "exp"}
    if fit_type not in supported:
        raise ValueError(f"unsupported fit type: {fit_type}")

    if feature_index < 0 or feature_index >= x.shape[1]:
        raise ValueError(f"feature_index {feature_index} is outside the input matrix")
    scalar_x = x[:, feature_index:feature_index + 1]
    if fit_type == "sigmoid":
        params = fit_sigmoid(scalar_x, y)
        return params, fit_type, False, predict_sigmoid(scalar_x, params)
    use_special_split = split_sign and allow_split
    if use_special_split and fit_type == "sym_log":
        x_fit, y_fit = get_medians(scalar_x, y, one_on_one)
        params = fit_sym_log(x_fit, y_fit)
        return params, fit_type, False, predict_sym_log(scalar_x, params)
    if use_special_split and fit_type == "sym_exp":
        params = fit_sym_exp(scalar_x, y)
        return params, fit_type, False, predict_sym_exp(scalar_x, params)
    if use_special_split and fit_type == "exp_log":
        params = fit_exp_log(scalar_x, y)
        return params, fit_type, True, predict_exp_log(scalar_x, params)
    if use_special_split and fit_type == "pchip":
        params = fit_pchip(scalar_x, y)
        return params, fit_type, False, predict_pchip(scalar_x, params)
    if not use_special_split and fit_type == "exp":
        params = fit_exp(scalar_x, y)
        return params, fit_type, False, predict_exp(scalar_x, params)
    if not use_special_split and fit_type == "pchip":
        params = fit_pchip(scalar_x, y)
        return params, fit_type, False, predict_pchip(scalar_x, params)
    if use_special_split:
        positive = scalar_x[:, 0] >= 0
        negative = ~positive
        if not np.any(positive) or not np.any(negative):
            raise ValueError("split_sign requires both positive and negative input samples")
        positive_params = fit_polynomial(x[positive], y[positive], poly_order)
        negative_params = fit_polynomial(x[negative], y[negative], poly_order)
        params = [positive_params, negative_params]
        prediction = np.zeros_like(y)
        prediction[positive] = predict(x[positive], positive_params, poly_order)
        prediction[negative] = predict(x[negative], negative_params, poly_order)
        return params, "poly", True, prediction
    params = fit_polynomial(x, y, poly_order)
    return params, "poly", False, predict(x, params, poly_order)


def _fit_group(
    x: np.ndarray,
    y: np.ndarray,
    input_cols: list[str],
    output_cols: list[str],
    fit_types: Sequence[str],
    config: TrainingConfig,
) -> list[tuple[list[str], list[str], Any, str, bool]]:
    """Fit one DIM group, preserving legacy scalar and multi-output paths.

    Polynomial and sigmoid are the legacy multi-output algorithms. Other
    curves are scalar curves, so each output is fitted independently while
    retaining the complete input-column metadata in its result.

    Args:
        x: Feature matrix of shape ``(n_rows, len(input_cols))``.
        y: Target matrix of shape ``(n_rows, len(output_cols))``.
        input_cols: Ordered feature column names stored in each result.
        output_cols: Ordered target names stored in each result.
        fit_types: Fit type per output column.
        config: Training options controlling polynomial order, sign splitting
            and median grouping.
    Returns:
        List of ``(input_cols, output_cols, params, fit_type, split)`` tuples
        ready for :func:`save_coefs` and runtime prediction.
    Raises:
        ValueError: Propagated for unsupported fit types or invalid data.
    """

    normalized = [fit_type.lower() for fit_type in fit_types]
    if len(set(normalized)) == 1 and normalized[0] in {"poly", "sigmoid"}:
        params, fitted_type, split, _ = _fit_one(
            x, y, normalized[0], config.poly_order, config.split_sign,
            config.one_on_one, True,
        )
        return [(input_cols, output_cols, params, fitted_type, split)]

    results = []
    for index, (output_col, fit_type) in enumerate(zip(output_cols, normalized)):
        params, fitted_type, split, _ = _fit_one(
            x, y[:, index:index + 1], fit_type, config.poly_order,
            config.split_sign, config.one_on_one, output_col != "delta_Force_Z",
            feature_index=min(index, x.shape[1] - 1),
        )
        results.append((input_cols, [output_col], params, fitted_type, split))
    return results


def _type_id(fit_type: str) -> int:
    """Map a textual fit type to the binary runtime type identifier.

    Args:
        fit_type: Lowercase or case-insensitive fit name.
    Returns:
        Integer identifier used in ``fit_coefs.bin``; unknown names use the
        legacy polynomial fallback identifier ``1``. No file is written.
    """
    return {"sigmoid": 0, "poly": 1, "exp_log": 2, "pchip": 3, "sym_exp": 4, "sym_log": 5, "exp": 6}.get(fit_type, 1)


def _ensure_parent(path: Path) -> None:
    """Create the parent directory for an output path if necessary.

    Args:
        path: Destination path whose parent directory should exist.
    Returns:
        ``None``. The directory tree may be created as a filesystem side
        effect; an existing tree is left unchanged.
    Raises:
        OSError: If the parent directory cannot be created.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def save_coefs(
    fit_results: Sequence[tuple[Sequence[str], Sequence[str], Any, str, bool]],
    path: str | os.PathLike[str],
) -> None:
    """Write calibration results in the runtime ``fit_coefs.bin`` format.

    Args:
        fit_results: Sequence of tuples ``(input_columns, output_columns,
            params, fit_type, split)`` returned by :func:`_fit_group`.
        path: Binary destination path; parent directories are created.
    Returns:
        ``None``.
    Side Effects:
        Creates or overwrites ``path`` with two int32 counts, per-output
        metadata and float64 fit parameters compatible with
        ``tangential.processing.calibration``.
    Raises:
        OSError: If the destination cannot be created or written.
        IndexError/ValueError: If fit results have an unsupported parameter
            structure.
    """

    path = Path(path)
    _ensure_parent(path)
    with path.open("wb") as stream:
        n_inputs = len(fit_results[0][0]) if fit_results else 1
        total_outputs = sum(len(item[1]) for item in fit_results)
        np.int32(n_inputs).tofile(stream)
        np.int32(total_outputs).tofile(stream)
        for _, outputs, params, fit_type, split in fit_results:
            if fit_type == "pchip":
                count = len(params[0].x)
            elif fit_type in ("sym_exp", "sym_log"):
                count = 3
            elif fit_type == "sigmoid":
                count = 4
            elif fit_type == "exp":
                count = 5
            elif fit_type == "exp_log":
                count = 3
            else:
                count = params[0].shape[0] if split else params.shape[0]
            for _ in outputs:
                np.asarray([_type_id(fit_type), count, int(bool(split))], dtype=np.int32).tofile(stream)
        for _, outputs, params, fit_type, split in fit_results:
            if fit_type == "exp":
                for item in params:
                    np.asarray(item, dtype=np.float64).tofile(stream)
            elif fit_type in ("sym_exp", "sym_log", "exp_log"):
                for negative, positive in params:
                    np.asarray(negative, dtype=np.float64).tofile(stream)
                    np.asarray(positive, dtype=np.float64).tofile(stream)
            elif fit_type == "pchip":
                for interp in params:
                    np.asarray(interp.x, dtype=np.float64).tofile(stream)
                    np.asarray(interp(interp.x), dtype=np.float64).tofile(stream)
            elif split:
                for index in range(len(outputs)):
                    if fit_type == "sigmoid":
                        np.asarray(params[0][index], dtype=np.float64).tofile(stream)
                        np.asarray(params[1][index], dtype=np.float64).tofile(stream)
                    else:
                        np.asarray(params[0].T[index], dtype=np.float64).tofile(stream)
                        np.asarray(params[1].T[index], dtype=np.float64).tofile(stream)
            elif fit_type == "sigmoid":
                for item in params:
                    np.asarray(item, dtype=np.float64).tofile(stream)
            else:
                for item in params.T:
                    np.asarray(item, dtype=np.float64).tofile(stream)


def _runtime_entries(params: Any, fit_type: str, split: bool, output_count: int) -> list[Any]:
    """Adapt fitted parameters to ``apply_fit_predict_multi`` entries.

    Args:
        params: In-memory parameter structure produced by a fit function.
        fit_type: Normalized fit type controlling parameter interpretation.
        split: Whether polynomial parameters contain positive/negative branches.
        output_count: Number of output columns represented by ``params``.
    Returns:
        List of runtime entries, one per output, containing parameters, fit
        type and split flag. No file is written.
    Raises:
        IndexError/ValueError: If ``params`` does not match the requested type.
    """

    if fit_type in ("sym_exp", "sym_log", "exp_log"):
        return [(item, fit_type, split) for item in params]
    if fit_type == "exp":
        return [(np.asarray(item, dtype=np.float64), fit_type, split) for item in params]
    if fit_type == "pchip":
        return [(item, fit_type, split) for item in params]
    if split:
        return [
            ((np.asarray(params[0][:, index]), np.asarray(params[1][:, index])), fit_type, True)
            for index in range(output_count)
        ]
    if fit_type == "sigmoid":
        return [(np.asarray(item), fit_type, False) for item in params]
    return [
        (np.asarray(params[:, index]), fit_type, False)
        for index in range(output_count)
    ]


def _poly_order_from_term_count(term_count: int, n_features: int) -> int:
    """Infer polynomial order from saved term count and feature count.

    Args:
        term_count: Number of coefficient rows in a saved polynomial.
        n_features: Number of input features used by that polynomial.
    Returns:
        Degree 1, 2 or 3 whose design matrix has ``term_count`` terms.
    Raises:
        ValueError: If no supported degree has the requested term count.
    """
    for order in (1, 2, 3):
        if build_design_matrix(np.zeros((1, n_features)), order).shape[1] == term_count:
            return order
    raise ValueError("cannot infer polynomial order from saved coefficients")


def _feature_index_for_output(input_cols: Sequence[str], output_col: str) -> int:
    """Choose the legacy scalar feature associated with an output column.

    Args:
        input_cols: Ordered available feature names.
        output_col: Target name, conventionally ``delta_Force_X/Y/Z``.
    Returns:
        Zero-based index of the preferred CoP-X, CoP-Y or ADC-sum feature, or
        ``0`` when the preferred feature is absent. No file I/O occurs.
    """
    preferred = {
        "delta_Force_X": "delta_CoP_X",
        "delta_Force_Y": "delta_CoP_Y",
        "delta_Force_Z": "adc_sum",
    }.get(output_col)
    if preferred in input_cols:
        return input_cols.index(preferred)
    return 0


def _predict_result_batch(
    result: tuple[list[str], list[str], Any, str, bool],
    values: np.ndarray,
) -> np.ndarray:
    """Predict every row in a fitted result while preserving output shape.

    Args:
        result: Tuple ``(input_cols, output_cols, params, fit_type, split)``.
        values: Feature matrix of shape ``(n_rows, len(input_cols))``.
    Returns:
        Prediction matrix of shape ``(n_rows, len(output_cols))``.
    Raises:
        ValueError/IndexError: If parameters, feature count or fit type is
            incompatible with ``values``.
    """

    input_cols, output_cols, params, fit_type, split = result
    output_count = len(output_cols)
    if fit_type == "poly":
        order = _poly_order_from_term_count(
            params[0].shape[0] if split else params.shape[0], len(input_cols)
        )
        if split:
            positive = values[:, 0] >= 0
            prediction = np.zeros((len(values), output_count))
            prediction[positive] = predict(values[positive], params[0], order)
            prediction[~positive] = predict(values[~positive], params[1], order)
            return prediction
        return predict(values, params, order)
    prediction = np.zeros((len(values), output_count))
    entries = _runtime_entries(params, fit_type, split, output_count)
    for output_index, output_col in enumerate(output_cols):
        scalar_index = _feature_index_for_output(input_cols, output_col)
        scalar_values = values[:, scalar_index:scalar_index + 1]
        for row_index, scalar_value in enumerate(scalar_values[:, 0]):
            prediction[row_index, output_index] = apply_fit_predict_multi(
                [float(scalar_value)], [entries[output_index]], fit_type, split
            )[0]
    return prediction


def _predict_result(result: tuple[list[str], list[str], Any, str, bool], values: Sequence[float]) -> list[float]:
    """Predict one row using a fitted result tuple.

    Args:
        result: Runtime fit tuple as accepted by :func:`_predict_result_batch`.
        values: One ordered feature vector of length ``len(result[0])``.
    Returns:
        Python list containing one prediction per output column.
    Raises:
        ValueError/IndexError: If the feature vector or fit parameters are
            incompatible.
    """
    return _predict_result_batch(result, np.asarray([values], dtype=np.float64))[0].tolist()


def write_back_csv(
    source_csv: str | os.PathLike[str],
    target_csv: str | os.PathLike[str],
    fit_results: Sequence[tuple[list[str], list[str], Any, str, bool]],
    valid_only: bool,
    force: bool,
) -> Path:
    """Copy a CSV while adding or updating calibrated force columns.

    Args:
        source_csv: Source CSV path; it is read by header names and never
            overwritten by this function.
        target_csv: Destination CSV path. Parent directories are created.
        fit_results: Fitted result tuples used to calculate ``Fx_cal`` and
            ``Fy_cal`` from each row.
        valid_only: If true, only rows accepted by :func:`_row_valid` receive
            predictions; other rows retain blank calibration fields.
        force: Permit an existing target or source/target identity. Without it,
            these cases raise before any write.
    Returns:
        Resolved target :class:`Path` after the complete CSV is written.
    Side Effects:
        Writes a copy with ``Fx_cal``, ``Fy_cal`` and ``Force_cal_angle``
        columns appended when absent. Existing target contents may be
        overwritten only when ``force=True``.
    Raises:
        FileNotFoundError/OSError: If source or target cannot be accessed.
        ValueError: If source has no header or a row cannot be interpreted.
        FileExistsError: If write-back protection rejects the target.
    """

    source = Path(source_csv)
    target = Path(target_csv)
    _validate_write_back(source, target, force)
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {source}")
        header = [name.strip() for name in reader.fieldnames]
        rows = list(reader)
    for column in ("Fx_cal", "Fy_cal", "Force_cal_angle"):
        if column not in header:
            header.append(column)
    for row in rows:
        for column in ("Fx_cal", "Fy_cal", "Force_cal_angle"):
            row.setdefault(column, "")
        if valid_only and not _row_valid(row, header):
            continue
        fx = fy = 0.0
        for result in fit_results:
            input_cols, output_cols, _, _, _ = result
            try:
                values = [float(row[column]) for column in input_cols]
                prediction = _predict_result(result, values)
            except (KeyError, TypeError, ValueError, NotImplementedError):
                continue
            for output, value in zip(output_cols, prediction):
                if "X" in output or "x" in output:
                    fx = value
                elif "Y" in output or "y" in output:
                    fy = value
        angle = float(np.degrees(np.arctan2(fy, fx + 1e-8)))
        if angle < 0:
            angle += 360
        row["Fx_cal"] = f"{fx:.6f}"
        row["Fy_cal"] = f"{fy:.6f}"
        row["Force_cal_angle"] = f"{angle:.6f}"
    _ensure_parent(target)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    return target


def _validate_write_back(source: Path, target: Path, force: bool) -> None:
    """Enforce the explicit write-back overwrite and collision policy.

    Args:
        source: Original CSV path.
        target: Requested output CSV path.
        force: Allow an existing target or identical source/target path.
    Returns:
        ``None`` when writing is permitted.
    Raises:
        FileExistsError: If the target exists or resolves to source while
            ``force`` is false.
    """
    if target.exists() and not force:
        raise FileExistsError(f"write-back target already exists: {target}")
    if target.resolve() == source.resolve() and not force:
        raise FileExistsError("write-back source and target are identical; use force=True")


def _plot_report(
    output_path: Path,
    training_data: Sequence[tuple[np.ndarray, np.ndarray, list[str], list[str], Any, str, bool]],
) -> None:
    """Create a compact PNG fit report using lazy Matplotlib import.

    Args:
        output_path: PNG destination; parent directories are created.
        training_data: Sequence of ``(x, y, input_cols, output_cols, params,
            fit_type, split)`` tuples. ``x`` is ``(n, n_features)`` and ``y``
            is ``(n, n_outputs)``.
    Returns:
        ``None``.
    Side Effects:
        Imports Matplotlib only when called and writes/overwrites one PNG.
        Failed diagnostic predictions are skipped while the report is still
        generated.
    Raises:
        OSError: If the report cannot be written.
        ImportError: If Matplotlib is unavailable in the environment.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(training_data), 1, figsize=(10, max(4, 4 * len(training_data))), squeeze=False)
    for index, (x, y, input_cols, output_cols, params, fit_type, split) in enumerate(training_data):
        axis = axes[index, 0]
        plot_index = _feature_index_for_output(input_cols, output_cols[0])
        axis.scatter(x[:, plot_index], y[:, 0], s=10, alpha=0.35, label="data")
        grid = np.linspace(np.min(x[:, plot_index]), np.max(x[:, plot_index]), 300)
        grid_x = np.zeros((len(grid), x.shape[1]))
        grid_x[:, plot_index] = grid
        try:
            prediction = _predict_result_batch(
                (input_cols, output_cols, params, fit_type, split), grid_x
            )[:, 0]
            if len(prediction) != len(grid):
                raise ValueError("plot prediction length does not match grid length")
            axis.plot(grid, prediction, "g-", linewidth=2, label="fit")
        except ValueError:
            pass
        axis.set_title(f"{output_cols[0]} <- {input_cols[0]}")
        axis.set_xlabel(input_cols[plot_index])
        axis.set_ylabel(output_cols[0])
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _csv_has_columns(path: Path, columns: Sequence[str]) -> bool:
    """Check whether a CSV header contains every requested column.

    Args:
        path: CSV path to inspect.
        columns: Required stripped header names.
    Returns:
        ``True`` if all names are present; unreadable files return ``False``.
        No file is modified.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            fieldnames = csv.DictReader(stream).fieldnames or []
    except OSError:
        return False
    return set(columns).issubset({name.strip() for name in fieldnames})


def _select_source(paths: Sequence[Path], columns: Sequence[str]) -> Path:
    """Select the first CSV whose header contains all requested columns.

    Args:
        paths: Candidate CSV paths in priority order.
        columns: Required header names.
    Returns:
        First matching path.
    Raises:
        ValueError: If no candidate contains all requested columns.
    """
    for path in paths:
        if _csv_has_columns(path, columns):
            return path
    joined = ", ".join(columns)
    raise ValueError(f"none of the training CSVs contains columns: {joined}")


def train_model(config: TrainingConfig) -> TrainingResult:
    """Fit configured calibration curves, write the model and report artifacts.

    Args:
        config: :class:`TrainingConfig` containing XY/Z CSV paths, dimensional
            mode, fit types, valid-row policy, output paths and write-back
            protection.
    Returns:
        :class:`TrainingResult` containing the written model path, optional PNG,
        runtime-compatible fit tuples, accepted sample counts per output, and
        the optional write-back target.
    Side Effects:
        Reads CSVs by the named columns in ``INPUT_COLS``/``OUTPUT_COLS``;
        when ``valid_only`` is true it selects nonzero ``valid`` rows or
        falls back to nonzero ``CoP_state``. Fits the configured ``poly``,
        ``sigmoid``, ``exp_log``, ``pchip``, ``sym_exp``, ``sym_log`` or
        ``exp`` models, writes ``output_model``, optionally writes a Matplotlib
        report to ``output_plot``, and only writes ``write_back`` when that
        path is explicitly configured and passes ``force`` protection.
    Raises:
        ValueError: For unsupported dimensions/orders, missing columns, no
            valid rows, unsupported fit types or incompatible dim=3 sources.
        FileExistsError: If write-back would overwrite without ``force=True``.
        OSError: If input or output files cannot be accessed.
    """

    if config.dim not in (1, 2, 3):
        raise ValueError("dim must be 1, 2, or 3")
    if config.poly_order not in (1, 2, 3):
        raise ValueError("poly_order must be 1, 2, or 3")
    xy_path = Path(config.xy_csv)
    z_path = Path(config.z_csv)
    fit_types = {"delta_Force_X": config.fx, "delta_Force_Y": config.fy, "delta_Force_Z": config.fz}
    if config.write_back is not None:
        _validate_write_back(xy_path, Path(config.write_back), config.force)
    fit_results: list[tuple[list[str], list[str], Any, str, bool]] = []
    training_data = []
    sample_counts: dict[str, int] = {}
    if config.dim == 1:
        groups = [
            ([input_col], [output_col], [fit_types[output_col]],
             z_path if output_col == "delta_Force_Z" else xy_path, None)
            for input_col, output_col in zip(INPUT_COLS, OUTPUT_COLS)
        ]
    else:
        group_inputs = list(INPUT_COLS[:config.dim])
        group_outputs = list(OUTPUT_COLS[:config.dim])
        group_types = [fit_types[name] for name in group_outputs]
        try:
            source = _select_source((xy_path, z_path), (*group_inputs, *group_outputs))
        except ValueError:
            if config.dim == 3:
                raise ValueError(
                    "dim=3 requires xy_csv or z_csv to contain all three "
                    "input columns and all three output columns"
                )
            raise
        groups = [(group_inputs, group_outputs, group_types, source, None)]

    for input_cols, output_cols, group_types, source, prepared in groups:
        if prepared is None:
            inputs, outputs = load_csv(source, input_cols, output_cols, config.valid_only)
        else:
            inputs, outputs = prepared
        if len(inputs) == 0:
            raise ValueError(f"no valid training rows for {', '.join(output_cols)} in {source}")
        results = _fit_group(inputs, outputs, input_cols, output_cols, group_types, config)
        fit_results.extend(results)
        for result in results:
            result_outputs = result[1]
            result_indices = [output_cols.index(name) for name in result_outputs]
            result_y = outputs[:, result_indices]
            training_data.append((inputs, result_y, *result[0:2], result[2], result[3], result[4]))
        for output_col in output_cols:
            sample_counts[output_col] = len(inputs)

    model_path = Path(config.output_model)
    save_coefs(fit_results, model_path)
    plot_path = None
    if config.output_plot is not None:
        plot_path = Path(config.output_plot)
        _plot_report(plot_path, training_data)
    written_path = None
    if config.write_back is not None:
        written_path = write_back_csv(
            xy_path,
            config.write_back,
            fit_results,
            config.valid_only,
            config.force,
        )
    return TrainingResult(model_path, plot_path, fit_results, sample_counts, written_path)


__all__ = [
    "INPUT_COLS", "OUTPUT_COLS", "TrainingConfig", "TrainingResult",
    "load_csv", "log_func", "log_func_o", "exp_func", "sigmoid",
    "build_design_matrix", "get_term_labels", "fit_polynomial", "predict",
    "fit_sym_exp", "predict_sym_exp", "fit_sym_log", "predict_sym_log",
    "fit_exp_log", "predict_exp_log", "fit_sigmoid", "predict_sigmoid",
    "fit_pchip", "predict_pchip", "fit_exp", "predict_exp", "get_medians",
    "save_coefs", "write_back_csv", "train_model",
]
