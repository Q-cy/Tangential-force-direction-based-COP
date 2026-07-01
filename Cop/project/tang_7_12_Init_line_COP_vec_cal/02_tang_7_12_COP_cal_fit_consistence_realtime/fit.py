import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, differential_evolution
from scipy.interpolate import PchipInterpolator


# ===================== Config =====================
TRAIN_CSV_xy = "/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0615_6.csv"
TARGET_CSV_xy = "/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0615_6.csv"
TRAIN_CSV_z = "/home/qcy/Project/data/2.PZT_tangential/weight/concat/concat_5_10_15.csv"
TARGET_CSV_z = "/home/qcy/Project/data/2.PZT_tangential/weight/concat/concat_5_10_15.csv"
FIT_PARAM_Save = "/home/qcy/Project/data/2.PZT_tangential/weight/png"

# Full input/output columns
INPUT_COLS = ["delta_CoP_X", "delta_CoP_Y", "adc_sum"]
OUTPUT_COLS = ["delta_Force_X", "delta_Force_Y", "delta_Force_Z"]

DIM = 1                      # 1=each pair independently, 2=first 2 together, 3=all 3 together
POLY_ORDER = 3               # 1=linear, 2=quadratic, 3=cubic (only used if FIT_TYPE="poly")
FIT_TYPE_FX = "sym_log"      # "poly"/"sigmoid"/"exp_log"/"pchip"/"sym_exp"/"sym_log"
FIT_TYPE_FY = "sym_log"
FIT_TYPE_FXY = "sym_log"
FIT_TYPE_FZ = "exp"    
FIT_TYPE_FXYZ = "sym_log"  

TRAIN_VALID_ONLY = True
WRITE_VALID_ONLY = True
ONE_ON_ONE = True           # 每个力值只取一个中位数
SAVE_COEFS = True
SPLIT_SIGN = True           # True=正负分开拟合，False=不分

# ===================== Read CSV =====================

def load_csv(csv_path, input_cols, output_cols, valid_only=True):
    """Read CSV, return (X[N, n_in], Y[N, n_out])"""
    X_rows, Y_rows = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)                                          # 迭代器，会逐行读取
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            try:
                if valid_only and float(row.get("valid", 0)) == 0:
                    continue
                x_vals = [float(row[c]) for c in input_cols]               # 字典按名取值
                y_vals = [float(row[c]) for c in output_cols]
                X_rows.append(x_vals)
                Y_rows.append(y_vals)
            except (KeyError, ValueError):
                continue
    return np.array(X_rows), np.array(Y_rows)                              # 转换前N个list，转换后[N,n_in](样本数，特征数)


#===================== Func define =====================
def log_func(x, a, b, c):
    """Logarithmic: y = a * ln(b * x + 1) + c"""
    return a * np.log(b * x + 1) + c

def exp_func(x, a, b, c):
    """Exponential: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


# ===================== Polynomial fit =====================

def build_design_matrix(X, order):
    n_vars = X.shape[1]
    if order == 1:
        return np.column_stack([np.ones(len(X))] + [X[:, i] for i in range(n_vars)])
    elif order == 2:
        cols = [np.ones(len(X))]
        for i in range(n_vars):
            cols.append(X[:, i])
        for i in range(n_vars):
            for j in range(i, n_vars):
                cols.append(X[:, i] * X[:, j])
        return np.column_stack(cols)
    elif order == 3:
        cols = [np.ones(len(X))]
        for i in range(n_vars):
            cols.append(X[:, i])
        for i in range(n_vars):
            for j in range(i, n_vars):
                cols.append(X[:, i] * X[:, j])
        for i in range(n_vars):
            for j in range(i, n_vars):
                for k in range(j, n_vars):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)
    else:
        raise ValueError(f"Unsupported order: {order}")

def get_term_labels(input_cols, order):
    labels = ["1"]
    n = len(input_cols)
    short = [c.replace("delta_", "").replace("_", "") for c in input_cols]
    if order >= 1:
        labels.extend(short)
    if order >= 2:
        for i in range(n):
            for j in range(i, n):
                labels.append(f"{short[i]}*{short[j]}")
    if order >= 3:
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    labels.append(f"{short[i]}*{short[j]}*{short[k]}")
    return labels

def fit_polynomial(X, Y, order):
    A = build_design_matrix(X, order)
    coefs, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    return coefs

def predict(X, coefs, order):
    A = build_design_matrix(X, order)
    return A @ coefs


# ===================== Symmetric Exponential fit =====================

def sym_exp_func(x, a, b, c):
    """Symmetric exponential: exp for x>=0, rotated exp for x<0. Works on scalar or array."""
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.zeros_like(x)
    pos = x >= 0
    neg = x < 0
    out[pos] = a * np.exp(b * x[pos]) + c
    out[neg] = -(a * np.exp(b * (-x[neg])) + c)
    return float(out[0]) if len(out) == 1 else out

def fit_sym_exp(X, Y):
    """Fit symmetric exp: exp for positive, rotated exp for negative."""
    n_out = Y.shape[1]
    all_params = []
    for i in range(n_out):
        x = X[:, 0]
        y = Y[:, i]
        pos_mask = x >= 0
        neg_mask = x < 0

        def _fit_exp_side(x_data, y_data):
            if len(x_data) < 3:
                return np.array([1.0, 1.0, 0.0])
            x_max = np.max(np.abs(x_data))
            if x_max > 10:
                scale = 1.0 / x_max
                x_scaled = x_data * scale
                b0 = 1.0
            else:
                scale = 1.0; x_scaled = x_data; b0 = 5.0
            try:
                popt, _ = curve_fit(exp_func, x_scaled, y_data, p0=[1.0, b0, np.min(y_data)], maxfev=10000)
                popt[1] *= scale; return popt
            except:
                return np.array([1.0, 1.0, 0.0])

        popt_pos = _fit_exp_side(x[pos_mask], y[pos_mask]) if np.sum(pos_mask) > 3 else np.array([1.0, 1.0, 0.0])
        if np.sum(neg_mask) > 3:
            popt_neg = _fit_exp_side(-x[neg_mask], -y[neg_mask])
        else:
            popt_neg = popt_pos.copy()
        all_params.append((popt_neg, popt_pos))
        print(f"  {i} +: a={popt_pos[0]:.4f}, b={popt_pos[1]:.4f}, c={popt_pos[2]:.4f}")
        print(f"  {i} -: a={popt_neg[0]:.4f}, b={popt_neg[1]:.4f}, c={popt_neg[2]:.4f}")
    return all_params

def predict_sym_exp(X, params_list):
    """Predict using sym_exp params. X[N,1], params_list: list of (p_neg, p_pos)."""
    Y_pred = np.zeros((len(X), len(params_list)))
    for i, (p_neg, p_pos) in enumerate(params_list):
        x = X[:, 0]
        neg_mask = x < 0
        pos_mask = x >= 0
        if np.any(pos_mask):
            Y_pred[pos_mask, i] = exp_func(x[pos_mask], *p_pos)
        if np.any(neg_mask):
            Y_pred[neg_mask, i] = -exp_func(-x[neg_mask], *p_neg)
    return Y_pred


# ===================== Symmetric Logarithmic fit =====================

def fit_sym_log(X, Y):
    """Fit symmetric log. Auto-normalizes if |x| > 100."""
    n_out = Y.shape[1]
    x_raw = X[:, 0]
    x = x_raw
    if np.max(np.abs(x_raw)) > 100:
        x_mean, x_std = float(np.mean(x_raw)), float(np.std(x_raw)) or 1.0
        x = (x_raw - x_mean) / x_std
    all_params = []
    for i in range(n_out):
        y = Y[:, i]; xi = x
        pos_mask = xi >= 0; neg_mask = xi < 0
        if np.sum(pos_mask) > 3:
            try: popt_pos, _ = curve_fit(log_func, xi[pos_mask], y[pos_mask],
                                          p0=[1.0, 1.0, np.min(y[pos_mask])], maxfev=10000)
            except: popt_pos = np.array([1.0, 1.0, np.min(y[pos_mask])])
        else: popt_pos = np.array([1.0, 1.0, 0.0])
        if np.sum(neg_mask) > 3:
            try: popt_neg, _ = curve_fit(log_func, -xi[neg_mask], -y[neg_mask],
                                          p0=[1.0, 1.0, np.min(-y[neg_mask])], maxfev=10000)
            except: popt_neg = np.array([1.0, 1.0, np.min(-y[neg_mask])])
        else: popt_neg = popt_pos.copy()
        all_params.append((popt_neg, popt_pos))
        print(f"  {i} +: a={popt_pos[0]:.4f}, b={popt_pos[1]:.4f}, c={popt_pos[2]:.4f}")
        print(f"  {i} -: a={popt_neg[0]:.4f}, b={popt_neg[1]:.4f}, c={popt_neg[2]:.4f}")
    return all_params

def predict_sym_log(X, params_list):
    """Predict using sym_log params. Auto-normalizes large x."""
    Y_pred = np.zeros((len(X), len(params_list)))
    x_raw = X[:, 0]
    x = x_raw
    if np.max(np.abs(x_raw)) > 100:
        x_mean, x_std = float(np.mean(x_raw)), float(np.std(x_raw)) or 1.0
        x = (x_raw - x_mean) / x_std
    for i, (p_neg, p_pos) in enumerate(params_list):
        pos_mask, neg_mask = x >= 0, x < 0
        if np.any(pos_mask): Y_pred[pos_mask, i] = log_func(x[pos_mask], *p_pos)
        if np.any(neg_mask): Y_pred[neg_mask, i] = -log_func(-x[neg_mask], *p_neg)
    return Y_pred


# ===================== EXP/LOG fit =====================

def fit_exp_log(X, Y):
    """Fit exp for negative inputs, log for positive inputs. X[N,1], Y[N,n_out]."""
    n_out = Y.shape[1]
    all_params = []
    for i in range(n_out):
        x = X[:, 0]
        y = Y[:, i]

        # Negative: exponential
        neg_mask = x < 0
        if np.sum(neg_mask) > 3:
            x_neg, y_neg = x[neg_mask], y[neg_mask]
            try:
                # Initial guess: a=1, b=1, c=y_min
                popt_neg, _ = curve_fit(exp_func, x_neg, y_neg,
                                        p0=[1.0, 1.0, np.min(y_neg)], maxfev=10000)
            except Exception:
                popt_neg = np.array([1.0, 1.0, np.min(y_neg)])
        else:
            popt_neg = np.array([1.0, 1.0, 0.0])

        # Positive: logarithmic
        pos_mask = x >= 0
        if np.sum(pos_mask) > 3:
            x_pos, y_pos = x[pos_mask], y[pos_mask]
            try:
                popt_pos, _ = curve_fit(log_func, x_pos, y_pos,
                                        p0=[1.0, 1.0, np.min(y_pos)], maxfev=10000)
            except Exception:
                popt_pos = np.array([1.0, 1.0, np.min(y_pos)])
        else:
            popt_pos = np.array([1.0, 1.0, 0.0])

        all_params.append((popt_neg, popt_pos))
    return all_params

def predict_exp_log(X, params_list):
    """Predict using exp/log params. X[N,1], params_list: list of (p_neg, p_pos)."""
    Y_pred = np.zeros((len(X), len(params_list)))
    for i, (p_neg, p_pos) in enumerate(params_list):
        x = X[:, 0]
        neg_mask = x < 0
        pos_mask = x >= 0
        if np.any(neg_mask):
            Y_pred[neg_mask, i] = exp_func(x[neg_mask], *p_neg)
        if np.any(pos_mask):
            Y_pred[pos_mask, i] = log_func(x[pos_mask], *p_pos)
    return Y_pred


# ===================== Sigmoid fit =====================

def sigmoid(x, L, k, x0, b):
    """Sigmoid function: y = L / (1 + exp(-k*(x-x0))) + b"""
    return L / (1 + np.exp(-k * (x - x0))) + b

def fit_sigmoid(X, Y):
    """Fit sigmoid for each output column. X[N,1], Y[N,n_out]. Returns list of param arrays."""
    n_out = Y.shape[1]
    all_params = []
    for i in range(n_out):
        x = X[:, 0]
        y = Y[:, i]
        # Initial guess
        y_min, y_max = np.min(y), np.max(y)
        L0 = y_max - y_min
        x0_0 = np.median(x)
        k0 = 10.0
        b0 = y_min
        try:
            popt, _ = curve_fit(sigmoid, x, y, p0=[L0, k0, x0_0, b0], maxfev=10000)
            all_params.append(popt)
        except Exception as e:
            print(f"  Sigmoid fit failed for output {i}: {e}")
            all_params.append(np.array([L0, k0, x0_0, b0]))
    return all_params

def predict_sigmoid(X, params_list):
    """Predict using sigmoid params. X[N,1], params_list: list of [L,k,x0,b]."""
    Y_pred = np.zeros((len(X), len(params_list)))
    for i, p in enumerate(params_list):
        Y_pred[:, i] = sigmoid(X[:, 0], *p)
    return Y_pred


# ===================== PCHIP fit =====================

def fit_pchip(X, Y):
    """Fit PCHIP for each output column. X[N,1], Y[N,n_out]. Returns list of interpolators."""
    interpolators = []
    for i in range(Y.shape[1]):
        x, y = X[:, 0].copy(), Y[:, i].copy()
        sort_idx = np.argsort(x)
        x_sorted, y_sorted = x[sort_idx], y[sort_idx]
        # Remove duplicate x values
        keep = np.diff(x_sorted, prepend=x_sorted[0]-1) != 0
        x_uniq, y_uniq = x_sorted[keep], y_sorted[keep]
        interpolators.append(PchipInterpolator(x_uniq, y_uniq))
    return interpolators

def predict_pchip(X, interpolators):
    """Predict using PCHIP interpolators. X[N,1]."""
    Y_pred = np.zeros((len(X), len(interpolators)))
    for i, interp in enumerate(interpolators):
        Y_pred[:, i] = interp(X[:, 0])
    return Y_pred


# ===================== Print formula =====================

def print_formulas(coefs, labels, output_cols):
    print(f"\n{'='*60}")
    print(f"  Fit result ({len(labels)-1} terms)")
    print(f"{'='*60}")
    for i, name in enumerate(output_cols):
        terms = []
        for c, label in zip(coefs[:, i], labels):
            if abs(c) < 1e-10:
                continue
            terms.append(f"{c:.6f}*{label}")
        formula = " + ".join(terms)
        print(f"  {name} = {formula}")
    print(f"{'='*60}\n")


# ===================== Error analysis =====================

def compute_errors(Y_true, Y_pred, output_cols):
    print(f"\n{'='*60}")
    print(f"  Error Analysis")
    print(f"{'='*60}")
    for i, name in enumerate(output_cols):
        err = Y_pred[:, i] - Y_true[:, i]
        abs_err = np.abs(err)
        ss_res = np.sum(err ** 2)
        ss_tot = np.sum((Y_true[:, i] - np.mean(Y_true[:, i])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        print(f"  {name}:")
        print(f"    MAE  = {np.mean(abs_err):.6f}")
        print(f"    RMSE = {np.sqrt(np.mean(err**2)):.6f}")
        print(f"    Max  = {np.max(abs_err):.6f}")
        print(f"    R2   = {r2:.6f}")
    print(f"{'='*60}\n")


# ===================== Save coefficients =====================

def save_coefs(fit_results, path):
    """Save fit results to .bin with per-output metadata."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        n_inputs = len(fit_results[0][0]) if fit_results else 1
        total_outputs = sum(len(out) for _, out, _, _, _ in fit_results)

        def _type_id(ft):
            m = {"sigmoid": 0, "poly": 1, "exp_log": 2, "pchip": 3,
                 "sym_exp": 4, "sym_log": 5, "exp": 6}
            return m.get(ft, 1)

        # Header: n_inputs, n_outputs
        f.write(np.int32(n_inputs).tobytes())
        f.write(np.int32(total_outputs).tobytes())

        # Per-output metadata: (fit_type_id, n_params, split) per output
        for _, out, params, ftype, split in fit_results:
            fid = _type_id(ftype)
            if ftype == "pchip":
                npar = len(params[0].x)
            elif ftype in ("sym_exp", "sym_log"):
                npar = 3
            elif ftype == "sigmoid":
                npar = 4
            elif ftype == "exp":
                npar = 5
            elif ftype == "exp_log":
                npar = 3
            else:  # poly
                npar = params[0].shape[0] if split else params.shape[0]
            for _ in range(len(out)):
                f.write(np.int32(fid).tobytes())
                f.write(np.int32(npar).tobytes())
                f.write(np.int32(1 if split else 0).tobytes())

        # Per-output params data
        for _, out, params, ftype, split in fit_results:
            n_out = len(out)
            if ftype == "exp":
                for p in params:
                    f.write(np.array(p, dtype=np.float64).tobytes())
            elif ftype in ("sym_exp", "sym_log"):
                if split:
                    # params = [(p_neg_pos, p_pos_pos), (p_neg_neg, p_pos_neg)]
                    # Combine: p_neg from neg-fit, p_pos from pos-fit → 1 entry per output
                    combined_neg = np.array(params[1][0], dtype=np.float64)
                    combined_pos = np.array(params[0][1], dtype=np.float64)
                    f.write(combined_neg.tobytes())
                    f.write(combined_pos.tobytes())
                else:
                    for p_neg, p_pos in params:
                        f.write(np.array(p_neg, dtype=np.float64).tobytes())
                        f.write(np.array(p_pos, dtype=np.float64).tobytes())
            elif ftype == "pchip":
                for interp in params:
                    x = np.array(interp.x, dtype=np.float64)
                    y = np.array(interp(x), dtype=np.float64)
                    f.write(x.tobytes())
                    f.write(y.tobytes())
            elif ftype == "exp_log":
                for p_neg, p_pos in params:
                    f.write(np.array(p_neg, dtype=np.float64).tobytes())
                    f.write(np.array(p_pos, dtype=np.float64).tobytes())
            elif split:
                # Interleave pos/neg per output to match reader order
                for i in range(n_out):
                    if ftype == "sigmoid":
                        f.write(np.array(params[0][i], dtype=np.float64).tobytes())
                        f.write(np.array(params[1][i], dtype=np.float64).tobytes())
                    else:
                        f.write(np.array(params[0].T[i], dtype=np.float64).tobytes())
                        f.write(np.array(params[1].T[i], dtype=np.float64).tobytes())
            else:
                if ftype == "sigmoid":
                    for p in params:
                        f.write(np.array(p, dtype=np.float64).tobytes())
                else:
                    for c in params.T:
                        f.write(np.array(c, dtype=np.float64).tobytes())
    print(f"  Coefs saved: {path} ({os.path.getsize(path)} bytes, {total_outputs} outputs)")

def load_coefs(path):
    """Load fit coefs from .bin. Returns (fit_type, n_inputs, params_list, split_sign).
    Each entry in params_list is a 3-tuple (params, fit_type_str, split_bool)."""
    _ID_MAP = {0: "sigmoid", 1: "poly", 2: "exp_log", 3: "pchip",
               4: "sym_exp", 5: "sym_log", 6: "exp"}

    with open(path, "rb") as f:
        n_inputs = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n_outputs = int(np.frombuffer(f.read(4), dtype=np.int32)[0])

        # Read per-output metadata
        meta = []  # list of (fit_type_str, n_params, split)
        for _ in range(n_outputs):
            fid = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            npar = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            spl = int(np.frombuffer(f.read(4), dtype=np.int32)[0]) == 1
            meta.append((_ID_MAP.get(fid, "poly"), npar, spl))

        # Read per-output params
        params_list = []
        for ft, npar, spl in meta:
            if ft in ("sym_exp", "sym_log"):
                p_neg = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                p_pos = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                params_list.append(((p_neg, p_pos), ft, spl))
            elif ft == "pchip":
                from scipy.interpolate import PchipInterpolator
                x_knots = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                y_knots = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                params_list.append((PchipInterpolator(x_knots, y_knots), ft, spl))
            elif ft == "exp_log":
                p_neg = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                p_pos = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                params_list.append(((p_neg, p_pos), ft, spl))
            elif ft == "exp":
                p = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                params_list.append((p, ft, spl))
            elif spl:
                p_pos = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                p_neg = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                params_list.append(((p_pos, p_neg), ft, spl))
            else:
                p = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                params_list.append((p, ft, spl))

    first_ft = meta[0][0] if meta else "poly"
    first_split = meta[0][2] if meta else False
    return first_ft, n_inputs, params_list, first_split

def _resolve_entry(p, fit_type, split_sign):
    """If p is a 3-tuple (params, ft, sp) from new load_coefs, use per-entry metadata.
    Otherwise fall back to the global fit_type/split_sign for old-format params."""
    if isinstance(p, tuple) and len(p) == 3:
        return p[0], p[1], p[2]
    return p, fit_type, split_sign


def apply_predict(x_val, params_list, fit_type, split_sign=False):
    """Predict using loaded params. x_val: scalar or 1D array. Returns list of predictions."""
    x = float(x_val) if np.isscalar(x_val) else float(x_val[0])
    results = []
    for p in params_list:
        params, ft, sp = _resolve_entry(p, fit_type, split_sign)
        if ft == "pchip":
            results.append(float(params(x)))
        elif ft in ("sym_exp", "sym_log"):
            p_neg, p_pos = params
            if x < 0:
                fn = exp_func if ft == "sym_exp" else log_func
                results.append(float(-fn(-x, *p_neg)))
            else:
                fn = exp_func if ft == "sym_exp" else log_func
                results.append(float(fn(x, *p_pos)))
        elif ft == "exp_log":
            p_neg, p_pos = params
            if x < 0:
                results.append(float(exp_func(x, *p_neg)))
            else:
                results.append(float(log_func(x, *p_pos)))
        elif ft == "exp":
            a, b, c, xm, xs = params
            results.append(float(-exp_func((x - xm) / xs, a, b, c)))
        elif sp:
            psel = params[0] if x >= 0 else params[1]
            if ft == "sigmoid":
                results.append(float(sigmoid(x, *psel)))
            else:
                basis = np.array([x**j for j in range(len(psel))])
                results.append(float(np.dot(psel, basis)))
        else:
            if ft == "sigmoid":
                results.append(float(sigmoid(x, *params)))
            else:
                basis = np.array([x**j for j in range(len(params))])
                results.append(float(np.dot(params, basis)))
    return results


def apply_predict_multi(x_vals_list, params_list, fit_type, split_sign=False):
    """Predict with different input per output. x_vals_list: list of scalars, one per output."""
    results = []
    for i, p in enumerate(params_list):
        x = float(x_vals_list[i]) if i < len(x_vals_list) else float(x_vals_list[0])
        params, ft, sp = _resolve_entry(p, fit_type, split_sign)
        if ft == "pchip":
            results.append(float(params(x)))
        elif ft in ("sym_exp", "sym_log"):
            p_neg, p_pos = params
            if x < 0:
                fn = exp_func if ft == "sym_exp" else log_func
                results.append(float(-fn(-x, *p_neg)))
            else:
                fn = exp_func if ft == "sym_exp" else log_func
                results.append(float(fn(x, *p_pos)))
        elif ft == "exp_log":
            p_neg, p_pos = params
            if x < 0:
                results.append(float(exp_func(x, *p_neg)))
            else:
                results.append(float(log_func(x, *p_pos)))
        elif ft == "exp":
            a, b, c, xm, xs = params
            results.append(float(-exp_func((x - xm) / xs, a, b, c)))
        elif sp:
            psel = params[0] if x >= 0 else params[1]
            if ft == "sigmoid":
                results.append(float(sigmoid(x, *psel)))
            else:
                basis = np.array([x**j for j in range(len(psel))])
                results.append(float(np.dot(psel, basis)))
        else:
            if ft == "sigmoid":
                results.append(float(sigmoid(x, *params)))
            else:
                basis = np.array([x**j for j in range(len(params))])
                results.append(float(np.dot(params, basis)))
    return results


# ===================== Write back CSV =====================

def write_back_csv(csv_path, input_cols, output_cols, fit_results, order, dim, write_valid_only=True):
    """Write back calibrated values. fit_results: list of (inp, out, params, ftype) for DIM=1, or single for DIM=2/3."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = [name.strip() for name in reader.fieldnames]
        rows = list(reader)

    for col in ["Fx_cal", "Fy_cal", "Force_cal_mag", "Force_cal_angle"]:
        if col not in header:
            header.append(col)
            for row in rows:
                row[col] = ""

    cnt = 0
    for row in rows:
        if write_valid_only and float(row.get("valid", 0)) == 0:
            continue

        cal_fx, cal_fy = 0.0, 0.0
        try:
            if dim == 1:
                for (inp_cols, out_cols, params, ftype, split) in fit_results:
                    x_val = float(row[inp_cols[0]])
                    if ftype in ("sym_exp", "sym_log"):
                        p_neg, p_pos = params[0]
                        fn = exp_func if ftype == "sym_exp" else log_func
                        if x_val < 0:
                            pred_val = float(-fn(-x_val, *p_neg))
                        else:
                            pred_val = float(fn(x_val, *p_pos))
                    elif ftype == "pchip":
                        pred_val = float(params[0](x_val))
                    elif ftype == "exp_log":
                        p_neg, p_pos = params
                        if x_val < 0:
                            pred_val = float(exp_func(x_val, *p_neg))
                        else:
                            pred_val = float(log_func(x_val, *p_pos))
                    elif split:
                        p = params[0] if x_val >= 0 else params[1]
                        if ftype == "sigmoid":
                            pred_val = float(sigmoid(x_val, *p))
                        else:
                            basis = np.array([x_val**j for j in range(len(p))])
                            pred_val = float(np.dot(p, basis))
                    else:
                        p = params
                        if ftype == "sigmoid":
                            pred_val = float(sigmoid(x_val, *p))
                        else:
                            basis = np.array([x_val**j for j in range(len(p))])
                            pred_val = float(np.dot(p, basis))
                    for j, oc in enumerate(out_cols):
                        if "X" in oc or "x" in oc:
                            cal_fx = pred_val
                        elif "Y" in oc or "y" in oc:
                            cal_fy = pred_val
            else:
                _, _, params, ftype, split = fit_results[0]
                x_vals = np.array([[float(row[c]) for c in input_cols]])
                if split:
                    x0 = float(row[input_cols[0]])
                    p = params[0] if x0 >= 0 else params[1]
                else:
                    p = params
                if ftype == "sigmoid":
                    pred_vals = []
                    for j in range(len(output_cols)):
                        pred_vals.append(float(sigmoid(float(row[input_cols[0]]), *p[j])))
                    pred = np.array([pred_vals])
                else:
                    pred = predict(x_vals, p, order)
                for j, oc in enumerate(output_cols):
                    if "X" in oc or "x" in oc:
                        cal_fx = float(pred[0, j])
                    elif "Y" in oc or "y" in oc:
                        cal_fy = float(pred[0, j])
        except (KeyError, ValueError):
            continue

        cal_mag = float(np.hypot(cal_fx, cal_fy))
        cal_angle = float(np.degrees(np.arctan2(cal_fy, cal_fx + 1e-8)))
        if cal_angle < 0:
            cal_angle += 360

        row["Fx_cal"] = f"{cal_fx:.6f}"
        row["Fy_cal"] = f"{cal_fy:.6f}"
        row["Force_cal_mag"] = f"{cal_mag:.6f}"
        row["Force_cal_angle"] = f"{cal_angle:.6f}"
        cnt += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Write back done: {cnt} rows -> {csv_path}")


# ===================== Plot =====================

def get_medians(csv_path, input_cols, output_cols):
    """Read CSV, group by similar force values (±0.5N), compute median per group."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = list(reader)

    X_all, Y_all = [], []
    for row in rows:
        try:
            v = float(row.get("valid", 0))
            if v == 0:
                continue
            x_vals = [float(row[c]) for c in input_cols]
            y_vals = [float(row[c]) for c in output_cols]
            X_all.append(x_vals)
            Y_all.append(y_vals)
        except (KeyError, ValueError):
            continue

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    X_med, Y_med = [], []
    if len(X_all) == 0:
        return X_all, Y_all, np.empty((0, len(input_cols))), np.empty((0, len(output_cols)))

    # Group by output force value similarity
    if ONE_ON_ONE:
        # Round to nearest 0.5N, one median per force level
        Y_round = np.round(Y_all[:,0] * 2) / 2
        for y_val in np.sort(np.unique(Y_round)):
            mask = Y_round == y_val
            X_med.append(np.median(X_all[mask], axis=0))
            Y_med.append(np.median(Y_all[mask], axis=0))
        return X_all, Y_all, np.array(X_med), np.array(Y_med)
    FORCE_BIN = 0.2
    # Sort by first output column for grouping
    if Y_all.shape[1] > 0:
        sort_idx = np.argsort(Y_all[:, 0])
        X_sorted = X_all[sort_idx]
        Y_sorted = Y_all[sort_idx]

        i = 0
        while i < len(Y_sorted):
            j = i
            while j + 1 < len(Y_sorted) and abs(Y_sorted[j + 1, 0] - Y_sorted[i, 0]) <= FORCE_BIN:
                j += 1
            X_med.append(np.median(X_sorted[i:j + 1], axis=0))
            Y_med.append(np.median(Y_sorted[i:j + 1], axis=0))
            i = j + 1

    return X_all, Y_all, np.array(X_med), np.array(Y_med)

def plot_single(ax, x_med, y_med, yi, xi, short_in, short_out, predict_fn=None):
    """Plot one subplot: scatter medians + fitted line."""
    if len(x_med) > 0:
        ax.scatter(x_med[:, xi], y_med[:, yi], s=30, c='red', zorder=5, label=f'{short_out[yi]} median')
    if predict_fn is not None:
        if x_med.size > 0 and xi < x_med.shape[1]:
            xr = x_med[:, xi]
            x_pad = (xr.max() - xr.min()) * 0.1 if xr.max() - xr.min() > 0 else 0.1
            x_dense = np.linspace(xr.min() - x_pad, xr.max() + x_pad, 500)
        else:
            x_dense = np.linspace(-2, 2, 500)
        y_dense = predict_fn(x_dense)
        ax.plot(x_dense, y_dense, 'g-', linewidth=2, label=f'{short_out[yi]} fitted')
    ax.set_title(f"{short_out[yi]}: True vs Fitted", fontsize=12)
    ax.set_xlabel(short_in[xi])
    ax.set_ylabel(short_out[yi])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if x_med.size > 0 and xi < x_med.shape[1]:
        xr = x_med[:, xi]
        if np.max(np.abs(xr)) < 10:
            ax.set_xlim(xr.min() - 0.1, xr.max() + 0.1)
        else:
            ax.set_xlim(xr.min() - 1000, xr.max() + 1000)

def _make_predict_fn(ftype, params, order=POLY_ORDER):
    """Create a predict function for dense x grid. Returns fn(x_1d) -> y_1d for output yi=0."""
    if ftype == "exp":
        def fn(x):
            x = np.atleast_1d(np.asarray(x, dtype=np.float64))
            a,b,c,xm,xs = params[0]
            return -exp_func((x - xm) / xs, a, b, c)
        return fn
    if ftype in ("sym_exp", "sym_log"):
        def fn(x):
            x = np.atleast_1d(np.asarray(x, dtype=np.float64))
            if np.max(np.abs(x)) > 100:
                x = (x - np.mean(x)) / (np.std(x) or 1.0)
            if ftype == "sym_exp": fn_base = exp_func
            else: fn_base = log_func
            y = np.zeros_like(x)
            pos, neg = x >= 0, x < 0
            p_neg, p_pos = params[0]
            y[pos] = fn_base(x[pos], *p_pos)
            y[neg] = -fn_base(-x[neg], *p_neg)
            return y
        return fn
    if ftype == "pchip":
        def fn(x):
            return params[0](np.atleast_1d(x))
        return fn
    if ftype == "sigmoid":
        def fn(x):
            x = np.atleast_1d(x); y = np.zeros(len(x))
            y[:] = sigmoid(x, *params[0])
            return y
        return fn
    # Default: polynomial
    def fn(x):
        x = np.atleast_1d(x); y = np.zeros(len(x))
        basis = np.array([x**j for j in range(len(params[0]))])
        y[:] = params[0] @ basis
        return y
    return fn


# ===================== Main =====================

if __name__ == "__main__":
    n_pairs = min(len(INPUT_COLS), len(OUTPUT_COLS))
    dim = min(DIM, n_pairs)

    print(f"  DIM = {dim}")

    if dim == 1:
        # Each pair independently
        fit_results = []  # list of (inp, out, params_or_coefs, ftype, split_sign)
        for i in range(n_pairs):
            inp = [INPUT_COLS[i]]
            out = [OUTPUT_COLS[i]]
            out_name = out[0]
            # 精确匹配完整列名，补全else防止变量未定义
            if out_name == "delta_Force_Z":
                train_file = TRAIN_CSV_z
                ft = FIT_TYPE_FZ
            elif out_name == "delta_Force_X":
                train_file = TRAIN_CSV_xy
                ft = FIT_TYPE_FX
            elif out_name == "delta_Force_Y":
                train_file = TRAIN_CSV_xy
                ft = FIT_TYPE_FY
            else:
                train_file = TRAIN_CSV_xy
                ft = FIT_TYPE_FXY
    
            X_train, Y_train = load_csv(train_file, inp, out, valid_only=TRAIN_VALID_ONLY)
            print(f"\n  --- {out_name} <- {inp[0]} ---")
    
            # 第一层优先判断是否开启正负分段 SPLIT_SIGN
            if SPLIT_SIGN and out_name != "delta_Force_Z":
                # 分割正负样本掩码
                pos_mask = X_train[:, 0] >= 0
                neg_mask = ~pos_mask
                X_pos, Y_pos = X_train[pos_mask], Y_train[pos_mask]
                X_neg, Y_neg = X_train[neg_mask], Y_train[neg_mask]
                print(f"  Positive: {len(X_pos)} samples, Negative: {len(X_neg)} samples")
    
                if ft == "exp_log":
                    params = fit_exp_log(X_train, Y_train)
                    for j, (p_neg, p_pos) in enumerate(params):
                        print(f"  {out[j]} -: a={p_neg[0]:.4f}, b={p_neg[1]:.4f}, c={p_neg[2]:.4f} (exp)")
                        print(f"  {out[j]} +: a={p_pos[0]:.4f}, b={p_pos[1]:.4f}, c={p_pos[2]:.4f} (log)")
                    Y_pred = predict_exp_log(X_train, params)
                    fit_results.append((inp, out, params, "exp_log", True))
                elif ft == "sym_exp":
                    params_pos = fit_sym_exp(X_pos, Y_pos)[0]
                    params_neg = fit_sym_exp(X_neg, Y_neg)[0]
                    params = [params_pos, params_neg]
                    Y_pred = np.zeros_like(Y_train)
                    Y_pred[pos_mask] = predict_sym_exp(X_pos, [params_pos])
                    Y_pred[neg_mask] = predict_sym_exp(X_neg, [params_neg])
                    fit_results.append((inp, out, params, "sym_exp", True))
                elif ft == "sym_log":
                    params_pos = fit_sym_log(X_pos, Y_pos)[0]
                    params_neg = fit_sym_log(X_neg, Y_neg)[0]
                    params = [params_pos, params_neg]
                    Y_pred = np.zeros_like(Y_train)
                    Y_pred[pos_mask] = predict_sym_log(X_pos, [params_pos])
                    Y_pred[neg_mask] = predict_sym_log(X_neg, [params_neg])
                    fit_results.append((inp, out, params, "sym_log", True))
                elif ft == "pchip":
                    params = fit_pchip(X_train, Y_train)
                    Y_pred = predict_pchip(X_train, params)
                    print(f"  PCHIP knots: {len(params[0].x)} points")
                    fit_results.append((inp, out, params, "pchip", False))
                else:
                    coefs_pos = fit_polynomial(X_pos, Y_pos, POLY_ORDER)
                    coefs_neg = fit_polynomial(X_neg, Y_neg, POLY_ORDER)
                    params = [coefs_pos, coefs_neg]
                    labels = get_term_labels(inp, POLY_ORDER)
                    print(f"  Positive:"); print_formulas(coefs_pos, labels, out)
                    print(f"  Negative:"); print_formulas(coefs_neg, labels, out)
                    Y_pred = np.zeros_like(Y_train)
                    Y_pred[pos_mask] = predict(X_pos, coefs_pos, POLY_ORDER)
                    Y_pred[neg_mask] = predict(X_neg, coefs_neg, POLY_ORDER)
                    fit_results.append((inp, out, params, "poly", True))
    
            else:
                # SPLIT_SIGN=False，不分正负，全局一套参数拟合
                if ft == "exp":
                    x_raw = X_train[:,0]; y = -Y_train[:,0]  # 取反
                    xm, xs = 0.0, 1.0
                    if np.max(np.abs(x_raw)) > 100:
                        xm = float(np.mean(x_raw)); xs = float(np.std(x_raw)) or 1.0
                        x_fit = (x_raw - xm) / xs
                    else:
                        x_fit = x_raw
                    try:
                        popt,_ = curve_fit(exp_func, x_fit, y, p0=[1.,1.,np.min(y)], maxfev=10000)
                    except:
                        popt = np.array([1.,1.,np.min(y)])
                    params = [(*popt, xm, xs)]
                    Y_pred = -np.array([exp_func(x_fit, *popt)]).T  # 取反还原
                    fit_results.append((inp, out, params, "exp", False))
    
                elif ft == "pchip":
                    params = fit_pchip(X_train, Y_train)
                    Y_pred = predict_pchip(X_train, params)
                    print(f"  PCHIP knots: {len(params[0].x)} points")
                    fit_results.append((inp, out, params, "pchip", False))
    
                elif ft == "sigmoid":
                    params = fit_sigmoid(X_train, Y_train)
                    Y_pred = predict_sigmoid(X_train, params)
                    for j, p in enumerate(params):
                        print(f"  {out[j]}: L={p[0]:.4f}, k={p[1]:.4f}, x0={p[2]:.4f}, b={p[3]:.4f}")
                    fit_results.append((inp, out, params, "sigmoid", False))
    
                else:
                    coefs = fit_polynomial(X_train, Y_train, POLY_ORDER)
                    labels = get_term_labels(inp, POLY_ORDER)
                    print_formulas(coefs, labels, out)
                    Y_pred = predict(X_train, coefs, POLY_ORDER)
                    fit_results.append((inp, out, coefs, "poly", False))
                compute_errors(Y_train, Y_pred, out)

        # Save coefs
        if SAVE_COEFS:
            coef_path = os.path.join(FIT_PARAM_Save, "fit_coefs.bin")
            save_coefs(fit_results, coef_path)

        # Write back
        print(f"  Write back to: {TARGET_CSV_xy}")
        write_back_csv(TARGET_CSV_xy, INPUT_COLS[:n_pairs], OUTPUT_COLS[:n_pairs], fit_results, POLY_ORDER, 1, WRITE_VALID_ONLY)

        # Plot
        fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 5 * n_pairs), squeeze=False)
        for i in range(n_pairs):
            inp, out, params, ftype, split = fit_results[i]
            short_in = [c.replace("delta_", "").replace("_", "") for c in inp]
            short_out = [c.replace("delta_", "").replace("_", "") for c in out]
            plot_csv = TRAIN_CSV_z if ("Z" in out[0] or "z" in out[0]) else TARGET_CSV_xy
            X_all, Y_all, X_med, Y_med = get_medians(plot_csv, inp, out)
            if len(X_all) > 0:
                if ftype == "exp":
                    a,b,c,xm,xs = params[0]
                    Y_pred = -np.array([exp_func((X_all[:,0]-xm)/xs, a,b,c)]).T
                elif ftype in ("sym_exp", "sym_log"):
                    Y_pred = predict_sym_exp(X_all, params) if ftype == "sym_exp" else predict_sym_log(X_all, params)
                elif ftype == "pchip":
                    Y_pred = predict_pchip(X_all, params)
                elif ftype == "exp_log":
                    Y_pred = predict_exp_log(X_all, params)
                elif split:
                    p_pos, p_neg = params
                    pos_mask_all = X_all[:, 0] >= 0
                    neg_mask_all = ~pos_mask_all
                    Y_pred = np.zeros_like(Y_all)
                    if ftype == "sigmoid":
                        Y_pred[pos_mask_all] = predict_sigmoid(X_all[pos_mask_all], p_pos)
                        Y_pred[neg_mask_all] = predict_sigmoid(X_all[neg_mask_all], p_neg)
                    else:
                        Y_pred[pos_mask_all] = predict(X_all[pos_mask_all], p_pos, POLY_ORDER)
                        Y_pred[neg_mask_all] = predict(X_all[neg_mask_all], p_neg, POLY_ORDER)
                else:
                    if ftype == "sigmoid":
                        Y_pred = predict_sigmoid(X_all, params)
                    else:
                        Y_pred = predict(X_all, params, POLY_ORDER)
                plot_single(axes[i, 0], X_med, Y_med, 0, 0, short_in, short_out, _make_predict_fn(ftype, params))
        plt.tight_layout()
        csv_stem = os.path.splitext(os.path.basename(TARGET_CSV_xy))[0]
        save_path = os.path.join(os.path.dirname(TARGET_CSV_xy), f"{csv_stem}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {save_path}")
