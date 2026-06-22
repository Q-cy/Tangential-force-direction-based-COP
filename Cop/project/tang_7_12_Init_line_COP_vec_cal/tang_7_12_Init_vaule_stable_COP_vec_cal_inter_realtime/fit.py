"""
Curve fitting tool: fit CSV calibration data, write back to CSV, output formula.
Standalone, no external dependencies.
"""

import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# ===================== Config =====================

TRAIN_CSV = "/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0615_5.csv"
TARGET_CSV = "/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0615_6.csv"

# Full input/output columns (always 3 pairs)
INPUT_COLS = ["delta_CoP_X", "delta_CoP_Y"]
OUTPUT_COLS = ["delta_Force_X", "delta_Force_Y"]
# INPUT_COLS = ["delta_CoP_X", "delta_CoP_Y", "adc_sum"]
# OUTPUT_COLS = ["delta_Force_X", "delta_Force_Y", "delta_Force_Z"]

DIM = 1          # 1=each pair independently, 2=first 2 together, 3=all 3 together
POLY_ORDER = 1   # 1=linear, 2=quadratic, 3=cubic (only used if FIT_TYPE="poly")
FIT_TYPE = "exp_log"  # "poly"=polynomial, "sigmoid"=S-curve, "exp_log"=指数+对数

TRAIN_VALID_ONLY = True
WRITE_VALID_ONLY = True
SAVE_COEFS = True
SPLIT_SIGN = True  # True=正负分开拟合，False=不分


# ===================== Read CSV =====================

def load_csv(csv_path, input_cols, output_cols, valid_only=True):
    """Read CSV, return (X[N, n_in], Y[N, n_out])"""
    X_rows, Y_rows = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            try:
                if valid_only and float(row.get("valid", 0)) == 0:
                    continue
                x_vals = [float(row[c]) for c in input_cols]
                y_vals = [float(row[c]) for c in output_cols]
                X_rows.append(x_vals)
                Y_rows.append(y_vals)
            except (KeyError, ValueError):
                continue
    return np.array(X_rows), np.array(Y_rows)


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


# ===================== Exp/Log fit =====================

def exp_func(x, a, b, c):
    """Exponential: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def log_func(x, a, b, c):
    """Logarithmic: y = a * ln(b * x + 1) + c"""
    return a * np.log(b * x + 1) + c


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
    """Save fit results to .bin. fit_results: list of (inp, out, params, ftype, split_sign)"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        total_outputs = 0
        first_ftype = fit_results[0][3] if fit_results else "sigmoid"
        n_inputs = len(fit_results[0][0]) if fit_results else 1
        has_split = fit_results[0][4] if fit_results else False
        for inp, out, params, ftype, split in fit_results:
            total_outputs += len(out)

        if first_ftype == "sigmoid":
            fit_type_id = 0
            n_params = 4
        elif first_ftype == "exp_log":
            fit_type_id = 2
            n_params = 3
        else:
            fit_type_id = 1
            n_params = fit_results[0][2][0].shape[0] if has_split else fit_results[0][2].shape[0]

        # Header
        f.write(np.int32(fit_type_id).tobytes())
        f.write(np.int32(n_inputs).tobytes())
        f.write(np.int32(total_outputs).tobytes())
        f.write(np.int32(n_params).tobytes())
        f.write(np.int32(1 if has_split else 0).tobytes())

        # Params
        for inp, out, params, ftype, split in fit_results:
            if ftype == "exp_log":
                # params = [(p_neg, p_pos), ...] per output
                for p_neg, p_pos in params:
                    f.write(np.array(p_neg, dtype=np.float64).tobytes())
                    f.write(np.array(p_pos, dtype=np.float64).tobytes())
            elif split:
                # params = [params_pos, params_neg]
                for sign_params in params:
                    if ftype == "sigmoid":
                        for p in sign_params:
                            f.write(np.array(p, dtype=np.float64).tobytes())
                    else:
                        for c in sign_params.T:
                            f.write(np.array(c, dtype=np.float64).tobytes())
            else:
                if ftype == "sigmoid":
                    for p in params:
                        f.write(np.array(p, dtype=np.float64).tobytes())
                else:
                    for c in params.T:
                        f.write(np.array(c, dtype=np.float64).tobytes())
    print(f"  Coefs saved: {path} ({os.path.getsize(path)} bytes, {total_outputs} outputs, split={has_split})")


def load_coefs(path):
    """Load fit coefs from .bin. Returns (fit_type, n_inputs, params_list, split_sign)"""
    with open(path, "rb") as f:
        fit_type_id = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n_inputs = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n_outputs = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n_params = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        split_sign = int(np.frombuffer(f.read(4), dtype=np.int32)[0]) == 1

        if fit_type_id == 0:
            fit_type = "sigmoid"
        elif fit_type_id == 2:
            fit_type = "exp_log"
        else:
            fit_type = "poly"

        if fit_type == "exp_log":
            # exp_log: always split, each output has (p_neg, p_pos)
            params_list = []
            for _ in range(n_outputs):
                p_neg = np.frombuffer(f.read(n_params * 8), dtype=np.float64).copy()
                p_pos = np.frombuffer(f.read(n_params * 8), dtype=np.float64).copy()
                params_list.append((p_neg, p_pos))
        elif split_sign:
            params_list = []
            for _ in range(n_outputs):
                p_pos = np.frombuffer(f.read(n_params * 8), dtype=np.float64).copy()
                p_neg = np.frombuffer(f.read(n_params * 8), dtype=np.float64).copy()
                params_list.append((p_pos, p_neg))
        else:
            params_list = []
            for _ in range(n_outputs):
                p = np.frombuffer(f.read(n_params * 8), dtype=np.float64).copy()
                params_list.append(p)
    return fit_type, n_inputs, params_list, split_sign


def apply_predict(x_val, params_list, fit_type, split_sign=False):
    """Predict using loaded params. x_val: scalar or 1D array. Returns list of predictions."""
    x = float(x_val) if np.isscalar(x_val) else float(x_val[0])
    results = []
    for p in params_list:
        if fit_type == "exp_log":
            # p = (p_neg, p_pos)
            p_neg, p_pos = p
            if x < 0:
                results.append(float(exp_func(x, *p_neg)))
            else:
                results.append(float(log_func(x, *p_pos)))
        elif split_sign:
            params = p[0] if x >= 0 else p[1]
            if fit_type == "sigmoid":
                results.append(float(sigmoid(x, *params)))
            else:
                basis = np.array([x**j for j in range(len(params))])
                results.append(float(np.dot(params, basis)))
        else:
            if fit_type == "sigmoid":
                results.append(float(sigmoid(x, *p)))
            else:
                basis = np.array([x**j for j in range(len(p))])
                results.append(float(np.dot(p, basis)))
    return results


def apply_predict_multi(x_vals_list, params_list, fit_type, split_sign=False):
    """Predict with different input per output. x_vals_list: list of scalars, one per output."""
    results = []
    for i, p in enumerate(params_list):
        x = float(x_vals_list[i]) if i < len(x_vals_list) else float(x_vals_list[0])
        if fit_type == "exp_log":
            p_neg, p_pos = p
            if x < 0:
                results.append(float(exp_func(x, *p_neg)))
            else:
                results.append(float(log_func(x, *p_pos)))
        elif split_sign:
            params = p[0] if x >= 0 else p[1]
            if fit_type == "sigmoid":
                results.append(float(sigmoid(x, *params)))
            else:
                basis = np.array([x**j for j in range(len(params))])
                results.append(float(np.dot(params, basis)))
        else:
            if fit_type == "sigmoid":
                results.append(float(sigmoid(x, *p)))
            else:
                basis = np.array([x**j for j in range(len(p))])
                results.append(float(np.dot(p, basis)))
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
                    if ftype == "exp_log":
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
    """Read CSV, compute median of consecutive valid=1 segments."""
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = list(reader)

    X_all, Y_all, valid_all = [], [], []
    for row in rows:
        try:
            x_vals = [float(row[c]) for c in input_cols]
            y_vals = [float(row[c]) for c in output_cols]
            v = float(row.get("valid", 0))
            X_all.append(x_vals)
            Y_all.append(y_vals)
            valid_all.append(v)
        except (KeyError, ValueError):
            continue

    X_all = np.array(X_all)
    Y_all = np.array(Y_all)
    valid_all = np.array(valid_all)

    v_mask = valid_all != 0
    changes = np.where(np.diff(v_mask.astype(int)))[0] + 1
    segments = np.split(np.arange(len(X_all)), changes)

    X_med, Y_med = [], []
    for seg in segments:
        if len(seg) == 0 or not v_mask[seg[0]]:
            continue
        X_med.append(np.median(X_all[seg], axis=0))
        Y_med.append(np.median(Y_all[seg], axis=0))

    return X_all, Y_all, np.array(X_med) if X_med else np.empty((0, len(input_cols))), np.array(Y_med) if Y_med else np.empty((0, len(output_cols)))


def plot_single(ax, x_all, x_med, y_med, x_pred, yi, xi, short_in, short_out):
    """Plot one subplot: scatter medians + fitted line."""
    if len(x_med) > 0:
        ax.scatter(x_med[:, xi], y_med[:, yi], s=30, c='red', zorder=5, label=f'{short_out[yi]} median')
    sort_idx = np.argsort(x_all[:, xi])
    ax.plot(x_all[sort_idx, xi], x_pred[sort_idx, yi], 'g-', linewidth=2, label=f'{short_out[yi]} fitted')
    ax.set_title(f"{short_out[yi]}: True vs Fitted", fontsize=12)
    ax.set_xlabel(short_in[xi])
    ax.set_ylabel(short_out[yi])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


# ===================== Main =====================

if __name__ == "__main__":
    n_pairs = min(len(INPUT_COLS), len(OUTPUT_COLS))
    dim = min(DIM, n_pairs)

    print(f"  DIM = {dim}")
    print(f"  Training CSV: {TRAIN_CSV}")

    if dim == 1:
        # Each pair independently
        fit_results = []  # list of (inp, out, params_or_coefs, ftype, split_sign)
        for i in range(n_pairs):
            inp = [INPUT_COLS[i]]
            out = [OUTPUT_COLS[i]]
            X_train, Y_train = load_csv(TRAIN_CSV, inp, out, valid_only=TRAIN_VALID_ONLY)
            print(f"\n  --- {out[0]} <- {inp[0]} ---")

            if SPLIT_SIGN or FIT_TYPE == "exp_log":
                # Split by sign of input
                pos_mask = X_train[:, 0] >= 0
                neg_mask = ~pos_mask
                X_pos, Y_pos = X_train[pos_mask], Y_train[pos_mask]
                X_neg, Y_neg = X_train[neg_mask], Y_train[neg_mask]
                print(f"  Positive: {len(X_pos)} samples, Negative: {len(X_neg)} samples")

                if FIT_TYPE == "exp_log":
                    params = fit_exp_log(X_train, Y_train)
                    for j, (p_neg, p_pos) in enumerate(params):
                        print(f"  {out[j]} -: a={p_neg[0]:.4f}, b={p_neg[1]:.4f}, c={p_neg[2]:.4f} (exp)")
                        print(f"  {out[j]} +: a={p_pos[0]:.4f}, b={p_pos[1]:.4f}, c={p_pos[2]:.4f} (log)")
                    Y_pred = predict_exp_log(X_train, params)
                    fit_results.append((inp, out, params, "exp_log", True))
                elif FIT_TYPE == "sigmoid":
                    params_pos = fit_sigmoid(X_pos, Y_pos)
                    params_neg = fit_sigmoid(X_neg, Y_neg)
                    params = [params_pos, params_neg]
                    for j in range(len(out)):
                        pp, pn = params_pos[j], params_neg[j]
                        print(f"  {out[j]} +: L={pp[0]:.4f}, k={pp[1]:.4f}, x0={pp[2]:.4f}, b={pp[3]:.4f}")
                        print(f"  {out[j]} -: L={pn[0]:.4f}, k={pn[1]:.4f}, x0={pn[2]:.4f}, b={pn[3]:.4f}")
                    Y_pred = np.zeros_like(Y_train)
                    Y_pred[pos_mask] = predict_sigmoid(X_pos, params_pos)
                    Y_pred[neg_mask] = predict_sigmoid(X_neg, params_neg)
                    fit_results.append((inp, out, params, "sigmoid", True))
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
                if FIT_TYPE == "sigmoid":
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
            out_dir = os.path.dirname(TRAIN_CSV)
            coef_path = os.path.join(out_dir, "fit_coefs.bin")
            save_coefs(fit_results, coef_path)

        # Write back
        print(f"  Target CSV: {TARGET_CSV}")
        write_back_csv(TARGET_CSV, INPUT_COLS[:n_pairs], OUTPUT_COLS[:n_pairs], fit_results, POLY_ORDER, 1, WRITE_VALID_ONLY)

        # Plot
        fig, axes = plt.subplots(n_pairs, 1, figsize=(10, 5 * n_pairs), squeeze=False)
        for i in range(n_pairs):
            inp, out, params, ftype, split = fit_results[i]
            short_in = [c.replace("delta_", "").replace("_", "") for c in inp]
            short_out = [c.replace("delta_", "").replace("_", "") for c in out]
            X_all, Y_all, X_med, Y_med = get_medians(TARGET_CSV, inp, out)
            if len(X_all) > 0:
                if ftype == "exp_log":
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
                plot_single(axes[i, 0], X_all, X_med, Y_med, Y_pred, 0, 0, short_in, short_out)
        plt.tight_layout()
        csv_stem = os.path.splitext(os.path.basename(TARGET_CSV))[0]
        save_path = os.path.join(os.path.dirname(TARGET_CSV), f"fit_{csv_stem}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {save_path}")

    else:
        # DIM=2 or 3: all inputs together
        inp = INPUT_COLS[:dim]
        out = OUTPUT_COLS[:dim]
        X_train, Y_train = load_csv(TRAIN_CSV, inp, out, valid_only=TRAIN_VALID_ONLY)
        print(f"  Training samples: {len(X_train)}")

        if SPLIT_SIGN:
            # Split by sign of first input
            pos_mask = X_train[:, 0] >= 0
            neg_mask = ~pos_mask
            X_pos, Y_pos = X_train[pos_mask], Y_train[pos_mask]
            X_neg, Y_neg = X_train[neg_mask], Y_train[neg_mask]
            print(f"  Positive: {len(X_pos)} samples, Negative: {len(X_neg)} samples")

            if FIT_TYPE == "exp_log":
                params = fit_exp_log(X_train, Y_train)
                Y_pred = predict_exp_log(X_train, params)
                fit_results = [(inp, out, params, "exp_log", True)]
            elif FIT_TYPE == "sigmoid":
                params_pos = fit_sigmoid(X_pos, Y_pos)
                params_neg = fit_sigmoid(X_neg, Y_neg)
                params = [params_pos, params_neg]
                Y_pred = np.zeros_like(Y_train)
                Y_pred[pos_mask] = predict_sigmoid(X_pos, params_pos)
                Y_pred[neg_mask] = predict_sigmoid(X_neg, params_neg)
                fit_results = [(inp, out, params, "sigmoid", True)]
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
                fit_results = [(inp, out, params, "poly", True)]
        else:
            if FIT_TYPE == "sigmoid":
                params = fit_sigmoid(X_train, Y_train)
                Y_pred = predict_sigmoid(X_train, params)
                fit_results = [(inp, out, params, "sigmoid", False)]
            else:
                coefs = fit_polynomial(X_train, Y_train, POLY_ORDER)
                labels = get_term_labels(inp, POLY_ORDER)
                print_formulas(coefs, labels, out)
                Y_pred = predict(X_train, coefs, POLY_ORDER)
                fit_results = [(inp, out, coefs, "poly", False)]

        compute_errors(Y_train, Y_pred, out)

        if SAVE_COEFS:
            out_dir = os.path.dirname(TRAIN_CSV)
            coef_path = os.path.join(out_dir, "fit_coefs.bin")
            save_coefs(fit_results, coef_path)

        print(f"  Target CSV: {TARGET_CSV}")
        write_back_csv(TARGET_CSV, inp, out, fit_results, POLY_ORDER, dim, WRITE_VALID_ONLY)

        # Plot
        short_in = [c.replace("delta_", "").replace("_", "") for c in inp]
        short_out = [c.replace("delta_", "").replace("_", "") for c in out]
        X_all, Y_all, X_med, Y_med = get_medians(TARGET_CSV, inp, out)
        _, _, params_plot, ftype_plot, split_plot = fit_results[0]
        if ftype_plot == "exp_log":
            Y_pred_all = predict_exp_log(X_all, params_plot)
        elif split_plot:
            p_pos, p_neg = params_plot
            pos_mask_all = X_all[:, 0] >= 0
            neg_mask_all = ~pos_mask_all
            Y_pred_all = np.zeros_like(Y_all)
            if ftype_plot == "sigmoid":
                Y_pred_all[pos_mask_all] = predict_sigmoid(X_all[pos_mask_all], p_pos)
                Y_pred_all[neg_mask_all] = predict_sigmoid(X_all[neg_mask_all], p_neg)
            else:
                Y_pred_all[pos_mask_all] = predict(X_all[pos_mask_all], p_pos, POLY_ORDER)
                Y_pred_all[neg_mask_all] = predict(X_all[neg_mask_all], p_neg, POLY_ORDER)
        else:
            if ftype_plot == "sigmoid":
                Y_pred_all = predict_sigmoid(X_all, params_plot)
            else:
                Y_pred_all = predict(X_all, params_plot, POLY_ORDER)
        fig, axes = plt.subplots(dim, 1, figsize=(10, 5 * dim), squeeze=False)
        for i in range(dim):
            plot_single(axes[i, 0], X_all, X_med, Y_med, Y_pred_all, i, i, short_in, short_out)
        plt.tight_layout()
        csv_stem = os.path.splitext(os.path.basename(TARGET_CSV))[0]
        save_path = os.path.join(os.path.dirname(TARGET_CSV), f"fit_{csv_stem}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"  Plot saved: {save_path}")
