"""
fitxy: Fx/Fy 对称对数拟合。正负以 X 轴为准。
"""
import os, csv, sys, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator

# ===================== Config =====================
TRAIN_CSV = "/home/qcy/Project/data/2.PZT_tangential/weight/test/COP_0615_6.csv"
TARGET_CSV = TRAIN_CSV
FIT_PARAM_Save = "/home/qcy/Project/data/2.PZT_tangential/weight/fit"

INPUT_COLS = ["delta_CoP_X", "delta_CoP_Y"]
OUTPUT_COLS = ["delta_Force_X", "delta_Force_Y"]
FIT_TYPE_FX = "sym_log"   # poly/sigmoid/exp_log/pchip/sym_exp/sym_log
FIT_TYPE_FY = "sym_log"

POLY_ORDER = 3
TRAIN_VALID_ONLY = True
WRITE_VALID_ONLY = True
ONE_ON_ONE = True
SAVE_COEFS = True
SPLIT_SIGN = True

# ===================== Funcs =====================
def log_func(x, a, b, c):
    return a * np.log(b * x + 1) + c

def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c

def load_csv(csv_path, inp, out):
    X, Y = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                if TRAIN_VALID_ONLY and float(row.get('valid',0)) == 0: continue
                X.append([float(row[c]) for c in inp])
                Y.append([float(row[c]) for c in out])
            except: pass
    return np.array(X), np.array(Y)

# --- Polynomial ---
def build_design_matrix(X, order):
    n = X.shape[1]
    if order == 1: return np.column_stack([np.ones(len(X))] + [X[:,i] for i in range(n)])
    elif order == 2:
        c = [np.ones(len(X))]
        for i in range(n): c.append(X[:,i])
        for i in range(n):
            for j in range(i,n): c.append(X[:,i]*X[:,j])
        return np.column_stack(c)
    elif order == 3:
        c = [np.ones(len(X))]
        for i in range(n): c.append(X[:,i])
        for i in range(n):
            for j in range(i,n): c.append(X[:,i]*X[:,j])
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n): c.append(X[:,i]*X[:,j]*X[:,k])
        return np.column_stack(c)
    raise ValueError(f"Bad order: {order}")

def get_term_labels(inp, order):
    labels = ["1"]
    n = len(inp); s = [c.replace("delta_","").replace("_","") for c in inp]
    if order>=1: labels.extend(s)
    if order>=2:
        for i in range(n):
            for j in range(i,n): labels.append(f"{s[i]}*{s[j]}")
    if order>=3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n): labels.append(f"{s[i]}*{s[j]}*{s[k]}")
    return labels

def fit_poly(X, Y, order):
    A = build_design_matrix(X, order)
    return np.linalg.lstsq(A, Y, rcond=None)[0]

def predict_poly(X, coefs, order):
    return build_design_matrix(X, order) @ coefs

# --- Sigmoid ---
def sigmoid(x, L, k, x0, b):
    return L/(1+np.exp(-k*(x-x0))) + b

def fit_sigmoid(X, Y):
    n_out = Y.shape[1]; all_p = []
    for i in range(n_out):
        x, y = X[:,0], Y[:,i]
        ym, yM = np.min(y), np.max(y)
        try: pp,_ = curve_fit(sigmoid, x, y, p0=[yM-ym,10.,np.median(x),ym], maxfev=10000)
        except: pp = np.array([yM-ym,10.,np.median(x),ym])
        all_p.append(pp)
    return all_p

def predict_sigmoid(X, params):
    Y = np.zeros((len(X), len(params)))
    for i, p in enumerate(params): Y[:,i] = sigmoid(X[:,0], *p)
    return Y

# --- Symmetric Log ---
def fit_sym_log(X, Y):
    n_out = Y.shape[1]; all_p = []
    x_raw = X[:,0]; x = x_raw
    if np.max(np.abs(x_raw)) > 100:
        x_mean = float(np.mean(x_raw)); x_std = float(np.std(x_raw)) or 1.0
        x = (x_raw - x_mean) / x_std
    for i in range(n_out):
        y = Y[:,i]; xi = x
        pos, neg = xi>=0, xi<0
        if np.sum(pos)>3:
            try: pp,_ = curve_fit(log_func, xi[pos], y[pos], p0=[1.,1.,np.min(y[pos])], maxfev=10000)
            except: pp = np.array([1.,1.,np.min(y[pos])])
        else: pp = np.array([1.,1.,0.])
        if np.sum(neg)>3:
            try: pn,_ = curve_fit(log_func, -xi[neg], -y[neg], p0=[1.,1.,np.min(-y[neg])], maxfev=10000)
            except: pn = np.array([1.,1.,np.min(-y[neg])])
        else: pn = pp.copy()
        all_p.append((pn, pp))
        print(f"  +: a={pp[0]:.4f} b={pp[1]:.4f} c={pp[2]:.4f}")
        print(f"  -: a={pn[0]:.4f} b={pn[1]:.4f} c={pn[2]:.4f}")
    return all_p

def predict_sym_log(X, params):
    Y = np.zeros((len(X), len(params)))
    x_raw = X[:,0]; x = x_raw
    if np.max(np.abs(x_raw))>100:
        x_mean = float(np.mean(x_raw)); x_std = float(np.std(x_raw)) or 1.0
        x = (x_raw - x_mean)/x_std
    for i, (pn, pp) in enumerate(params):
        pos, neg = x>=0, x<0
        if np.any(pos): Y[pos,i] = log_func(x[pos], *pp)
        if np.any(neg): Y[neg,i] = -log_func(-x[neg], *pn)
    return Y

# --- Symmetric Exp ---
def fit_sym_exp(X, Y):
    n_out = Y.shape[1]; all_p = []
    for i in range(n_out):
        x, y = X[:,0], Y[:,i]; pos, neg = x>=0, x<0
        def _fit(xd, yd):
            if len(xd)<3: return np.array([1.,1.,0.])
            try: pp,_ = curve_fit(exp_func, xd, yd, p0=[1.,1.,np.min(yd)], maxfev=10000)
            except: pp = np.array([1.,1.,np.min(yd)])
            return pp
        pp = _fit(x[pos], y[pos]) if np.sum(pos)>3 else np.array([1.,1.,0.])
        pn = _fit(-x[neg], -y[neg]) if np.sum(neg)>3 else pp.copy()
        all_p.append((pn, pp))
        print(f"  +: a={pp[0]:.4f} b={pp[1]:.4f} c={pp[2]:.4f}")
        print(f"  -: a={pn[0]:.4f} b={pn[1]:.4f} c={pn[2]:.4f}")
    return all_p

def predict_sym_exp(X, params):
    Y = np.zeros((len(X), len(params)))
    for i, (pn, pp) in enumerate(params):
        x = X[:,0]; pos, neg = x>=0, x<0
        if np.any(pos): Y[pos,i] = exp_func(x[pos], *pp)
        if np.any(neg): Y[neg,i] = -exp_func(-x[neg], *pn)
    return Y

# --- Exp/Log ---
def fit_exp_log(X, Y):
    n_out = Y.shape[1]; all_p = []
    for i in range(n_out):
        x, y = X[:,0], Y[:,i]; neg, pos = x<0, x>=0
        if np.sum(neg)>3:
            try: pn,_ = curve_fit(exp_func, x[neg], y[neg], p0=[1.,1.,np.min(y[neg])], maxfev=10000)
            except: pn = np.array([1.,1.,np.min(y[neg])])
        else: pn = np.array([1.,1.,0.])
        if np.sum(pos)>3:
            try: pp,_ = curve_fit(log_func, x[pos], y[pos], p0=[1.,1.,np.min(y[pos])], maxfev=10000)
            except: pp = np.array([1.,1.,np.min(y[pos])])
        else: pp = np.array([1.,1.,0.])
        all_p.append((pn, pp))
    return all_p

def predict_exp_log(X, params):
    Y = np.zeros((len(X), len(params)))
    for i, (pn, pp) in enumerate(params):
        x = X[:,0]; neg, pos = x<0, x>=0
        if np.any(neg): Y[neg,i] = exp_func(x[neg], *pn)
        if np.any(pos): Y[pos,i] = log_func(x[pos], *pp)
    return Y

# --- PCHIP ---
def fit_pchip(X, Y):
    interps = []
    for i in range(Y.shape[1]):
        x, y = X[:,0].copy(), Y[:,i].copy()
        si = np.argsort(x); xs, ys = x[si], y[si]
        keep = np.diff(xs, prepend=xs[0]-1)!=0
        interps.append(PchipInterpolator(xs[keep], ys[keep]))
    return interps

def predict_pchip(X, interps):
    Y = np.zeros((len(X), len(interps)))
    for i, ip in enumerate(interps): Y[:,i] = ip(X[:,0])
    return Y

# --- Save/Load ---
def save_coefs(fit_results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        total = 0; first_ft = fit_results[0][3]; has_sp = fit_results[0][4]
        for _, o, _, _, _ in fit_results: total += len(o)
        if first_ft == "pchip":
            fid, npp = 3, len(fit_results[0][2][0].x)
        elif first_ft in ("sym_exp","sym_log"):
            fid, npp = (4 if first_ft=="sym_exp" else 5), 3
        elif first_ft == "sigmoid": fid, npp = 0, 4
        elif first_ft == "exp_log": fid, npp = 2, 3
        else: fid, npp = 1, fit_results[0][2][0].shape[0] if has_sp else fit_results[0][2].shape[0]
        ni = len(fit_results[0][0])
        f.write(np.int32(fid).tobytes()); f.write(np.int32(ni).tobytes())
        f.write(np.int32(total).tobytes()); f.write(np.int32(npp).tobytes())
        f.write(np.int32(1 if has_sp else 0).tobytes())
        for _, _, params, ft, sp in fit_results:
            if ft in ("sym_exp","sym_log","exp_log"):
                for pn, pp in params:
                    f.write(np.array(pn, np.float64).tobytes())
                    f.write(np.array(pp, np.float64).tobytes())
            elif ft == "pchip":
                for ip in params:
                    x = np.array(ip.x, np.float64); f.write(x.tobytes())
                    y = np.array(ip(x), np.float64); f.write(y.tobytes())
            elif sp:
                for spm in params:
                    if ft == "sigmoid":
                        for p in spm: f.write(np.array(p, np.float64).tobytes())
                    else:
                        for cc in spm.T: f.write(np.array(cc, np.float64).tobytes())
            else:
                if ft == "sigmoid":
                    for p in params: f.write(np.array(p, np.float64).tobytes())
                else:
                    for cc in params.T: f.write(np.array(cc, np.float64).tobytes())
    print(f"  Coefs saved: {path} ({os.path.getsize(path)} bytes, {total} outputs)")

# --- Errors ---
def compute_errors(Yt, Yp, out):
    print(f"\n{'='*60}\n  Error Analysis\n{'='*60}")
    for i, nm in enumerate(out):
        e = Yp[:,i]-Yt[:,i]; ae = np.abs(e)
        ssr = np.sum(e**2); sst = np.sum((Yt[:,i]-np.mean(Yt[:,i]))**2)
        r2 = 1-ssr/sst if sst>0 else np.nan
        print(f"  {nm}: MAE={np.mean(ae):.6f} RMSE={np.sqrt(np.mean(e**2)):.6f} R2={r2:.6f}")
    print(f"{'='*60}\n")

# --- Print formulas ---
def print_formulas(coefs, labels, out):
    print(f"\n{'='*60}\n  Fit result ({len(labels)-1} terms)\n{'='*60}")
    for i, nm in enumerate(out):
        terms = []
        for c, l in zip(coefs[:,i], labels):
            if abs(c)<1e-10: continue
            terms.append(f"{c:.6f}*{l}")
        print(f"  {nm} = {' + '.join(terms)}")
    print(f"{'='*60}\n")

# --- Write back ---
def write_back_csv(csv_path, inp_cols, out_cols, fit_results, order, write_valid_only=True):
    with open(csv_path) as f:
        reader = csv.DictReader(f); hdr = [c.strip() for c in reader.fieldnames]; rows = list(reader)
    for col in ["Fx_cal","Fy_cal","Force_cal_mag","Force_cal_angle"]:
        if col not in hdr: hdr.append(col); [r.__setitem__(col,"") for r in rows]
    cnt = 0
    for row in rows:
        if write_valid_only and float(row.get('valid',0))==0: continue
        cal_fx, cal_fy = 0., 0.
        try:
            for (ic, oc, params, ft, sp) in fit_results:
                xv = float(row[ic[0]])
                if ft in ("sym_exp","sym_log"):
                    pn, pp = params[0]
                    fn = exp_func if ft=="sym_exp" else log_func
                    pv = float(-fn(-xv,*pn)) if xv<0 else float(fn(xv,*pp))
                elif ft == "exp_log": pn,pp=params; pv=float(exp_func(xv,*pn) if xv<0 else log_func(xv,*pp))
                elif ft == "pchip": pv = float(params[0](xv))
                elif sp:
                    p = params[0] if xv>=0 else params[1]
                    if ft=="sigmoid": pv=float(sigmoid(xv,*p))
                    else: pv=float(np.dot(np.array([xv**j for j in range(len(p))]), p))
                else:
                    if ft=="sigmoid": pv=float(sigmoid(xv,*params[0]))
                    else: pv=float(np.dot(np.array([xv**j for j in range(len(params[0]))]), params[0]))
                for j, oc in enumerate(oc):
                    if "X" in oc or "x" in oc: cal_fx = pv
                    elif "Y" in oc or "y" in oc: cal_fy = pv
        except: continue
        cal_mag = float(np.hypot(cal_fx, cal_fy))
        cal_angle = float(np.degrees(np.arctan2(cal_fy, cal_fx+1e-8))) % 360
        row["Fx_cal"]=f"{cal_fx:.6f}"; row["Fy_cal"]=f"{cal_fy:.6f}"
        row["Force_cal_mag"]=f"{cal_mag:.6f}"; row["Force_cal_angle"]=f"{cal_angle:.6f}"
        cnt+=1
    with open(csv_path,"w",newline="") as f:
        csv.DictWriter(f,hdr).writeheader(); csv.DictWriter(f,hdr).writerows(rows)
    print(f"  Write back done: {cnt} rows -> {csv_path}")

# --- Get medians ---
def get_medians(csv_path, inp, out):
    Xa, Ya = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            try:
                if float(row.get('valid',0))==0: continue
                Xa.append([float(row[c]) for c in inp])
                Ya.append([float(row[c]) for c in out])
            except: pass
    Xa, Ya = np.array(Xa), np.array(Ya)
    if len(Xa)==0: return Xa, Ya, np.empty((0,1)), np.empty((0,1))
    Xm, Ym = [], []
    if ONE_ON_ONE:
        Yr = np.round(Ya[:,0]*2)/2
        for yv in np.sort(np.unique(Yr)):
            m = Yr==yv; Xm.append(np.median(Xa[m], axis=0)); Ym.append(np.median(Ya[m], axis=0))
        return Xa, Ya, np.array(Xm), np.array(Ym)
    si = np.argsort(Ya[:,0]); Xs, Ys = Xa[si], Ya[si]
    i, BIN = 0, 0.2
    while i < len(Ys):
        j = i
        while j+1<len(Ys) and abs(Ys[j+1,0]-Ys[i,0])<=BIN: j+=1
        Xm.append(np.median(Xs[i:j+1],axis=0)); Ym.append(np.median(Ys[i:j+1],axis=0)); i=j+1
    return Xa, Ya, np.array(Xm), np.array(Ym)

# --- Plot ---
def plot_single(ax, x_med, y_med, yi, xi, si, so, predict_fn=None):
    if len(x_med)>0: ax.scatter(x_med[:,xi], y_med[:,yi], s=30, c='red', zorder=5, label=f'{so[yi]} median')
    if predict_fn:
        if x_med.size>0 and xi<x_med.shape[1]:
            xr = x_med[:,xi]; pad = max((xr.max()-xr.min())*0.1, 0.1)
            xd = np.linspace(xr.min()-pad, xr.max()+pad, 500)
        else: xd = np.linspace(-2,2,500)
        ax.plot(xd, predict_fn(xd), 'g-', lw=2, label=f'{so[yi]} fitted')
    ax.set_title(f"{so[yi]}: True vs Fitted"); ax.set_xlabel(si[xi]); ax.set_ylabel(so[yi])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

def _make_predict_fn(ftype, params):
    if ftype in ("sym_exp","sym_log"):
        def fn(x):
            x = np.atleast_1d(np.asarray(x, float))
            fnb = exp_func if ftype=="sym_exp" else log_func
            y = np.zeros_like(x); pn, pp = params[0]
            pos, neg = x>=0, x<0
            y[pos]=fnb(x[pos],*pp); y[neg]=-fnb(-x[neg],*pn)
            return y
        return fn
    if ftype == "pchip":
        return lambda x: params[0](np.atleast_1d(x))
    if ftype == "sigmoid":
        return lambda x: sigmoid(np.atleast_1d(x), *params[0])
    def fn(x):
        x = np.atleast_1d(np.asarray(x, float)); y = np.zeros(len(x))
        b = np.array([x**j for j in range(len(params[0]))])
        y[:] = params[0] @ b; return y
    return fn

# ===================== Main =====================
if __name__ == "__main__":
    fit_results = []
    names = ["delta_Force_X", "delta_Force_Y"]
    ftypes = [FIT_TYPE_FX, FIT_TYPE_FY]
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), squeeze=False)
    for i in range(2):
        inp, out = [INPUT_COLS[i]], [OUTPUT_COLS[i]]
        ft = ftypes[i]
        Xt, Yt = load_csv(TRAIN_CSV, inp, out)
        print(f"\n--- {out[0]} <- {inp[0]} ({len(Xt)} samples) ---")

        if SPLIT_SIGN:
            pos_mask = Xt[:,0]>=0; neg_mask = ~pos_mask
            Xp, Yp = Xt[pos_mask], Yt[pos_mask]
            Xn, Yn = Xt[neg_mask], Yt[neg_mask]
            print(f"  Positive: {len(Xp)}, Negative: {len(Xn)}")
            if ft == "exp_log":
                params = fit_exp_log(Xt, Yt)
                Y_pred = predict_exp_log(Xt, params)
                fit_results.append((inp, out, params, "exp_log", True))
            elif ft == "sym_exp":
                pp = fit_sym_exp(Xp, Yp)[0]; pn = fit_sym_exp(Xn, Yn)[0]
                params = [pp, pn]
                Y_pred = np.zeros_like(Yt)
                Y_pred[pos_mask] = predict_sym_exp(Xp, [pp])
                Y_pred[neg_mask] = predict_sym_exp(Xn, [pn])
                fit_results.append((inp, out, params, "sym_exp", True))
            elif ft == "sym_log":
                pp = fit_sym_log(Xp, Yp)[0]; pn = fit_sym_log(Xn, Yn)[0]
                params = [pp, pn]
                Y_pred = np.zeros_like(Yt)
                Y_pred[pos_mask] = predict_sym_log(Xp, [pp])
                Y_pred[neg_mask] = predict_sym_log(Xn, [pn])
                fit_results.append((inp, out, params, "sym_log", True))
            elif ft == "pchip":
                params = fit_pchip(Xt, Yt)
                Y_pred = predict_pchip(Xt, params)
                print(f"  PCHIP knots: {len(params[0].x)}")
                fit_results.append((inp, out, params, "pchip", False))
            else:
                cp = fit_poly(Xp, Yp, POLY_ORDER); cn = fit_poly(Xn, Yn, POLY_ORDER)
                params = [cp, cn]
                labels = get_term_labels(inp, POLY_ORDER)
                print("  Positive:"); print_formulas(cp, labels, out)
                print("  Negative:"); print_formulas(cn, labels, out)
                Y_pred = np.zeros_like(Yt)
                Y_pred[pos_mask] = predict_poly(Xp, cp, POLY_ORDER)
                Y_pred[neg_mask] = predict_poly(Xn, cn, POLY_ORDER)
                fit_results.append((inp, out, params, "poly", True))
        else:
            if ft == "pchip":
                params = fit_pchip(Xt, Yt)
                Y_pred = predict_pchip(Xt, params)
                fit_results.append((inp, out, params, "pchip", False))
            elif ft == "sigmoid":
                params = fit_sigmoid(Xt, Yt)
                Y_pred = predict_sigmoid(Xt, params)
                fit_results.append((inp, out, params, "sigmoid", False))
            else:
                coefs = fit_poly(Xt, Yt, POLY_ORDER)
                labels = get_term_labels(inp, POLY_ORDER)
                print_formulas(coefs, labels, out)
                Y_pred = predict_poly(Xt, coefs, POLY_ORDER)
                fit_results.append((inp, out, coefs, "poly", False))
        compute_errors(Yt, Y_pred, out)

        # Plot subplot
        _, _, Xm, Ym = get_medians(TARGET_CSV, inp, out)
        plot_single(axes[i,0], Xm, Ym, 0, 0, inp, out, _make_predict_fn(ftypes[i], fit_results[-1][2]))

    if SAVE_COEFS:
        save_coefs(fit_results, os.path.join(FIT_PARAM_Save, "fitxy_coefs.bin"))
    print(f"  Write back to: {TARGET_CSV}")
    write_back_csv(TARGET_CSV, INPUT_COLS[:2], OUTPUT_COLS[:2], fit_results, POLY_ORDER, WRITE_VALID_ONLY)
    plt.tight_layout()
    sp = os.path.join(os.path.dirname(TARGET_CSV), os.path.splitext(os.path.basename(TARGET_CSV))[0]+"_xy.png")
    plt.savefig(sp, dpi=150); plt.close(fig)
    print(f"  Plot saved: {sp}")
