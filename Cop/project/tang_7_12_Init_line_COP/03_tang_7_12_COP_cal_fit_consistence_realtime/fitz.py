"""
fitz: Fz(adc_sum) 指数拟合。Fz取反+归一化+exp拟合，不分段。
"""
import os, csv, sys, numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ===================== Config =====================
TRAIN_CSV = "/home/qcy/Project/data/2.PZT_tangential/weight/concat/concat_5_10_15.csv"
TARGET_CSV = TRAIN_CSV
FIT_PARAM_Save = "/home/qcy/Project/data/2.PZT_tangential/weight/png"

INPUT_COLS = ["adc_sum"]
OUTPUT_COLS = ["delta_Force_Z"]
FIT_TYPE_FZ = "exp"
POLY_ORDER = 1
TRAIN_VALID_ONLY = True
WRITE_VALID_ONLY = True
ONE_ON_ONE = True
SAVE_COEFS = True

# ===================== Funcs =====================
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

def fit_exp_norm(X, Y):
    """Fz取反+归一化+exp拟合。返回 a,b,c,xm,xs"""
    x_raw, y = X[:,0], -Y[:,0]  # Fz取反
    xm, xs = 0.0, 1.0
    if np.max(np.abs(x_raw)) > 100:
        xm = float(np.mean(x_raw)); xs = float(np.std(x_raw)) or 1.0
        x_fit = (x_raw - xm) / xs
    else:
        x_fit = x_raw
    try: popt,_ = curve_fit(exp_func, x_fit, y, p0=[1.,1.,np.min(y)], maxfev=10000)
    except: popt = np.array([1.,1.,np.min(y)])
    return (*popt, xm, xs)

def predict_exp(x, a, b, c, xm=0, xs=1):
    x = np.atleast_1d(x)
    return -exp_func((x - xm) / xs, a, b, c)  # 取反还原

def compute_errors(Yt, Yp, out):
    print(f"\n{'='*60}\n  Error Analysis\n{'='*60}")
    for i, nm in enumerate(out):
        e = Yp[:,i]-Yt[:,i]; ae = np.abs(e)
        ssr = np.sum(e**2); sst = np.sum((Yt[:,i]-np.mean(Yt[:,i]))**2)
        r2 = 1-ssr/sst if sst>0 else np.nan
        print(f"  {nm}: MAE={np.mean(ae):.6f} RMSE={np.sqrt(np.mean(e**2)):.6f} R2={r2:.6f}")
    print(f"{'='*60}\n")

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

def save_coefs(params, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(np.int32(6).tobytes()); f.write(np.int32(1).tobytes())  # type=exp, 1 input
        f.write(np.int32(1).tobytes()); f.write(np.int32(5).tobytes())  # 1 output, 5 params
        f.write(np.int32(0).tobytes())  # no split
        f.write(np.array(params, dtype=np.float64).tobytes())
    print(f"  Coefs saved: {path} ({os.path.getsize(path)} bytes)")

def write_back_csv(csv_path, inp_cols, out_cols, popt, write_valid_only=True):
    a,b,c,xm,xs = popt
    with open(csv_path) as f:
        reader = csv.DictReader(f); hdr = [c.strip() for c in reader.fieldnames]; rows = list(reader)
    cnt = 0
    for row in rows:
        if write_valid_only and float(row.get('valid',0))==0: continue
        try:
            xv = float(row[inp_cols[0]])
            fz = float(predict_exp(xv, a, b, c, xm, xs))
        except: continue
        row["delta_Force_Z"] = f"{fz:.6f}"
        cnt+=1
    with open(csv_path,"w",newline="") as f:
        csv.DictWriter(f,hdr).writeheader(); csv.DictWriter(f,hdr).writerows(rows)
    print(f"  Write back done: {cnt} rows -> {csv_path}")

# ===================== Main =====================
if __name__ == "__main__":
    Xt, Yt = load_csv(TRAIN_CSV, INPUT_COLS, OUTPUT_COLS)
    print(f"--- {OUTPUT_COLS[0]} <- {INPUT_COLS[0]} ({len(Xt)} samples) ---")

    a, b, c, xm, xs = fit_exp_norm(Xt, Yt)
    print(f"  exp: a={a:.4f} b={b:.6f} c={c:.4f} norm(m={xm:.0f},s={xs:.0f})")
    Yp = np.array([predict_exp(Xt[:,0], a, b, c, xm, xs)]).T
    compute_errors(Yt, Yp, OUTPUT_COLS)

    if SAVE_COEFS:
        save_coefs((a,b,c,xm,xs), os.path.join(FIT_PARAM_Save, "fitz_coefs.bin"))

    print(f"  Write back to: {TARGET_CSV}")
    write_back_csv(TARGET_CSV, INPUT_COLS, OUTPUT_COLS, (a,b,c,xm,xs), WRITE_VALID_ONLY)

    # Plot
    fig, ax = plt.subplots(1,1,figsize=(10,5))
    _, _, Xm, Ym = get_medians(TARGET_CSV, INPUT_COLS, OUTPUT_COLS)
    if len(Xm)>0: ax.scatter(Xm[:,0], Ym[:,0], s=30, c='red', zorder=5, label='median')
    if len(Xm)>0:
        xr = Xm[:,0]; pad = max((xr.max()-xr.min())*0.1, 1000)
        xd = np.linspace(xr.min()-pad, xr.max()+pad, 500)
    else: xd = np.linspace(-2,2,500)
    ax.plot(xd, predict_exp(xd, a, b, c, xm, xs), 'g-', lw=2, label='fitted')
    ax.set_title(f"{OUTPUT_COLS[0]}: True vs Fitted"); ax.set_xlabel(INPUT_COLS[0]); ax.set_ylabel(OUTPUT_COLS[0])
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sp = os.path.join(os.path.dirname(TARGET_CSV), os.path.splitext(os.path.basename(TARGET_CSV))[0]+"_z.png")
    plt.savefig(sp, dpi=150); plt.close(fig)
    print(f"  Plot saved: {sp}")
