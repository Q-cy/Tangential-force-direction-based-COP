"""F1 联动 + 训练/预测一致性: _row_valid fallback + load_csv + save/load coefs + 冒烟"""
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fit as F


def test_row_valid_fallback():
    # 新版 recorder: valid 列存在
    assert F._row_valid({"valid": "1", "CoP_state": "2"}, ["valid", "CoP_state"]) is True
    assert F._row_valid({"valid": "0", "CoP_state": "1"}, ["valid", "CoP_state"]) is False
    # 旧版: 只有 CoP_state → fallback
    assert F._row_valid({"CoP_state": "2"}, ["CoP_state"]) is True
    assert F._row_valid({"CoP_state": "0"}, ["CoP_state"]) is False
    # 都没有 → 视为有效
    assert F._row_valid({"x": "1"}, ["x"]) is True


def test_load_csv_valid_column():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "new.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("delta_CoP_X,delta_CoP_Y,delta_Force_X,CoP_state,valid\n")
            for i in range(1, 6):
                f.write(f"{i},{i},{i * 2},2,1\n")     # 有效行
            f.write("0,0,0,0,0\n")                     # 无效行
        X, Y = F.load_csv(path, ["delta_CoP_X", "delta_CoP_Y"], ["delta_Force_X"], valid_only=True)
        assert len(X) == 5 and len(Y) == 5
        Xa, Ya = F.load_csv(path, ["delta_CoP_X"], ["delta_Force_X"], valid_only=False)
        assert len(Xa) == 6


def test_load_csv_fallback_coP_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "old.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("delta_CoP_X,delta_Force_X,CoP_state\n")
            f.write("1,2,2\n")     # 接触
            f.write("0,0,0\n")     # 未接触
        X, Y = F.load_csv(path, ["delta_CoP_X"], ["delta_Force_X"], valid_only=True)
        assert len(X) == 1


def test_load_csv_no_valid_column():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "none.csv")
        with open(path, "w", encoding="utf-8") as f:
            f.write("delta_CoP_X,delta_Force_X\n")
            f.write("1,2\n")
        X, Y = F.load_csv(path, ["delta_CoP_X"], ["delta_Force_X"], valid_only=True)
        assert len(X) == 1   # 无任何筛选列 → 全视为有效


def test_coefs_roundtrip_sym_log_and_exp():
    """save_coefs → load_coefs 往返一致 (与 fit_coefs.bin 相同结构: sym_log/sym_log/exp)"""
    p1 = (np.array([1.0, 2.0, 0.0]), np.array([1.5, 2.5, 0.0]))
    p2 = (np.array([0.5, 1.0, 0.0]), np.array([0.8, 1.2, 0.0]))
    p3 = np.array([3.0, 0.001, 1.0, 0.0, 1.0])   # exp: (a, b, c, xm, xs)
    fit_results = [
        (["delta_CoP_X"], ["delta_Force_X"], [p1], "sym_log", False),
        (["delta_CoP_Y"], ["delta_Force_Y"], [p2], "sym_log", False),
        (["adc_sum"], ["delta_Force_Z"], [p3], "exp", False),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "coefs.bin")
        F.save_coefs(fit_results, path)
        ft, n_in, params_list, split = F.load_coefs(path)
        assert ft == "sym_log" and n_in == 1 and split is False
        assert len(params_list) == 3
        assert params_list[0][1] == "sym_log" and params_list[2][1] == "exp"
        assert np.allclose(params_list[0][0][0], p1[0])   # p_neg 还原
        assert np.allclose(params_list[2][0], p3)         # exp 参数还原
        # 预测冒烟
        preds = F.apply_predict_multi([1.5, 2.0, 3000.0], params_list, ft, split)
        assert len(preds) == 3
        assert all(np.isfinite(p) for p in preds)
        assert preds[2] < 0   # exp 类型: 训练对 -Y 拟合 → 预测带负号


def test_real_bin_smoke():
    """当前 fit_coefs.bin 加载 + 预测冒烟 (无 NaN/inf)"""
    here = os.path.dirname(os.path.abspath(__file__))
    bin_path = os.path.join(os.path.dirname(here), "fit_coefs.bin")
    ft, n_in, params_list, split = F.load_coefs(bin_path)
    assert len(params_list) == 3
    for x_vals in ([1.5, 2.0, 3000.0], [0.0, 0.0, 0.0], [-1.5, -2.0, 8000.0]):
        preds = F.apply_predict_multi(x_vals, params_list, ft, split)
        assert len(preds) == 3
        assert all(np.isfinite(p) for p in preds), f"非有限预测: {preds} @ {x_vals}"
