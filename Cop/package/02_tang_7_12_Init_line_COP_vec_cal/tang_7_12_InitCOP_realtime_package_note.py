"""
tang_7_12_InitCOP_realtime_package_note
=======================================
压阻阵列数据处理库入口(CoP 计算 + 角度 + 拟合 inference)

包含模块:
    COP            - 压阻 CoP 计算核心(状态机、baseline、初始接触检测、静置精修)
    angle          - 向量角度/幅值工具
    fit_inference  - 拟合预测(load_coefs + predict_sym_log / predict_exp / apply_predict_multi)

使用:
    import sys
    sys.path.insert(0, '/path/to/01_tang_7_12_Init_line_COP_vec_cal')
    import tang_7_12_InitCOP_realtime_package_note as cop

    cop.reset_cop_state()
    res = cop.compute_pressure_direction(adc_data)
    angle, mag = cop.compute_PZT_angle(res[6], res[7])

    fit_type, _, params_list, split_sign = cop.load_coefs('fit_coefs.bin')
    results = cop.apply_predict_multi([dx, dy], params_list, fit_type, split_sign)

依赖:
    numpy, threading, 标准库
"""

import os
import threading
from collections import deque

import numpy as np


# =====================================================================
# Section 1: 算法常量
# =====================================================================

# ---------- CoP 状态机常量(从 COP.py) ----------
COP_INIT_MEDIAN_FRAMES = 1
# 初始 CoP 取中位数的帧数(决定何时锁定初始接触点)

COP_BASELINE_COLLECT_FRAMES = 20
# 启动后采集多少帧的 total_pressure 用于计算动态阈值

COP_THRESH_K = 5
# 动态阈值乘数:threshold = K × mean(baseline),K 越大越不敏感

COP_SENSOR_ROW_CNT = 12
# 压阻传感器阵列行数

COP_SENSOR_COL_CNT = 7
# 压阻传感器阵列列数(12 × 7 = 84 通道)

COP_POST_INIT_WINDOW_CNT = 600000
# 初始 CoP 确定后,精修监测的最大帧数(超时则强制完成)

COP_POST_INIT_STABLE_CNT = 10
# 原始模式:精修阶段需连续保持静止的帧数

COP_POST_INIT_STABLE_THRESH = 0.1
# 精修判据:CoP 偏移距离阈值(欧氏距离,小于此值算"静止")

COP_POST_INIT_TRIGGER_CNT = 20
# 触发模式:收到 trigger_cop_refine 信号后需连续保持静止的帧数

COP_SNAP_CENTER_X, COP_SNAP_CENTER_Y = 3.0, 5.5
# 吸附目标(阵列中心),初始 CoP 在此范围内时被吸附到此点

COP_SNAP_RANGE_X = 0.0
# X 方向吸附范围(0 = 禁用吸附)

COP_SNAP_RANGE_Y = 0.0
# Y 方向吸附范围(0 = 禁用吸附)

# ---------- 拟合类型常量(用户指定的 3 个) ----------
FIT_TYPE_SYM_LOG = "sym_log"
# 对称对数模型: y = a*ln(b*x+1) + c,正负 x 各自拟合参数

FIT_TYPE_EXP = "exp"
# 指数模型: y = a*exp(b*x) + c

FIT_TYPE_FX = "sym_log"
# Fx 输出使用的拟合类型

FIT_TYPE_FY = "sym_log"
# Fy 输出使用的拟合类型

FIT_TYPE_FZ = "exp"
# Fz 输出使用的拟合类型

# 拟合类型 ID 映射(用于 .bin 序列化,只支持 sym_log 和 exp)
_TYPE_ID_MAP = {
    "sym_log": 5,  # .bin 中 type_id = 5
    "exp": 6,        # .bin 中 type_id = 6
}
_ID_TO_TYPE = {v: k for k, v in _TYPE_ID_MAP.items()}
# _ID_TO_TYPE 是 _TYPE_ID_MAP 的反向映射,用于 .bin 读取时把 int ID 转换回 str


# =====================================================================
# Section 2: CoP 状态机(类封装,只导出梯度表)
# =====================================================================

class _CopState:
    """CoP 计算的内部状态机(单例:模块级 _cop_state)。"""

    def __init__(self):
        self.contact_init_x = None
        # 初始接触点 X 坐标(锁定后不再变,除非重置或精修)

        self.contact_init_y = None
        # 初始接触点 Y 坐标(锁定后不再变,除非重置或精修)

        self.contact_init_flag = False
        # 初始接触点是否已稳定确定;True 后 CoP 输出相对偏移(dx, dy)

        self.init_x_buf = deque(maxlen=COP_INIT_MEDIAN_FRAMES)
        # 候选初始 CoP X 序列缓冲(收集前 N 帧取中位数)

        self.init_y_buf = deque(maxlen=COP_INIT_MEDIAN_FRAMES)
        # 候选初始 CoP Y 序列缓冲

        self.post_init_frame_cnt = 0
        # 精修阶段已监测的帧数(超过 COP_POST_INIT_WINDOW_CNT 则强制结束)

        self.post_stable_cnt = 0
        # 精修阶段连续满足静止判据的帧数(达到阈值后完成精修)

        self.post_refined_flag = False
        # 精修是否已完成;True 后 CoP 输出用精修后的基准点

        self.post_cand_x = None
        # 精修候选静止点的 X 坐标(满足静止判据时更新)

        self.post_cand_y = None
        # 精修候选静止点的 Y 坐标

        self.noise_sum_buf = deque(maxlen=COP_BASELINE_COLLECT_FRAMES)
        # 启动期 total_pressure 缓冲,用于计算动态阈值

        self.dynamic_thresh = None
        # 动态计算后的低压阈值 = K × mean(baseline);None = 未校准

        self.post_trigger_signal = False
        # 外部触发信号标志(由 trigger_cop_refine 设置)

        self.post_triggered = False
        # 是否已进入触发模式(True 后精修阈值切换为 TRIGGER_CNT)

        self.filtered_dir = None
        # 滤波后的方向向量(暂未使用,保留供将来扩展)

        self.grad_table_arr = np.zeros((COP_SENSOR_ROW_CNT, COP_SENSOR_COL_CNT, 2))
        # 梯度表 (rows, cols, 2):每个 cell 的 (dx, dy) 梯度,供实时图可视化

        self.grad_table_lock = threading.Lock()
        # 梯度表读写锁(compute_pressure_direction 写、实时图读)

# 模块级单例(整个库共享同一份 CoP 状态)
_cop_state = _CopState()

# 公共全局(供外部代码,如 realtime.py,直接读梯度表)
g_cop_grad_table_arr = _cop_state.grad_table_arr
g_cop_grad_table_lock = _cop_state.grad_table_lock


# =====================================================================
# Section 3: 角度工具(从 angle.py)
# =====================================================================

def compute_vector_angle(x, y):
    """计算向量(x,y)的角度(0~360°)和幅值"""
    epsilon = 1e-8
    mag = np.hypot(x, y)
    angle = np.degrees(np.arctan2(y, x + epsilon))
    if angle < 0:
        angle += 360
    return angle, mag


def compute_PZT_angle(Px, Py):
    """计算压阻传感器(Px,Py)的角度(0~360°)和幅值"""
    return compute_vector_angle(Px, Py)


def angle_difference(a1, a2):
    """计算两个角度的最小差值(0~180°)"""
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)


# =====================================================================
# Section 4: CoP 计算核心(从 COP.py)
# =====================================================================

def reset_cop_state():
    """压力过低/离开接触面 → 重置所有状态"""
    s = _cop_state
    s.filtered_dir = None
    s.contact_init_x = None
    s.contact_init_y = None
    s.contact_init_flag = False
    s.init_x_buf.clear()
    s.init_y_buf.clear()
    s.post_init_frame_cnt = 0
    s.post_stable_cnt = 0
    s.post_refined_flag = False
    s.post_cand_x = None
    s.post_cand_y = None
    s.post_trigger_signal = False
    s.post_triggered = False
    with s.grad_table_lock:
        s.grad_table_arr.fill(0)


def trigger_cop_refine():
    """外部调用:触发二次精修切换为触发模式(20 帧)"""
    s = _cop_state
    if s.contact_init_flag and not s.post_refined_flag:
        s.post_trigger_signal = True


def compute_pressure_direction(raw_frame):
    """
    输入:84 通道原始 ADC 数据
    输出:11-tuple (cop_x, cop_y, _, row_max, _, col_max, dx, dy, base_x, base_y, state)
    """
    s = _cop_state

    sensor_rows, sensor_cols = COP_SENSOR_ROW_CNT, COP_SENSOR_COL_CNT
    frame_flat_arr = np.asarray(raw_frame, dtype=np.float32).flatten()
    frame_2d_arr = frame_flat_arr.reshape(sensor_rows, sensor_cols)

    # 计算梯度(用于可视化)
    grad_arr = np.zeros((sensor_rows, sensor_cols, 2), dtype=np.float32)
    for row_idx in range(sensor_rows):
        for col_idx in range(sensor_cols):
            center_val = frame_2d_arr[row_idx, col_idx]
            left_val = frame_2d_arr[row_idx, col_idx - 1] if col_idx - 1 >= 0 else center_val
            right_val = frame_2d_arr[row_idx, col_idx + 1] if col_idx + 1 < sensor_cols else center_val
            up_val = frame_2d_arr[row_idx - 1, col_idx] if row_idx - 1 >= 0 else center_val
            down_val = frame_2d_arr[row_idx + 1, col_idx] if row_idx + 1 < sensor_rows else center_val
            grad_x = right_val - left_val
            grad_y = up_val - down_val
            grad_arr[row_idx, col_idx] = (grad_x, grad_y)
    with s.grad_table_lock:
        s.grad_table_arr[:] = grad_arr[:]

    # 总压力
    total_press_val = np.sum(frame_2d_arr)

    # 动态阈值:启动后收集前 N 帧的 total_press_val,计算 mean + K*std
    if s.dynamic_thresh is None:
        s.noise_sum_buf.append(total_press_val)
        if len(s.noise_sum_buf) >= COP_BASELINE_COLLECT_FRAMES:
            sums = np.array(s.noise_sum_buf)
            s.dynamic_thresh = COP_THRESH_K * float(np.mean(sums))

    # 总压力判断:动态阈值就绪后才启用低压重置
    if s.dynamic_thresh is not None and total_press_val < s.dynamic_thresh:
        if s.contact_init_flag:
            reset_cop_state()
        return 0.0, 0.0, 0, sensor_rows - 1, 0, sensor_cols - 1, 0.0, 0.0, 0.0, 0.0, 0

    if total_press_val == 0:
        return 0.0, 0.0, 0, sensor_rows - 1, 0, sensor_cols - 1, 0.0, 0.0, 0.0, 0.0, 0

    # 计算 CoP 中心
    grid_x_arr = np.tile(np.arange(sensor_cols), (sensor_rows, 1))
    grid_y_arr = np.repeat(np.arange(sensor_rows), sensor_cols).reshape(sensor_rows, sensor_cols)
    cop_curr_x = np.sum(frame_2d_arr * grid_x_arr) / total_press_val
    cop_curr_y = np.sum(frame_2d_arr * grid_y_arr) / total_press_val

    cop_delta_x = 0.0
    cop_delta_y = 0.0
    cop_base_x = cop_curr_x
    cop_base_y = cop_curr_y

    # ============ 初始点稳定判断(中位数判定) ============
    if not s.contact_init_flag:
        s.init_x_buf.append(cop_curr_x)
        s.init_y_buf.append(cop_curr_y)
        if len(s.init_x_buf) >= COP_INIT_MEDIAN_FRAMES:
            s.contact_init_x = float(np.median(s.init_x_buf))
            s.contact_init_y = float(np.median(s.init_y_buf))
            s.contact_init_flag = True
            s.init_x_buf.clear()
            s.init_y_buf.clear()
            if (abs(s.contact_init_x - COP_SNAP_CENTER_X) <= COP_SNAP_RANGE_X and
                abs(s.contact_init_y - COP_SNAP_CENTER_Y) <= COP_SNAP_RANGE_Y):
                s.contact_init_x = COP_SNAP_CENTER_X
                s.contact_init_y = COP_SNAP_CENTER_Y

    # ========== 计算偏移量 ==========
    else:  # s.contact_init_flag 为 True
        # 二次静置精修:检测静止,修正初始 CoP
        s.post_init_frame_cnt += 1

        # 检查触发信号:原始模式计数中收到触发 → 切换为触发模式
        if s.post_trigger_signal and not s.post_refined_flag and not s.post_triggered:
            s.post_trigger_signal = False
            s.post_triggered = True
            s.post_cand_x = cop_curr_x
            s.post_cand_y = cop_curr_y
            s.post_stable_cnt = 1

        # 确定当前精修阈值
        stable_thresh = COP_POST_INIT_TRIGGER_CNT if s.post_triggered else COP_POST_INIT_STABLE_CNT

        if not s.post_refined_flag and s.post_init_frame_cnt <= COP_POST_INIT_WINDOW_CNT:
            if s.post_cand_x is not None:
                dist_val = np.hypot(cop_curr_x - s.post_cand_x,
                                    cop_curr_y - s.post_cand_y)
                if dist_val <= COP_POST_INIT_STABLE_THRESH:
                    s.post_stable_cnt += 1
                else:
                    s.post_cand_x = cop_curr_x
                    s.post_cand_y = cop_curr_y
                    s.post_stable_cnt = 1
            else:
                s.post_cand_x = cop_curr_x
                s.post_cand_y = cop_curr_y
                s.post_stable_cnt = 1

            if s.post_stable_cnt >= stable_thresh:
                s.contact_init_x = s.post_cand_x
                s.contact_init_y = s.post_cand_y
                s.post_refined_flag = True
        else:
            s.post_refined_flag = True  # 超时或已完成

        cop_delta_x = cop_curr_x - s.contact_init_x
        cop_delta_y = s.contact_init_y - cop_curr_y
        cop_base_x = s.contact_init_x
        cop_base_y = s.contact_init_y

    cop_state = 2 if s.post_refined_flag else 1

    return (cop_curr_x, cop_curr_y,
            0, sensor_rows - 1, 0, sensor_cols - 1,
            cop_delta_x, cop_delta_y,
            cop_base_x, cop_base_y,
            cop_state)


# =====================================================================
# Section 5: 拟合 inference(从 fit.py,只保留 inference)
# =====================================================================

# ---------- 基础数学函数 ----------
def log_func(x, a, b, c):
    """Logarithmic: y = a * ln(b * x + 1) + c"""
    return a * np.log(b * x + 1) + c


def exp_func(x, a, b, c):
    """Exponential: y = a * exp(b * x) + c, with overflow protection."""
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(over="ignore"):
        return a * np.exp(np.clip(b * x, -700, 700)) + c


# ---------- 预测函数(只保留 sym_log 和 exp) ----------
def predict_sym_log(x, params_3tuple):
    """
    用 sym_log 模型预测单点。
    params_3tuple: load_coefs 返回的 3-tuple 格式 ((p_neg, p_pos), "sym_log", split)
    """
    (p_neg, p_pos), ft, _ = params_3tuple
    x = float(x)
    if x < 0:
        return float(-log_func(-x, *p_neg))
    else:
        return float(log_func(x, *p_pos))


def predict_exp(x, params_3tuple):
    """
    用 exp 模型预测单点。
    params_3tuple: load_coefs 返回的 3-tuple 格式 (p_5tuple, "exp", split)
    其中 p_5tuple = (a, b, c, xm, xs),预测 y = -exp_func((x - xm) / xs, a, b, c)
    """
    p, ft, _ = params_3tuple
    a, b, c, xm, xs = p
    return float(-exp_func((x - xm) / xs, a, b, c))


# ---------- save/load .bin ----------
def save_coefs(fit_results, path):
    """
    保存拟合结果到 .bin。fit_results: list of (inp, out, params, ftype, split)。
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "wb") as f:
        n_inputs = len(fit_results[0][0]) if fit_results else 1
        total_outputs = sum(len(out) for _, out, _, _, _ in fit_results)

        def _type_id(ft):
            return _TYPE_ID_MAP.get(ft, 1)

        # Header
        f.write(np.int32(n_inputs).tobytes())
        f.write(np.int32(total_outputs).tobytes())

        # Per-output metadata
        for _, out, params, ftype, split in fit_results:
            fid = _type_id(ftype)
            if ftype == "sym_log":
                npar = 3
            elif ftype == "exp":
                npar = 5
            else:
                raise ValueError(
                    f"Unsupported fit type for save_coefs: '{ftype}'. "
                    f"Only 'sym_log' and 'exp' are supported."
                )
            for _ in range(len(out)):
                f.write(np.int32(fid).tobytes())
                f.write(np.int32(npar).tobytes())
                f.write(np.int32(1 if split else 0).tobytes())

        # Per-output params data
        for _, out, params, ftype, split in fit_results:
            n_out = len(out)
            if ftype == "sym_log":
                for p_neg, p_pos in params:
                    f.write(np.array(p_neg, dtype=np.float64).tobytes())
                    f.write(np.array(p_pos, dtype=np.float64).tobytes())
            elif ftype == "exp":
                for p in params:
                    f.write(np.array(p, dtype=np.float64).tobytes())
            else:
                raise ValueError(
                    f"Unsupported fit type for save_coefs: '{ftype}'. "
                    f"Only 'sym_log' and 'exp' are supported."
                )


def load_coefs(path):
    """
    从 .bin 加载拟合系数。
    返回 (fit_type, n_inputs, params_list, split_sign)。
    params_list 中每项是 3-tuple: (params, fit_type_str, split_bool)。
    """
    with open(path, "rb") as f:
        n_inputs = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        n_outputs = int(np.frombuffer(f.read(4), dtype=np.int32)[0])

        meta = []
        for _ in range(n_outputs):
            fid = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            npar = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            spl = int(np.frombuffer(f.read(4), dtype=np.int32)[0]) == 1
            ft_name = _ID_TO_TYPE.get(fid)
            if ft_name is None:
                raise ValueError(
                    f"Unknown fit type ID in .bin: {fid}. "
                    f"Expected one of {_ID_TO_TYPE.keys()}."
                )
            meta.append((ft_name, npar, spl))

        params_list = []
        for ft, npar, spl in meta:
            if ft == "sym_log":
                p_neg = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                p_pos = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                params_list.append(((p_neg, p_pos), ft, spl))
            elif ft == "exp":
                p = np.frombuffer(f.read(npar * 8), dtype=np.float64).copy()
                params_list.append((p, ft, spl))
            else:
                raise ValueError(
                    f"Unsupported fit type in .bin: '{ft}'. "
                    f"Only 'sym_log' and 'exp' are supported."
                )

    first_ft = meta[0][0] if meta else "sym_log"
    first_split = meta[0][2] if meta else False
    return first_ft, n_inputs, params_list, first_split


# ---------- 应用预测 ----------
def apply_predict(x_val, params_list, fit_type, split_sign=False):
    """
    预测单输入多输出。x_val: 标量或 1D array。返回 list[float]。
    支持 fit 类型: "sym_log" 和 "exp"。其他类型抛 ValueError。
    """
    x = float(x_val) if np.isscalar(x_val) else float(x_val[0])
    return apply_predict_multi([x], params_list, fit_type, split_sign)


def apply_predict_multi(x_vals_list, params_list, fit_type, split_sign=False):
    """
    每个输出用不同输入预测。x_vals_list: list of scalars, one per output。
    支持 fit 类型: "sym_log" 和 "exp"。其他类型抛 ValueError。
    """
    results = []
    for i, p in enumerate(params_list):
        x = float(x_vals_list[i]) if i < len(x_vals_list) else float(x_vals_list[0])
        params, ft, _ = p  # 3-tuple: (params, fit_type_str, split_bool)
        if ft == "sym_log":
            p_neg, p_pos = params
            if x < 0:
                results.append(float(-log_func(-x, *p_neg)))
            else:
                results.append(float(log_func(x, *p_pos)))
        elif ft == "exp":
            a, b, c, xm, xs = params
            results.append(float(-exp_func((x - xm) / xs, a, b, c)))
        else:
            raise ValueError(
                f"Unsupported fit type: '{ft}'. Only 'sym_log' and 'exp' are supported."
            )
    return results


# =====================================================================
# Section 6: CoP 顶层封装(用户主入口)
# =====================================================================

def _compute_cop_delta(adc_data):
    """
    内部:84 通道 ADC → (dx, dy) CoP 偏移。
    每次调用前会 reset_cop_state()(确保无历史状态污染)。
    """
    reset_cop_state()
    res = compute_pressure_direction(adc_data)
    return res[6], res[7]


def _compute_cop_xyz_from_delta(dx, dy, adc_data, model_path):
    """
    内部:(dx, dy, adc) + 模型路径 → (Fx, Fy, Fz)。
    接受已算好的 dx, dy,避免与 compute_cop_angle 重复计算 CoP。
    """
    fit_type, _, params_list, split_sign = load_coefs(model_path)
    total = float(np.sum(adc_data))
    return apply_predict_multi([dx, dy, total], params_list, fit_type, split_sign)


def _compute_cop_xyz(adc_data, model_path):
    """
    内部:84 通道 ADC + 模型路径 → (Fx, Fy, Fz)。
    内部调用 _compute_cop_delta(供 compute_cop_fx/fy/fz 单独调用)。
    """
    dx, dy = _compute_cop_delta(adc_data)
    return _compute_cop_xyz_from_delta(dx, dy, adc_data, model_path)


def compute_cop_angle(adc_data):
    """84 通道 ADC → 压阻传感器角度(°)。不需要模型。"""
    dx, dy = _compute_cop_delta(adc_data)
    angle, _ = compute_PZT_angle(dx, dy)
    return angle


def compute_cop_fx(adc_data, model_path):
    """84 通道 ADC + 拟合模型 → Fx (N)。"""
    return _compute_cop_xyz(adc_data, model_path)[0]


def compute_cop_fy(adc_data, model_path):
    """84 通道 ADC + 拟合模型 → Fy (N)。"""
    return _compute_cop_xyz(adc_data, model_path)[1]


def compute_cop_fz(adc_data, model_path):
    """84 通道 ADC + 拟合模型 → Fz (N)。"""
    return _compute_cop_xyz(adc_data, model_path)[2]


def compute_cop_data(adc_data, model_path):
    """
    总入口:84 通道 ADC + 拟合模型路径 → (angle°, Fx, Fy, Fz)。

    这是推荐的 user-facing 接口 —— 一次调用得到 4 个值。

    :param adc_data: 84 通道 ADC 原始数据(list/array)
    :param model_path: 拟合模型 .bin 文件路径(由 fit.py 训练产生)
    :return: (angle_deg, Fx_N, Fy_N, Fz_N)
    """
    dx, dy = _compute_cop_delta(adc_data)
    angle, _ = compute_PZT_angle(dx, dy)
    Fx, Fy, Fz = _compute_cop_xyz_from_delta(dx, dy, adc_data, model_path)
    return angle, Fx, Fy, Fz
