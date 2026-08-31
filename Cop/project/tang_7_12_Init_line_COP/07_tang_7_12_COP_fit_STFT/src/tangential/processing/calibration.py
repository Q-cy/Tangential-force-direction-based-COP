"""fit_coefs.bin 的轻量运行时加载与预测。"""

import io
import os
from importlib import resources

import numpy as np


def _open_binary_source(source):
    """把模型来源规范化为可读取的二进制流并标记是否需要关闭。

    Args:
        source: ``bytes``/``bytearray``/``memoryview``、具有 ``read`` 方法的
            二进制流，或文件路径/路径对象。

    Returns:
        tuple：``(file_obj, should_close)``；前者提供 ``read``，后者表示调用方
        是否负责在读取结束后关闭该流。

    Raises:
        OSError: 路径无法以二进制方式打开。
        TypeError: 来源既不是 bytes-like、文件对象，也不是合法路径。
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return io.BytesIO(bytes(source)), True
    if hasattr(source, "read"):
        return source, False
    return open(source, "rb"), True


def load_fit_coefs(source):
    """从路径、二进制流或内存字节解析 ``fit_coefs.bin`` 模型。

    Args:
        source: 模型文件路径、二进制可读流或 bytes-like 数据。文件格式包含
            输入/输出数量、拟合类型元数据和 float64 参数；类型编号支持
            sigmoid、poly、exp_log、pchip、sym_exp、sym_log、exp。

    Returns:
        tuple：``(first_fit_type, n_inputs, params_list, first_split)``。其中
            ``params_list`` 的每项为 ``(params, fit_type, split_sign)``，
            pchip 的 params 是 SciPy 插值器，其余为 numpy 数组或正负分支数组。

    Raises:
        OSError: 文件或流读取失败。
        IndexError/ValueError: 文件头、元数据或参数内容不完整/非法。
        ImportError: 模型包含 pchip 且 SciPy 插值器不可用。

    Side Effects:
        对路径或 bytes 创建的临时流在解析结束后关闭；调用方提供的流不关闭。
    """
    id_map = {
        0: "sigmoid", 1: "poly", 2: "exp_log", 3: "pchip",
        4: "sym_exp", 5: "sym_log", 6: "exp",
    }
    file_obj, should_close = _open_binary_source(source)
    try:
        n_inputs = int(np.frombuffer(file_obj.read(4), dtype=np.int32)[0])
        n_outputs = int(np.frombuffer(file_obj.read(4), dtype=np.int32)[0])
        metadata = []
        for _ in range(n_outputs):
            fit_id = int(np.frombuffer(file_obj.read(4), dtype=np.int32)[0])
            count = int(np.frombuffer(file_obj.read(4), dtype=np.int32)[0])
            split = int(np.frombuffer(file_obj.read(4), dtype=np.int32)[0]) == 1
            metadata.append((id_map.get(fit_id, "poly"), count, split))

        params_list = []
        for fit_type, count, split in metadata:
            if fit_type in ("sym_exp", "sym_log", "exp_log"):
                negative = np.frombuffer(
                    file_obj.read(count * 8), dtype=np.float64
                ).copy()
                positive = np.frombuffer(
                    file_obj.read(count * 8), dtype=np.float64
                ).copy()
                params_list.append(((negative, positive), fit_type, split))
            elif fit_type == "pchip":
                from scipy.interpolate import PchipInterpolator

                x_knots = np.frombuffer(
                    file_obj.read(count * 8), dtype=np.float64
                ).copy()
                y_knots = np.frombuffer(
                    file_obj.read(count * 8), dtype=np.float64
                ).copy()
                params_list.append(
                    (PchipInterpolator(x_knots, y_knots), fit_type, split)
                )
            elif fit_type == "exp":
                params = np.frombuffer(
                    file_obj.read(count * 8), dtype=np.float64
                ).copy()
                params_list.append((params, fit_type, split))
            elif split:
                positive = np.frombuffer(
                    file_obj.read(count * 8), dtype=np.float64
                ).copy()
                negative = np.frombuffer(
                    file_obj.read(count * 8), dtype=np.float64
                ).copy()
                params_list.append(((positive, negative), fit_type, split))
            else:
                params = np.frombuffer(
                    file_obj.read(count * 8), dtype=np.float64
                ).copy()
                params_list.append((params, fit_type, split))

        first_type = metadata[0][0] if metadata else "poly"
        first_split = metadata[0][2] if metadata else False
        return first_type, n_inputs, params_list, first_split
    finally:
        if should_close:
            file_obj.close()


def _resolve_entry(entry, fit_type, split_sign):
    """将旧式参数项或带元数据参数项统一解析为三元组。

    Args:
        entry: 参数项；可为 ``(params, type, split)``，也可为裸 params。
        fit_type: 裸 params 使用的默认拟合类型。
        split_sign: 裸 params 使用的默认正负分支标志。

    Returns:
        tuple：``(params, effective_fit_type, effective_split_sign)``。
    """
    if isinstance(entry, tuple) and len(entry) == 3:
        return entry[0], entry[1], entry[2]
    return entry, fit_type, split_sign


def _log(x, a, b, c):
    """计算单侧对数拟合函数 ``a*log(b*x+1)+c``。

    Args:
        x: 标量输入。
        a: 输出幅值系数。
        b: 输入缩放系数。
        c: 输出偏置。

    Returns:
        float 或 numpy 标量：拟合函数值；实际类型由 NumPy 运算决定。
    """
    return a * np.log(b * x + 1) + c


def _exp(x, a, b, c):
    """计算指数拟合函数 ``a*exp(b*x)+c``。

    Args:
        x: 标量输入。
        a: 输出幅值系数。
        b: 指数斜率。
        c: 输出偏置。

    Returns:
        float 或 numpy 标量：指数拟合值。
    """
    return a * np.exp(b * x) + c


def _sigmoid(x, level, slope, center, bias):
    """计算四参数 sigmoid 拟合函数。

    Args:
        x: 标量输入。
        level: sigmoid 幅值。
        slope: 斜率。
        center: 中心位置。
        bias: 输出偏置。

    Returns:
        float 或 numpy 标量：``level/(1+exp(-slope*(x-center)))+bias``。
    """
    return level / (1 + np.exp(-slope * (x - center))) + bias


def apply_fit_predict_multi(x_values, params_list, fit_type, split_sign=False):
    """按每个输出通道的拟合参数执行多输出标定预测。

    Args:
        x_values: 输入标量序列；第 ``index`` 个输出使用同索引输入，输入不足
            时回退使用第一个输入。通常对应 dx、dy、总压力等标定量。
        params_list: 每个输出通道的参数项；可含 ``(params, type, split)`` 元组。
        fit_type: 裸参数项使用的默认拟合类型。
        split_sign: 裸参数项使用的默认正负输入分支标志。

    Returns:
        list[float]：与 ``params_list`` 等长的预测值；pchip、poly、exp、
        sigmoid 及正负分支均按模型元数据选择公式。

    Raises:
        ValueError/TypeError: 输入值或参数无法转换为拟合公式所需标量。
        FloatingPointError: NumPy 浮点错误配置为抛出时，公式出现非法数值。
    """
    results = []
    for index, entry in enumerate(params_list):
        x = float(x_values[index] if index < len(x_values) else x_values[0])
        params, current_type, current_split = _resolve_entry(
            entry, fit_type, split_sign
        )
        if current_type == "pchip":
            value = float(params(x))
        elif current_type in ("sym_exp", "sym_log"):
            negative, positive = params
            function = _exp if current_type == "sym_exp" else _log
            value = (
                float(-function(-x, *negative))
                if x < 0 else float(function(x, *positive))
            )
        elif current_type == "exp_log":
            negative, positive = params
            value = (
                float(_exp(x, *negative))
                if x < 0 else float(_log(x, *positive))
            )
        elif current_type == "exp":
            a, b, c, mean, scale = params
            value = float(-_exp((x - mean) / scale, a, b, c))
        elif current_split:
            selected = params[0] if x >= 0 else params[1]
            if current_type == "sigmoid":
                value = float(_sigmoid(x, *selected))
            else:
                basis = np.array([x**power for power in range(len(selected))])
                value = float(np.dot(selected, basis))
        elif current_type == "sigmoid":
            value = float(_sigmoid(x, *params))
        else:
            basis = np.array([x**power for power in range(len(params))])
            value = float(np.dot(params, basis))
        results.append(value)
    return results


class FitCalibrationModel:
    """``fit_coefs.bin`` 的运行时标定模型封装。

    Attributes:
        fit_type: 默认拟合类型字符串，模型不可用时为 ``None``。
        params_list: 每个输出通道的参数列表；加载失败时为 ``None``。
        split_sign: 默认是否按输入正负选择参数分支。
        path: 模型来源路径或 package resource 标识。
        error: 加载失败时保存的异常对象，否则为 ``None``。
        n_inputs: 多输入 poly 模型的输入数量。

    该类只读取模型和执行预测，不训练、不修改输入 CSV，也不会改变模型文件。
    """

    def __init__(self, fit_type=None, params_list=None, split_sign=False,
                 path=None, error=None, n_inputs=1):
        """创建一个已加载或不可用的标定模型对象。

        Args:
            fit_type: 默认拟合类型；通常由 ``load_fit_coefs`` 返回。
            params_list: 输出通道参数列表；``None`` 表示模型不可用。
            split_sign: 默认正负分支标志。
            path: 模型路径或资源标识，用于诊断。
            error: 加载失败时保存的异常对象。
            n_inputs: 模型期望的输入数量，默认为 1。

        Returns:
            None。模型状态保存到实例属性。
        """
        self.fit_type = fit_type
        self.params_list = params_list
        self.split_sign = bool(split_sign)
        self.path = path
        self.error = error
        self.n_inputs = int(n_inputs)

    @property
    def available(self):
        """报告模型参数是否已成功加载。

        Returns:
            bool：``params_list`` 非 ``None`` 时为 ``True``，否则为 ``False``。
        """
        return self.params_list is not None

    @classmethod
    def from_path(cls, path):
        """从外部文件路径加载标定模型。

        Args:
            path: ``fit_coefs.bin`` 文件路径；空路径或不存在时返回不可用模型。

        Returns:
            FitCalibrationModel：成功时包含解析参数；失败时返回 ``available``
            为 ``False`` 且 ``error`` 保存异常的对象，不向调用方抛出读取异常。

        Side Effects:
            只读取指定文件，不写入路径或修改模型内容。
        """
        if not path or not os.path.exists(path):
            return cls(path=path)
        try:
            fit_type, n_inputs, params_list, split_sign = load_fit_coefs(path)
            return cls(
                fit_type, params_list, split_sign, path=path,
                n_inputs=n_inputs,
            )
        except Exception as exc:
            return cls(path=path, error=exc)

    @classmethod
    def from_default(cls):
        """从 wheel 内置 package resource 加载默认模型。

        Returns:
            FitCalibrationModel：从 ``tangential.resources/fit_coefs.bin`` 加载的
            模型；资源缺失或解析失败时返回不可用对象并保存异常。

        Side Effects:
            通过 ``importlib.resources`` 读取 package resource，不依赖当前工作
            目录，也不修改静态资源。
        """
        resource_name = "tangential.resources/fit_coefs.bin"
        try:
            resource = resources.files("tangential.resources").joinpath(
                "fit_coefs.bin"
            )
            fit_type, n_inputs, params_list, split_sign = load_fit_coefs(
                resource.read_bytes()
            )
            return cls(
                fit_type,
                params_list,
                split_sign,
                path=resource_name,
                n_inputs=n_inputs,
            )
        except Exception as exc:
            return cls(path=resource_name, error=exc)

    def predict(self, dx, dy, adc_sum, cal_dim="3D"):
        """根据 CoP 位移和阵列 ADC 总和预测 Fx、Fy、Fz。

        Args:
            dx: CoP X 位移，单位为传感器 cell。
            dy: CoP Y 位移，单位为传感器 cell。
            adc_sum: 当前阵列全部通道 ADC 总和；``cal_dim="3D"`` 时作为第三输入。
            cal_dim: 标定维度；``"3D"`` 使用 dx/dy/adc_sum，否则只使用 dx/dy。

        Returns:
            tuple[float, float, float]：按 Fx、Fy、Fz 顺序的预测值；模型不可用
            或输出通道不足 3 个时，缺失位置填 ``NaN``。

        Raises:
            ValueError: 多输入 poly 模型的系数数量无法推断多项式阶数。
            TypeError/OverflowError: 输入无法转换为标量或参与浮点运算。
        """
        if not self.available:
            return (float("nan"),) * 3
        inputs = [dx, dy]
        if cal_dim == "3D":
            inputs.append(adc_sum)
        resolved_entries = [
            _resolve_entry(entry, self.fit_type, self.split_sign)
            for entry in self.params_list
        ]
        if (
            self.n_inputs > 1
            and resolved_entries
            and all(entry[1] == "poly" for entry in resolved_entries)
        ):
            values = []
            inputs = np.asarray(inputs[:self.n_inputs], dtype=np.float64)
            first_params, _, first_split = resolved_entries[0]
            term_count = (
                len(first_params[0]) if first_split else len(first_params)
            )
            order = _infer_poly_order(term_count, self.n_inputs)
            for params, _, current_split in resolved_entries:
                if current_split:
                    params = params[0] if inputs[0] >= 0 else params[1]
                values.append(float(np.dot(params, _poly_basis(inputs, order))))
            values.extend([float("nan")] * (3 - len(values)))
            return tuple(float(value) for value in values[:3])
        values = list(apply_fit_predict_multi(
            inputs, self.params_list, self.fit_type, self.split_sign
        ))
        values.extend([float("nan")] * (3 - len(values)))
        return tuple(float(value) for value in values[:3])


def _poly_basis(values, order):
    """构造常数项及总次数不超过指定阶数的多项式基。

    Args:
        values: 输入向量，可为列表或 ndarray；会展平为一维 float64。
        order: 多项式阶数；当前实现支持 1、2、3，分别加入一次、二次和三次
            组合项。

    Returns:
        np.ndarray：一维 float64 基向量，第一项恒为 1，项顺序与训练/预测约定一致。

    Raises:
        ValueError: ``order`` 为负数等不符合调用约定的值时可能产生非法基。
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    columns = [1.0]
    if order >= 1:
        columns.extend(values[index] for index in range(len(values)))
    if order >= 2:
        columns.extend(
            values[i] * values[j]
            for i in range(len(values))
            for j in range(i, len(values))
        )
    if order >= 3:
        columns.extend(
            values[i] * values[j] * values[k]
            for i in range(len(values))
            for j in range(i, len(values))
            for k in range(j, len(values))
        )
    return np.asarray(columns, dtype=np.float64)


def _infer_poly_order(term_count, n_inputs):
    """根据输入维数和系数项数量推断 1 到 3 阶多项式阶数。

    Args:
        term_count: 模型系数数量。
        n_inputs: 输入变量数量。

    Returns:
        int：匹配的阶数，取 1、2 或 3。

    Raises:
        ValueError: 没有任何支持阶数能产生 ``term_count`` 个基项。
    """
    for order in (1, 2, 3):
        if len(_poly_basis(np.zeros(n_inputs), order)) == term_count:
            return order
    raise ValueError(
        f"无法从 {n_inputs} 个输入和 {term_count} 个系数推断 poly 阶数"
    )
