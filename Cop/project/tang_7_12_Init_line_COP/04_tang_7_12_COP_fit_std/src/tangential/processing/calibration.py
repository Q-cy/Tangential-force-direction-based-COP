"""fit_coefs.bin 的轻量运行时加载与预测。"""

import os

import numpy as np


def load_fit_coefs(path):
    id_map = {
        0: "sigmoid", 1: "poly", 2: "exp_log", 3: "pchip",
        4: "sym_exp", 5: "sym_log", 6: "exp",
    }
    with open(path, "rb") as file_obj:
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


def _resolve_entry(entry, fit_type, split_sign):
    if isinstance(entry, tuple) and len(entry) == 3:
        return entry[0], entry[1], entry[2]
    return entry, fit_type, split_sign


def _log(x, a, b, c):
    return a * np.log(b * x + 1) + c


def _exp(x, a, b, c):
    return a * np.exp(b * x) + c


def _sigmoid(x, level, slope, center, bias):
    return level / (1 + np.exp(-slope * (x - center))) + bias


def apply_fit_predict_multi(x_values, params_list, fit_type, split_sign=False):
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
    def __init__(self, fit_type=None, params_list=None, split_sign=False,
                 path=None, error=None):
        self.fit_type = fit_type
        self.params_list = params_list
        self.split_sign = bool(split_sign)
        self.path = path
        self.error = error

    @property
    def available(self):
        return self.params_list is not None

    @classmethod
    def from_path(cls, path):
        if not path or not os.path.exists(path):
            return cls(path=path)
        try:
            fit_type, _, params_list, split_sign = load_fit_coefs(path)
            return cls(fit_type, params_list, split_sign, path=path)
        except Exception as exc:
            return cls(path=path, error=exc)

    def predict(self, dx, dy, total, cal_dim="3D"):
        if not self.available:
            return (float("nan"),) * 3
        inputs = [dx, dy]
        if cal_dim == "3D":
            inputs.append(total)
        values = list(apply_fit_predict_multi(
            inputs, self.params_list, self.fit_type, self.split_sign
        ))
        values.extend([float("nan")] * (3 - len(values)))
        return tuple(float(value) for value in values[:3])
