"""84通道压阻阵列多量程分段一致性标定与运行时系数加载。

离线阶段把配置目录中的每个CSV视为一个量程端点：读取最后若干个非空
数据行并按通道求均值，再把各通道端点单调化后建立分段线性映射。运行时
只加载安全的v2 NPZ并应用到一帧原始ADC，不访问离线CSV。
"""

from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from collections import deque
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import ConsistenceCalibrationConfig


CHANNEL_COUNT = 84
FORMAT_VERSION = 2
DEFAULT_RESOURCE_NAME = "consistence_coeffs.npz"
_CHANNEL_NAMES = tuple(f"channel{index}" for index in range(1, CHANNEL_COUNT + 1))
_SEGMENT_VALUE_PATTERN = re.compile(r"-(\d+(?:\.\d+)?)G$", re.IGNORECASE)


def _as_finite_float(value: Any, *, label: str) -> float:
    """把CSV单元格转换为有限浮点数。"""
    try:
        converted = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字，实际为 {value!r}") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{label} 必须是有限数字")
    return converted


def _segment_value_from_path(path: Path) -> float:
    """从 ``*-<数值>G.csv`` 文件名提取用于排序的量程值。"""
    match = _SEGMENT_VALUE_PATTERN.search(path.stem)
    if match is None:
        raise ValueError(
            f"一致性标定CSV文件名必须以-<数值>G结尾: {path.name}"
        )
    value = float(match.group(1))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"一致性标定量程值必须大于0: {path.name}")
    return value


def _read_segment_endpoint(
    path: Path,
    *,
    tail_rows: int,
) -> tuple[np.ndarray, int, int, str]:
    """读取一个CSV最后 ``tail_rows`` 个非空行并返回84通道均值。"""
    source_bytes = path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"一致性标定CSV为空: {path}") from exc
        headers = [str(item).strip() for item in raw_header]
        if any(not item for item in headers):
            raise ValueError(f"一致性标定CSV表头不能包含空列名: {path.name}")
        if len(set(headers)) != len(headers):
            raise ValueError(f"一致性标定CSV表头包含重复列名: {path.name}")
        missing = [name for name in _CHANNEL_NAMES if name not in headers]
        if missing:
            raise ValueError(
                f"一致性标定CSV {path.name} 缺少列: " + ", ".join(missing)
            )
        channel_indices = [headers.index(name) for name in _CHANNEL_NAMES]
        tail: deque[tuple[int, list[str]]] = deque(maxlen=tail_rows)
        nonempty_count = 0
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(str(cell).strip() for cell in row):
                continue
            nonempty_count += 1
            tail.append((line_number, row))

    if nonempty_count < tail_rows:
        raise ValueError(
            f"一致性标定CSV {path.name} 至少需要{tail_rows}个非空数据行，"
            f"实际为{nonempty_count}"
        )
    values: list[list[float]] = []
    for line_number, row in tail:
        if len(row) != len(headers):
            raise ValueError(
                f"一致性标定CSV {path.name} 第{line_number}行列数错误："
                f"期望{len(headers)}，实际{len(row)}"
            )
        values.append(
            [
                _as_finite_float(
                    row[index],
                    label=f"{path.name}第{line_number}行{_CHANNEL_NAMES[offset]}",
                )
                for offset, index in enumerate(channel_indices)
            ]
        )
    endpoint = np.mean(np.asarray(values, dtype=np.float64), axis=0)
    invalid = np.flatnonzero(endpoint <= 0.0)
    if invalid.size:
        channels = ", ".join(f"channel{int(index) + 1}" for index in invalid)
        raise ValueError(
            f"一致性标定CSV {path.name} 的末尾均值必须大于0，问题通道: {channels}"
        )
    return endpoint, nonempty_count, tail[-1][0], source_sha256


def _read_calibration_directory(
    csv_directory: str | Path,
    *,
    csv_pattern: str,
    tail_rows: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """读取全部量程CSV并返回按量程升序排列的原始端点。"""
    directory = Path(csv_directory)
    if not directory.exists():
        raise FileNotFoundError(f"一致性标定CSV目录不存在: {directory}")
    if not directory.is_dir():
        raise ValueError(f"一致性标定CSV路径不是目录: {directory}")
    paths = [path for path in directory.glob(csv_pattern) if path.is_file()]
    if not paths:
        raise ValueError(
            f"一致性标定目录没有匹配 {csv_pattern!r} 的CSV: {directory}"
        )
    valued_paths = [(_segment_value_from_path(path), path) for path in paths]
    valued_paths.sort(key=lambda item: (item[0], item[1].name))
    segment_values = np.asarray([item[0] for item in valued_paths], dtype=np.float64)
    if np.any(np.diff(segment_values) <= 0.0):
        raise ValueError("一致性标定CSV文件名中的量程值必须唯一且严格递增")

    endpoints: list[np.ndarray] = []
    row_counts: list[int] = []
    last_line_numbers: list[int] = []
    source_hashes: list[str] = []
    combined_hash = hashlib.sha256()
    for _segment_value, path in valued_paths:
        endpoint, row_count, last_line, source_hash = _read_segment_endpoint(
            path,
            tail_rows=tail_rows,
        )
        endpoints.append(endpoint)
        row_counts.append(row_count)
        last_line_numbers.append(last_line)
        source_hashes.append(source_hash)
        combined_hash.update(path.name.encode("utf-8"))
        combined_hash.update(b"\0")
        combined_hash.update(path.read_bytes())
        combined_hash.update(b"\0")

    metadata = {
        "calibration_method": "piecewise_tail_mean_isotonic_v2",
        "source_directory": str(directory.resolve()),
        "source_pattern": csv_pattern,
        "source_file_count": len(valued_paths),
        "source_file_names": np.asarray(
            [path.name for _value, path in valued_paths], dtype=np.str_
        ),
        "source_file_sha256": np.asarray(source_hashes, dtype=np.str_),
        "source_combined_sha256": combined_hash.hexdigest(),
        "source_row_counts": np.asarray(row_counts, dtype=np.int64),
        "source_last_line_numbers": np.asarray(last_line_numbers, dtype=np.int64),
        "tail_rows": int(tail_rows),
    }
    return np.stack(endpoints), segment_values, metadata


def _fit_channel_breakpoints(
    raw_endpoints: np.ndarray,
    reference_endpoints: np.ndarray,
    *,
    minimum_step: float,
    max_segment_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按原始ADC升序拟合每通道单调且有增益上限的分段断点。"""
    from scipy.optimize import isotonic_regression

    segment_count = raw_endpoints.shape[0]
    input_breakpoints = np.zeros(
        (segment_count + 1, CHANNEL_COUNT), dtype=np.float64
    )
    target_breakpoints = np.zeros_like(input_breakpoints)
    for channel in range(CHANNEL_COUNT):
        order = np.argsort(raw_endpoints[:, channel], kind="stable")
        sorted_inputs = raw_endpoints[order, channel]
        fitted_targets = np.asarray(
            isotonic_regression(
                reference_endpoints[order], increasing=True
            ).x,
            dtype=np.float64,
        )
        previous_input = 0.0
        previous_target = 0.0
        for segment, (raw_input, fitted_target) in enumerate(
            zip(sorted_inputs, fitted_targets), start=1
        ):
            current_input = max(
                float(raw_input), previous_input + minimum_step
            )
            maximum_target = previous_target + max_segment_scale * (
                current_input - previous_input
            )
            current_target = min(float(fitted_target), maximum_target)
            current_target = max(current_target, previous_target)
            input_breakpoints[segment, channel] = current_input
            target_breakpoints[segment, channel] = current_target
            previous_input = current_input
            previous_target = current_target
    return input_breakpoints, target_breakpoints


class ConsistenceCalibrator:
    """保存并应用84通道多量程分段一致性标定系数。"""

    def __init__(
        self,
        input_breakpoints: Any,
        target_breakpoints: Any,
        segment_scale: Any,
        segment_offset: Any,
        *,
        segment_values: Any | None = None,
        clip_min: float | None = 0.0,
        clip_max: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        """校验并保存分段断点、系数、裁剪范围和审计元数据。"""
        inputs = np.asarray(input_breakpoints, dtype=np.float64)
        targets = np.asarray(target_breakpoints, dtype=np.float64)
        scales = np.asarray(segment_scale, dtype=np.float64)
        offsets = np.asarray(segment_offset, dtype=np.float64)
        if inputs.ndim != 2 or inputs.shape[1] != CHANNEL_COUNT or inputs.shape[0] < 2:
            raise ValueError(
                "input_breakpoints 必须是形状(segment_count+1, 84)的二维数组"
            )
        segment_count = inputs.shape[0] - 1
        if targets.shape != inputs.shape:
            raise ValueError("target_breakpoints 必须与input_breakpoints形状相同")
        expected_coefficients = (segment_count, CHANNEL_COUNT)
        if scales.shape != expected_coefficients or offsets.shape != expected_coefficients:
            raise ValueError(
                f"segment_scale/segment_offset 必须是形状{expected_coefficients}"
            )
        if not all(np.all(np.isfinite(array)) for array in (inputs, targets, scales, offsets)):
            raise ValueError("一致性标定断点和系数不能包含NaN或无穷值")
        if np.any(np.diff(inputs, axis=0) <= 0.0):
            raise ValueError("每个通道的input_breakpoints必须严格递增")
        if np.any(np.diff(targets, axis=0) < 0.0):
            raise ValueError("每个通道的target_breakpoints必须非递减")
        if np.any(scales < 0.0):
            raise ValueError("segment_scale不能为负数")
        if np.any(scales > 100.0):
            raise ValueError("segment_scale不能大于100")
        if not np.allclose(inputs[0], 0.0) or not np.allclose(targets[0], 0.0):
            raise ValueError("分段断点必须包含每个通道的(0, 0)锚点")
        if segment_values is None:
            values = np.arange(1, segment_count + 1, dtype=np.float64)
        else:
            values = np.asarray(segment_values, dtype=np.float64)
        if values.shape != (segment_count,) or not np.all(np.isfinite(values)):
            raise ValueError("segment_values必须是长度等于分段数的有限一维数组")
        if np.any(np.diff(values) <= 0.0):
            raise ValueError("segment_values必须严格递增")
        if clip_min is not None and not np.isfinite(clip_min):
            raise ValueError("clip_min必须是有限数字或None")
        if clip_max is not None and not np.isfinite(clip_max):
            raise ValueError("clip_max必须是有限数字或None")
        if clip_min is not None and clip_max is not None and clip_max < clip_min:
            raise ValueError("clip_max不能小于clip_min")
        self.input_breakpoints = inputs.copy()
        self.target_breakpoints = targets.copy()
        self.segment_scale = scales.copy()
        self.segment_offset = offsets.copy()
        self.segment_values = values.copy()
        self.clip_min = None if clip_min is None else float(clip_min)
        self.clip_max = None if clip_max is None else float(clip_max)
        self.metadata = dict(metadata or {})
        self.output_path = None if output_path is None else Path(output_path)

    @classmethod
    def fit_from_directory(
        cls,
        config: ConsistenceCalibrationConfig,
    ) -> "ConsistenceCalibrator":
        """从多个量程CSV的末尾均值拟合分段一致性标定器。"""
        config.validate()
        raw_endpoints, segment_values, metadata = _read_calibration_directory(
            config.csv_directory,
            csv_pattern=config.csv_pattern,
            tail_rows=config.tail_rows,
        )
        reference_endpoints = np.mean(raw_endpoints, axis=1)
        if not np.all(np.isfinite(reference_endpoints)) or np.any(reference_endpoints <= 0.0):
            raise ValueError("各量程的84通道目标均值必须是有限正数")
        if np.any(np.diff(reference_endpoints) <= 0.0):
            raise ValueError("按文件名排序后，各量程的84通道目标均值必须严格递增")
        input_breakpoints, target_breakpoints = _fit_channel_breakpoints(
            raw_endpoints,
            reference_endpoints,
            minimum_step=config.minimum_breakpoint_step,
            max_segment_scale=config.max_segment_scale,
        )
        input_delta = np.diff(input_breakpoints, axis=0)
        target_delta = np.diff(target_breakpoints, axis=0)
        segment_scale = target_delta / input_delta
        segment_offset = (
            target_breakpoints[:-1]
            - input_breakpoints[:-1] * segment_scale
        )
        fitted_at_source = np.empty_like(raw_endpoints)
        for segment in range(raw_endpoints.shape[0]):
            for channel in range(CHANNEL_COUNT):
                fitted_at_source[segment, channel] = np.interp(
                    raw_endpoints[segment, channel],
                    input_breakpoints[:, channel],
                    target_breakpoints[:, channel],
                )
        residual = fitted_at_source - reference_endpoints[:, None]
        metadata.update(
            {
                "minimum_breakpoint_step": config.minimum_breakpoint_step,
                "max_segment_scale": config.max_segment_scale,
                "raw_segment_endpoints": raw_endpoints,
                "reference_endpoints": reference_endpoints,
                "fitted_at_source_endpoints": fitted_at_source,
                "fit_residual_rms": np.sqrt(np.mean(residual * residual, axis=0)),
                "fit_residual_max_abs": np.max(np.abs(residual), axis=0),
            }
        )
        return cls(
            input_breakpoints,
            target_breakpoints,
            segment_scale,
            segment_offset,
            segment_values=segment_values,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
            metadata=metadata,
            output_path=config.output_path,
        )

    @classmethod
    def fit(cls, config: ConsistenceCalibrationConfig) -> "ConsistenceCalibrator":
        """``fit_from_directory``的简洁别名。"""
        return cls.fit_from_directory(config)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        clip_min: float | None = 0.0,
        clip_max: float | None = None,
    ) -> "ConsistenceCalibrator":
        """从安全的分段v2 NPZ加载运行时一致性系数。"""
        coefficient_path = Path(path)
        if not coefficient_path.is_file():
            raise FileNotFoundError(f"一致性标定系数不存在: {coefficient_path}")
        required = {
            "format_version", "input_breakpoints", "target_breakpoints",
            "segment_scale", "segment_offset", "segment_values",
        }
        try:
            with np.load(coefficient_path, allow_pickle=False) as archive:
                version = (
                    int(archive["format_version"].item())
                    if "format_version" in archive else None
                )
                if version != FORMAT_VERSION:
                    raise ValueError(
                        "旧单段一致性系数不再兼容，请重新运行多量程一致性标定"
                    )
                missing = sorted(required.difference(archive.files))
                if missing:
                    raise ValueError("分段系数文件缺少字段: " + ", ".join(missing))
                core = {
                    name: np.asarray(archive[name], dtype=np.float64)
                    for name in (
                        "input_breakpoints", "target_breakpoints",
                        "segment_scale", "segment_offset", "segment_values",
                    )
                }
                metadata: dict[str, Any] = {}
                for key in archive.files:
                    if key in required:
                        continue
                    value = archive[key]
                    if value.ndim == 0:
                        value = value.item()
                    metadata[key] = value
        except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(f"无法加载一致性标定系数 {coefficient_path}: {exc}") from exc
        return cls(
            core["input_breakpoints"], core["target_breakpoints"],
            core["segment_scale"], core["segment_offset"],
            segment_values=core["segment_values"],
            clip_min=clip_min, clip_max=clip_max,
            metadata=metadata, output_path=coefficient_path,
        )

    @classmethod
    def from_default(
        cls,
        *,
        clip_min: float | None = 0.0,
        clip_max: float | None = None,
    ) -> "ConsistenceCalibrator":
        """加载wheel内置的分段v2一致性标定资源。"""
        resource = resources.files("tangential.resources").joinpath(DEFAULT_RESOURCE_NAME)
        if not resource.is_file():
            raise FileNotFoundError(
                f"wheel中缺少内置一致性标定资源: tangential.resources/{DEFAULT_RESOURCE_NAME}"
            )
        with resources.as_file(resource) as resource_path:
            calibrator = cls.from_path(
                resource_path, clip_min=clip_min, clip_max=clip_max
            )
        calibrator.output_path = Path(f"tangential.resources/{DEFAULT_RESOURCE_NAME}")
        return calibrator

    @classmethod
    def from_config(
        cls, config: ConsistenceCalibrationConfig
    ) -> "ConsistenceCalibrator":
        """按运行时配置加载外部或内置分段系数。"""
        config.validate()
        if not config.enabled:
            raise ValueError(
                "ConsistenceCalibrationConfig.enabled=False时不应加载标定器"
            )
        if config.coefficients_path is None:
            return cls.from_default(
                clip_min=config.clip_min, clip_max=config.clip_max
            )
        return cls.from_path(
            config.coefficients_path,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    def apply(self, raw_data: Any) -> np.ndarray:
        """按每通道输入断点选择量程段并应用对应线性系数。"""
        values = np.asarray(raw_data, dtype=np.float64).reshape(-1)
        if values.shape != (CHANNEL_COUNT,):
            raise ValueError(
                f"一致性标定输入必须是形状({CHANNEL_COUNT},)，实际为{values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("一致性标定输入不能包含NaN或无穷值")
        segment_count = self.segment_scale.shape[0]
        segment_indices = np.empty(CHANNEL_COUNT, dtype=np.int64)
        for channel in range(CHANNEL_COUNT):
            index = int(
                np.searchsorted(
                    self.input_breakpoints[:, channel],
                    values[channel],
                    side="right",
                ) - 1
            )
            segment_indices[channel] = min(max(index, 0), segment_count - 1)
        channels = np.arange(CHANNEL_COUNT)
        corrected = (
            values * self.segment_scale[segment_indices, channels]
            + self.segment_offset[segment_indices, channels]
        )
        if self.clip_min is not None or self.clip_max is not None:
            corrected = np.clip(corrected, self.clip_min, self.clip_max)
        return corrected

    __call__ = apply

    def save(
        self,
        path: str | Path | None = None,
        *,
        force: bool = False,
    ) -> Path:
        """以不含pickle的NPZ v2格式保存分段系数和审计元数据。"""
        output = Path(path) if path is not None else self.output_path
        if output is None:
            raise ValueError("必须提供一致性标定系数输出路径")
        if output.exists() and not force:
            raise FileExistsError(f"一致性标定系数已存在，使用force覆盖: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Any] = {
            "format_version": np.array(FORMAT_VERSION, dtype=np.int64),
            "input_breakpoints": self.input_breakpoints,
            "target_breakpoints": self.target_breakpoints,
            "segment_scale": self.segment_scale,
            "segment_offset": self.segment_offset,
            "segment_values": self.segment_values,
            "clip_min": np.array(
                np.nan if self.clip_min is None else self.clip_min,
                dtype=np.float64,
            ),
            "clip_max": np.array(
                np.nan if self.clip_max is None else self.clip_max,
                dtype=np.float64,
            ),
        }
        for key, value in self.metadata.items():
            if key in arrays or value is None:
                continue
            if isinstance(value, (str, Path)):
                arrays[key] = np.array(str(value))
            else:
                array = np.asarray(value)
                if array.dtype.hasobject:
                    raise ValueError(
                        f"一致性标定元数据{key!r}不能保存为object/pickle数组"
                    )
                arrays[key] = array
        with output.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        self.output_path = output
        return output


def fit_consistence(config: ConsistenceCalibrationConfig) -> ConsistenceCalibrator:
    """离线拟合并按统一配置的覆盖策略保存分段一致性系数。"""
    calibrator = ConsistenceCalibrator.fit_from_directory(config)
    calibrator.save(config.output_path, force=config.force)
    return calibrator


def main() -> int:
    """按 ``config.py`` 的维护者默认配置生成或更新分段一致性系数。"""
    config = ConsistenceCalibrationConfig()
    print(f"一致性标定源目录: {config.csv_directory}")
    calibrator = fit_consistence(config)
    print(
        "一致性标定量程: "
        + ", ".join(f"{value:g}G" for value in calibrator.segment_values)
    )
    print(
        f"一致性标定输出NPZ: {config.output_path} "
        f"({calibrator.segment_scale.shape[0]}段, {CHANNEL_COUNT}通道)"
    )
    return 0


__all__ = [
    "CHANNEL_COUNT", "FORMAT_VERSION", "DEFAULT_RESOURCE_NAME",
    "ConsistenceCalibrationConfig", "ConsistenceCalibrator",
    "fit_consistence", "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
