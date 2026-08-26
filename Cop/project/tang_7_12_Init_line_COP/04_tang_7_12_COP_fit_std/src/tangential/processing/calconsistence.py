"""84 通道压阻阵列离线一致性标定与运行时系数加载。

离线阶段从调用方指定的 CSV 读取 ``CoP_state`` 和 ``ch1`` 到 ``ch84``，
根据卸载/加载两组样本生成每通道两点仿射系数。运行时只加载安全的
``.npz`` 系数文件并应用到一帧原始 ADC，不访问离线 CSV。
"""

from __future__ import annotations

import csv
import hashlib
import zipfile
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import ConsistenceCalibrationConfig


CHANNEL_COUNT = 84
DEFAULT_RESOURCE_NAME = "consistence_coeffs.npz"
_CHANNEL_NAMES = tuple(f"ch{index}" for index in range(1, CHANNEL_COUNT + 1))


def _as_finite_float(value: Any, *, label: str) -> float:
    """把 CSV 单元格转换为有限浮点数。"""
    try:
        converted = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数字，实际为 {value!r}") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{label} 必须是有限数字")
    return converted


def _read_calibration_csv(
    csv_path: str | Path,
    *,
    state_column: str,
    baseline_state: int,
    loaded_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """严格读取两种状态下的 84 通道数据和源文件摘要。"""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"一致性标定 CSV 不存在: {path}")

    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    baseline_rows: list[list[float]] = []
    loaded_rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        try:
            raw_header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"一致性标定 CSV 为空: {path}") from exc
        headers = [str(item).strip() for item in raw_header]
        if any(not item for item in headers):
            raise ValueError("一致性标定 CSV 表头不能包含空列名")
        if len(set(headers)) != len(headers):
            raise ValueError("一致性标定 CSV 表头包含重复列名")
        required = (state_column.strip(), *_CHANNEL_NAMES)
        missing = [name for name in required if name not in headers]
        if missing:
            raise ValueError("一致性标定 CSV 缺少列: " + ", ".join(missing))
        state_index = headers.index(state_column.strip())
        channel_indices = [headers.index(name) for name in _CHANNEL_NAMES]

        row_count = 0
        for line_number, row in enumerate(reader, start=2):
            if not row or not any(str(cell).strip() for cell in row):
                continue
            row_count += 1
            if len(row) != len(headers):
                raise ValueError(
                    f"一致性标定 CSV 第 {line_number} 行列数错误："
                    f"期望 {len(headers)}，实际 {len(row)}"
                )
            state_value = _as_finite_float(
                row[state_index], label=f"第 {line_number} 行 {state_column}"
            )
            if not state_value.is_integer():
                raise ValueError(
                    f"第 {line_number} 行 {state_column} 必须是整数状态"
                )
            state = int(state_value)
            if state not in (baseline_state, loaded_state):
                continue
            values = [
                _as_finite_float(
                    row[index], label=f"第 {line_number} 行 {_CHANNEL_NAMES[offset]}"
                )
                for offset, index in enumerate(channel_indices)
            ]
            if state == baseline_state:
                baseline_rows.append(values)
            elif state == loaded_state:
                loaded_rows.append(values)

    if row_count == 0:
        raise ValueError("一致性标定 CSV 没有有效数据行")
    if not baseline_rows:
        raise ValueError(f"一致性标定 CSV 没有 state={baseline_state} 的样本")
    if not loaded_rows:
        raise ValueError(f"一致性标定 CSV 没有 state={loaded_state} 的样本")

    metadata = {
        "source_sha256": source_sha256,
        "baseline_count": len(baseline_rows),
        "loaded_count": len(loaded_rows),
        "row_count": row_count,
    }
    return (
        np.asarray(baseline_rows, dtype=np.float64),
        np.asarray(loaded_rows, dtype=np.float64),
        metadata,
    )


class ConsistenceCalibrator:
    """保存并应用 84 通道一致性标定系数。

    Args:
        scale: 每通道乘法系数，形状必须为 ``(84,)``。
        offset: 每通道偏移系数，形状必须为 ``(84,)``。
        clip_min: 应用结果的可选下限，默认 0。
        clip_max: 应用结果的可选上限，默认不裁剪。
        metadata: 离线拟合元数据；只用于诊断和保存，不参与计算。

    Raises:
        ValueError: 系数维度错误、包含非有限值或裁剪范围非法。
    """

    def __init__(
        self,
        scale: Any,
        offset: Any,
        *,
        clip_min: float | None = 0.0,
        clip_max: float | None = None,
        metadata: Mapping[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        self.scale = self._validate_coefficients(scale, "scale")
        self.offset = self._validate_coefficients(offset, "offset")
        if clip_min is not None and not np.isfinite(clip_min):
            raise ValueError("clip_min 必须是有限数字或 None")
        if clip_max is not None and not np.isfinite(clip_max):
            raise ValueError("clip_max 必须是有限数字或 None")
        if clip_min is not None and clip_max is not None and clip_max < clip_min:
            raise ValueError("clip_max 不能小于 clip_min")
        self.clip_min = None if clip_min is None else float(clip_min)
        self.clip_max = None if clip_max is None else float(clip_max)
        self.metadata = dict(metadata or {})
        self.output_path = None if output_path is None else Path(output_path)

    @staticmethod
    def _validate_coefficients(values: Any, name: str) -> np.ndarray:
        """校验并复制一组 84 通道系数。"""
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (CHANNEL_COUNT,):
            raise ValueError(f"{name} 必须是形状 ({CHANNEL_COUNT},)，实际为 {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} 不能包含 NaN 或无穷值")
        return array.copy()

    @classmethod
    def fit_from_csv(
        cls,
        config: ConsistenceCalibrationConfig,
    ) -> "ConsistenceCalibrator":
        """根据配置指定的 CSV 计算两点仿射一致性标定器。

        Args:
            config: 包含 CSV 路径、状态列、目标范围和裁剪设置的离线配置。

        Returns:
            ConsistenceCalibrator: 已拟合但尚未写出文件的标定器。

        Raises:
            FileNotFoundError: 输入 CSV 不存在。
            ValueError: CSV 列、状态、数值或通道跨度不符合要求。
        """
        config.validate()
        baseline, loaded, metadata = _read_calibration_csv(
            config.csv_path,
            state_column=config.state_column,
            baseline_state=config.baseline_state,
            loaded_state=config.loaded_state,
        )
        baseline_median = np.median(baseline, axis=0)
        loaded_median = np.median(loaded, axis=0)
        spans = loaded_median - baseline_median
        if not np.all(np.isfinite(spans)):
            raise ValueError("一致性标定的通道跨度必须是有限数字")
        invalid = np.flatnonzero(spans <= 0)
        if invalid.size:
            channels = ", ".join(f"ch{int(index) + 1}" for index in invalid)
            raise ValueError(
                "加载状态中位数必须严格大于卸载状态中位数，问题通道: " + channels
            )
        target_span = config.target_max - config.target_min
        scale = target_span / spans
        offset = config.target_min - baseline_median * scale
        metadata.update(
            {
                "state_column": config.state_column.strip(),
                "baseline_state": config.baseline_state,
                "loaded_state": config.loaded_state,
                "target_min": config.target_min,
                "target_max": config.target_max,
                "states": np.asarray(
                    [config.baseline_state, config.loaded_state], dtype=np.int64
                ),
                "targets": np.asarray(
                    [config.target_min, config.target_max], dtype=np.float64
                ),
                "sample_counts": np.asarray(
                    [metadata["baseline_count"], metadata["loaded_count"]],
                    dtype=np.int64,
                ),
                "baseline_median": baseline_median,
                "loaded_median": loaded_median,
            }
        )
        return cls(
            scale,
            offset,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
            metadata=metadata,
            output_path=config.output_path,
        )

    @classmethod
    def fit(cls, config: ConsistenceCalibrationConfig) -> "ConsistenceCalibrator":
        """``fit_from_csv`` 的简洁别名。"""
        return cls.fit_from_csv(config)

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        clip_min: float | None = 0.0,
        clip_max: float | None = None,
    ) -> "ConsistenceCalibrator":
        """从安全 NPZ 文件加载运行时一致性系数。

        ``allow_pickle=False`` 严格禁止从系数文件反序列化任意 Python 对象。
        """
        coefficient_path = Path(path)
        if not coefficient_path.is_file():
            raise FileNotFoundError(f"一致性标定系数不存在: {coefficient_path}")
        try:
            with np.load(coefficient_path, allow_pickle=False) as archive:
                if "scale" not in archive or "offset" not in archive:
                    raise ValueError("系数文件必须包含 scale 和 offset")
                scale = np.asarray(archive["scale"], dtype=np.float64)
                offset = np.asarray(archive["offset"], dtype=np.float64)
                metadata: dict[str, Any] = {}
                for key in archive.files:
                    if key in {"scale", "offset"}:
                        continue
                    value = archive[key]
                    if value.ndim == 0:
                        value = value.item()
                    metadata[key] = value
        except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
            raise ValueError(f"无法加载一致性标定系数 {coefficient_path}: {exc}") from exc
        return cls(
            scale,
            offset,
            clip_min=clip_min,
            clip_max=clip_max,
            metadata=metadata,
            output_path=coefficient_path,
        )

    @classmethod
    def from_default(
        cls,
        *,
        clip_min: float | None = 0.0,
        clip_max: float | None = None,
    ) -> "ConsistenceCalibrator":
        """加载 wheel 内置的 ``resources/consistence_coeffs.npz``。"""
        resource = resources.files("tangential.resources").joinpath(DEFAULT_RESOURCE_NAME)
        if not resource.is_file():
            raise FileNotFoundError(
                f"wheel 中缺少内置一致性标定资源: tangential.resources/{DEFAULT_RESOURCE_NAME}"
            )
        with resources.as_file(resource) as resource_path:
            calibrator = cls.from_path(
                resource_path, clip_min=clip_min, clip_max=clip_max
            )
        calibrator.output_path = Path(
            f"tangential.resources/{DEFAULT_RESOURCE_NAME}"
        )
        return calibrator

    @classmethod
    def from_config(
        cls, config: ConsistenceCalibrationConfig
    ) -> "ConsistenceCalibrator":
        """按运行时配置加载外部或内置系数。"""
        config.validate()
        if not config.enabled:
            raise ValueError(
                "ConsistenceCalibrationConfig.enabled=False 时不应加载标定器"
            )
        if config.coefficients_path is None:
            return cls.from_default(
                clip_min=config.clip_min,
                clip_max=config.clip_max,
            )
        return cls.from_path(
            config.coefficients_path,
            clip_min=config.clip_min,
            clip_max=config.clip_max,
        )

    def apply(self, raw_data: Any) -> np.ndarray:
        """把一帧原始 ADC 转换为一致性标定后的独立数组。

        Args:
            raw_data: 长度为84的一维原始 ADC 序列。

        Returns:
            numpy.ndarray: ``float64`` 的84通道校正数据；不会修改输入。

        Raises:
            ValueError: 输入不是84通道或包含非有限值。
        """
        values = np.asarray(raw_data, dtype=np.float64).reshape(-1)
        if values.shape != (CHANNEL_COUNT,):
            raise ValueError(
                f"一致性标定输入必须是形状 ({CHANNEL_COUNT},)，实际为 {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("一致性标定输入不能包含 NaN 或无穷值")
        corrected = values * self.scale + self.offset
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
        """以不含 pickle 的 NPZ 格式保存系数和可审计元数据。

        Args:
            path: 输出路径；省略时使用拟合配置中的 ``output_path``。
            force: 是否允许覆盖已有文件。底层安全默认值为 ``False``；维护者
                无参数入口会显式传递统一配置中的 ``force``。

        Returns:
            pathlib.Path: 实际写出的文件路径。

        Raises:
            FileExistsError: 目标存在且未指定 ``force``。
            OSError: 目录创建或文件写入失败。
        """
        output = Path(path) if path is not None else self.output_path
        if output is None:
            raise ValueError("必须提供一致性标定系数输出路径")
        if output.exists() and not force:
            raise FileExistsError(f"一致性标定系数已存在，使用 force 覆盖: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.metadata
        arrays: dict[str, Any] = {
            "scale": self.scale,
            "offset": self.offset,
            "clip_min": np.array(
                np.nan if self.clip_min is None else self.clip_min, dtype=np.float64
            ),
            "clip_max": np.array(
                np.nan if self.clip_max is None else self.clip_max, dtype=np.float64
            ),
        }
        for key, value in metadata.items():
            if isinstance(value, str):
                arrays[key] = np.array(value)
            elif value is None:
                continue
            else:
                array = np.asarray(value)
                if array.dtype.hasobject:
                    raise ValueError(
                        f"一致性标定元数据 {key!r} 不能保存为 object/pickle 数组"
                    )
                arrays[key] = array
        with output.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        self.output_path = output
        return output


def fit_consistence(config: ConsistenceCalibrationConfig) -> ConsistenceCalibrator:
    """离线拟合并按统一配置的覆盖策略保存一致性系数。

    ``ConsistenceCalibrationConfig.force`` 默认是 ``True``，因此维护者无参数
    命令可以重复生成同一输出；显式构造 ``force=False`` 的配置时，已有目标
    仍由底层 ``save()`` 拒绝覆盖。
    """
    calibrator = ConsistenceCalibrator.fit_from_csv(config)
    calibrator.save(config.output_path, force=config.force)
    return calibrator


def main() -> int:
    """按 ``config.py`` 中的维护者默认配置生成或更新一致性系数。

    Returns:
        int: 标定和保存成功后返回 ``0``。

    Raises:
        FileNotFoundError: 配置的标定 CSV 不存在。
        ValueError: CSV 数据或标定参数不合法。
        FileExistsError: 显式把配置 ``force`` 设为 ``False`` 且输出已存在。

    Side Effects:
        读取配置的 CSV、写入配置的 NPZ，并打印输入与输出路径；默认覆盖同名
        旧文件，不访问硬件。
    """
    config = ConsistenceCalibrationConfig()
    print(f"一致性标定源 CSV: {config.csv_path}")
    calibrator = fit_consistence(config)
    print(
        f"一致性标定输出 NPZ: {config.output_path} "
        f"({len(calibrator.scale)} 通道)"
    )
    return 0


__all__ = [
    "CHANNEL_COUNT",
    "DEFAULT_RESOURCE_NAME",
    "ConsistenceCalibrationConfig",
    "ConsistenceCalibrator",
    "fit_consistence",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
