"""Tangential SDK 的集中配置定义。

协议帧头、CRC、动态帧长度和 CSV 列布局等协议不变量留在各自实现中；
本模块只保存用户能够调整的设备、处理、同步、输出和离线工具参数。
环境变量只提供默认值，显式构造的 dataclass 字段始终具有更高优先级。
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


# 一致性标定路径基于源码项目根目录解析，不受当前工作目录影响。
_SOURCE_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    """读取非空环境变量，否则返回字符串默认值。"""
    value = os.environ.get(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    """读取整数环境变量，格式错误时抛出明确配置异常。"""
    try:
        return int(_env(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是整数") from exc


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量，格式错误时抛出明确配置异常。"""
    try:
        return float(_env(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字") from exc


def _env_optional_float(name: str, default: float | None) -> float | None:
    """读取可选浮点环境变量；空字符串和 ``none`` 表示无上限。"""
    value = os.environ.get(name)
    if value is None or not value.strip() or value.strip().lower() in {"none", "null"}:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"环境变量 {name} 必须是数字或 none") from exc


def _coerce_finite(value: Any, label: str, *, allow_none: bool = False) -> float | None:
    """把配置数值规范化为有限浮点数并给出统一错误。"""
    if value is None and allow_none:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} 必须是有限数字") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{label} 必须是有限数字")
    return converted


def _validate_array_dimensions(rows: Any, cols: Any, *, label: str = "阵列") -> tuple[int, int]:
    """严格校验压力阵列尺寸并返回 ``(rows, cols)``。

    ``rows`` 和 ``cols`` 同时决定协议中的传感器字节数、二维处理矩阵、CSV
    通道列和 GUI 网格，因此所有需要尺寸的内部模块都应复用这个校验。布尔值
    虽然是 Python ``int`` 的子类，但不是合法的阵列尺寸；浮点值也不做隐式
    截断。协议响应的 payload 长度是 ``2 * rows * cols + 10``，长度字段为
    16 位小端整数。
    """
    if isinstance(rows, bool) or not isinstance(rows, Integral):
        raise ValueError(f"{label} rows 必须是正整数")
    if isinstance(cols, bool) or not isinstance(cols, Integral):
        raise ValueError(f"{label} cols 必须是正整数")
    rows = int(rows)
    cols = int(cols)
    if rows <= 0 or cols <= 0:
        raise ValueError(f"{label} rows 和 cols 必须是正整数")
    if 2 * rows * cols + 10 > 0xFFFF:
        raise ValueError(
            f"{label}协议长度溢出：2*rows*cols+10 必须不超过 65535"
        )
    return rows, cols


@dataclass
class ArrayConfig:
    """整个项目共用的压力阵列布局配置。

    ``rows`` 和 ``cols`` 同时决定压力协议请求长度、解码通道数、二维算法
    矩阵、CSV 通道列和实时 GUI 网格。所有运行时组件都应接收同一个
    ``ArrayConfig`` 实例；组件内部的 ``rows``/``cols`` 只允许作为该对象的
    派生只读语义使用，不能再创建第二套默认尺寸。

    Attributes:
        rows: 阵列行数，默认 12；可由 ``TANGENTIAL_ARRAY_ROWS`` 提供默认值。
        cols: 阵列列数，默认 7；可由 ``TANGENTIAL_ARRAY_COLS`` 提供默认值。
    """

    DEFAULT_SHAPE: ClassVar[tuple[int, int]] = (12, 7)

    rows: int = field(
        default_factory=lambda: _env_int("TANGENTIAL_ARRAY_ROWS", 12)
    )
    cols: int = field(
        default_factory=lambda: _env_int("TANGENTIAL_ARRAY_COLS", 7)
    )

    def __post_init__(self) -> None:
        """创建后立即执行严格尺寸校验。"""
        self.validate()

    @property
    def shape(self) -> tuple[int, int]:
        """返回 ``(rows, cols)`` 二维数组形状。"""
        return self.rows, self.cols

    @property
    def channel_count(self) -> int:
        """返回压力通道总数 ``rows * cols``。"""
        return self.rows * self.cols

    @property
    def sensor_bytes(self) -> int:
        """返回一帧 ADC 数据占用的字节数 ``channel_count * 2``。"""
        return self.channel_count * 2

    def validate(self) -> "ArrayConfig":
        """严格校验并规范化阵列尺寸，返回当前对象。

        Raises:
            ValueError: 行列不是严格正整数，或动态压力协议长度超过 16 位。
        """
        self.rows, self.cols = _validate_array_dimensions(
            self.rows, self.cols, label="ArrayConfig"
        )
        return self


def _env_bool(name: str, default: bool) -> bool:
    """读取常见布尔环境变量，未知值时抛出明确配置异常。"""
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"环境变量 {name} 必须是 true/false、yes/no、on/off 或 1/0")


def _env_frequency_bands(
    name: str,
    default: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    """读取逗号分隔的频段环境默认值。

    环境变量格式为 ``low:high,low:high``，同时接受 ``low-high`` 作为
    人工输入的简写。这里只负责解析，不负责 Nyquist 和频段重叠校验；
    这些校验由 :class:`SpectrumConfig` 统一完成。
    """
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    return _parse_frequency_bands(value, name)


def _env_frequency_band(
    name: str,
    default: tuple[float, float],
) -> tuple[float, float]:
    """读取恰好一个 ``low:high`` 目标频段的环境默认值。"""
    bands = _env_frequency_bands(name, (default,))
    if len(bands) != 1:
        raise ValueError(f"环境变量 {name} 必须只包含一个 low:high 频段")
    return bands[0]


def _parse_frequency_bands(value: str, label: str) -> tuple[tuple[float, float], ...]:
    """解析 ``low:high,low:high`` 形式的频段文本。"""
    bands: list[tuple[float, float]] = []
    for item in value.split(","):
        text = item.strip().strip("()[]")
        parts = re.split(r"\s*(?::|-)\s*", text, maxsplit=1)
        if len(parts) != 2:
            raise ValueError(
                f"{label} 必须使用 low:high,low:high 格式"
            )
        try:
            bands.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise ValueError(
                f"{label} 的频段必须是数字: {item!r}"
            ) from exc
    return tuple(bands)


def default_model_path() -> str | None:
    """返回外部模型覆盖路径；未设置时返回 ``None``。"""
    return os.environ.get("TANGENTIAL_MODEL_PATH") or None


def default_save_dir() -> str:
    """返回 CSV 默认保存目录，不创建目录。"""
    return _env("TANGENTIAL_DATA_DIR", str((Path.cwd() / "data").resolve()))


@dataclass
class PressureConfig:
    """压力阵列设备和轮询参数。"""

    port: str = field(default_factory=lambda: _env("TANGENTIAL_PRESSURE_PORT", "/dev/ttyUSB0"))
    baudrate: int = field(default_factory=lambda: _env_int("TANGENTIAL_PRESSURE_BAUDRATE", 921600))
    target_hz: float = field(default_factory=lambda: _env_float("TANGENTIAL_PRESSURE_HZ", 200.0))
    response_timeout_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_PRESSURE_TIMEOUT_S", 0.050))
    frame_queue_size: int = field(default_factory=lambda: _env_int("TANGENTIAL_PRESSURE_QUEUE_SIZE", 256))
    startup_timeout_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_PRESSURE_STARTUP_TIMEOUT_S", 2.0))

    @property
    def period_s(self) -> float:
        """返回由目标频率计算出的轮询周期。"""
        return 1.0 / self.target_hz

    def validate(self) -> "PressureConfig":
        """校验压力配置并返回自身。"""
        if not self.port:
            raise ValueError("PressureConfig.port 不能为空")
        if self.baudrate <= 0 or self.target_hz <= 0 or self.response_timeout_s <= 0:
            raise ValueError("压力波特率、频率和响应超时必须大于 0")
        if self.frame_queue_size <= 0 or self.startup_timeout_s <= 0:
            raise ValueError("压力队列容量和启动超时必须大于 0")
        return self


@dataclass
class ForceConfig:
    """六维力设备、轮询和软件校零参数。"""

    enabled: bool = field(default_factory=lambda: _env_bool("TANGENTIAL_FORCE_ENABLED", True))
    port: str = field(default_factory=lambda: _env("TANGENTIAL_FORCE_PORT", "/dev/ttyUSB1"))
    baudrate: int = field(default_factory=lambda: _env_int("TANGENTIAL_FORCE_BAUDRATE", 460800))
    target_hz: float = field(default_factory=lambda: _env_float("TANGENTIAL_FORCE_HZ", 200.0))
    response_timeout_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_FORCE_TIMEOUT_S", 0.050))
    frame_queue_size: int = field(default_factory=lambda: _env_int("TANGENTIAL_FORCE_QUEUE_SIZE", 256))
    startup_timeout_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_FORCE_STARTUP_TIMEOUT_S", 2.0))
    zero_sample_count: int = field(default_factory=lambda: _env_int("TANGENTIAL_FORCE_ZERO_SAMPLES", 10))
    zero_timeout_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_FORCE_ZERO_TIMEOUT_S", 1.0))
    rezero_timeout_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_FORCE_REZERO_TIMEOUT_S", 1.0))

    @property
    def period_s(self) -> float:
        """返回由目标频率计算出的轮询周期。"""
        return 1.0 / self.target_hz

    def validate(self) -> "ForceConfig":
        """校验六维力配置并返回自身。"""
        if self.enabled and not self.port:
            raise ValueError("ForceConfig.port 不能为空")
        if self.baudrate <= 0 or self.target_hz <= 0 or self.response_timeout_s <= 0:
            raise ValueError("力传感器波特率、频率和响应超时必须大于 0")
        if self.frame_queue_size <= 0 or self.startup_timeout_s <= 0:
            raise ValueError("力传感器队列容量和启动超时必须大于 0")
        if self.zero_sample_count <= 0 or self.zero_timeout_s <= 0 or self.rezero_timeout_s <= 0:
            raise ValueError("力传感器校零样本数和超时必须大于 0")
        return self


@dataclass
class CopConfig:
    """CoP、动态阈值、区域和二次精修参数。"""

    total_threshold_factor: float = 3.0
    pixel_threshold_factor: float = 5.0
    collect_frames: int = 10
    stability_frames: int = 5
    reset_at_frame: int = 0
    refine_cnt: int = 10
    refine_distance: float = 0.1
    merge_ratio: float = 0.6
    region_match_dist: float = 5.0
    region_min_area: int = 4
    region_peak_ratio: float = 1.0
    region_peak_dist: int = 3

    def validate(self) -> "CopConfig":
        """校验 CoP 阈值、稳定、区域和精修参数。"""
        if self.collect_frames <= 0:
            raise ValueError("CoP 背景帧数必须大于 0")
        if self.stability_frames <= 0 or self.region_min_area <= 0:
            raise ValueError("CoP 稳定帧数和区域最小面积必须大于 0")
        if self.refine_cnt < 0 or self.reset_at_frame < 0:
            raise ValueError("CoP 精修/重置帧数不能为负数")
        return self

    def as_kwargs(self) -> dict[str, Any]:
        """返回可直接传给 ``PRSensorAngle`` 的参数字典。"""
        return {
            "total_threshold_factor": self.total_threshold_factor,
            "pixel_threshold_factor": self.pixel_threshold_factor,
            "collect_frames": self.collect_frames,
            "stability_frames": self.stability_frames,
            "reset_at_frame": self.reset_at_frame,
            "refine_cnt": self.refine_cnt,
            "refine_distance": self.refine_distance,
            "merge_ratio": self.merge_ratio,
            "region_match_dist": self.region_match_dist,
            "region_min_area": self.region_min_area,
            "region_peak_ratio": self.region_peak_ratio,
            "region_peak_dist": self.region_peak_dist,
        }


@dataclass
class SlipConfig:
    """全局滑移检测参数。

    参数单位中的 distance/search radius 均为压力阵列 cell。环境变量使用
    ``TANGENTIAL_SLIP_*`` 前缀，例如 ``TANGENTIAL_SLIP_ENTER_DISTANCE``。

    Attributes:
        enabled: 是否启用滑移判定；关闭后接触帧保持 ``STICK``。
        window_frames: CoP 和压力斑块运动比较使用的短窗帧数。
        enter_distance: 进入滑移所需的最小短窗 CoP 位移，单位为 cell。
        exit_distance: 判定运动稳定所需的最大短窗 CoP 位移，单位为 cell。
        reanchor_distance: 当斑块相关性不足时，相对静摩擦锚点触发滑移的
            兜底距离，单位为 cell。
        enter_frames: 连续满足滑移证据后进入 ``SLIP`` 的窗口数。
        exit_frames: 连续满足稳定条件后退出 ``SLIP`` 的窗口数。
        direction_smoothing: 滑移方向指数移动平均系数，范围为 ``(0, 1]``。
        patch_search_radius: 压力斑块平移搜索半径，单位为 cell。
        patch_min_correlation: 确认斑块平移所需的最小余弦相关值。
        patch_min_improvement: 最优非零平移相对零平移的最小相关提升。
        angle_deadband: 小于该向量模长时把输出方向角置零，单位为 cell。
    """

    enabled: bool = field(default_factory=lambda: _env_bool("TANGENTIAL_SLIP_ENABLED", True))
    window_frames: int = field(default_factory=lambda: _env_int("TANGENTIAL_SLIP_WINDOW_FRAMES", 5))
    enter_distance: float = field(default_factory=lambda: _env_float("TANGENTIAL_SLIP_ENTER_DISTANCE", 0.20))
    exit_distance: float = field(default_factory=lambda: _env_float("TANGENTIAL_SLIP_EXIT_DISTANCE", 0.05))
    reanchor_distance: float = field(default_factory=lambda: _env_float("TANGENTIAL_SLIP_REANCHOR_DISTANCE", 3.0))
    enter_frames: int = field(default_factory=lambda: _env_int("TANGENTIAL_SLIP_ENTER_FRAMES", 3))
    exit_frames: int = field(default_factory=lambda: _env_int("TANGENTIAL_SLIP_EXIT_FRAMES", 8))
    direction_smoothing: float = field(default_factory=lambda: _env_float("TANGENTIAL_SLIP_DIRECTION_SMOOTHING", 0.6))
    patch_search_radius: int = field(default_factory=lambda: _env_int("TANGENTIAL_SLIP_PATCH_SEARCH_RADIUS", 2))
    patch_min_correlation: float = field(default_factory=lambda: _env_float("TANGENTIAL_SLIP_PATCH_MIN_CORRELATION", 0.75))
    patch_min_improvement: float = field(default_factory=lambda: _env_float("TANGENTIAL_SLIP_PATCH_MIN_IMPROVEMENT", 0.03))
    angle_deadband: float = field(default_factory=lambda: _env_float("TANGENTIAL_SLIP_ANGLE_DEADBAND", 0.3))

    def validate(self) -> "SlipConfig":
        """严格校验滑移窗口、阈值、相关性和角度死区参数。"""
        if self.window_frames < 2:
            raise ValueError("SlipConfig.window_frames 必须至少为 2")
        if self.enter_frames <= 0 or self.exit_frames <= 0:
            raise ValueError("SlipConfig.enter_frames/exit_frames 必须大于 0")
        if self.enter_distance < 0 or self.exit_distance < 0:
            raise ValueError("SlipConfig.enter_distance/exit_distance 不能为负数")
        if self.reanchor_distance < self.enter_distance:
            raise ValueError("SlipConfig.reanchor_distance 不能小于 enter_distance")
        if not 0.0 < self.direction_smoothing <= 1.0:
            raise ValueError("SlipConfig.direction_smoothing 必须在 (0, 1] 内")
        if self.patch_search_radius < 0:
            raise ValueError("SlipConfig.patch_search_radius 不能为负数")
        if not 0.0 <= self.patch_min_correlation <= 1.0:
            raise ValueError("SlipConfig.patch_min_correlation 必须在 [0, 1] 内")
        if self.patch_min_improvement < 0:
            raise ValueError("SlipConfig.patch_min_improvement 不能为负数")
        if self.angle_deadband < 0:
            raise ValueError("SlipConfig.angle_deadband 不能为负数")
        return self


@dataclass
class ConsistenceCalibrationConfig:
    """维护者使用的运行时与离线一致性标定统一配置。

    所有可编辑默认值都集中在下面的字段定义中。``csv_path`` 和
    ``output_path`` 基于源码项目根目录；``coefficients_path=None`` 表示
    运行时加载 package resource。环境变量只覆盖运行期开关、外部系数路径
    和裁剪范围。``force=True`` 使维护者无参数离线命令可重复生成并覆盖旧
    NPZ；调用方显式传入 ``force=False`` 时仍拒绝覆盖。
    """

    enabled: bool = field(
        default_factory=lambda: _env_bool("TANGENTIAL_CONSISTENCE_ENABLED", False)
    )
    csv_path: str | os.PathLike[str] = field(
        default_factory=lambda: (
            _SOURCE_PROJECT_ROOT / "data" / "COP_test_0825_3.csv"
        ).resolve()
    )
    output_path: str | os.PathLike[str] = field(
        default_factory=lambda: (
            _SOURCE_PROJECT_ROOT
            / "src"
            / "tangential"
            / "resources"
            / "consistence_coeffs.npz"
        ).resolve()
    )
    coefficients_path: str | os.PathLike[str] | None = field(
        default_factory=lambda: (
            os.environ.get("TANGENTIAL_CONSISTENCE_COEFFICIENTS") or None
        )
    )
    state_column: str = "CoP_state"
    baseline_state: int = 0
    loaded_state: int = 2
    target_min: float = 0.0
    target_max: float = 4000.0
    clip_min: float | None = field(
        default_factory=lambda: _env_optional_float(
            "TANGENTIAL_CONSISTENCE_CLIP_MIN", 0.0
        )
    )
    clip_max: float | None = field(
        default_factory=lambda: _env_optional_float(
            "TANGENTIAL_CONSISTENCE_CLIP_MAX", None
        )
    )
    force: bool = True

    def validate(self) -> "ConsistenceCalibrationConfig":
        """校验离线标定输入、目标范围和裁剪范围。"""
        if not str(self.csv_path).strip():
            raise ValueError("ConsistenceCalibrationConfig.csv_path 不能为空")
        if not str(self.output_path).strip():
            raise ValueError("ConsistenceCalibrationConfig.output_path 不能为空")
        if self.coefficients_path is not None and not str(
            self.coefficients_path
        ).strip():
            raise ValueError(
                "ConsistenceCalibrationConfig.coefficients_path 不能为空字符串"
            )
        if not self.state_column.strip():
            raise ValueError("ConsistenceCalibrationConfig.state_column 不能为空")
        if self.baseline_state == self.loaded_state:
            raise ValueError("baseline_state 和 loaded_state 不能相同")
        self.target_min = _coerce_finite(
            self.target_min, "ConsistenceCalibrationConfig.target_min"
        )
        self.target_max = _coerce_finite(
            self.target_max, "ConsistenceCalibrationConfig.target_max"
        )
        if self.target_max <= self.target_min:
            raise ValueError("target_max 必须大于 target_min")
        self.clip_min = _coerce_finite(
            self.clip_min, "ConsistenceCalibrationConfig.clip_min", allow_none=True
        )
        self.clip_max = _coerce_finite(
            self.clip_max, "ConsistenceCalibrationConfig.clip_max", allow_none=True
        )
        if (
            self.clip_min is not None
            and self.clip_max is not None
            and self.clip_max < self.clip_min
        ):
            raise ValueError("clip_max 不能小于 clip_min")
        return self


@dataclass
class ProcessingConfig:
    """单帧处理和标定调用参数。"""

    cal_dim: str = "3D"
    region_mode: str = "full"
    median_window: int = 5
    refine_rezero_force: bool = True
    cop: CopConfig = field(default_factory=CopConfig)
    slip: SlipConfig = field(default_factory=SlipConfig)
    consistence: ConsistenceCalibrationConfig = field(
        default_factory=ConsistenceCalibrationConfig
    )

    def validate(self) -> "ProcessingConfig":
        """校验处理模式并递归校验 CoP 配置。"""
        if self.cal_dim not in {"1D", "2D", "3D"}:
            raise ValueError("ProcessingConfig.cal_dim 必须是 1D、2D 或 3D")
        if self.region_mode not in {"full", "region", "both"}:
            raise ValueError("ProcessingConfig.region_mode 必须是 full、region 或 both")
        if self.median_window <= 0:
            raise ValueError("ProcessingConfig.median_window 必须大于 0")
        self.cop.validate()
        self.slip.validate()
        self.consistence.validate()
        return self


@dataclass
class CalibrationConfig:
    """运行时标定模型路径配置。"""

    model_path: str | None = field(default_factory=default_model_path)


@dataclass
class SyncConfig:
    """压力—六维力匹配和主循环时序参数。"""

    target_fps: float = field(default_factory=lambda: _env_float("TANGENTIAL_TARGET_FPS", 100.0))
    plot_fps: float = field(default_factory=lambda: _env_float("TANGENTIAL_PLOT_FPS", 60.0))
    max_time_diff_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_MAX_TIME_DIFF_S", 0.015))
    timing_log_interval_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_TIMING_LOG_INTERVAL_S", 1.0))
    buffer_size: int = field(default_factory=lambda: _env_int("TANGENTIAL_BUFFER_SIZE", 500))

    def validate(self) -> "SyncConfig":
        """校验时序和缓存参数。"""
        if self.target_fps <= 0 or self.plot_fps <= 0 or self.timing_log_interval_s <= 0:
            raise ValueError("同步和绘图频率、统计周期必须大于 0")
        if self.max_time_diff_s < 0 or self.buffer_size <= 0:
            raise ValueError("匹配窗口不能为负数，缓存容量必须大于 0")
        return self


@dataclass
class OutputConfig:
    """CSV 输出参数。"""

    save_dir: str = field(default_factory=default_save_dir)


@dataclass
class GuiConfig:
    """实时 GUI 显示参数。"""

    window_title: str = field(default_factory=lambda: _env("TANGENTIAL_GUI_WINDOW_TITLE", "RealTime"))
    timer_interval_ms: int = field(default_factory=lambda: _env_int("TANGENTIAL_GUI_TIMER_MS", 10))
    history_size: int = field(default_factory=lambda: _env_int("TANGENTIAL_GUI_HISTORY_SIZE", 100))
    error_history_size: int = field(default_factory=lambda: _env_int("TANGENTIAL_GUI_ERROR_HISTORY_SIZE", 100))
    max_region_arrows: int = field(default_factory=lambda: _env_int("TANGENTIAL_GUI_MAX_REGION_ARROWS", 8))
    heat_vmax: float = field(default_factory=lambda: _env_float("TANGENTIAL_GUI_HEAT_VMAX", 500.0))
    window_width: int = field(default_factory=lambda: _env_int("TANGENTIAL_GUI_WINDOW_WIDTH", 1900))
    window_height: int = field(default_factory=lambda: _env_int("TANGENTIAL_GUI_WINDOW_HEIGHT", 1050))
    region_palette: tuple[tuple[int, int, int], ...] = (
        (0, 102, 255), (0, 204, 51), (255, 128, 0), (153, 0, 255),
        (0, 204, 204), (255, 204, 0), (255, 0, 153), (255, 61, 61),
    )

    def validate(self) -> "GuiConfig":
        """校验 GUI 刷新、历史长度、窗口和区域配色。"""
        if not self.window_title.strip():
            raise ValueError("GuiConfig.window_title 不能为空")
        positive = (
            self.timer_interval_ms, self.history_size, self.error_history_size,
            self.max_region_arrows, self.heat_vmax, self.window_width,
            self.window_height,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("GUI 定时、历史、色阶、箭头和窗口尺寸必须大于 0")
        if not self.region_palette:
            raise ValueError("GuiConfig.region_palette 不能为空")
        if any(len(color) != 3 or any(channel < 0 or channel > 255 for channel in color)
               for color in self.region_palette):
            raise ValueError("GUI 区域颜色必须是 0..255 的 RGB 三元组")
        return self


@dataclass
class SpectrumConfig:
    """单路 CoP 速度 STFT 与目标频带功率占比判定参数。"""

    # 是否启用单路频谱分析、窗口和可选 NPZ 保存。
    enabled: bool = field(default_factory=lambda: _env_bool("TANGENTIAL_SPECTRUM_ENABLED", True))
    # 双传感器配置意图；双路运行入口仍强制关闭频谱，避免共享状态。
    enabled_in_dual: bool = field(default_factory=lambda: _env_bool("TANGENTIAL_SPECTRUM_ENABLED_IN_DUAL", False))
    # CoP 真实时间序列线性重采样频率，单位 Hz；不是单点扫描频率。
    sample_rate_hz: float = field(default_factory=lambda: _env_float("TANGENTIAL_SPECTRUM_SAMPLE_RATE_HZ", 160.0))
    # 速度 STFT 窗长，单位秒；默认 0.5 秒对应 80 个速度点。
    window_duration_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_SPECTRUM_WINDOW_S", 0.5))
    # 相邻频谱快照的最小时间间隔，单位秒。
    update_interval_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_SPECTRUM_UPDATE_S", 0.05))
    # 完整分析频带下限，单位 Hz；该范围内全部频点进入比值分母。
    analysis_min_frequency_hz: float = field(default_factory=lambda: _env_float("TANGENTIAL_SPECTRUM_ANALYSIS_MIN_HZ", 2.0))
    # 完整分析频带上限，单位 Hz；不得超过 Nyquist 频率。
    analysis_max_frequency_hz: float = field(default_factory=lambda: _env_float("TANGENTIAL_SPECTRUM_ANALYSIS_MAX_HZ", 70.0))
    # 滑移候选频带，单位 Hz；边界包含，必须位于完整分析频带内。
    slip_band_hz: tuple[float, float] = field(
        default_factory=lambda: _env_frequency_band(
            "TANGENTIAL_SPECTRUM_SLIP_BAND_HZ", (24.0, 28.0)
        )
    )
    # 滑移频带功率占完整分析频带功率的判定阈值，范围严格为 (0, 1)。
    slip_band_power_ratio_threshold: float = field(
        default_factory=lambda: _env_float(
            "TANGENTIAL_SPECTRUM_SLIP_BAND_POWER_RATIO_THRESHOLD", 0.16
        )
    )
    # STICK 状态下连续达到或超过阈值后进入 SLIP 的窗口数。
    enter_windows: int = field(default_factory=lambda: _env_int("TANGENTIAL_SPECTRUM_ENTER_WINDOWS", 5))
    # SLIP 状态下连续低于同一阈值后返回 STICK 的窗口数。
    exit_windows: int = field(default_factory=lambda: _env_int("TANGENTIAL_SPECTRUM_EXIT_WINDOWS", 5))
    # 每次接触开始后旁路收集逐频点静态基线的时长，单位秒；不阻塞ratio状态。
    baseline_duration_s: float = field(
        default_factory=lambda: _env_float(
            "TANGENTIAL_SPECTRUM_BASELINE_DURATION_S", 1.0
        )
    )
    # 相对基线功率计算的正数地板，避免零功率除法和无穷dB；不参与ratio。
    baseline_power_floor: float = field(
        default_factory=lambda: _env_float(
            "TANGENTIAL_SPECTRUM_BASELINE_POWER_FLOOR", 1e-6
        )
    )
    # 允许线性插值的相邻真实 CoP 最大时间间隔，单位秒；超过后重积完整窗。
    max_gap_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_SPECTRUM_MAX_GAP_S", 0.160))
    # 参与频谱分析的 CoP 状态值；默认只分析 state=2 的稳定精修帧。
    required_cop_state: int = field(default_factory=lambda: _env_int("TANGENTIAL_SPECTRUM_COP_STATE", 2))
    # GUI 瀑布图保留的最近历史时长，单位秒；不限制会话 NPZ 历史。
    history_duration_s: float = field(default_factory=lambda: _env_float("TANGENTIAL_SPECTRUM_HISTORY_S", 30.0))
    # 相对基线 dB 瀑布图色阶上限使用的百分位，范围为 (0, 100]。
    color_percentile: float = field(default_factory=lambda: _env_float("TANGENTIAL_SPECTRUM_COLOR_PERCENTILE", 95.0))
    # 会话退出时是否保存新 schema 的频谱 NPZ；没有快照时不创建文件。
    save_npz: bool = field(default_factory=lambda: _env_bool("TANGENTIAL_SPECTRUM_SAVE_NPZ", True))
    # 频谱 NPZ 相对 CSV stem 的文件名后缀。
    output_suffix: str = field(default_factory=lambda: _env("TANGENTIAL_SPECTRUM_OUTPUT_SUFFIX", "_spectrum"))
    # 频谱窗口宽度，单位像素。
    window_width: int = field(default_factory=lambda: _env_int("TANGENTIAL_SPECTRUM_WINDOW_WIDTH", 1200))
    # 频谱窗口高度，单位像素。
    window_height: int = field(default_factory=lambda: _env_int("TANGENTIAL_SPECTRUM_WINDOW_HEIGHT", 800))

    @property
    def window_samples(self) -> int:
        """返回速度 STFT 的速度样本数。"""
        return int(round(self.sample_rate_hz * self.window_duration_s))

    @property
    def required_samples(self) -> int:
        """返回形成速度窗所需的 CoP 位置点数。"""
        return self.window_samples + 1

    def validate(self) -> "SpectrumConfig":
        """规范化并校验频带、时间、滞回和显示参数。"""
        for name in ("enabled", "enabled_in_dual", "save_npz"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"SpectrumConfig.{name} 必须是布尔值")
        for name in (
            "sample_rate_hz", "window_duration_s", "update_interval_s",
            "baseline_duration_s", "baseline_power_floor", "max_gap_s",
            "history_duration_s",
        ):
            value = _coerce_finite(getattr(self, name), f"SpectrumConfig.{name}")
            if value <= 0:
                raise ValueError(f"SpectrumConfig.{name} 必须大于 0")
            setattr(self, name, value)
        for name in ("analysis_min_frequency_hz", "analysis_max_frequency_hz"):
            setattr(self, name, _coerce_finite(getattr(self, name), f"SpectrumConfig.{name}"))
        nyquist = self.sample_rate_hz / 2.0
        if not 0 <= self.analysis_min_frequency_hz < self.analysis_max_frequency_hz:
            raise ValueError("SpectrumConfig 分析频带必须满足 0 <= min < max")
        if self.analysis_max_frequency_hz > nyquist:
            raise ValueError("SpectrumConfig.analysis_max_frequency_hz 不能超过 Nyquist 频率")
        raw_band = self.slip_band_hz
        if isinstance(raw_band, str):
            parsed = _parse_frequency_bands(raw_band, "SpectrumConfig.slip_band_hz")
            if len(parsed) != 1:
                raise ValueError("SpectrumConfig.slip_band_hz 必须只有一个 low:high 频段")
            raw_band = parsed[0]
        try:
            low, high = raw_band
        except (TypeError, ValueError) as exc:
            raise ValueError("SpectrumConfig.slip_band_hz 必须是 (low, high)") from exc
        low = _coerce_finite(low, "SpectrumConfig.slip_band_hz.low")
        high = _coerce_finite(high, "SpectrumConfig.slip_band_hz.high")
        if not self.analysis_min_frequency_hz <= low < high <= self.analysis_max_frequency_hz:
            raise ValueError("SpectrumConfig.slip_band_hz 必须位于完整分析频带内并满足 low < high")
        self.slip_band_hz = (low, high)
        samples = self.sample_rate_hz * self.window_duration_s
        if abs(samples - round(samples)) > 1e-9 or round(samples) < 2:
            raise ValueError("SpectrumConfig.sample_rate_hz * window_duration_s 必须是至少 2 的整数")
        if self.update_interval_s > self.window_duration_s:
            raise ValueError("SpectrumConfig.update_interval_s 不能大于窗长")
        if self.baseline_duration_s < self.update_interval_s:
            raise ValueError("SpectrumConfig.baseline_duration_s 不能小于更新间隔")
        if self.history_duration_s < self.update_interval_s:
            raise ValueError("SpectrumConfig.history_duration_s 不能小于更新间隔")
        frequency_hz = np.fft.rfftfreq(self.window_samples, d=1.0 / self.sample_rate_hz)
        if not np.any((frequency_hz >= low - 1e-12) & (frequency_hz <= high + 1e-12)):
            raise ValueError("SpectrumConfig.slip_band_hz 没有对应的 FFT 频点")
        threshold = _coerce_finite(
            self.slip_band_power_ratio_threshold,
            "SpectrumConfig.slip_band_power_ratio_threshold",
        )
        if not 0 < threshold < 1:
            raise ValueError("SpectrumConfig.slip_band_power_ratio_threshold 必须在 (0,1) 内")
        self.slip_band_power_ratio_threshold = threshold
        required_state = _coerce_finite(self.required_cop_state, "SpectrumConfig.required_cop_state")
        if not required_state.is_integer() or required_state < 0:
            raise ValueError("SpectrumConfig.required_cop_state 必须是非负整数")
        self.required_cop_state = int(required_state)
        self.color_percentile = _coerce_finite(self.color_percentile, "SpectrumConfig.color_percentile")
        if not 0 < self.color_percentile <= 100:
            raise ValueError("SpectrumConfig.color_percentile 必须在 (0,100] 内")
        if not isinstance(self.output_suffix, str) or not self.output_suffix.strip():
            raise ValueError("SpectrumConfig.output_suffix 不能为空")
        for name in ("enter_windows", "exit_windows", "window_width", "window_height"):
            value = _coerce_finite(getattr(self, name), f"SpectrumConfig.{name}")
            if not value.is_integer() or value <= 0:
                raise ValueError(f"SpectrumConfig.{name} 必须是大于 0 的整数")
            setattr(self, name, int(value))
        return self

@dataclass(init=False)
class FullApplicationConfig:
    """完整应用的分层配置聚合对象。"""

    array: ArrayConfig = field(default_factory=ArrayConfig)
    pressure: PressureConfig = field(default_factory=PressureConfig)
    force: ForceConfig = field(default_factory=ForceConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    spectrum: SpectrumConfig = field(default_factory=SpectrumConfig)

    def __init__(
        self,
        array: ArrayConfig | None = None,
        pressure: PressureConfig | None = None,
        force: ForceConfig | None = None,
        processing: ProcessingConfig | None = None,
        calibration: CalibrationConfig | None = None,
        sync: SyncConfig | None = None,
        output: OutputConfig | None = None,
        gui: GuiConfig | None = None,
        spectrum: SpectrumConfig | None = None,
        **legacy_overrides: Any,
    ) -> None:
        """创建分层配置，并接受一次性的旧扁平字段覆盖。

        扁平参数只是构造适配，不再保存第二套默认值；新代码应优先传入
        ``PressureConfig``、``SyncConfig`` 等分类对象。
        """
        self.array = ArrayConfig() if array is None else array
        self.pressure = PressureConfig() if pressure is None else pressure
        self.force = ForceConfig() if force is None else force
        self.processing = ProcessingConfig() if processing is None else processing
        self.calibration = CalibrationConfig() if calibration is None else calibration
        self.sync = SyncConfig() if sync is None else sync
        self.output = OutputConfig() if output is None else output
        self.gui = GuiConfig() if gui is None else gui
        self.spectrum = SpectrumConfig() if spectrum is None else spectrum
        mapping = {
            "pressure_port": (self.pressure, "port"),
            "force_port": (self.force, "port"),
            "force_enabled": (self.force, "enabled"),
            "model_path": (self.calibration, "model_path"),
            "save_dir": (self.output, "save_dir"),
            "cal_dim": (self.processing, "cal_dim"),
            "region_mode": (self.processing, "region_mode"),
            "refine_rezero_force": (self.processing, "refine_rezero_force"),
            "target_fps": (self.sync, "target_fps"),
            "plot_fps": (self.sync, "plot_fps"),
            "max_time_diff_s": (self.sync, "max_time_diff_s"),
            "timing_log_interval_s": (self.sync, "timing_log_interval_s"),
            "buffer_size": (self.sync, "buffer_size"),
            "zero_sample_count": (self.force, "zero_sample_count"),
            "zero_timeout_s": (self.force, "zero_timeout_s"),
            "rezero_timeout_s": (self.force, "rezero_timeout_s"),
        }
        unknown = set(legacy_overrides) - set(mapping)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"未知 FullApplicationConfig 参数: {names}")
        for name, value in legacy_overrides.items():
            target, attribute = mapping[name]
            setattr(target, attribute, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        """在应用启动前验证所有嵌套配置。"""
        self.validate()

    def validate(self) -> "FullApplicationConfig":
        """递归校验所有嵌套配置并返回当前对象。"""
        if not isinstance(self.array, ArrayConfig):
            raise TypeError("FullApplicationConfig.array 必须是 ArrayConfig")
        self.array.validate()
        self.pressure.validate()
        self.force.validate()
        self.processing.validate()
        self.sync.validate()
        self.gui.validate()
        self.spectrum.validate()
        return self

    @property
    def pressure_port(self) -> str:
        """返回压力串口路径。"""
        return self.pressure.port

    @property
    def force_port(self) -> str:
        """返回六维力串口路径。"""
        return self.force.port

    @property
    def force_enabled(self) -> bool:
        """返回是否启用六维力通道。"""
        return self.force.enabled

    @property
    def model_path(self) -> str | None:
        """返回外部模型路径或 ``None``。"""
        return self.calibration.model_path

    @property
    def save_dir(self) -> str:
        """返回 CSV 保存目录。"""
        return self.output.save_dir

    @property
    def cal_dim(self) -> str:
        """返回标定维度模式。"""
        return self.processing.cal_dim

    @property
    def region_mode(self) -> str:
        """返回区域处理模式。"""
        return self.processing.region_mode

    @property
    def refine_rezero_force(self) -> bool:
        """返回是否在 CoP 精修/卸载时重新归零。"""
        return self.processing.refine_rezero_force

    @property
    def target_fps(self) -> float:
        """返回主循环目标频率。"""
        return self.sync.target_fps

    @property
    def plot_fps(self) -> float:
        """返回 GUI 更新上限。"""
        return self.sync.plot_fps

    @property
    def max_time_diff_s(self) -> float:
        """返回压力—力最大匹配时间差。"""
        return self.sync.max_time_diff_s

    @property
    def timing_log_interval_s(self) -> float:
        """返回时序日志间隔。"""
        return self.sync.timing_log_interval_s

    @property
    def buffer_size(self) -> int:
        """返回每路时间戳缓存容量。"""
        return self.sync.buffer_size

    @property
    def zero_sample_count(self) -> int:
        """返回启动校零样本数。"""
        return self.force.zero_sample_count

    @property
    def zero_timeout_s(self) -> float:
        """返回启动校零超时。"""
        return self.force.zero_timeout_s

    @property
    def rezero_timeout_s(self) -> float:
        """返回运行期重新归零超时。"""
        return self.force.rezero_timeout_s

@dataclass
class TrainingConfig:
    """离线拟合配置；算法位于 ``tangential.tools.training``。"""

    xy_csv: str | os.PathLike[str]
    z_csv: str | os.PathLike[str]
    output_model: str | os.PathLike[str] = "fit_coefs.bin"
    output_plot: str | os.PathLike[str] | None = "fit_report.png"
    dim: int = 1
    poly_order: int = 3
    fx: str = "sym_log"
    fy: str = "sym_log"
    fz: str = "exp"
    valid_only: bool = True
    split_sign: bool = True
    one_on_one: bool = True
    write_back: str | os.PathLike[str] | None = None
    force: bool = False


@dataclass
class PlotConfig:
    """离线绘图参数；算法位于 ``tangential.tools.plotting``。"""

    files: str | Path | list[str | Path] | None = None
    directory: str | Path = field(default_factory=lambda: Path.cwd() / "data")
    columns: tuple[str | int, ...] = ("Fy_cal", "delta_Force_Y")
    rows: Any = None
    x_column: str | int | None = "rel_ms"
    title: str | None = None
    save_path: str | Path | None = None
    error_ref: str | int | None = None
    mode: str = "plot"
    highlight_valid: bool = True
    show_annotations: bool = True
    force_min: float = 0.2


__all__ = [
    "ArrayConfig", "PressureConfig", "ForceConfig", "CopConfig", "SlipConfig",
    "ConsistenceCalibrationConfig", "ProcessingConfig",
    "CalibrationConfig", "SyncConfig", "OutputConfig", "GuiConfig",
    "SpectrumConfig",
    "TrainingConfig", "PlotConfig", "FullApplicationConfig",
    "default_model_path", "default_save_dir",
]
