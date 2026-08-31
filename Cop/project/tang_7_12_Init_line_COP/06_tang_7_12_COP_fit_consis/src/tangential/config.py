"""Tangential SDK 的集中配置定义。

协议帧头、CRC、固定阵列尺寸和 108 列 CSV 等协议不变量留在各自实现中；
本模块只保存用户能够调整的设备、处理、同步、输出和离线工具参数。
环境变量只提供默认值，显式构造的 dataclass 字段始终具有更高优先级。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是有限数字") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{label} 必须是有限数字")
    return converted


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

    rows: int = 12
    cols: int = 7
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
        """校验阵列尺寸和 CoP 参数。"""
        if self.rows <= 0 or self.cols <= 0 or self.collect_frames <= 0:
            raise ValueError("CoP 阵列尺寸和背景帧数必须大于 0")
        if self.stability_frames <= 0 or self.region_min_area <= 0:
            raise ValueError("CoP 稳定帧数和区域最小面积必须大于 0")
        if self.refine_cnt < 0 or self.reset_at_frame < 0:
            raise ValueError("CoP 精修/重置帧数不能为负数")
        return self

    def as_kwargs(self) -> dict[str, Any]:
        """返回可直接传给 ``PRSensorAngle`` 的参数字典。"""
        return {
            "rows": self.rows, "cols": self.cols,
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

    所有可编辑默认值都集中在下面的字段定义中。``csv_directory`` 和
    ``output_path`` 基于源码项目根目录；``coefficients_path=None`` 表示
    运行时加载 package resource。环境变量只覆盖运行期开关、外部系数路径
    和裁剪范围。``force=True`` 使维护者无参数离线命令可重复生成并覆盖旧
    NPZ；调用方显式传入 ``force=False`` 时仍拒绝覆盖。
    """

    enabled: bool = field(
        default_factory=lambda: _env_bool("TANGENTIAL_CONSISTENCE_ENABLED", True)
    )
    csv_directory: str | os.PathLike[str] = field(
        default_factory=lambda: (
            _SOURCE_PROJECT_ROOT
            / "src"
            / "tangential"
            / "resources"
            / "consistence"
        ).resolve()
    )
    csv_pattern: str = "*.csv"
    tail_rows: int = 10
    minimum_breakpoint_step: float = 1.0
    max_segment_scale: float = 100.0
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
        """校验多段离线标定输入、断点间距和裁剪范围。"""
        if not str(self.csv_directory).strip():
            raise ValueError("ConsistenceCalibrationConfig.csv_directory 不能为空")
        if not self.csv_pattern.strip():
            raise ValueError("ConsistenceCalibrationConfig.csv_pattern 不能为空")
        if self.tail_rows <= 0:
            raise ValueError("ConsistenceCalibrationConfig.tail_rows 必须大于0")
        self.minimum_breakpoint_step = _coerce_finite(
            self.minimum_breakpoint_step,
            "ConsistenceCalibrationConfig.minimum_breakpoint_step",
        )
        if self.minimum_breakpoint_step <= 0:
            raise ValueError(
                "ConsistenceCalibrationConfig.minimum_breakpoint_step 必须大于0"
            )
        self.max_segment_scale = _coerce_finite(
            self.max_segment_scale,
            "ConsistenceCalibrationConfig.max_segment_scale",
        )
        if self.max_segment_scale <= 0:
            raise ValueError(
                "ConsistenceCalibrationConfig.max_segment_scale 必须大于0"
            )
        if self.max_segment_scale > 100.0:
            raise ValueError(
                "ConsistenceCalibrationConfig.max_segment_scale 不能大于100"
            )
        if not str(self.output_path).strip():
            raise ValueError("ConsistenceCalibrationConfig.output_path 不能为空")
        if self.coefficients_path is not None and not str(
            self.coefficients_path
        ).strip():
            raise ValueError(
                "ConsistenceCalibrationConfig.coefficients_path 不能为空字符串"
            )
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


@dataclass(init=False)
class FullApplicationConfig:
    """完整应用的分层配置聚合对象。"""

    pressure: PressureConfig = field(default_factory=PressureConfig)
    force: ForceConfig = field(default_factory=ForceConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)

    def __init__(
        self,
        pressure: PressureConfig | None = None,
        force: ForceConfig | None = None,
        processing: ProcessingConfig | None = None,
        calibration: CalibrationConfig | None = None,
        sync: SyncConfig | None = None,
        output: OutputConfig | None = None,
        gui: GuiConfig | None = None,
        **legacy_overrides: Any,
    ) -> None:
        """创建分层配置，并接受一次性的旧扁平字段覆盖。

        扁平参数只是构造适配，不再保存第二套默认值；新代码应优先传入
        ``PressureConfig``、``SyncConfig`` 等分类对象。
        """
        self.pressure = pressure or PressureConfig()
        self.force = force or ForceConfig()
        self.processing = processing or ProcessingConfig()
        self.calibration = calibration or CalibrationConfig()
        self.sync = sync or SyncConfig()
        self.output = output or OutputConfig()
        self.gui = gui or GuiConfig()
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
        self.pressure.validate()
        self.force.validate()
        self.processing.validate()
        self.sync.validate()
        self.gui.validate()

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
    "PressureConfig", "ForceConfig", "CopConfig", "SlipConfig",
    "ConsistenceCalibrationConfig", "ProcessingConfig",
    "CalibrationConfig", "SyncConfig", "OutputConfig", "GuiConfig",
    "TrainingConfig", "PlotConfig", "FullApplicationConfig",
    "default_model_path", "default_save_dir",
]
