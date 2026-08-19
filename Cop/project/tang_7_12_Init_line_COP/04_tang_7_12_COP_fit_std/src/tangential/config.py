"""公共配置与默认资源路径。"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_model_path() -> str | None:
    """返回外部模型覆盖路径；未设置时由运行时加载内置模型。"""
    return os.environ.get("TANGENTIAL_MODEL_PATH") or None


def default_save_dir() -> str:
    override = os.environ.get("TANGENTIAL_DATA_DIR")
    if override:
        return override
    return str((Path.cwd() / "data").resolve())


@dataclass
class FullApplicationConfig:
    save_dir: str = field(default_factory=default_save_dir)
    model_path: str | None = field(default_factory=default_model_path)
    pressure_port: str = "/dev/ttyUSB0"
    force_port: str = "/dev/ttyUSB1"
    cal_dim: str = "3D"
    refine_rezero_force: bool = True
    target_fps: int = 100
    plot_fps: int = 60
    max_time_diff_s: float = 0.015
    timing_log_interval_s: float = 1.0
    region_mode: str = "full"
    zero_sample_count: int = 10
    zero_timeout_s: float = 1.0
    rezero_timeout_s: float = 1.0
    buffer_size: int = 500
