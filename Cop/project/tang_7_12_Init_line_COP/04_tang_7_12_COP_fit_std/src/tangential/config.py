"""公共配置与默认资源路径。"""

import os
import sysconfig
from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_model_path() -> str:
    override = os.environ.get("TANGENTIAL_MODEL_PATH")
    if override:
        return override
    source_model = PROJECT_ROOT / "fit_coefs.bin"
    if source_model.exists():
        return str(source_model)
    target_model = (
        Path(__file__).resolve().parents[1]
        / "share"
        / "tangential"
        / "fit_coefs.bin"
    )
    if target_model.exists():
        return str(target_model)
    installed_model = (
        Path(sysconfig.get_path("data"))
        / "share"
        / "tangential"
        / "fit_coefs.bin"
    )
    return str(installed_model)


def default_save_dir() -> str:
    override = os.environ.get("TANGENTIAL_DATA_DIR")
    if override:
        return override
    if (PROJECT_ROOT / "main.py").exists():
        return str(
            (PROJECT_ROOT / "../../../../../../../data/2.PZT_tangential/"
             "weight/test").resolve()
        )
    return str((Path.cwd() / "data").resolve())


@dataclass
class FullApplicationConfig:
    save_dir: str = field(default_factory=default_save_dir)
    fit_coefs_path: str = field(default_factory=default_model_path)
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
