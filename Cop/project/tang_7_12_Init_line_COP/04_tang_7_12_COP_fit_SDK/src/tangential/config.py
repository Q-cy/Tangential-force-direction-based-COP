"""公共配置与默认资源路径。

本模块集中保存完整采集应用的默认端口、目录、时间窗口和校零参数；不
创建传感器、不启动线程，也不加载 Qt。
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


def default_model_path() -> str | None:
    """读取外部拟合模型覆盖路径。

    Returns:
        str | None: ``TANGENTIAL_MODEL_PATH`` 的非空值；环境变量未设置或
            为空字符串时返回 ``None``，表示使用 package 内置模型。

    Side Effects:
        只读取当前进程环境变量，不修改环境或文件。
    """
    return os.environ.get("TANGENTIAL_MODEL_PATH") or None


def default_save_dir() -> str:
    """解析默认 CSV 保存目录。

    Returns:
        str: ``TANGENTIAL_DATA_DIR`` 的非空值；未设置时返回当前工作目录
            下 ``data`` 子目录的绝对路径。

    Side Effects:
        只读取环境变量和当前工作目录，不创建目录；目录由 CSV 初始化函数
        在真正保存时创建。
    """
    override = os.environ.get("TANGENTIAL_DATA_DIR")
    if override:
        return override
    return str((Path.cwd() / "data").resolve())


@dataclass
class FullApplicationConfig:
    """完整采集会话的设备、算法、时序和输出配置。

    Attributes:
        save_dir (str): CSV 保存目录，默认由 ``default_save_dir`` 计算。
        model_path (str | None): 外部模型路径；``None`` 使用内置模型。
        pressure_port (str): 压力传感器串口路径。
        force_port (str): 六维力传感器串口路径。
        cal_dim (str): 标定输出维度模式。
        refine_rezero_force (bool): CoP 精修后是否触发 Fx/Fy 重新归零。
        target_fps (int): 主采集循环目标频率，单位为 Hz。
        plot_fps (int): GUI 更新上限，单位为 Hz。
        max_time_diff_s (float): 压力/六维力匹配窗口，单位为秒。
        timing_log_interval_s (float): 时序统计日志间隔，单位为秒。
        region_mode (str): CoP 区域计算模式。
        zero_sample_count (int): 启动校零需要的有效普通力帧数。
        zero_timeout_s (float): 启动校零最长等待时间，单位为秒。
        rezero_timeout_s (float): 运行期重新归零最长等待时间，单位为秒。
        buffer_size (int): 每个传感器时间戳缓存的最大帧数。

    Notes:
        这是数据类，构造时只保存配置值；设备连接、目录创建和模型加载由
        完整采集会话负责。
    """
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
