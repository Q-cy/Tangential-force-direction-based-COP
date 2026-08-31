"""固定格式 CSV 持久化入口。

本子包公开默认 108 列（动态阵列为 ``channel_count + 24`` 列）的
``TABLE_CSV_HEADER``、路径生成、文件初始化和
行构造函数。完整采集应复用这些定义，不在 ``full``、CLI 或 GUI 中手工
复制列顺序；本包不负责读取串口、执行 CoP/标定或绘图。
"""

from .csv import (
    TABLE_CSV_HEADER, build_csv_header,
    auto_get_csv_path,
    build_csv_row,
    full_analysis_png_path,
    init_csv_file,
)

__all__ = [
    "TABLE_CSV_HEADER", "build_csv_header", "auto_get_csv_path", "build_csv_row",
    "full_analysis_png_path", "init_csv_file",
]
