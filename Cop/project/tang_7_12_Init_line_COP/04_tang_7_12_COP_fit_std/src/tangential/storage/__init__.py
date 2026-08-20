"""固定格式 CSV 持久化入口。

本子包公开唯一的 108 列 ``TABLE_CSV_HEADER``、路径生成、文件初始化和
行构造函数。完整采集应复用这些定义，不在 ``full``、CLI 或 GUI 中手工
复制列顺序；本包不负责读取串口、执行 CoP/标定或绘图。
"""

from .csv import TABLE_CSV_HEADER, auto_get_csv_path, build_csv_row, init_csv_file

__all__ = [
    "TABLE_CSV_HEADER", "auto_get_csv_path", "build_csv_row", "init_csv_file"
]
