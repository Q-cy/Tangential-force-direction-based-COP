# file_name: table.py

import os
import csv
import numpy as np

# 定义CSV表头（严格匹配要求的格式）
CSV_HEADER = [
    "timestamp", "rel_ms",
    # ch1 ~ ch84
    "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7",
    "ch8", "ch9", "ch10", "ch11", "ch12", "ch13", "ch14",
    "ch15", "ch16", "ch17", "ch18", "ch19", "ch20", "ch21",
    "ch22", "ch23", "ch24", "ch25", "ch26", "ch27", "ch28",
    "ch29", "ch30", "ch31", "ch32", "ch33", "ch34", "ch35",
    "ch36", "ch37", "ch38", "ch39", "ch40", "ch41", "ch42",
    "ch43", "ch44", "ch45", "ch46", "ch47", "ch48", "ch49",
    "ch50", "ch51", "ch52", "ch53", "ch54", "ch55", "ch56",
    "ch57", "ch58", "ch59", "ch60", "ch61", "ch62", "ch63",
    "ch64", "ch65", "ch66", "ch67", "ch68", "ch69", "ch70",
    "ch71", "ch72", "ch73", "ch74", "ch75", "ch76", "ch77",
    "ch78", "ch79", "ch80", "ch81", "ch82", "ch83", "ch84",
    # 力传感器数据
    "Fx", "Fy", "Fz", "Mx", "My", "Mz",
    # 时间戳相关
    "press_t", "force_t", "dt",
    # 新增 CoP 偏移分量
    "delta_CoP_X", "delta_CoP_Y",
    # 新增 Force 分量
    "delta_Force_X", "delta_Force_Y", # <--- ADDED THESE TWO LINES
    # 角度和幅值
    "ADC_angle", "ADC_mag", "Force_angle", "Force_mag"
]

def auto_get_csv_path(save_dir: str) -> str:
    """
    自动生成不重复的CSV文件路径（格式：data_1.csv, data_2.csv...）
    :param save_dir: 保存目录
    :return: 完整的CSV文件路径
    """
    os.makedirs(save_dir, exist_ok=True)
    idx = 1
    while os.path.exists(f"{save_dir}/data_{idx}.csv"):
        idx += 1
    return f"{save_dir}/data_{idx}.csv"

def init_csv_file(file_path: str) -> tuple[csv.writer, object]:
    """
    初始化CSV文件，写入表头并返回writer和文件对象
    :param file_path: CSV文件路径
    :return: (csv_writer, csv_file_object)
    """
    csv_file_obj = open(file_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file_obj)
    csv_writer.writerow(CSV_HEADER)
    print(f"📂 CSV文件已初始化：{file_path}")
    return csv_writer, csv_file_obj

def build_csv_row(
    press_timestamp: float,  # 压力传感器时间戳（秒）
    rel_ms: int,             # 相对毫秒数
    ch_data: list,           # 84通道压力数据
    force_data: list,        # 六维力传感器数据 [Fx,Fy,Fz,Mx,My,Mz]
    force_timestamp: float,  # 力传感器时间戳（秒）
    delta_cop_x: float,      # 新增 CoP 偏移X分量
    delta_cop_y: float,      # 新增 CoP 偏移Y分量
    delta_force_x: float,    # <--- ADDED THIS PARAMETER
    delta_force_y: float,    # <--- ADDED THIS PARAMETER
    adc_angle: float,        # ADC角度
    adc_mag: float,          # ADC幅值
    force_angle: float,      # 力传感器角度
    force_mag: float         # 力传感器幅值
) -> list:
    """
    构造符合表头格式的CSV行数据
    :return: 完整的CSV行列表
    """
    # 计算时间差
    dt = abs(press_timestamp - force_timestamp)
    
    # 构造行数据
    csv_row = [
        press_timestamp * 1000,  # timestamp：转换为毫秒级
        rel_ms,                  # rel_ms：相对开始时间的毫秒数
        *ch_data,                # ch1~ch84：压力传感器84通道数据
        *force_data,             # Fx,Fy,Fz,Mx,My,Mz：力传感器数据
        press_timestamp,         # press_t：压力传感器原始时间戳（秒）
        force_timestamp,         # force_t：力传感器原始时间戳（秒）
        dt,                      # dt：时间戳差值（秒）
        delta_cop_x,             # delta_CoP_X
        delta_cop_y,             # delta_CoP_Y
        delta_force_x,           # <--- ADDED THIS TO THE ROW
        delta_force_y,           # <--- ADDED THIS TO THE ROW
        adc_angle,               # ADC_angle：PZT计算的角度
        adc_mag,                 # ADC_mag：CoP偏移幅值
        force_angle,             # Force_angle：力传感器计算的角度
        force_mag                # Force_mag：力传感器幅值
    ]
    return csv_row
