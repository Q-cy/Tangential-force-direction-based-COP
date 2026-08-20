"""完整采集 CSV 的唯一表头、文件初始化和行构造函数。

CSV 列顺序是对外数据格式的一部分；采集和离线工具都应复用本模块，
不要在其他模块手工拼接 108 列行数据。
"""

import os
import csv

TABLE_CSV_HEADER = [  # CSV 文件表头（84通道 + 时间差 + 力/角度/标定数据）
    "rel_ms", "delta_ms", "adc_sum",
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
    "delta_Force_X", "delta_Force_Y", "delta_Force_Z",
    # 角度
    "ADC_angle", "Force_angle",
    # 标定后的切向力
    "Fx_cal", "Fy_cal", "Force_cal_angle",
    # 接触状态
    "CoP_state",
    # 有效行标记 (1=接触帧有效, 0=无效); fit.py/plot_static.py 训练筛选依赖此列
    "valid"
]

def auto_get_csv_path(save_dir: str) -> str:
    """在保存目录中生成当天不重复的 CSV 文件路径。

    Args:
        save_dir (str): 保存目录路径；目录不存在时会创建。

    Returns:
        str: 形如 ``<save_dir>/COP_test_MMDD_N.csv`` 的第一个未占用路径，
            其中 ``MMDD`` 为本地日期，``N`` 从 1 开始。

    Raises:
        OSError: 保存目录无法创建或访问时抛出。

    Side Effects:
        调用 ``os.makedirs(..., exist_ok=True)`` 创建目录；只检查路径存在性，
        不创建最终 CSV 文件。
    """
    from datetime import datetime
    os.makedirs(save_dir, exist_ok=True)
    date_str = datetime.now().strftime("%m%d")
    idx = 1
    while True:
        full_path = f"{save_dir}/COP_test_{date_str}_{idx}.csv"
        if not os.path.exists(full_path):
            return full_path
        idx += 1

def init_csv_file(file_path: str) -> tuple[csv.writer, object]:
    """创建并初始化 CSV 文件，写入唯一固定表头。

    Args:
        file_path (str): 要写入的 CSV 文件路径；以写入模式打开，已有文件
            内容会被覆盖。

    Returns:
        tuple[csv.writer, TextIO]: CSV writer 和保持打开状态的文本文件对象。
            调用方负责持续写入、flush，并最终调用文件对象的 ``close``。

    Raises:
        OSError: 文件无法创建或打开时抛出。
        UnicodeError: 文件编码初始化失败时可能抛出。

    Side Effects:
        以 UTF-8、换行兼容模式打开文件，写入 ``TABLE_CSV_HEADER``，并向
        标准输出打印初始化路径。
    """
    csv_file_obj = open(file_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file_obj)
    csv_writer.writerow(TABLE_CSV_HEADER)
    print(f"📂 CSV文件已初始化：{file_path}")
    return csv_writer, csv_file_obj

def build_csv_row(
    press_timestamp: float,  # 压力传感器时间戳（秒）
    rel_ms: float,           # 相对首个保存行的毫秒数
    delta_ms: float,         # 与上一保存行的压力时间差（毫秒）
    ch_data: list,           # 84通道压力数据
    force_data: list,        # 六维力传感器数据 [Fx,Fy,Fz,Mx,My,Mz]
    force_timestamp: float,  # 力传感器时间戳（秒）
    delta_cop_x: float,      # 新增 CoP 偏移X分量
    delta_cop_y: float,      # 新增 CoP 偏移Y分量
    delta_force_x: float,
    delta_force_y: float,
    delta_force_z: float,
    adc_angle: float,        # ADC角度
    force_angle: float,      # 力传感器角度
    fx_cal: float = None,    # 标定后切向力 X (N)
    fy_cal: float = None,    # 标定后切向力 Y (N)
    force_cal_angle: float = None, # 标定后角度 (deg)
    cop_state: int = 0,            # 接触状态: 0=未接触, 1=等待稳定, 2=测量中
    adc_sum: float = 0.0,          # ADC 84通道之和
    valid: int = 0,                # 有效行标记: 1=接触帧有效, 0=无效 (训练筛选用)
) -> list:
    """按 ``TABLE_CSV_HEADER`` 顺序构造一行完整 CSV 数据。

    Args:
        press_timestamp (float): 压力帧原始时间戳，单位为秒。
        rel_ms (float): 相对首个保存压力帧的时间，单位为毫秒。
        delta_ms (float): 与上一保存压力帧的时间差，单位为毫秒。
        ch_data (Sequence): 84 个压力 ADC 通道值，按原始线序排列。
        force_data (Sequence): 六维力值 ``[Fx, Fy, Fz, Mx, My, Mz]``。
        force_timestamp (float): 力帧原始时间戳，单位为秒；无匹配时可为
            ``NaN``。
        delta_cop_x (float): CoP X 偏移分量。
        delta_cop_y (float): CoP Y 偏移分量。
        delta_force_x (float): 力 X 偏移/滤波分量。
        delta_force_y (float): 力 Y 偏移/滤波分量。
        delta_force_z (float): 力 Z 偏移/滤波分量。
        adc_angle (float): 压力阵列方向角，单位为度。
        force_angle (float): 六维力方向角，单位为度。
        fx_cal (float | None): 标定后的切向力 X，单位由模型定义；默认
            ``None``，输出 ``NaN``。
        fy_cal (float | None): 标定后的切向力 Y；默认 ``None``，输出
            ``NaN``。
        force_cal_angle (float | None): 标定后的方向角，单位为度；默认
            ``None``，输出 ``NaN``。
        cop_state (int): CoP 接触状态，默认 0。
        adc_sum (float): 84 通道 ADC 总和，默认 0.0。
        valid (int): 有效行标记，默认 0；通常 1 表示接触帧有效。

    Returns:
        list: 按固定 108 列顺序排列的 CSV 行列表。时间戳差 ``dt`` 由压力
            和力时间戳的绝对差计算，单位为秒。

    Raises:
        TypeError: 输入序列无法展开或数值无法参与时间计算时可能抛出。
        ValueError: 输入容器或元素不符合调用方约定时可能抛出。

    Notes:
        本函数不写文件、不校验通道数量，也不修改输入序列；通道数量和数据
        类型由调用方保证。
    """
    # 计算时间差
    dt = abs(press_timestamp - force_timestamp)

    # 构造行数据
    csv_row = [
        rel_ms,
        delta_ms,
        adc_sum,                 # adc_sum：ADC 84通道之和
        *ch_data,                # ch1~ch84：压力传感器84通道数据
        *force_data,             # Fx,Fy,Fz,Mx,My,Mz：力传感器数据
        press_timestamp,         # press_t：压力传感器原始时间戳（秒）
        force_timestamp,         # force_t：力传感器原始时间戳（秒）
        dt,                      # dt：时间戳差值（秒）
        delta_cop_x,             # delta_CoP_X
        delta_cop_y,             # delta_CoP_Y
        delta_force_x,
        delta_force_y,
        delta_force_z,
        adc_angle,               # ADC_angle：PZT计算的角度
        force_angle,             # Force_angle：力传感器计算的角度
        fx_cal if fx_cal is not None else float('nan'),
        fy_cal if fy_cal is not None else float('nan'),
        force_cal_angle if force_cal_angle is not None else float('nan'),
        cop_state,
        valid,
    ]
    return csv_row
