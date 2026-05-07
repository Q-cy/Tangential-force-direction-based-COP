# file_name: table.py

import os
import csv
import numpy as np

N_COP_CSV = 5  # CSV 中保存的 COP 数量

# 构建主COP字段（保持向后兼容）
_MAIN_COP_FIELDS = [
    "delta_CoP_X", "delta_CoP_Y",
    "delta_Force_X", "delta_Force_Y",
    "ADC_angle", "ADC_mag", "Force_angle", "Force_mag"
]

# 构建额外COP字段 (COP2~COP5)
_EXTRA_COP_FIELDS = []
for i in range(2, N_COP_CSV + 1):
    _EXTRA_COP_FIELDS += [
        f"COP{i}_x", f"COP{i}_y",
        f"COP{i}_dx", f"COP{i}_dy",
        f"COP{i}_total",
        f"COP{i}_angle", f"COP{i}_mag",
    ]

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
    # 主 COP 字段（COP1，向后兼容）
    *_MAIN_COP_FIELDS,
    # 额外 COP 字段（COP2~COP5）
    *_EXTRA_COP_FIELDS,
]

_EMPTY_COP_FIELDS = [""] * 7  # 每个 COP 有 7 个字段


def auto_get_csv_path(save_dir: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    idx = 1
    while os.path.exists(f"{save_dir}/data_{idx}.csv"):
        idx += 1
    return f"{save_dir}/data_{idx}.csv"


def init_csv_file(file_path: str) -> tuple[csv.writer, object]:
    csv_file_obj = open(file_path, "w", encoding="utf-8", newline="")
    csv_writer = csv.writer(csv_file_obj)
    csv_writer.writerow(CSV_HEADER)
    print(f"📂 CSV文件已初始化：{file_path}")
    return csv_writer, csv_file_obj


def build_csv_row(
    press_timestamp: float,
    rel_ms: int,
    ch_data: list,
    force_data: list,
    force_timestamp: float,
    regions: list,
    delta_force_x: float,
    delta_force_y: float,
    force_angle: float,
    force_mag: float
) -> list:
    """
    构造符合表头格式的CSV行数据，支持多COP。
    :param regions: COP 区域列表，regions[0] 为主COP
    """
    dt = abs(press_timestamp - force_timestamp)

    # 主 COP 数据
    if regions:
        p = regions[0]
        delta_cop_x = p['delta_cop_x']
        delta_cop_y = p['delta_cop_y']
        adc_angle = p.get('angle', 0.0)
        adc_mag = p.get('mag', 0.0)
    else:
        delta_cop_x = delta_cop_y = adc_angle = adc_mag = 0.0

    # 构造额外 COP 字段 (COP2 ~ COP5)
    extra_cop_flat = []
    for i in range(1, N_COP_CSV):
        if i < len(regions):
            r = regions[i]
            extra_cop_flat += [
                r['cop_x'], r['cop_y'],
                r['delta_cop_x'], r['delta_cop_y'],
                r['total_pressure'],
                r.get('angle', 0.0), r.get('mag', 0.0),
            ]
        else:
            extra_cop_flat += _EMPTY_COP_FIELDS

    csv_row = [
        press_timestamp * 1000,
        rel_ms,
        *ch_data,
        *force_data,
        press_timestamp,
        force_timestamp,
        dt,
        delta_cop_x,
        delta_cop_y,
        delta_force_x,
        delta_force_y,
        adc_angle,
        adc_mag,
        force_angle,
        force_mag,
        *extra_cop_flat,
    ]
    return csv_row
