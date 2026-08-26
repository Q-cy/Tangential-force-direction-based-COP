from typing import Any
import csv

TABLE_CSV_HEADER: list[str]
def auto_get_csv_path(save_dir: str) -> str: ...
def init_csv_file(file_path: str) -> tuple[csv.writer, Any]: ...
def build_csv_row(press_timestamp: float, rel_ms: float, delta_ms: float,
                  ch_data: list[Any], force_data: list[Any],
                  force_timestamp: float, delta_cop_x: float,
                  delta_cop_y: float, delta_force_x: float,
                  delta_force_y: float, delta_force_z: float,
                  adc_angle: float, force_angle: float,
                  fx_cal: float | None = ..., fy_cal: float | None = ...,
                  force_cal_angle: float | None = ..., cop_state: int = ...,
                  adc_sum: float = ..., valid: int = ...) -> list[Any]: ...
