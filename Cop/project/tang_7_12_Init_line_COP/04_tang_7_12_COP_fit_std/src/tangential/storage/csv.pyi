from typing import Any
import csv
from os import PathLike
from pathlib import Path
from ..config import ArrayConfig

TABLE_CSV_HEADER: list[str]
def build_csv_header(array_config: ArrayConfig | None = ...) -> list[str]: ...
def auto_get_csv_path(save_dir: str) -> str: ...
def full_analysis_png_path(csv_path: str | PathLike[str]) -> Path: ...
def init_csv_file(file_path: str, array_config: ArrayConfig | None = ...) -> tuple[csv.writer, Any]: ...
def build_csv_row(press_timestamp: float, rel_ms: float, delta_ms: float,
                  ch_data: list[Any], force_data: list[Any],
                  force_timestamp: float, delta_cop_x: float,
                  delta_cop_y: float, delta_force_x: float,
                  delta_force_y: float, delta_force_z: float,
                  adc_angle: float, force_angle: float,
                  fx_cal: float | None = ..., fy_cal: float | None = ...,
                  force_cal_angle: float | None = ..., cop_state: int = ...,
                  adc_sum: float = ..., valid: int = ...,
                  *, array_config: ArrayConfig | None = ...) -> list[Any]: ...
