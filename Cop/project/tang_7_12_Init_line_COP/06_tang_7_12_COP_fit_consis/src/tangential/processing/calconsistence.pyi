from typing import Any, Mapping
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..config import ConsistenceCalibrationConfig

CHANNEL_COUNT: int
FORMAT_VERSION: int
DEFAULT_RESOURCE_NAME: str


class ConsistenceCalibrator:
    input_breakpoints: NDArray[np.float64]
    target_breakpoints: NDArray[np.float64]
    segment_scale: NDArray[np.float64]
    segment_offset: NDArray[np.float64]
    segment_values: NDArray[np.float64]
    clip_min: float | None
    clip_max: float | None
    metadata: Mapping[str, Any]
    output_path: Path | None

    def __init__(self, input_breakpoints: Any, target_breakpoints: Any,
                 segment_scale: Any, segment_offset: Any, *,
                 segment_values: Any | None = ...,
                 clip_min: float | None = ..., clip_max: float | None = ...,
                 metadata: Mapping[str, Any] | None = ...,
                 output_path: str | Path | None = ...) -> None: ...
    @classmethod
    def fit_from_directory(cls, config: ConsistenceCalibrationConfig) -> "ConsistenceCalibrator": ...
    @classmethod
    def fit(cls, config: ConsistenceCalibrationConfig) -> "ConsistenceCalibrator": ...
    @classmethod
    def from_path(cls, path: str | Path, *, clip_min: float | None = ...,
                  clip_max: float | None = ...) -> "ConsistenceCalibrator": ...
    @classmethod
    def from_default(cls, *, clip_min: float | None = ...,
                     clip_max: float | None = ...) -> "ConsistenceCalibrator": ...
    @classmethod
    def from_config(cls, config: ConsistenceCalibrationConfig) -> "ConsistenceCalibrator": ...
    def apply(self, raw_data: Any) -> NDArray[np.float64]: ...
    def __call__(self, raw_data: Any) -> NDArray[np.float64]: ...
    def save(self, path: str | Path | None = ..., *, force: bool = False) -> Path: ...


def fit_consistence(config: ConsistenceCalibrationConfig) -> ConsistenceCalibrator: ...
def main() -> int: ...
