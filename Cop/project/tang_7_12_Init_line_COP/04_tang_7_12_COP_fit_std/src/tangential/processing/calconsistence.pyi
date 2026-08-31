from typing import Any, Mapping
from pathlib import Path

from ..config import ConsistenceCalibrationConfig

DEFAULT_RESOURCE_NAME: str


class ConsistenceCalibrator:
    scale: Any
    offset: Any
    channel_count: int
    clip_min: float | None
    clip_max: float | None
    metadata: Mapping[str, Any]
    output_path: Path | None

    def __init__(self, scale: Any, offset: Any, *,
                 clip_min: float | None = ..., clip_max: float | None = ...,
                 metadata: Mapping[str, Any] | None = ...,
                 output_path: str | Path | None = ...) -> None: ...
    @classmethod
    def fit_from_csv(cls, config: ConsistenceCalibrationConfig) -> "ConsistenceCalibrator": ...
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
    def apply(self, raw_data: Any) -> Any: ...
    def __call__(self, raw_data: Any) -> Any: ...
    def save(self, path: str | Path | None = ..., *, force: bool = False) -> Path: ...


def fit_consistence(config: ConsistenceCalibrationConfig) -> ConsistenceCalibrator: ...
def main() -> int: ...
