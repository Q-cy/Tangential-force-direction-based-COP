from enum import IntEnum
from os import PathLike

import numpy as np

from ..config import SpectrumConfig


class SpectralFrictionState(IntEnum):
    WAITING: int
    STICK: int
    SLIP: int


class SpectrumSnapshot:
    frequency_hz: np.ndarray
    spectrum_time_s: float
    velocity_amplitude_x: np.ndarray
    velocity_amplitude_y: np.ndarray
    velocity_amplitude_combined: np.ndarray
    baseline_power: np.ndarray
    relative_power_db: np.ndarray
    baseline_established: bool
    slip_band_power_ratio: float
    friction_state: SpectralFrictionState
    threshold: float
    revision: int
    @property
    def time_s(self) -> float: ...
    @property
    def state(self) -> SpectralFrictionState: ...
    @property
    def state_name(self) -> str: ...
    @property
    def amplitude_x(self) -> np.ndarray: ...
    @property
    def amplitude_y(self) -> np.ndarray: ...
    @property
    def amplitude_combined(self) -> np.ndarray: ...


class CopSpectrumAnalyzer:
    config: SpectrumConfig
    sample_rate_hz: float
    sample_period_s: float
    window_samples: int
    required_samples: int
    def __init__(self, config: SpectrumConfig | None = ...) -> None: ...
    @property
    def frequencies_hz(self) -> np.ndarray: ...
    @property
    def ready_samples(self) -> int: ...
    @property
    def snapshots(self) -> tuple[SpectrumSnapshot, ...]: ...
    @property
    def recent_snapshots(self) -> tuple[SpectrumSnapshot, ...]: ...
    def get_snapshots(self) -> tuple[SpectrumSnapshot, ...]: ...
    def get_recent_snapshots(self) -> tuple[SpectrumSnapshot, ...]: ...
    @property
    def current_snapshot(self) -> SpectrumSnapshot | None: ...
    @property
    def friction_state(self) -> SpectralFrictionState: ...
    @property
    def baseline_power(self) -> np.ndarray | None: ...
    def reset(self, *, reset_time_origin: bool = ...) -> None: ...
    def process(self, rx_t: float, cop_x: float, cop_y: float, state: int) -> SpectrumSnapshot | None: ...
    def save_npz(self, path: str | PathLike[str], csv_file_name: str = ...) -> bool: ...
