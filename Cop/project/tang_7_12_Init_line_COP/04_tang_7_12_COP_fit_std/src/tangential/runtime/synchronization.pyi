from typing import Any
from ..acquisition.buffer import TimestampedBuffer

def match_force_frame(force_buffer: TimestampedBuffer, pressure_t: float,
                      max_time_diff_s: float, *,
                      min_seq: int = ...) -> dict[str, Any] | None: ...
