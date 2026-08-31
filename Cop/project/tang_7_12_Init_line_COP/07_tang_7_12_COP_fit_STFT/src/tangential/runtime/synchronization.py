"""压力与六维力帧的时间同步薄适配层。

同步算法的唯一实现仍在 ``TimestampedBuffer``；本模块只提供面向运行时
代码的语义化入口，避免在会话和测试中复制匹配规则。
"""

from __future__ import annotations

from typing import Any

from ..acquisition.buffer import TimestampedBuffer, match_closest


def match_force_frame(
    force_buffer: TimestampedBuffer,
    pressure_t: float,
    max_time_diff_s: float,
    *,
    min_seq: int = -1,
) -> dict[str, Any] | None:
    """查找一个尚未使用且处于时间窗口内的六维力帧。

    Args:
        force_buffer: 六维力时间戳缓存。
        pressure_t: 压力帧接收时间，使用同一 ``perf_counter`` 时钟。
        max_time_diff_s: 最大允许绝对时间差，单位为秒。
        min_seq: 只考虑大于该序号的力帧。

    Returns:
        dict | None: 最近的未使用力帧；没有满足条件的帧时返回 ``None``。
    """
    return match_closest(
        force_buffer,
        pressure_t,
        max_time_diff_s,
        min_seq=min_seq,
    )


__all__ = ["match_force_frame"]
