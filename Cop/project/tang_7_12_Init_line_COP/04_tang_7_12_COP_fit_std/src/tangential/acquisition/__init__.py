"""采集、缓存与同步。"""

from .buffer import TimestampedBuffer, match_closest

__all__ = ["TimestampedBuffer", "match_closest"]
