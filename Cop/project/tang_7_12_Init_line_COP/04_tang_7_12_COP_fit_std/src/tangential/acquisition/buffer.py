"""线程安全的时间戳缓存与一对一最近帧匹配。"""

import threading
from collections import deque


# ===================== 带时间戳的线程安全缓存 =====================
class TimestampedBuffer:
    def __init__(self, maxlen=500):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self._next_seq = 0

    def append(self, item):
        """追加一帧并自动赋予当前 buffer 内单调递增的 seq。"""
        with self.lock:
            stored = dict(item)
            stored["seq"] = self._next_seq
            self._next_seq += 1
            self.buf.append(stored)
            return stored["seq"]

    def get_latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def get_after(self, seq):
        """按 seq 顺序返回尚未处理的帧快照。"""
        with self.lock:
            return [item for item in self.buf if item["seq"] > seq]

    def find_closest(self, ts, max_diff_s=None, min_seq=-1):
        """返回未使用帧中时间最接近 ts 的一帧。

        min_seq 是已消费的最后序号，候选帧必须满足 seq > min_seq。
        max_diff_s 非 None 时，超出时间窗口直接返回 None。
        """
        with self.lock:
            best = None
            best_dt = float("inf")
            for item in self.buf:
                if item["seq"] <= min_seq:
                    continue
                dt = abs(item["t"] - ts)
                if dt < best_dt:
                    best_dt = dt
                    best = item
            if best is not None and max_diff_s is not None and best_dt > max_diff_s:
                return None
            return best


def match_closest(buf: "TimestampedBuffer", ts: float, max_diff_s: float,
                  min_seq: int = -1):
    """在 buf 中找与 ts 最接近的帧; 时间差超过 max_diff_s 返回 None (F5 严格同步)"""
    return buf.find_closest(ts, max_diff_s=max_diff_s, min_seq=min_seq)
