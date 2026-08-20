"""线程安全的时间戳缓存与一对一最近帧匹配。

本模块只负责带序号帧的缓存和时间匹配，不负责传感器协议解析、数据
转换或 CSV 写入。
"""

import threading
from collections import deque


# ===================== 带时间戳的线程安全缓存 =====================
class TimestampedBuffer:
    """保存带时间戳帧并提供按序消费和最近时间匹配。

    Attributes:
        buf (collections.deque): 按追加顺序保存的帧；每帧会复制为字典并
            增加本缓存内部的 ``seq`` 字段。
        lock (threading.Lock): 保护 ``buf`` 和 ``_next_seq`` 的互斥锁。
        _next_seq (int): 下一帧将使用的缓存内序号。
    """

    def __init__(self, maxlen=500):
        """初始化有界线程安全缓存。

        Args:
            maxlen (int): 最多保留的帧数；传给 ``collections.deque``，超过
                容量时自动淘汰最旧帧。

        Returns:
            None: 构造函数只初始化对象状态。

        Raises:
            TypeError: ``maxlen`` 不能被 ``deque`` 接受时抛出。
            ValueError: ``maxlen`` 为非法容量时由 ``deque`` 抛出。

        Side Effects:
            创建缓存队列和线程锁；不会启动线程，也不会复制外部数据。
        """
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self._next_seq = 0

    def append(self, item):
        """复制并追加一帧，同时分配缓存内单调递增的序号。

        Args:
            item (Mapping): 待缓存帧。通常至少包含 ``t`` 时间戳字段；函数
                会浅复制为字典，不修改调用方传入的映射。

        Returns:
            int: 本次追加帧获得的 ``seq``。

        Raises:
            TypeError: ``item`` 不能转换为字典时抛出。

        Side Effects:
            在锁保护下修改缓存和下一序号；队列已满时由 ``deque`` 淘汰最旧
            帧。
        """
        with self.lock:
            stored = dict(item)
            stored["seq"] = self._next_seq
            self._next_seq += 1
            self.buf.append(stored)
            return stored["seq"]

    def get_latest(self):
        """返回当前缓存中的最新帧引用。

        Args:
            None: 此方法不接收业务参数。

        Returns:
            dict | None: 最新缓存字典；缓存为空时返回 ``None``。返回的是
                字典对象引用，调用方应将其视为只读。

        Side Effects:
            只在锁内读取，不改变缓存内容。
        """
        with self.lock:
            return self.buf[-1] if self.buf else None

    def get_after(self, seq):
        """按 ``seq`` 升序返回尚未处理的帧列表。

        Args:
            seq (int): 已处理的最后缓存序号；只返回 ``item["seq"] > seq``
                的帧。

        Returns:
            list[dict]: 当前缓存中满足条件的帧字典列表；没有符合项时为空
                列表。列表是在锁内创建的快照。

        Side Effects:
            不推进消费游标，也不删除或修改缓存中的帧。
        """
        with self.lock:
            return [item for item in self.buf if item["seq"] > seq]

    def find_closest(self, ts, max_diff_s=None, min_seq=-1):
        """返回符合序号条件且时间最接近目标的帧。

        ``min_seq`` 是已消费的最后序号，候选帧必须满足
        ``seq > min_seq``。此方法只查找，不会标记帧已使用；调用方需自行
        保存返回帧的序号以保证一对一消费。

        Args:
            ts (float): 目标时间戳，单位为秒，必须与帧的 ``t`` 使用同一时钟。
            max_diff_s (float | None): 允许的最大绝对时间差，单位为秒；为
                ``None`` 时不限制时间窗口。
            min_seq (int): 候选帧的最小序号下界；默认 ``-1`` 表示允许所有
                非负序号帧。

        Returns:
            dict | None: 时间差最小的候选帧；没有候选帧，或最近帧超出
                ``max_diff_s`` 时返回 ``None``。

        Side Effects:
            只读访问缓存，不改变缓存或帧的使用状态。
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
    """在缓存中查找满足严格时间窗口的一对一匹配候选。

    Args:
        buf (TimestampedBuffer): 要查询的时间戳缓存。
        ts (float): 压力帧或其他基准帧的目标时间，单位为秒。
        max_diff_s (float): 允许的最大绝对时间差，单位为秒；超过该窗口
            返回 ``None``。
        min_seq (int): 已使用的最后缓存序号；默认 ``-1``。

    Returns:
        dict | None: 与 ``ts`` 最近且序号大于 ``min_seq`` 的帧，或无匹配时
        的 ``None``。本函数本身不标记帧已使用。

    Raises:
        AttributeError: ``buf`` 不具备 ``find_closest`` 方法时抛出。
    """
    return buf.find_closest(ts, max_diff_s=max_diff_s, min_seq=min_seq)
