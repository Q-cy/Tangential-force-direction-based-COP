"""采集进程之间的时间戳缓存和一对一同步工具。

本子包公开 ``TimestampedBuffer`` 与 ``match_closest``：前者负责带锁保存
带 ``seq``/时间戳的数据帧并按顺序消费，后者提供按时间窗口寻找最近未用
帧的薄适配。串口协议和传感器生命周期属于 ``tangential.sensors``，本包
不负责读取硬件、运行 Qt 或写 CSV。
"""

from .buffer import TimestampedBuffer, match_closest

__all__ = ["TimestampedBuffer", "match_closest"]
