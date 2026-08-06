"""F5 覆盖: TimestampedBuffer.find_closest + match_closest 15ms 窗口配对"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import TimestampedBuffer, match_closest


def test_find_closest_empty():
    buf = TimestampedBuffer()
    assert buf.find_closest(1.0) is None
    assert match_closest(buf, 1.0, 0.015) is None


def test_find_closest_picks_nearest():
    buf = TimestampedBuffer()
    buf.append({"t": 1.000, "data": [0]})
    buf.append({"t": 1.020, "data": [1]})
    buf.append({"t": 1.040, "data": [2]})
    assert buf.find_closest(1.022)["data"] == [1]
    assert buf.find_closest(1.039)["data"] == [2]


def test_match_closest_within_window():
    buf = TimestampedBuffer()
    buf.append({"t": 1.000, "data": [0]})
    buf.append({"t": 1.010, "data": [1]})
    assert match_closest(buf, 1.015, 0.015)["data"] == [1]    # dt=5ms ✓
    assert match_closest(buf, 1.040, 0.015) is None           # dt=30ms 超窗
    assert match_closest(buf, 0.987, 0.015)["data"] == [0]    # dt=13ms ✓
    assert match_closest(buf, 0.975, 0.015) is None           # dt=25ms 超窗


def test_dual_stream_pairing():
    """模拟 data_loop 配对: press 帧率 83Hz (12ms), 力帧率 100Hz, 偏差 4ms"""
    press_buf = TimestampedBuffer()
    for i in range(5):
        press_buf.append({"t": i * 0.012, "data": [i]})   # 0, 12, 24, 36, 48 ms
    force_ts = 0.030                                       # 位于 press 24 与 36ms 之间
    matched = match_closest(press_buf, force_ts, 0.015)
    assert matched is not None and matched["t"] == 0.024   # dt=6ms ✓
    assert match_closest(press_buf, force_ts, 0.005) is None   # dt=6ms > 5ms 超窗
    # 力帧停滞: 最新力帧远早于最新压阻帧 → 超窗, 行被跳过
    assert match_closest(press_buf, 0.100, 0.015) is None
