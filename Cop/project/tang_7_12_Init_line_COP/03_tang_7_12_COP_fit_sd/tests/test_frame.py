"""F2 覆盖: 压阻帧解析 (CRC-8 校验 + 对齐滑字节) 与力帧解析 (28B 帧头尾 + 粘包滑字节)"""
import os
import struct
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import PressureSensor, SixAxisForceSensor


def _make_press_frame(payload: bytes) -> bytes:
    """构造 183B 合法压阻帧: AA55 头 + buf[13]=0 + 168B payload + CRC-8-ITU"""
    assert len(payload) == 168
    header = bytearray(14)
    header[0:2] = b'\xaa\x55'
    header[13] = 0
    body = bytes(header) + payload
    return body + bytes([PressureSensor.crc8_itu(body)])


def _make_force_frame(f6) -> bytes:
    """构造 28B 合法力帧: 49 AA + 6×float32 + 0D 0A"""
    return b'\x49\xaa' + b''.join(struct.pack('<f', v) for v in f6) + b'\x0d\x0a'


def _press_sensor() -> PressureSensor:
    s = PressureSensor.__new__(PressureSensor)
    s._rx_buf = bytearray()
    s._rx_lock = threading.Lock()
    return s


def _force_sensor():
    s = SixAxisForceSensor.__new__(SixAxisForceSensor)
    s.zero_data = [0.0] * 6
    s._rx_buf = bytearray()
    s._rx_lock = threading.Lock()
    return s


def test_crc8_known_vector():
    # CRC-8/ITU 标准校验值: poly 0x07, init 0x00, xorout 0x55 → b"123456789" = 0xA1
    assert PressureSensor.crc8_itu(b"123456789") == 0xA1


def test_crc8_detects_bit_flip():
    frame = _make_press_frame(bytes(168))
    bad = bytearray(frame)
    bad[50] ^= 0x01
    assert PressureSensor.crc8_itu(bytes(bad[:182])) != bad[182]


def test_press_parse_valid_frame():
    s = _press_sensor()
    payload = bytes(range(168))
    s._rx_buf.extend(_make_press_frame(payload))
    assert s.read_data() == payload
    assert len(s._rx_buf) == 0


def test_press_parse_crc_failure_drops_frame():
    s = _press_sensor()
    frame = bytearray(_make_press_frame(bytes(168)))
    frame[100] ^= 0xFF   # 破坏 payload → CRC 错
    s._rx_buf.extend(frame)
    assert s.read_data() is None
    assert len(s._rx_buf) == 0   # 整段错误帧被丢弃


def test_press_parse_skips_garbage_then_syncs():
    s = _press_sensor()
    payload = bytes(range(168))
    s._rx_buf.extend(b'\xde\xad\xbe\xef' + _make_press_frame(payload))
    assert s.read_data() == payload
    assert len(s._rx_buf) == 0


def test_press_parse_partial_frame_returns_none():
    s = _press_sensor()
    s._rx_buf.extend(_make_press_frame(bytes(168))[:50])
    assert s.read_data() is None
    assert len(s._rx_buf) == 50   # 字节保留, 等补齐


def test_force_parse_valid():
    s = _force_sensor()
    raw = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    assert s._parse_frame(_make_force_frame(raw)) == [round(v * 9.8, 2) for v in raw]


def test_force_parse_subtracts_zero():
    s = _force_sensor()
    s.zero_data = [9.8, 0.0, 0.0, 0.0, 0.0, 0.0]   # 1.0 N 偏置
    out = s._parse_frame(_make_force_frame([1.0, 2.0, 3.0, 0.1, 0.2, 0.3]))
    assert out[0] == 0.0   # 9.8 - 9.8


def test_force_pop_frame_sticky_and_garbage():
    s = _force_sensor()
    f1 = _make_force_frame([1, 2, 3, 4, 5, 6])
    f2 = _make_force_frame([7, 8, 9, 10, 11, 12])
    s._rx_buf.extend(f1 + f2)                 # 粘包: 两帧连续
    assert s._try_pop_frame() == f1
    assert s._try_pop_frame() == f2
    s._rx_buf.extend(b'\x01\x02\x03' + f1)    # 垃圾前缀
    assert s._try_pop_frame() == f1
    assert s._try_pop_frame() is None


def test_force_pop_frame_partial():
    s = _force_sensor()
    s._rx_buf.extend(_make_force_frame([1, 2, 3, 4, 5, 6])[:10])
    assert s._try_pop_frame() is None
    assert len(s._rx_buf) == 10   # 不丢字节
