import os
import ctypes
from ctypes import (
    Structure, POINTER, c_void_p, c_char, c_char_p, c_uint8, c_uint16,
    c_uint32, c_uint64, c_bool
)

LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libeskin_finger_sdk.so")

class EskinSdkVersion(Structure):
    _fields_ = [
        ("major", c_uint16),
        ("minor", c_uint16),
        ("patch", c_uint16),
    ]


class CForce3D(Structure):
    _fields_ = [
        ("fx", c_uint32),
        ("fy", c_uint32),
        ("fz", c_uint32),
    ]


class CCombinedForce(Structure):
    _fields_ = [
        ("module", c_uint32),
        ("force", CForce3D),
    ]


class CFingerSample(Structure):
    _fields_ = [
        ("timestamp_us", c_uint64),
        ("sequence", c_uint32),
        ("combined_force", CCombinedForce),
    ]


class EskinDevice:
    def __init__(self):
        self._lib = ctypes.CDLL(LIB_PATH)
        self._setup_functions()
        self._handle = None

    def _setup_functions(self):
        lib = self._lib

        lib.eskin_version.restype = EskinSdkVersion
        lib.eskin_version.argtypes = []

        lib.eskin_open.restype = c_void_p
        lib.eskin_open.argtypes = [c_char_p, c_void_p]

        lib.eskin_close.restype = c_uint32
        lib.eskin_close.argtypes = [c_void_p]

        lib.eskin_read_register.restype = c_uint32
        lib.eskin_read_register.argtypes = [
            c_void_p, c_uint32, c_uint16,
            POINTER(c_uint8), c_uint32, POINTER(c_uint32)
        ]

        lib.eskin_write_register.restype = c_uint32
        lib.eskin_write_register.argtypes = [
            c_void_p, c_uint32, POINTER(c_uint8), c_uint16, POINTER(c_uint16)
        ]

        # EskinDeviceFunc bindings

        lib.eskin_read_hdw_version.restype = c_uint32
        lib.eskin_read_hdw_version.argtypes = [
            c_void_p, POINTER(c_char), c_uint32, POINTER(c_uint32)
        ]

        lib.eskin_read_matrix_row.restype = c_uint32
        lib.eskin_read_matrix_row.argtypes = [
            c_void_p, POINTER(c_uint8)
        ]

        lib.eskin_read_matrix_col.restype = c_uint32
        lib.eskin_read_matrix_col.argtypes = [
            c_void_p, POINTER(c_uint8)
        ]

        lib.eskin_read_device_config1.restype = c_uint32
        lib.eskin_read_device_config1.argtypes = [
            c_void_p, POINTER(c_uint8)
        ]

        lib.eskin_read_device_config2.restype = c_uint32
        lib.eskin_read_device_config2.argtypes = [
            c_void_p, POINTER(c_uint8)
        ]

        lib.eskin_write_device_config1.restype = c_uint32
        lib.eskin_write_device_config1.argtypes = [
            c_void_p, c_bool, POINTER(c_uint16)
        ]

        lib.eskin_write_device_config2.restype = c_uint32
        lib.eskin_write_device_config2.argtypes = [
            c_void_p, c_bool, POINTER(c_uint16)
        ]

        lib.eskin_write_matrix_row.restype = c_uint32
        lib.eskin_write_matrix_row.argtypes = [
            c_void_p, c_uint8, POINTER(c_uint16)
        ]

        lib.eskin_write_matrix_col.restype = c_uint32
        lib.eskin_write_matrix_col.argtypes = [
            c_void_p, c_uint8, POINTER(c_uint16)
        ]

        # Streaming bindings

        lib.eskin_start_stream.restype = c_uint32
        lib.eskin_start_stream.argtypes = [c_void_p]

        lib.eskin_stop_stream.restype = c_uint32
        lib.eskin_stop_stream.argtypes = [c_void_p]

        lib.eskin_read_sample.restype = c_uint32
        lib.eskin_read_sample.argtypes = [
            c_void_p, c_uint32, POINTER(CFingerSample)
        ]

        lib.eskin_read_stream_frame.restype = c_uint32
        lib.eskin_read_stream_frame.argtypes = [
            c_void_p, c_uint32, POINTER(c_uint8), c_uint32, POINTER(c_uint32)
        ]

        lib.eskin_get_mode.restype = c_uint32
        lib.eskin_get_mode.argtypes = [c_void_p, POINTER(c_uint32)]

    def version(self) -> tuple:
        v = self._lib.eskin_version()
        return (v.major, v.minor, v.patch)

    def open(self, path: str):
        handle = self._lib.eskin_open(path.encode("utf-8"), None)
        if not handle:
            raise RuntimeError(f"Failed to open device: {path}")
        self._handle = handle

    def close(self):
        if self._handle:
            self._lib.eskin_close(self._handle)
            self._handle = None

    def read_register(self, addr: int, length: int) -> bytes:
        """读寄存器，返回原始字节"""
        buf = (c_uint8 * 256)()
        actual = c_uint32(0)
        err = self._lib.eskin_read_register(
            self._handle, addr, length, buf, len(buf), ctypes.byref(actual)
        )
        if err != 0:
            raise RuntimeError(f"read_register failed: error={err}")
        return bytes(buf[:actual.value])

    def write_register(self, addr: int, data: bytes) -> int:
        """写寄存器，返回设备确认的字节数"""
        arr = (c_uint8 * len(data))(*data)
        ret = c_uint16(0)
        err = self._lib.eskin_write_register(
            self._handle, addr, arr, len(data), ctypes.byref(ret)
        )
        if err != 0:
            raise RuntimeError(f"write_register failed: error={err}")
        return ret.value

    def read_hdw_version(self) -> str:
        """读取硬件版本号"""
        buf = (c_char * 64)()
        actual = c_uint32(0)
        err = self._lib.eskin_read_hdw_version(
            self._handle, buf, len(buf), ctypes.byref(actual)
        )
        if err != 0:
            raise RuntimeError(f"read_hdw_version failed: error={err}")
        return buf[:actual.value].decode("utf-8")

    def read_matrix_row(self) -> int:
        """读取矩阵行数"""
        out = c_uint8(0)
        err = self._lib.eskin_read_matrix_row(self._handle, ctypes.byref(out))
        if err != 0:
            raise RuntimeError(f"read_matrix_row failed: error={err}")
        return out.value

    def read_matrix_col(self) -> int:
        """读取矩阵列数"""
        out = c_uint8(0)
        err = self._lib.eskin_read_matrix_col(self._handle, ctypes.byref(out))
        if err != 0:
            raise RuntimeError(f"read_matrix_col failed: error={err}")
        return out.value

    def read_device_config1(self) -> int:
        """读取设备配置寄存器1"""
        out = c_uint8(0)
        err = self._lib.eskin_read_device_config1(self._handle, ctypes.byref(out))
        if err != 0:
            raise RuntimeError(f"read_device_config1 failed: error={err}")
        return out.value

    def read_device_config2(self) -> int:
        """读取设备配置寄存器2"""
        out = c_uint8(0)
        err = self._lib.eskin_read_device_config2(self._handle, ctypes.byref(out))
        if err != 0:
            raise RuntimeError(f"read_device_config2 failed: error={err}")
        return out.value

    def write_device_config1(self, enable: bool) -> int:
        """写入设备配置寄存器1"""
        ret = c_uint16(0)
        err = self._lib.eskin_write_device_config1(
            self._handle, enable, ctypes.byref(ret)
        )
        if err != 0:
            raise RuntimeError(f"write_device_config1 failed: error={err}")
        return ret.value

    def write_device_config2(self, enable: bool) -> int:
        """写入设备配置寄存器2"""
        ret = c_uint16(0)
        err = self._lib.eskin_write_device_config2(
            self._handle, enable, ctypes.byref(ret)
        )
        if err != 0:
            raise RuntimeError(f"write_device_config2 failed: error={err}")
        return ret.value

    def write_matrix_row(self, row: int) -> int:
        """写入矩阵行数"""
        ret = c_uint16(0)
        err = self._lib.eskin_write_matrix_row(
            self._handle, row, ctypes.byref(ret)
        )
        if err != 0:
            raise RuntimeError(f"write_matrix_row failed: error={err}")
        return ret.value

    def write_matrix_col(self, col: int) -> int:
        """写入矩阵列数"""
        ret = c_uint16(0)
        err = self._lib.eskin_write_matrix_col(
            self._handle, col, ctypes.byref(ret)
        )
        if err != 0:
            raise RuntimeError(f"write_matrix_col failed: error={err}")
        return ret.value

    # Streaming methods

    def start_stream(self):
        """启动流式采集"""
        err = self._lib.eskin_start_stream(self._handle)
        if err != 0:
            raise RuntimeError(f"start_stream failed: error={err}")

    def stop_stream(self):
        """停止流式采集"""
        err = self._lib.eskin_stop_stream(self._handle)
        if err != 0:
            raise RuntimeError(f"stop_stream failed: error={err}")

    def read_sample(self, timeout_ms: int = 200) -> CFingerSample:
        """读取一个采样数据（流模式下调用）"""
        sample = CFingerSample()
        err = self._lib.eskin_read_sample(self._handle, timeout_ms, ctypes.byref(sample))
        if err != 0:
            raise RuntimeError(f"read_sample failed: error={err}")
        return sample

    def read_stream_frame(self, timeout_ms: int = 200, max_len: int = 512) -> bytes:
        """读取一个 stream 原始完整协议帧（流模式下调用）"""
        buf = (c_uint8 * max_len)()
        actual = c_uint32(0)
        err = self._lib.eskin_read_stream_frame(
            self._handle, timeout_ms, buf, len(buf), ctypes.byref(actual)
        )
        if err != 0:
            raise RuntimeError(f"read_stream_frame failed: error={err}")
        return bytes(buf[:min(actual.value, len(buf))])

    def get_mode(self) -> int:
        """查询当前设备模式（0=Command, 1=Streaming）"""
        out = c_uint32(0)
        err = self._lib.eskin_get_mode(self._handle, ctypes.byref(out))
        if err != 0:
            raise RuntimeError(f"get_mode failed: error={err}")
        return out.value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
