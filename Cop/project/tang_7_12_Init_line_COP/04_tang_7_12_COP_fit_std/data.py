"""旧版 data 模块兼容层；规范实现位于 tangential 包。"""

import multiprocessing
import os
import queue
import select
import serial
import struct
import threading
import time

try:
    from tangential.acquisition.buffer import TimestampedBuffer, match_closest
    from tangential.sensors.force import (
        DATA_BAUDRATE_FORCE,
        FORCE_FRAME_QUEUE_SIZE,
        FORCE_PERIOD_S,
        FORCE_RESPONSE_TIMEOUT_S,
        FORCE_SENSOR_PORT,
        FORCE_TARGET_HZ,
        SixAxisForceSensor,
        _force_process_main,
    )
    from tangential.sensors.pressure import (
        DATA_BAUDRATE_PRESS,
        PRESSURE_FRAME_QUEUE_SIZE,
        PRESSURE_PERIOD_S,
        PRESSURE_RESPONSE_TIMEOUT_S,
        PRESSURE_SENSOR_PORT,
        PRESSURE_TARGET_HZ,
        PressureSensor,
        _pressure_process_main,
    )
except ModuleNotFoundError:
    from src.tangential.acquisition.buffer import TimestampedBuffer, match_closest
    from src.tangential.sensors.force import (
        DATA_BAUDRATE_FORCE,
        FORCE_FRAME_QUEUE_SIZE,
        FORCE_PERIOD_S,
        FORCE_RESPONSE_TIMEOUT_S,
        FORCE_SENSOR_PORT,
        FORCE_TARGET_HZ,
        SixAxisForceSensor,
        _force_process_main,
    )
    from src.tangential.sensors.pressure import (
        DATA_BAUDRATE_PRESS,
        PRESSURE_FRAME_QUEUE_SIZE,
        PRESSURE_PERIOD_S,
        PRESSURE_RESPONSE_TIMEOUT_S,
        PRESSURE_SENSOR_PORT,
        PRESSURE_TARGET_HZ,
        PressureSensor,
        _pressure_process_main,
    )

__all__ = [
    "PressureSensor", "SixAxisForceSensor", "TimestampedBuffer", "match_closest",
    "DATA_BAUDRATE_PRESS", "DATA_BAUDRATE_FORCE",
    "PRESSURE_SENSOR_PORT", "FORCE_SENSOR_PORT",
    "PRESSURE_TARGET_HZ", "PRESSURE_PERIOD_S",
    "PRESSURE_RESPONSE_TIMEOUT_S", "PRESSURE_FRAME_QUEUE_SIZE",
    "FORCE_TARGET_HZ", "FORCE_PERIOD_S",
    "FORCE_RESPONSE_TIMEOUT_S", "FORCE_FRAME_QUEUE_SIZE",
]
