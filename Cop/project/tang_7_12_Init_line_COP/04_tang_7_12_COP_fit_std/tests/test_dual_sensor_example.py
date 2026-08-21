"""双压力传感器示例的并发和资源隔离测试。"""

from __future__ import annotations

import io
import threading
import unittest
from types import SimpleNamespace

from tangential.config import PressureConfig
from tangential.examples.dual_sensor import run


class FakeSensorAPI:
    """要求两路 read 同时到达屏障的假压力API。"""

    barrier = threading.Barrier(2)
    instances: list["FakeSensorAPI"] = []

    def __init__(self, *, config, model_path=None):
        """记录每个实例自己的配置和模型参数。"""
        self.config = config
        self.model_path = model_path
        self.closed = False
        self.read_thread = None
        type(self).instances.append(self)

    def __enter__(self):
        """返回当前假API实例。"""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """记录上下文退出，模拟幂等资源关闭。"""
        self.closed = True

    def read(self, timeout_s=0.1):
        """等待另一路读取线程，并返回带当前端口标识的样本。"""
        self.read_thread = threading.get_ident()
        type(self).barrier.wait(timeout=1.0)
        value = 1.0 if self.config.port.endswith("0") else 2.0
        return SimpleNamespace(
            request_seq=0,
            total=value,
            cop_x=value,
            cop_y=value + 0.5,
            angle=value * 10.0,
        )


class DualPressureExampleTests(unittest.TestCase):
    """验证两路配置、线程和生命周期互相独立。"""

    def setUp(self):
        """为每个测试重置假实例和两方屏障。"""
        FakeSensorAPI.instances.clear()
        FakeSensorAPI.barrier = threading.Barrier(2)

    def test_two_sensors_use_independent_configs_threads_and_cleanup(self):
        """两路读取并发执行，配置不串线且退出时全部关闭。"""
        output = io.StringIO()
        result = run(
            PressureConfig(port="/dev/fake0", target_hz=120),
            PressureConfig(port="/dev/fake1", target_hz=180),
            model_path="model.bin",
            stream=output,
            sensor_factory=FakeSensorAPI,
            max_iterations=1,
        )

        self.assertEqual(result, 0)
        self.assertEqual(len(FakeSensorAPI.instances), 2)
        first, second = FakeSensorAPI.instances
        self.assertEqual(first.config.port, "/dev/fake0")
        self.assertEqual(second.config.port, "/dev/fake1")
        self.assertEqual(first.config.target_hz, 120)
        self.assertEqual(second.config.target_hz, 180)
        self.assertNotEqual(first.read_thread, second.read_thread)
        self.assertTrue(first.closed)
        self.assertTrue(second.closed)
        self.assertIn("A(/dev/fake0)", output.getvalue())
        self.assertIn("B(/dev/fake1)", output.getvalue())

    def test_same_physical_port_is_rejected_before_opening(self):
        """同一路径不能被两个压力实例重复打开。"""
        with self.assertRaisesRegex(ValueError, "同一个物理串口"):
            run(
                PressureConfig(port="/dev/fake0"),
                PressureConfig(port="/dev/fake0"),
                sensor_factory=FakeSensorAPI,
                max_iterations=1,
            )
        self.assertEqual(FakeSensorAPI.instances, [])


if __name__ == "__main__":
    unittest.main()
