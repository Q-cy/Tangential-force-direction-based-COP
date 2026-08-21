"""双路完整 GUI 示例、配置和生命周期测试。"""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tangential.config import ForceConfig, FullApplicationConfig, GuiConfig, OutputConfig
from tangential.examples.dual_sensor import (
    _build_parser,
    build_configs_from_args,
    build_config,
    run,
)
from tangential.runtime import session as session_module
from tangential.runtime.session import DualApplicationRunner


class DualSensorExampleTests(unittest.TestCase):
    """验证双路完整应用不会共享设备、状态或输出目录。"""

    def _configs(self, root: str):
        """创建两份禁用力通道的隔离完整配置。"""
        return (
            build_config(
                pressure_port="/dev/pressure-a",
                force_port=None,
                save_dir=Path(root) / "sensor_a",
                model_path=None,
                window_title="Sensor A",
            ),
            build_config(
                pressure_port="/dev/pressure-b",
                force_port=None,
                save_dir=Path(root) / "sensor_b",
                model_path=None,
                window_title="Sensor B",
            ),
        )

    def test_cli_builds_full_configs_with_separate_default_directories(self):
        """未提供力端口时两路完整配置都明确禁用力传感器。"""
        args = _build_parser().parse_args([
            "--port-a", "/dev/a", "--port-b", "/dev/b",
            "--save-dir", "/tmp/dual-output",
        ])
        config_a, config_b = build_configs_from_args(args)
        self.assertFalse(config_a.force.enabled)
        self.assertFalse(config_b.force.enabled)
        self.assertEqual(config_a.output.save_dir, "/tmp/dual-output/sensor_a")
        self.assertEqual(config_b.output.save_dir, "/tmp/dual-output/sensor_b")
        self.assertEqual(config_a.gui.window_title, "Sensor A")
        self.assertEqual(config_b.gui.window_title, "Sensor B")

    def test_optional_force_ports_enable_two_independent_force_configs(self):
        """提供两路不同力端口时才启用对应力通道。"""
        args = _build_parser().parse_args([
            "--port-a", "/dev/a", "--port-b", "/dev/b",
            "--force-port-a", "/dev/force-a", "--force-port-b", "/dev/force-b",
        ])
        config_a, config_b = build_configs_from_args(args)
        self.assertTrue(config_a.force.enabled)
        self.assertTrue(config_b.force.enabled)
        self.assertEqual(config_a.force.port, "/dev/force-a")
        self.assertEqual(config_b.force.port, "/dev/force-b")

    def test_same_physical_pressure_or_force_port_is_rejected(self):
        """相同压力端口和相同启用力端口都在打开设备前拒绝。"""
        with tempfile.TemporaryDirectory() as root:
            config_a, config_b = self._configs(root)
            config_b.pressure.port = "/dev/pressure-a"
            with self.assertRaisesRegex(ValueError, "压力串口"):
                DualApplicationRunner(config_a, config_b)

            config_a, config_b = self._configs(root)
            config_a.force = ForceConfig(enabled=True, port="/dev/force")
            config_b.force = ForceConfig(enabled=True, port="/dev/force")
            with self.assertRaisesRegex(ValueError, "力串口"):
                DualApplicationRunner(config_a, config_b)

    def test_run_delegates_to_public_dual_application_entry(self):
        """示例只组装配置并委托给公共双路入口，不复制采集循环。"""
        with tempfile.TemporaryDirectory() as root:
            config_a, config_b = self._configs(root)
            observed = {}

            def fake_runner(first, second):
                observed["configs"] = (first, second)
                return 0

            self.assertEqual(run(config_a, config_b, runner=fake_runner), 0)
            self.assertEqual(observed["configs"], (config_a, config_b))
            self.assertEqual(config_a.gui.window_title, "Sensor A")
            self.assertEqual(config_b.gui.window_title, "Sensor B")

    def test_run_preserves_custom_window_titles(self):
        """示例层不得覆盖代码传入的可调 GUI 标题。"""
        with tempfile.TemporaryDirectory() as root:
            config_a, config_b = self._configs(root)
            config_a.gui.window_title = "Left Hand"
            config_b.gui.window_title = "Right Hand"
            observed = {}

            def fake_runner(first, second):
                observed["titles"] = (
                    first.gui.window_title,
                    second.gui.window_title,
                )
                return 0

            self.assertEqual(run(config_a, config_b, runner=fake_runner), 0)
            self.assertEqual(observed["titles"], ("Left Hand", "Right Hand"))


class _FakePlot:
    """不依赖 Qt 的完整窗口替身。"""

    def __init__(self, config=None):
        self.config = config
        self.statuses = []
        self.analysis_dirs = []

    def set_status(self, status):
        self.statuses.append(status)

    def plot_full_analysis(self, save_dir):
        self.analysis_dirs.append(save_dir)


class _FakeTimer:
    """只记录定时器生命周期的 Qt 替身。"""

    def __init__(self):
        self.callback = None

    class _Signal:
        def __init__(self, owner):
            self.owner = owner

        def connect(self, callback):
            self.owner.callback = callback

    @property
    def timeout(self):
        return self._signal

    def start(self, interval):
        self._signal = getattr(self, "_signal", self._Signal(self))

    def stop(self):
        pass


class DualRunnerLifecycleTests(unittest.TestCase):
    """验证一个 QApplication 管理两个 plot、线程和停止事件。"""

    def test_two_plots_and_workers_are_joined_and_analyzed_independently(self):
        """正常退出时两路都启动、停止、join 并生成各自分析图。"""
        with tempfile.TemporaryDirectory() as root:
            config_a = FullApplicationConfig(
                output=OutputConfig(save_dir=str(Path(root) / "a")),
                gui=GuiConfig(window_title="Sensor A"),
                force=ForceConfig(enabled=False),
            )
            config_b = FullApplicationConfig(
                pressure_port="/dev/b",
                output=OutputConfig(save_dir=str(Path(root) / "b")),
                gui=GuiConfig(window_title="Sensor B"),
                force=ForceConfig(enabled=False),
            )
            started = [threading.Event(), threading.Event()]
            stopped = [threading.Event(), threading.Event()]
            plots = []

            class App:
                @staticmethod
                def instance():
                    return None

                def __init__(self, argv):
                    self.quit_called = False

                def exec(self):
                    self.started = started
                    time.sleep(0.03)

                def quit(self):
                    self.quit_called = True

            def plot_factory(config=None):
                plot = _FakePlot(config)
                plots.append(plot)
                return plot

            def worker(plot, stop_event, config):
                index = 0 if config is config_a else 1
                started[index].set()
                stop_event.wait(1)
                stopped[index].set()

            class TimerFactory(_FakeTimer):
                def __init__(self):
                    super().__init__()
                    self._signal = self._Signal(self)

            with mock.patch.object(session_module.QtWidgets, "QApplication", App), \
                 mock.patch.object(session_module.QtCore, "QTimer", TimerFactory):
                DualApplicationRunner(
                    config_a,
                    config_b,
                    worker_target=worker,
                    plot_factory=plot_factory,
                ).run()

            self.assertEqual(len(plots), 2)
            self.assertTrue(all(event.is_set() for event in started))
            self.assertTrue(all(event.is_set() for event in stopped))
            self.assertEqual(
                [plot.analysis_dirs for plot in plots],
                [[config_a.save_dir], [config_b.save_dir]],
            )

    def test_worker_error_sets_both_stop_events(self):
        """任一路后台异常都会让另一路停止并完成清理。"""
        with tempfile.TemporaryDirectory() as root:
            config_a = FullApplicationConfig(
                output=OutputConfig(save_dir=str(Path(root) / "a")),
                force=ForceConfig(enabled=False),
            )
            config_b = FullApplicationConfig(
                pressure_port="/dev/b",
                output=OutputConfig(save_dir=str(Path(root) / "b")),
                force=ForceConfig(enabled=False),
            )
            other_stopped = threading.Event()

            class App:
                @staticmethod
                def instance():
                    return None

                def __init__(self, argv):
                    pass

                def exec(self):
                    time.sleep(0.03)

                def quit(self):
                    pass

            def worker(plot, stop_event, config):
                if config is config_a:
                    raise OSError("A serial disconnected")
                stop_event.wait(1)
                other_stopped.set()

            class TimerFactory(_FakeTimer):
                def __init__(self):
                    super().__init__()
                    self._signal = self._Signal(self)

            with mock.patch.object(session_module.QtWidgets, "QApplication", App), \
                 mock.patch.object(session_module.QtCore, "QTimer", TimerFactory), \
                 self.assertRaisesRegex(RuntimeError, "Sensor A 数据线程异常"):
                DualApplicationRunner(
                    config_a,
                    config_b,
                    worker_target=worker,
                    plot_factory=_FakePlot,
                ).run()

            self.assertTrue(other_stopped.is_set())


if __name__ == "__main__":
    unittest.main()
