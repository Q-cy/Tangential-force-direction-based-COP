"""完整采集应用的线程、会话、同步、CSV 和 GUI 辅助层。

本模块依赖可选的 Qt/PyQtGraph，仅在运行完整 GUI 应用时使用。压力和六维
力分别由各自的传感器对象读取，父线程通过时间戳缓存按压力帧顺序处理。
"""

import inspect
import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from ..acquisition.buffer import TimestampedBuffer
from .sensor import TangentialSampleProcessor, compute_vector_angle
from ..config import FullApplicationConfig
from ..gui.realtime import RealTimePlot
from ..processing.calibration import FitCalibrationModel
from ..processing.cop import PRSensorAngle
from ..sensors.force import SixAxisForceSensor
from ..sensors.pressure import PressureSensor
from ..storage.csv import auto_get_csv_path, build_csv_row, init_csv_file
from .synchronization import match_force_frame


g_main_stop_flag = threading.Event()


def _construct_sensor(factory, port):
    """按工厂签名构造传感器并兼容无参测试工厂。

    Args:
        factory (callable): 传感器类或工厂；可能接受 ``port`` 关键字，或
            完全不接受端口参数。
        port (str): 要传给支持端口参数的工厂的串口路径。

    Returns:
        object: 工厂创建的传感器实例。

    Raises:
        Exception: 工厂签名检查失败且调用工厂失败时，工厂原始异常向上传播。
        TypeError: 工厂参数不兼容且回退调用仍失败时可能抛出。

    Side Effects:
        调用一次传感器工厂，可能打开串口或创建子进程；不修改工厂对象。
    """
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        parameters = signature.parameters.values()
        accepts_port = (
            "port" in signature.parameters
            or any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                   for parameter in parameters)
        )
        if accepts_port:
            return factory(port=port)
        return factory()
    try:
        return factory(port=port)
    except TypeError:
        return factory()


class PressureThread(threading.Thread):
    """从压力传感器读取合法帧并写入父进程时间戳缓存。

    Attributes:
        sensor (object): 提供 ``read_frame`` 和 ``decode`` 的压力传感器。
        buffer (TimestampedBuffer): 接收解码后压力帧的缓存。
        stop_event (threading.Event): 请求线程停止的事件。
        error (BaseException | None): 线程未处理异常；正常运行时为 ``None``。
    """

    def __init__(self, sensor, buffer, stop_event):
        """初始化压力消费线程，不立即启动线程。

        Args:
            sensor (object): 压力传感器实例。
            buffer (TimestampedBuffer): 压力帧目标缓存。
            stop_event (threading.Event): 线程循环使用的停止事件。

        Returns:
            None: 初始化线程对象和错误状态。

        Side Effects:
            调用 ``threading.Thread`` 初始化；不会读取串口或启动线程，需由
            调用方显式调用 ``start``。
        """
        super().__init__(daemon=True, name="pressure-consumer")
        self.sensor = sensor
        self.buffer = buffer
        self.stop_event = stop_event
        self.error = None

    def run(self):
        """持续读取、解码并缓存压力帧直到停止或发生异常。

        Returns:
            None: 线程结束后返回；单帧类型/长度错误会跳过该帧。

        Side Effects:
            调用传感器的 ``read_frame``/``decode``，向缓存追加带 ``t``、
            ``data``、请求序号和时序元数据的帧；未处理异常保存到 ``error``。
        """
        try:
            while not self.stop_event.is_set():
                frame = self.sensor.read_frame(timeout_s=0.1)
                if frame is None:
                    continue
                try:
                    data = np.asarray(
                        self.sensor.decode(frame["raw"]), dtype=np.float64
                    )
                    self.buffer.append({
                        "t": frame["rx_t"],
                        "data": data,
                        "request_seq": frame["request_seq"],
                        "tx_t": frame["tx_t"],
                        "latency_s": frame["latency_s"],
                    })
                except (TypeError, ValueError, IndexError):
                    continue
        except Exception as exc:
            self.error = exc


class ForceThread(threading.Thread):
    """从六维力传感器读取合法普通帧并写入时间戳缓存。

    Attributes:
        sensor (object): 提供 ``read_frame`` 的六维力传感器。
        buffer (TimestampedBuffer): 接收六维力帧的缓存。
        stop_event (threading.Event): 请求线程停止的事件。
        error (BaseException | None): 线程未处理异常；正常运行时为 ``None``。
    """

    def __init__(self, sensor, buffer, stop_event):
        """初始化六维力消费线程，不立即启动线程。

        Args:
            sensor (object): 六维力传感器实例。
            buffer (TimestampedBuffer): 六维力帧目标缓存。
            stop_event (threading.Event): 线程循环使用的停止事件。

        Returns:
            None: 初始化线程对象和错误状态。

        Side Effects:
            只初始化线程对象；不会读取串口或启动线程。
        """
        super().__init__(daemon=True, name="force-consumer")
        self.sensor = sensor
        self.buffer = buffer
        self.stop_event = stop_event
        self.error = None

    def run(self):
        """持续读取并缓存六维力帧直到停止或发生异常。

        Returns:
            None: 线程结束后返回。

        Side Effects:
            调用传感器 ``read_frame``，把六维力数据和时序元数据追加到缓存；
            未处理异常保存到 ``error``。
        """
        try:
            while not self.stop_event.is_set():
                frame = self.sensor.read_frame(timeout_s=0.1)
                if frame is None:
                    continue
                self.buffer.append({
                    "t": frame["rx_t"],
                    "data": np.asarray(frame["data"], dtype=np.float64),
                    "request_seq": frame["request_seq"],
                    "tx_t": frame["tx_t"],
                    "latency_s": frame["latency_s"],
                })
        except Exception as exc:
            self.error = exc


class FullAcquisitionSession:
    """完整应用的一次采集会话；循环由 ``acquisition_loop`` 显式驱动。

    Attributes:
        plot: 提供实时显示方法 ``set_data``、``append_full_data`` 的绘图对象。
        config (FullApplicationConfig): 设备、时序、模型和输出配置。
        stop_event (threading.Event): 会话停止信号。
        sensor_press/sensor_force: 压力/六维力传感器实例；六维力不可用时为
            ``None``。
        buf_press/buf_force (TimestampedBuffer | None): 两路输入缓存。
        csv_writer/csv_file_obj/csv_path: 当前 CSV 输出资源。
        has_force (bool): 是否已启用六维力通道。
        pending_press (deque): 等待六维力一对一匹配的压力样本队列。
        sample_processor (TangentialSampleProcessor): 生成内部详细样本的处理器。
        latest_sample: 最近处理的压力样本，供 GUI 使用。
    """

    def __init__(
        self,
        plot,
        config=None,
        stop_event=None,
        pressure_factory=PressureSensor,
        force_factory=SixAxisForceSensor,
        sample_processor=None,
    ):
        """初始化完整采集会话及其运行时状态。

        Args:
            plot (object): 实时绘图对象。
            config (FullApplicationConfig | None): 会话配置；为 ``None`` 时
                创建默认配置。
            stop_event (threading.Event | None): 外部停止事件；为 ``None`` 时
                使用模块级 ``g_main_stop_flag``。
            pressure_factory (callable): 压力传感器工厂，默认
                ``PressureSensor``。
            force_factory (callable): 六维力传感器工厂，默认
                ``SixAxisForceSensor``。
            sample_processor (object | None): 可选的测试注入对象；必须提供
                ``_process_sample(raw, frame=None)``。未注入时，``start`` 会创建
                独立的 ``TangentialSampleProcessor``。

        Returns:
            None: 只建立会话状态，不连接设备、不创建 CSV。

        Side Effects:
            创建内部队列、锁、统计计数器和停止状态；实际资源由 ``start``
            创建，由 ``close`` 释放。
        """
        self.plot = plot
        self.config = config or FullApplicationConfig()
        self.stop_event = stop_event or g_main_stop_flag
        self.pressure_factory = pressure_factory
        self.force_factory = force_factory

        self.sensor_press = None
        self.sensor_force = None
        self.thread_press = None
        self.thread_force = None
        self.buf_press = None
        self.buf_force = None
        self.csv_writer = None
        self.csv_file_obj = None
        self.csv_path = None
        self.row_count = 0
        self.has_force = False
        if sample_processor is not None and not callable(
            getattr(sample_processor, "_process_sample", None)
        ):
            raise TypeError(
                "sample_processor 必须提供 _process_sample(raw, frame=None)"
            )
        self.sample_processor = sample_processor
        self.pending_press = deque()
        self.rezero_guard = threading.Lock()
        self.rezero_threads = []

        self.last_press_seq = -1
        self.last_force_seq = -1
        self.prev_refined = False
        self.prev_contact = False
        self.pressure_start_t = None
        self.last_rel_ms = 0
        self.first_saved_press_t = None
        self.previous_saved_press_t = None

        self.force_fx_values = deque(maxlen=5)
        self.force_fy_values = deque(maxlen=5)
        self.force_fz_values = deque(maxlen=5)
        self.force_fx_filt = float("nan")
        self.force_fy_filt = float("nan")
        self.force_fz_filt = float("nan")
        self.force_angle_deg = float("nan")

        self.last_plot_t = 0.0
        self.latest_sample = None
        self.last_stats_log_t = None
        self.last_stats_frames = 0
        self.last_force_stats_frames = 0
        self.iteration_started_t = None
        self._started = False
        self._closed = False

    def start(self):
        """连接设备、初始化模型/CSV并启动采集线程。

        Returns:
            FullAcquisitionSession: 当前已启动会话，便于链式调用。

        Raises:
            RuntimeError: 压力传感器连接失败时抛出；压力是必需设备。
            Exception: 压力 CSV、模型、线程或其他启动步骤失败时向上传播。

        Side Effects:
            清除停止事件，连接压力传感器；六维力连接或校零失败时关闭该
            通道并降级为压力模式；创建 CSV、处理器和后台线程。重复调用在
            已启动状态下直接返回当前会话。
        """
        if self._started:
            return self
        self.stop_event.clear()
        try:
            if self.pressure_factory is PressureSensor:
                self.sensor_press = self.pressure_factory(
                    port=self.config.pressure.port,
                    period_s=self.config.pressure.period_s,
                    response_timeout_s=self.config.pressure.response_timeout_s,
                    queue_size=self.config.pressure.frame_queue_size,
                    baudrate=self.config.pressure.baudrate,
                    _startup_timeout_s=self.config.pressure.startup_timeout_s,
                )
            else:
                self.sensor_press = _construct_sensor(
                    self.pressure_factory, self.config.pressure_port
                )
        except Exception as exc:
            raise RuntimeError(f"压力传感器未连接: {exc}") from exc
        self.buf_press = TimestampedBuffer(self.config.buffer_size)
        print("✅ 压力传感器就绪")

        if not self.config.force.enabled:
            print("ℹ️ 六维力通道已禁用，使用压力模式")
        else:
            try:
                if self.force_factory is SixAxisForceSensor:
                    self.sensor_force = self.force_factory(
                        port=self.config.force.port,
                        period_s=self.config.force.period_s,
                        response_timeout_s=self.config.force.response_timeout_s,
                        queue_size=self.config.force.frame_queue_size,
                        baudrate=self.config.force.baudrate,
                        _startup_timeout_s=self.config.force.startup_timeout_s,
                    )
                else:
                    self.sensor_force = _construct_sensor(
                        self.force_factory, self.config.force_port
                    )
                if not self.sensor_force.calibrate_zero(
                    sample_count=self.config.zero_sample_count,
                    timeout_s=self.config.zero_timeout_s,
                ):
                    raise RuntimeError(
                        f"{self.config.zero_timeout_s:.1f}s 内未收到 "
                        f"{self.config.zero_sample_count} 个有效校零帧"
                    )
                self.buf_force = TimestampedBuffer(self.config.buffer_size)
                self.has_force = True
                print("✅ 六维力传感器就绪，启动零点校准完成")
            except Exception as exc:
                print(f"⚠️ 六维力传感器不可用，降级为压力模式: {exc}")
                if self.sensor_force is not None:
                    try:
                        self.sensor_force.close()
                    except Exception:
                        pass
                self.sensor_force = None
                self.has_force = False

        self.csv_path = auto_get_csv_path(self.config.save_dir)
        self.csv_writer, self.csv_file_obj = init_csv_file(self.csv_path)

        if self.sample_processor is None:
            calibration = (
                FitCalibrationModel.from_default()
                if self.config.model_path is None
                else FitCalibrationModel.from_path(self.config.model_path)
            )
            if calibration.available:
                summary = ", ".join(
                    f"{entry[1]}{'(split)' if entry[2] else ''}"
                    for entry in calibration.params_list
                )
                print(
                    f"📐 fit模型已加载: "
                    f"{calibration.path or 'tangential.resources/fit_coefs.bin'} "
                    f"(outputs: {summary})"
                )
            elif calibration.error is not None:
                print(f"⚠️ fit模型加载失败: {calibration.error}")
            else:
                print("💡 未找到 fit 模型文件")
            self.sample_processor = TangentialSampleProcessor(
                cop_sensor=PRSensorAngle(**self.config.processing.cop.as_kwargs()),
                calibration=calibration,
                processing_config=self.config.processing,
            )

        self.thread_press = PressureThread(
            self.sensor_press, self.buf_press, self.stop_event
        )
        self.thread_press.start()
        if self.has_force:
            self.thread_force = ForceThread(
                self.sensor_force, self.buf_force, self.stop_event
            )
            self.thread_force.start()

        now = time.perf_counter()
        self.last_stats_log_t = now
        self.last_stats_frames = self.sensor_press.get_timing_stats()["frames"]
        if self.has_force:
            self.last_force_stats_frames = (
                self.sensor_force.get_timing_stats()["frames"]
            )
        self._started = True
        return self

    def should_stop(self) -> bool:
        """查询会话是否收到停止请求。

        Returns:
            bool: ``stop_event`` 当前状态；已设置时为 ``True``。
        """
        return self.stop_event.is_set()

    def check_errors(self):
        """检查后台压力/六维力线程是否报告未处理异常。

        Returns:
            None: 两个线程均无异常时返回。

        Raises:
            RuntimeError: 任一消费线程的 ``error`` 不为 ``None``，异常信息会
                包含对应通道名称。

        Side Effects:
            更新本轮 ``iteration_started_t``，供主循环的节拍等待使用；不清除
            线程错误状态。
        """
        self.iteration_started_t = time.perf_counter()
        if self.thread_press is not None and self.thread_press.error is not None:
            raise RuntimeError(f"压力采集线程异常: {self.thread_press.error}")
        if self.thread_force is not None and self.thread_force.error is not None:
            raise RuntimeError(f"六维力采集线程异常: {self.thread_force.error}")

    def schedule_rezero(self, reason: str):
        """异步安排一次基于新六维力普通帧的 Fx/Fy 重新归零。

        Args:
            reason (str): 触发原因，用于日志，例如 CoP 精修或力卸载。

        Returns:
            None: 请求被忽略、合并或已启动后台归零任务时均无返回值。

        Side Effects:
            若六维力通道可用且会话未停止，创建并启动一个 daemon 线程；该线程
            从 ``buf_force`` 读取新的普通帧，收集配置数量后调用
            ``sensor_force.add_zero_bias``。单一锁会把并发请求合并，校零不足
            时仅打印失败日志。
        """
        if not self.has_force or self.stop_event.is_set():
            return

        def worker():
            """收集新六维力帧并在数量足够时更新 Fx/Fy 零偏。"""
            if not self.rezero_guard.acquire(blocking=False):
                print(f"ℹ️ {reason}归零请求已合并到正在执行的任务")
                return
            try:
                latest = self.buf_force.get_latest()
                seq = latest["seq"] if latest is not None else -1
                values = []
                deadline = time.perf_counter() + self.config.rezero_timeout_s
                while (
                    len(values) < self.config.zero_sample_count
                    and time.perf_counter() < deadline
                    and not self.stop_event.is_set()
                ):
                    for item in self.buf_force.get_after(seq):
                        values.append(item["data"])
                        seq = item["seq"]
                        if len(values) >= self.config.zero_sample_count:
                            break
                    if len(values) < self.config.zero_sample_count:
                        time.sleep(0.002)
                if len(values) < self.config.zero_sample_count:
                    print(f"⚠️ {reason}归零失败：有效力帧不足")
                    return
                average = np.mean(values, axis=0)
                self.sensor_force.add_zero_bias(
                    float(average[0]), float(average[1])
                )
                print(f"🔄 {reason}，Fx/Fy已归零")
            finally:
                self.rezero_guard.release()

        task = threading.Thread(target=worker, daemon=True, name="force-rezero")
        self.rezero_threads.append(task)
        task.start()

    def _process_pressure(self, press_item):
        """处理单个缓存压力帧并更新接触/精修/相对时间状态。

        Args:
            press_item (dict): 压力缓存帧，必须含 ``t`` 和 ``data``，可含
                ``request_seq``、``tx_t``、``latency_s``。

        Returns:
            TangentialSample: 经过 CoP、梯度、状态机、平滑和标定处理的内部
                结果；``rel_ms`` 会基于真实接收时间设置为单调不减毫秒值。

        Raises:
            Exception: 单帧处理器或归零调度相关错误向上传播。

        Side Effects:
            更新处理器状态、首帧时间、相对时间、上一接触/精修状态；状态
            边沿可能创建异步重新归零任务。
        """
        metadata = {
            "request_seq": press_item.get("request_seq", -1),
            "tx_t": press_item.get("tx_t", float("nan")),
            "rx_t": press_item["t"],
            "latency_s": press_item.get("latency_s", float("nan")),
        }
        # 完整应用直接消费内部样本；公开 FrameProcessor 只负责对外投影，
        # 这里不能再次执行 CoP、滑移或标定。
        sample = self.sample_processor._process_sample(press_item["data"], metadata)
        actual_contact = sample.state > 0
        if (
            self.config.refine_rezero_force
            and sample.refined
            and not self.prev_refined
        ):
            self.schedule_rezero("COP精修完成")
        self.prev_refined = sample.refined
        if self.prev_contact and not actual_contact:
            self.schedule_rezero("力卸载")
        self.prev_contact = actual_contact

        if self.pressure_start_t is None:
            self.pressure_start_t = sample.rx_t
        sample.rel_ms = max(
            self.last_rel_ms,
            int(round((sample.rx_t - self.pressure_start_t) * 1000.0)),
        )
        self.last_rel_ms = sample.rel_ms
        return sample

    def process_new_pressure_frames(self) -> int:
        """按缓存序号顺序处理所有尚未消费的压力帧。

        Returns:
            int: 本次调用实际处理的压力帧数量。

        Raises:
            RuntimeError: 单帧处理或后台错误检查相关异常向上传播。

        Side Effects:
            推进 ``last_press_seq``，更新 ``latest_sample``；有六维力时将样本
            放入待匹配队列，无六维力时立即写入 NaN 力字段的 CSV 行。
        """
        new_items = self.buf_press.get_after(self.last_press_seq)
        processed = 0
        for press_item in new_items:
            self.last_press_seq = press_item["seq"]
            sample = self._process_pressure(press_item)
            self.latest_sample = sample
            if self.has_force:
                self.pending_press.append(sample)
            else:
                self.write_snapshot(sample, None)
            processed += 1
        return processed

    def write_snapshot(self, sample, force_item):
        """把压力样本和可选匹配力帧写成一行 108 列 CSV。

        Args:
            sample (TangentialSample): 要保存的内部压力结果，使用其真实 ``rx_t``、
                ADC、CoP、角度和标定结果。
            force_item (dict | None): 已匹配六维力帧，需含 ``t`` 和长度至少为
                6 的 ``data``；为 ``None`` 时六维力及其派生字段写为 ``NaN``。

        Returns:
            None: 写入并刷新当前 CSV 文件。

        Raises:
            AttributeError: 会话尚未成功初始化 CSV，或样本缺字段时可能抛出。
            OSError: CSV 写入或刷新失败时抛出。
            ValueError: 力帧数据无法转换为所需数值时可能抛出。

        Side Effects:
            通过唯一的 ``build_csv_row`` 构造行，写入并 flush 文件；更新已
            保存时间、力中值滤波状态和 ``row_count``。
        """
        press_timestamp = float(sample.rx_t)
        if self.first_saved_press_t is None:
            csv_rel_ms = 0.0
            csv_delta_ms = 0.0
        else:
            csv_rel_ms = max(
                0.0,
                round(
                    (press_timestamp - self.first_saved_press_t) * 1000.0, 6
                ),
            )
            csv_delta_ms = max(
                0.0,
                round(
                    (press_timestamp - self.previous_saved_press_t) * 1000.0, 6
                ),
            )

        if force_item is None:
            force_data = [float("nan")] * 6
            force_ts = float("nan")
            row_fx = row_fy = row_fz = float("nan")
            row_angle = float("nan")
        else:
            force_data = force_item["data"]
            force_ts = force_item["t"]
            raw_fx, raw_fy, raw_fz = force_data[:3]
            self.force_fx_values.append(raw_fx)
            self.force_fy_values.append(raw_fy)
            self.force_fz_values.append(raw_fz)
            self.force_fx_filt = float(np.median(self.force_fx_values))
            self.force_fy_filt = float(np.median(self.force_fy_values))
            self.force_fz_filt = float(np.median(self.force_fz_values))
            self.force_angle_deg = compute_vector_angle(
                self.force_fx_filt, self.force_fy_filt
            )
            row_fx = self.force_fx_filt
            row_fy = self.force_fy_filt
            row_fz = self.force_fz_filt
            row_angle = self.force_angle_deg

        self.csv_writer.writerow(build_csv_row(
            press_timestamp=press_timestamp,
            rel_ms=csv_rel_ms,
            delta_ms=csv_delta_ms,
            ch_data=sample.raw,
            force_data=force_data,
            force_timestamp=force_ts,
            delta_cop_x=sample.dx,
            delta_cop_y=sample.dy,
            delta_force_x=row_fx,
            delta_force_y=row_fy,
            delta_force_z=row_fz,
            adc_angle=sample.angle,
            force_angle=row_angle,
            fx_cal=sample.calibrated_fx,
            fy_cal=sample.calibrated_fy,
            force_cal_angle=sample.calibrated_angle,
            cop_state=sample.state,
            adc_sum=sample.adc_sum,
            valid=1 if sample.state > 0 else 0,
        ))
        self.csv_file_obj.flush()
        self.row_count += 1
        if self.first_saved_press_t is None:
            self.first_saved_press_t = press_timestamp
        self.previous_saved_press_t = press_timestamp

    def drain_force_matches(self, now=None):
        """按待处理压力队列顺序匹配并保存六维力帧。

        Args:
            now (float | None): 当前单调时钟秒数；为 ``None`` 时调用
                ``time.perf_counter`` 获取。用于判断等待是否超过匹配窗口。

        Returns:
            None: 匹配和写入通过副作用完成。

        Side Effects:
            对队首压力样本寻找未消费且时间差不超过配置窗口的最近力帧；匹配
            成功时推进 ``last_force_seq`` 并写 CSV。当前实现对超过等待窗口
            的未匹配样本移出待匹配队列但不写入 CSV；无六维力通道时直接返回。
        """
        if not self.has_force:
            return
        now = time.perf_counter() if now is None else now
        while self.pending_press:
            sample = self.pending_press[0]
            force_item = match_force_frame(
                self.buf_force,
                sample.rx_t,
                self.config.max_time_diff_s,
                min_seq=self.last_force_seq,
            )
            if force_item is not None:
                self.pending_press.popleft()
                self.last_force_seq = force_item["seq"]
                self.write_snapshot(sample, force_item)
                continue
            if now - sample.rx_t > self.config.max_time_diff_s:
                self.pending_press.popleft()
                continue
            break

    @staticmethod
    def _percentile_ms(values, percentile):
        """计算秒值序列的指定百分位并转换为毫秒。

        Args:
            values (Sequence[float]): 以秒为单位的数值序列。
            percentile (float): NumPy 百分位参数，通常为 50 或 95。

        Returns:
            float: 以毫秒为单位的百分位；输入为空时返回 ``NaN``。

        Raises:
            ValueError: ``percentile`` 超出 NumPy 允许范围时可能抛出。
        """
        if not values:
            return float("nan")
        return float(np.percentile(values, percentile) * 1000.0)

    def _print_pressure_stats(self, stats, fps):
        """格式化并打印压力传感器时序统计。

        Args:
            stats (Mapping): ``PressureSensor.get_timing_stats`` 返回的统计字典。
            fps (float): 当前日志区间内的实际压力帧率，单位为 Hz。

        Returns:
            None: 统计写入标准输出。

        Raises:
            KeyError: ``stats`` 缺少日志所需字段时抛出。
        """
        print(
            "⏱ 压力时序: "
            f"{fps:.1f} Hz, 请求间隔 P50/P95="
            f"{self._percentile_ms(stats['tx_intervals_s'], 50):.2f}/"
            f"{self._percentile_ms(stats['tx_intervals_s'], 95):.2f} ms, "
            "响应延迟 P50/P95="
            f"{self._percentile_ms(stats['latencies_s'], 50):.2f}/"
            f"{self._percentile_ms(stats['latencies_s'], 95):.2f} ms, "
            f"超时={stats['response_timeouts']}, "
            f"CRC={stats['crc_errors']}, 状态={stats['status_errors']}, "
            f"队列丢帧={stats['queue_drops']}, "
            f"跳过周期={stats['schedule_skips']}"
        )

    def _print_force_stats(self, stats, fps):
        """格式化并打印六维力传感器时序统计。

        Args:
            stats (Mapping): ``SixAxisForceSensor.get_timing_stats`` 返回的统计字典。
            fps (float): 当前日志区间内的实际六维力帧率，单位为 Hz。

        Returns:
            None: 统计写入标准输出。

        Raises:
            KeyError: ``stats`` 缺少日志所需字段时抛出。
        """
        print(
            "⏱ 六维力时序: "
            f"{fps:.1f} Hz, 请求间隔 P50/P95="
            f"{self._percentile_ms(stats['tx_intervals_s'], 50):.2f}/"
            f"{self._percentile_ms(stats['tx_intervals_s'], 95):.2f} ms, "
            "响应延迟 P50/P95="
            f"{self._percentile_ms(stats['latencies_s'], 50):.2f}/"
            f"{self._percentile_ms(stats['latencies_s'], 95):.2f} ms, "
            f"超时={stats['response_timeouts']}, "
            f"帧头错误={stats['framing_errors']}, "
            f"尾部错误={stats['tail_errors']}, "
            f"读错={stats['serial_read_errors']}, "
            f"写错={stats['serial_write_errors']}, "
            f"队列丢帧={stats['queue_drops']}, "
            f"跳过周期={stats['schedule_skips']}"
        )

    def log_timing_stats(self, now=None):
        """按配置间隔打印压力和六维力时序统计。

        Args:
            now (float | None): 当前单调时钟秒数；为 ``None`` 时自动读取。

        Returns:
            None: 未到日志间隔时静默返回，否则打印本区间帧率、延迟和错误计数。

        Raises:
            AttributeError/KeyError: 传感器未初始化或统计字典缺字段时可能抛出。

        Side Effects:
            读取传感器统计并更新 ``last_stats_log_t`` 及上一统计区间的帧数。
        """
        now = time.perf_counter() if now is None else now
        if now - self.last_stats_log_t < self.config.timing_log_interval_s:
            return
        elapsed = now - self.last_stats_log_t
        stats = self.sensor_press.get_timing_stats()
        frame_count = stats["frames"]
        self._print_pressure_stats(
            stats, (frame_count - self.last_stats_frames) / elapsed
        )
        self.last_stats_frames = frame_count
        if self.has_force:
            force_stats = self.sensor_force.get_timing_stats()
            force_count = force_stats["frames"]
            self._print_force_stats(
                force_stats,
                (force_count - self.last_force_stats_frames) / elapsed,
            )
            self.last_force_stats_frames = force_count
        self.last_stats_log_t = now

    def update_plot(self):
        """按 GUI 频率限制刷新最新压力样本和力/标定曲线。

        Returns:
            None: 没有样本或尚未到绘图周期时直接返回。

        Raises:
            AttributeError: 绘图对象缺少所需更新方法时抛出。
            Exception: 绘图层更新失败时向上传播。

        Side Effects:
            可能调用 ``plot.set_data`` 和 ``plot.append_full_data``，更新绘图
            时间；刷新后清除 ``latest_sample``，但不影响 CSV 或串口采集。
        """
        sample = self.latest_sample
        if sample is None:
            return
        now = time.perf_counter()
        if now - self.last_plot_t < 1.0 / self.config.plot_fps:
            return
        self.plot.set_data(
            sample.angle,
            self.force_angle_deg,
            sample.raw,
            sample.adc_sum,
            sample.cop_x,
            sample.cop_y,
            sample.origin_x,
            sample.origin_y,
            sample.dx,
            sample.dy,
            self.force_fx_filt,
            self.force_fy_filt,
            self.force_fz_filt,
            sample.calibrated_fx,
            sample.calibrated_fy,
            sample.calibrated_fz,
            cop_state=sample.state,
            gradient=sample.gradient,
            contact_init=sample.display_contact,
            refined=sample.refined,
            pzt_table_angle_deg=(-sample.angle) % 360.0,
            region_mask=sample.region_mask,
            regions=sample.regions,
            centroid=sample.centroid,
            motion_state=sample.motion_state,
            is_slipping=sample.is_slipping,
            slip_motion_distance=sample.slip_motion_distance,
            slip_confidence=sample.slip_confidence,
            angle_vector_magnitude=sample.angle_vector_magnitude,
        )
        if sample.contact:
            self.plot.append_full_data(
                sample.rel_ms,
                sample.angle,
                sample.adc_sum,
                sample.dx,
                sample.dy,
                self.force_angle_deg,
                self.force_fz_filt,
                self.force_fx_filt,
                self.force_fy_filt,
                sample.calibrated_angle,
                sample.calibrated_fx,
                sample.calibrated_fy,
                sample.calibrated_fz,
            )
        self.last_plot_t = now
        self.latest_sample = None

    def wait_for_next_iteration(self):
        """根据目标主循环频率等待下一次会话迭代。

        Returns:
            None: 睡眠至少 1 ms 或当前周期剩余时间。

        Side Effects:
            调用 ``time.sleep`` 阻塞当前主采集循环；不阻塞传感器消费线程。
        """
        started = self.iteration_started_t or time.perf_counter()
        elapsed = time.perf_counter() - started
        time.sleep(max(0.001, 1.0 / self.config.target_fps - elapsed))

    def close(self):
        """幂等地停止会话并释放线程、传感器、CSV 和空文件资源。

        Returns:
            None: 重复调用直接返回。

        Side Effects:
            设置停止事件，等待消费/归零线程，关闭两个传感器和 CSV；若本次
            没有写入任何行则删除已创建的空 CSV，否则打印已保存行数。清理
            传感器时的单项关闭异常会被忽略以保证继续释放其他资源。
        """
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        if self.thread_press is not None and self.thread_press.is_alive():
            self.thread_press.join(timeout=2)
        if self.thread_force is not None and self.thread_force.is_alive():
            self.thread_force.join(timeout=2)
        for task in self.rezero_threads:
            if task.is_alive():
                task.join(timeout=1)
        for sensor in (self.sensor_press, self.sensor_force):
            if sensor is not None:
                try:
                    sensor.close()
                except Exception:
                    pass
        if self.csv_file_obj is not None:
            self.csv_file_obj.close()
        if self.csv_path is not None:
            if self.row_count == 0 and os.path.exists(self.csv_path):
                os.remove(self.csv_path)
                print("⚠️ 无数据，CSV 已删除")
            elif self.row_count > 0:
                print(f"✅ CSV已关闭（{self.row_count} 行）")


class FullApplicationRunner:
    """管理 Qt 生命周期并把采集线程异常转发到 GUI 主线程。

    该类不包含采集 ``while`` 循环；循环由传入的 ``worker_target``（通常是
    ``acquisition_loop``）执行。

    Attributes:
        worker_target (callable): 接受绘图对象、停止事件和配置的采集入口。
        config (FullApplicationConfig): 传给采集入口的配置。
        plot_factory (callable): 创建实时绘图对象的工厂。
    """

    def __init__(self, worker_target, config=None, plot_factory=RealTimePlot):
        """初始化 Qt 应用运行器。

        Args:
            worker_target (callable): 采集工作函数；通常为
                ``acquisition_loop``。
            config (FullApplicationConfig | None): 采集配置；为 ``None`` 时
                创建默认配置。
            plot_factory (callable): 无参绘图对象工厂，默认 ``RealTimePlot``。

        Returns:
            None: 保存工作函数、配置和绘图工厂。

        Side Effects:
            只初始化 Python 对象；不会创建 Qt 应用、窗口或后台线程。
        """
        self.worker_target = worker_target
        self.config = config or FullApplicationConfig()
        self.plot_factory = plot_factory

    def run(self):
        """创建 Qt 窗口并运行 GUI 事件循环直到退出。

        Returns:
            None: Qt 事件循环退出后完成线程等待和绘图分析。

        Raises:
            Exception: 绘图对象创建或最终完整分析失败时可能抛出；采集线程
                内部异常会通过窗口日志和停止事件处理。

        Side Effects:
            清除全局停止事件，创建 QApplication/绘图窗口，启动 daemon 采集
            线程和错误轮询定时器；退出时停止定时器、等待采集线程，并调用
            ``plot_full_analysis``。
        """
        g_main_stop_flag.clear()
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        plot = (
            self.plot_factory(config=self.config.gui)
            if self.plot_factory is RealTimePlot
            else self.plot_factory()
        )
        errors = queue.Queue()

        def worker():
            """在线程中执行采集入口并把异常放入线程安全队列。"""
            try:
                self.worker_target(
                    plot,
                    stop_event=g_main_stop_flag,
                    config=self.config,
                )
            except Exception as exc:
                errors.put(exc)
                g_main_stop_flag.set()

        data_thread = threading.Thread(
            target=worker, daemon=True, name="full-acquisition"
        )
        data_thread.start()
        error_timer = QtCore.QTimer()

        def poll_errors():
            """在 Qt 主线程轮询采集线程错误并请求应用退出。"""
            try:
                exc = errors.get_nowait()
            except queue.Empty:
                return
            print(f"❌ 数据线程异常: {exc}")
            if hasattr(plot, "set_status"):
                plot.set_status(f"数据线程异常: {exc}")
            else:
                plot.win.setWindowTitle(f"{self.config.gui.window_title} — 数据线程异常: {exc}")
            g_main_stop_flag.set()
            app.quit()

        error_timer.timeout.connect(poll_errors)
        error_timer.start(100)
        try:
            app.exec()
        except KeyboardInterrupt:
            pass
        finally:
            error_timer.stop()
            g_main_stop_flag.set()
            data_thread.join(timeout=5)
            plot.plot_full_analysis(self.config.save_dir)


class DualApplicationRunner:
    """在同一个 Qt 应用中运行两个相互隔离的完整采集会话。

    两路分别使用独立的 ``FullApplicationConfig``、停止事件、绘图窗口和
    ``acquisition_loop`` 后台线程；任一路出现异常都会请求两路一起退出。
    该类只编排生命周期，不复制压力/六维力采集、CoP、标定或 CSV 算法。
    """

    def __init__(
        self,
        config_a: FullApplicationConfig,
        config_b: FullApplicationConfig,
        worker_target=None,
        plot_factory=RealTimePlot,
    ) -> None:
        """创建双路完整应用运行器。

        Args:
            config_a: Sensor A 的完整分层配置。
            config_b: Sensor B 的完整分层配置。
            worker_target: 完整采集循环，默认使用 ``acquisition_loop``；
                测试可注入替代实现。
            plot_factory: 实时窗口工厂，默认创建 ``RealTimePlot``。

        Raises:
            ValueError: 两路压力或两路启用的六维力指向同一物理串口，或
                输出目录相同。
        """
        self.config_a = config_a
        self.config_b = config_b
        if "Sensor A" not in self.config_a.gui.window_title:
            self.config_a.gui.window_title = f"Sensor A — {self.config_a.gui.window_title}"
        if "Sensor B" not in self.config_b.gui.window_title:
            self.config_b.gui.window_title = f"Sensor B — {self.config_b.gui.window_title}"
        self.worker_target = acquisition_loop if worker_target is None else worker_target
        self.plot_factory = plot_factory
        _validate_dual_configs(config_a, config_b)

    def run(self) -> None:
        """启动两个窗口和后台采集循环，直到 Qt 退出或任一路异常。

        Returns:
            None: 两路会话关闭、CSV 刷新并分别生成结束分析图后返回。

        Side Effects:
            创建一个 ``QApplication``、两个 ``RealTimePlot``、两个后台
            采集线程和两个独立 CSV；任一路异常会在 Qt 主线程报告并联动
            设置两路停止事件。
        """
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        stop_events = [threading.Event(), threading.Event()]
        configs = [self.config_a, self.config_b]
        labels = ["Sensor A", "Sensor B"]
        plots = []
        for config in configs:
            if self.plot_factory is RealTimePlot:
                plots.append(self.plot_factory(config=config.gui))
            else:
                plots.append(self.plot_factory(config=config.gui))

        errors = queue.Queue()
        worker_errors = []
        worker_errors_lock = threading.Lock()

        def worker(index: int) -> None:
            """运行一路完整循环，并把异常传给 Qt 主线程。"""
            try:
                self.worker_target(
                    plots[index],
                    stop_event=stop_events[index],
                    config=configs[index],
                )
            except Exception as exc:
                with worker_errors_lock:
                    worker_errors.append((index, exc))
                errors.put((index, exc))
                stop_events[0].set()
                stop_events[1].set()

        threads = [
            threading.Thread(
                target=worker,
                args=(index,),
                daemon=True,
                name=f"{label.lower().replace(' ', '-')}-acquisition",
            )
            for index, label in enumerate(labels)
        ]
        for thread in threads:
            thread.start()

        error_timer = QtCore.QTimer()

        def poll_errors() -> None:
            """在 Qt 主线程报告 A/B 错误并退出应用。"""
            try:
                index, exc = errors.get_nowait()
            except queue.Empty:
                return
            message = f"{labels[index]} 数据线程异常: {exc}"
            print(f"❌ {message}")
            for plot in plots:
                if hasattr(plot, "set_status"):
                    plot.set_status(message)
            stop_events[0].set()
            stop_events[1].set()
            app.quit()

        error_timer.timeout.connect(poll_errors)
        error_timer.start(100)
        try:
            app.exec()
        except KeyboardInterrupt:
            pass
        finally:
            error_timer.stop()
            for event in stop_events:
                event.set()
            for thread in threads:
                thread.join(timeout=5)
            for plot, config in zip(plots, configs):
                plot.plot_full_analysis(config.save_dir)
        if worker_errors:
            index, exc = worker_errors[0]
            raise RuntimeError(f"{labels[index]} 数据线程异常: {exc}") from exc


def _validate_dual_configs(
    config_a: FullApplicationConfig,
    config_b: FullApplicationConfig,
) -> None:
    """校验双路完整应用的物理端口和输出目录隔离。

    Args:
        config_a: Sensor A 配置。
        config_b: Sensor B 配置。

    Returns:
        None: 配置合法时正常返回。

    Raises:
        ValueError: 压力端口、启用的力端口或 CSV 输出目录发生冲突。
    """
    pressure_a = os.path.realpath(os.path.abspath(config_a.pressure.port))
    pressure_b = os.path.realpath(os.path.abspath(config_b.pressure.port))
    if pressure_a == pressure_b:
        raise ValueError("Sensor A 和 Sensor B 不能使用同一个物理压力串口")
    if config_a.force.enabled and config_b.force.enabled:
        force_a = os.path.realpath(os.path.abspath(config_a.force.port))
        force_b = os.path.realpath(os.path.abspath(config_b.force.port))
        if force_a == force_b:
            raise ValueError("Sensor A 和 Sensor B 不能使用同一个物理力串口")
    active_ports = [pressure_a, pressure_b]
    if config_a.force.enabled:
        active_ports.append(os.path.realpath(os.path.abspath(config_a.force.port)))
    if config_b.force.enabled:
        active_ports.append(os.path.realpath(os.path.abspath(config_b.force.port)))
    if len(active_ports) != len(set(active_ports)):
        raise ValueError("压力串口和启用的六维力串口不能指向同一物理设备")
    save_a = os.path.realpath(os.path.abspath(config_a.save_dir))
    save_b = os.path.realpath(os.path.abspath(config_b.save_dir))
    if save_a == save_b:
        raise ValueError("Sensor A 和 Sensor B 必须使用不同的 CSV 输出目录")


def acquisition_loop(
    plot,
    stop_event=None,
    config=None,
    session_factory=FullAcquisitionSession,
    **kwargs,
):
    """运行完整采集循环并在退出时可靠关闭会话。

    Args:
        plot (object): 实时绘图对象，传给 ``FullAcquisitionSession``。
        stop_event (threading.Event | None): 外部停止事件；为 ``None`` 时使用
            模块级 ``g_main_stop_flag``。
        config (FullApplicationConfig | None): 完整会话配置；为 ``None`` 时
            创建默认配置。
        session_factory (callable): 会话工厂，默认 ``FullAcquisitionSession``；
            测试可注入替代实现。
        **kwargs: 继续传给会话工厂的其他构造参数，例如设备工厂。

    Returns:
        None: 停止事件设置或会话循环结束后返回。

    Raises:
        Exception: 会话启动、错误检查、压力处理、匹配、绘图或等待失败时
            向上传播；无论是否异常都会执行 ``session.close``。

    Side Effects:
        创建并启动会话，按顺序执行错误检查、压力帧处理、力匹配、统计和
        GUI 更新循环；最终释放线程、传感器、CSV 和其他会话资源。
    """
    active_stop_event = g_main_stop_flag if stop_event is None else stop_event
    session = session_factory(
        plot,
        config=config or FullApplicationConfig(),
        stop_event=active_stop_event,
        **kwargs,
    )
    try:
        session.start()
        while not session.should_stop():
            session.check_errors()
            session.process_new_pressure_frames()
            session.drain_force_matches()
            session.log_timing_stats()
            session.update_plot()
            session.wait_for_next_iteration()
    finally:
        session.close()
