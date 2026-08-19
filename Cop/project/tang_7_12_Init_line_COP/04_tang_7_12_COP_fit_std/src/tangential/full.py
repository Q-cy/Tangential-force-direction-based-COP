"""完整采集示例使用的线程、会话、CSV、同步和 GUI 辅助层。"""

import inspect
import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from .acquisition.buffer import TimestampedBuffer, match_closest
from .api import TangentialFrameProcessor, compute_vector_angle
from .config import FullApplicationConfig
from .gui.realtime import RealTimePlot
from .processing.calibration import FitCalibrationModel
from .sensors.force import SixAxisForceSensor
from .sensors.pressure import PressureSensor
from .storage.csv import auto_get_csv_path, build_csv_row, init_csv_file


g_main_stop_flag = threading.Event()


def _construct_sensor(factory, port):
    """构造真实传感器并保留无参测试工厂的注入能力。"""
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
    def __init__(self, sensor, buffer, stop_event):
        super().__init__(daemon=True, name="pressure-consumer")
        self.sensor = sensor
        self.buffer = buffer
        self.stop_event = stop_event
        self.error = None

    def run(self):
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
    def __init__(self, sensor, buffer, stop_event):
        super().__init__(daemon=True, name="force-consumer")
        self.sensor = sensor
        self.buffer = buffer
        self.stop_event = stop_event
        self.error = None

    def run(self):
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
    """完整应用的一次采集会话；循环由 acquisition_loop 显式驱动。"""

    def __init__(
        self,
        plot,
        config=None,
        stop_event=None,
        pressure_factory=PressureSensor,
        force_factory=SixAxisForceSensor,
    ):
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
        self.processor = None
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
        if self._started:
            return self
        self.stop_event.clear()
        try:
            self.sensor_press = _construct_sensor(
                self.pressure_factory, self.config.pressure_port
            )
        except Exception as exc:
            raise RuntimeError(f"压力传感器未连接: {exc}") from exc
        self.buf_press = TimestampedBuffer(self.config.buffer_size)
        print("✅ 压力传感器就绪")

        try:
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
        self.processor = TangentialFrameProcessor(
            calibration=calibration,
            cal_dim=self.config.cal_dim,
            region_mode=self.config.region_mode,
            median_window=5,
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
        return self.stop_event.is_set()

    def check_errors(self):
        self.iteration_started_t = time.perf_counter()
        if self.thread_press is not None and self.thread_press.error is not None:
            raise RuntimeError(f"压力采集线程异常: {self.thread_press.error}")
        if self.thread_force is not None and self.thread_force.error is not None:
            raise RuntimeError(f"六维力采集线程异常: {self.thread_force.error}")

    def schedule_rezero(self, reason: str):
        """从 ForceThread 缓冲区取新帧，合并并发归零请求。"""
        if not self.has_force or self.stop_event.is_set():
            return

        def worker():
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
        metadata = {
            "request_seq": press_item.get("request_seq", -1),
            "tx_t": press_item.get("tx_t", float("nan")),
            "rx_t": press_item["t"],
            "latency_s": press_item.get("latency_s", float("nan")),
        }
        sample = self.processor.process(press_item["data"], metadata)
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
            adc_sum=sample.total,
            valid=1 if sample.state > 0 else 0,
        ))
        self.csv_file_obj.flush()
        self.row_count += 1
        if self.first_saved_press_t is None:
            self.first_saved_press_t = press_timestamp
        self.previous_saved_press_t = press_timestamp

    def drain_force_matches(self, now=None):
        if not self.has_force:
            return
        now = time.perf_counter() if now is None else now
        while self.pending_press:
            sample = self.pending_press[0]
            force_item = match_closest(
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
        if not values:
            return float("nan")
        return float(np.percentile(values, percentile) * 1000.0)

    def _print_pressure_stats(self, stats, fps):
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
            sample.total,
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
        )
        if sample.contact:
            self.plot.append_full_data(
                sample.rel_ms,
                sample.angle,
                sample.total,
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
        started = self.iteration_started_t or time.perf_counter()
        elapsed = time.perf_counter() - started
        time.sleep(max(0.001, 1.0 / self.config.target_fps - elapsed))

    def close(self):
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
    """Qt 生命周期和数据线程错误转发；不包含采集 while 循环。"""

    def __init__(self, worker_target, config=None, plot_factory=RealTimePlot):
        self.worker_target = worker_target
        self.config = config or FullApplicationConfig()
        self.plot_factory = plot_factory

    def run(self):
        g_main_stop_flag.clear()
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        plot = self.plot_factory()
        errors = queue.Queue()

        def worker():
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
            try:
                exc = errors.get_nowait()
            except queue.Empty:
                return
            print(f"❌ 数据线程异常: {exc}")
            plot.win.setWindowTitle(f"RealTime — 数据线程异常: {exc}")
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


def acquisition_loop(
    plot,
    stop_event=None,
    config=None,
    session_factory=FullAcquisitionSession,
    **kwargs,
):
    """运行完整采集循环；Qt 线程和测试都通过这个正式入口调用。"""
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
