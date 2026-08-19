import os
import queue
import sys
import threading
import time
from collections import deque

import numpy as np
from pyqtgraph.Qt import QtCore, QtWidgets

from data import PressureSensor, SixAxisForceSensor, TimestampedBuffer, match_closest
from fit import apply_predict_multi, load_coefs
from realtime import RealTimePlot
from table import auto_get_csv_path, build_csv_row, init_csv_file
import tangential_other_package as pzt
from tangential_package import PRSensorAngle


# ===================== 配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SAVE_DIR = os.path.join(
    BASE_DIR, "../../../../../../../data/2.PZT_tangential/weight/test"
)
fit_coefs_path = os.path.join(BASE_DIR, "fit_coefs.bin")
MAIN_CAL_DIM = "3D"                         # "2D"=Fx/Fy, "3D"=Fz/Fx/Fy
MAIN_REFINE_REZERO_FORCE = True

MAIN_TARGET_FPS = 100
MAIN_PLOT_FPS = 60
MAIN_MAX_TIME_DIFF_S = 0.015
MAIN_TIMING_LOG_INTERVAL_S = 1.0
MAIN_REGION_MODE = "full"                  # "full" / "region" / "both"
MAIN_ZERO_SAMPLE_COUNT = 10
MAIN_ZERO_TIMEOUT_S = 1.0
MAIN_REZERO_TIMEOUT_S = 1.0

g_main_stop_flag = threading.Event()


# ===================== 采集线程 =====================
class PressureThread(threading.Thread):
    def __init__(self, sensor, buf):
        super().__init__(daemon=True)
        self.s = sensor
        self.buf = buf
        self.error = None

    def run(self):
        try:
            while not g_main_stop_flag.is_set():
                frame = self.s.read_frame(timeout_s=0.1)
                if frame is not None:
                    try:
                        data = np.asarray(self.s.decode(frame["raw"]), dtype=np.float64)
                        self.buf.append({
                            "t": frame["rx_t"],
                            "data": data,
                            "request_seq": frame["request_seq"],
                            "tx_t": frame["tx_t"],
                            "latency_s": frame["latency_s"],
                        })
                    except (TypeError, ValueError, IndexError):
                        pass
        except Exception as exc:
            self.error = exc


class ForceThread(threading.Thread):
    def __init__(self, sensor, buf):
        super().__init__(daemon=True)
        self.s = sensor
        self.buf = buf
        self.error = None

    def run(self):
        try:
            while not g_main_stop_flag.is_set():
                frame = self.s.read_frame(timeout_s=0.1)
                if frame is not None:
                    data = np.asarray(frame["data"], dtype=np.float64)
                    self.buf.append({
                        "t": frame["rx_t"],
                        "data": data,
                        "request_seq": frame["request_seq"],
                        "tx_t": frame["tx_t"],
                        "latency_s": frame["latency_s"],
                    })
        except Exception as exc:
            self.error = exc


def _load_fit_model():
    fit_type = None
    params_list = None
    split_sign = False
    if not os.path.exists(fit_coefs_path):
        print("💡 未找到 fit 模型文件")
        return fit_type, params_list, split_sign
    try:
        fit_type, _, params_list, split_sign = load_coefs(fit_coefs_path)
        type_summary = ", ".join(
            f"{entry[1]}{'(split)' if entry[2] else ''}" for entry in params_list
        )
        print(f"📐 fit模型已加载: {fit_coefs_path} (outputs: {type_summary})")
    except Exception as exc:
        print(f"⚠️ fit模型加载失败: {exc}")
        params_list = None
        split_sign = False
    return fit_type, params_list, split_sign


# ===================== 数据循环 =====================
def data_loop(plot):
    """压力帧驱动的数据循环；每个 pressure seq 只处理一次。"""
    sensor_press = None
    sensor_force = None
    thread_press = None
    thread_force = None
    csv_file_obj = None
    csv_path = None
    row_count = 0
    rezero_threads = []
    buf_press = None
    buf_force = None
    pending_press = None
    has_force = False
    process_pressure = None
    drain_pending = None

    try:
        # 压力传感器是系统的必需输入。失败时不创建 CSV。
        try:
            sensor_press = PressureSensor()
        except Exception as exc:
            raise RuntimeError(f"压力传感器未连接: {exc}") from exc

        buf_press = TimestampedBuffer(500)
        print("✅ 压力传感器就绪")

        # 六维力可选；连接或校零失败均降级为压力单传感器模式。
        has_force = False
        try:
            sensor_force = SixAxisForceSensor()
            if not sensor_force.calibrate_zero(
                sample_count=MAIN_ZERO_SAMPLE_COUNT,
                timeout_s=MAIN_ZERO_TIMEOUT_S,
            ):
                raise RuntimeError(
                    f"{MAIN_ZERO_TIMEOUT_S:.1f}s 内未收到 "
                    f"{MAIN_ZERO_SAMPLE_COUNT} 个有效校零帧"
                )
            buf_force = TimestampedBuffer(500)
            has_force = True
            print("✅ 六维力传感器就绪，启动零点校准完成")
        except Exception as exc:
            print(f"⚠️ 六维力传感器不可用，降级为压力模式: {exc}")
            if sensor_force is not None:
                try:
                    sensor_force.close()
                except Exception:
                    pass
            sensor_force = None

        # 设备状态确定后再创建文件，避免连接失败留下空 CSV。
        csv_path = auto_get_csv_path(MAIN_SAVE_DIR)
        csv_writer, csv_file_obj = init_csv_file(csv_path)

        thread_press = PressureThread(sensor_press, buf_press)
        thread_press.start()
        if has_force:
            thread_force = ForceThread(sensor_force, buf_force)
            thread_force.start()

        fit_type, fit_params_list, fit_split_sign = _load_fit_model()

        median_filt_window = 5
        buf_cop_delta_x = deque(maxlen=median_filt_window)
        buf_cop_delta_y = deque(maxlen=median_filt_window)
        buf_force_fx = deque(maxlen=median_filt_window)
        buf_force_fy = deque(maxlen=median_filt_window)
        buf_force_fz = deque(maxlen=median_filt_window)

        cop_sensor = PRSensorAngle()
        nan6 = [float("nan")] * 6
        prev_refined = False
        prev_contact = False
        last_press_seq = -1
        last_force_seq = -1
        last_plot_t = 0.0
        plot_interval_s = 1.0 / MAIN_PLOT_FPS
        pending_press = deque()
        rezero_guard = threading.Lock()
        pressure_start_t = None
        last_rel_ms = 0
        first_saved_press_t = None
        previous_saved_press_t = None
        last_stats_log_t = time.perf_counter()
        last_stats_frames = sensor_press.get_timing_stats()["frames"]
        last_force_stats_frames = (
            sensor_force.get_timing_stats()["frames"] if has_force else 0
        )

        force_fx_filt = float("nan")
        force_fy_filt = float("nan")
        force_fz_filt = float("nan")
        force_angle_deg = float("nan")

        def schedule_rezero(reason: str):
            """从 ForceThread 缓冲区收集10个新帧，串口始终保持单消费者。"""
            if not has_force or g_main_stop_flag.is_set():
                return

            def worker():
                if not rezero_guard.acquire(blocking=False):
                    print(f"ℹ️ {reason}归零请求已合并到正在执行的任务")
                    return
                try:
                    latest = buf_force.get_latest()
                    seq = latest["seq"] if latest is not None else -1
                    values = []
                    deadline = time.perf_counter() + MAIN_REZERO_TIMEOUT_S
                    while (
                        len(values) < MAIN_ZERO_SAMPLE_COUNT
                        and time.perf_counter() < deadline
                        and not g_main_stop_flag.is_set()
                    ):
                        items = buf_force.get_after(seq)
                        for item in items:
                            values.append(item["data"])
                            seq = item["seq"]
                            if len(values) >= MAIN_ZERO_SAMPLE_COUNT:
                                break
                        if len(values) < MAIN_ZERO_SAMPLE_COUNT:
                            time.sleep(0.002)
                    if len(values) < MAIN_ZERO_SAMPLE_COUNT:
                        print(f"⚠️ {reason}归零失败：有效力帧不足")
                        return
                    avg = np.mean(values, axis=0)
                    sensor_force.add_zero_bias(float(avg[0]), float(avg[1]))
                    print(f"🔄 {reason}，Fx/Fy已归零")
                finally:
                    rezero_guard.release()

            task = threading.Thread(target=worker, daemon=True)
            rezero_threads.append(task)
            task.start()

        def process_pressure(press_item):
            """推进一次 CoP 状态并返回可延迟匹配力帧的完整快照。"""
            nonlocal prev_refined, prev_contact
            nonlocal pressure_start_t, last_rel_ms

            base_sub_arr = np.asarray(press_item["data"], dtype=np.float64)
            frame2d = base_sub_arr.reshape(cop_sensor.rows, cop_sensor.cols)
            cop_sensor.dynamic_threshold(frame2d)

            use_full = MAIN_REGION_MODE in ("full", "both")
            use_region = MAIN_REGION_MODE in ("region", "both")

            if use_full:
                pzt_angle_deg, cop_delta_x, cop_delta_y, cop_curr_x, cop_curr_y = (
                    cop_sensor.get_all(base_sub_arr)
                )
                origin_x, origin_y = cop_sensor.get_origin()
                cop_state = cop_sensor.get_state()
                gradient_arr = cop_sensor.get_gradient(base_sub_arr)
                centroid_xy = cop_sensor._compute_centroid(frame2d)
            else:
                pzt_angle_deg = 0.0
                cop_delta_x = cop_delta_y = 0.0
                cop_curr_x = cop_curr_y = float("nan")
                origin_x = origin_y = None
                cop_state = 0
                gradient_arr = np.zeros((12, 7, 2), dtype=np.float32)
                centroid_xy = None

            contact_init = cop_state > 0
            if use_region:
                region_list = cop_sensor._compute_region_delta_cop(frame2d)
                region_mask = np.zeros((cop_sensor.rows, cop_sensor.cols), dtype=np.int32)
                for region in region_list:
                    for row, col in region["coords"]:
                        region_mask[row, col] = region["id"]
            else:
                region_list = []
                region_mask = np.zeros((cop_sensor.rows, cop_sensor.cols), dtype=np.int32)

            contact_init_display = contact_init
            if use_region and not use_full:
                contact_init_display = any(
                    region.get("contact_init", False) for region in region_list
                )

            refined = cop_state == 2
            if MAIN_REFINE_REZERO_FORCE and refined and not prev_refined:
                schedule_rezero("COP精修完成")
            prev_refined = refined
            if prev_contact and not contact_init:
                schedule_rezero("力卸载")
            prev_contact = contact_init

            buf_cop_delta_x.append(cop_delta_x)
            buf_cop_delta_y.append(cop_delta_y)
            cop_delta_x_filt = float(np.median(buf_cop_delta_x))
            cop_delta_y_filt = float(np.median(buf_cop_delta_y))
            total_press_val = float(np.sum(base_sub_arr))

            cal_fx_val = cal_fy_val = cal_fz_val = cal_angle_deg = None
            if fit_params_list is not None:
                inputs = [cop_delta_x_filt, cop_delta_y_filt]
                if MAIN_CAL_DIM == "3D":
                    inputs.append(total_press_val)
                results = apply_predict_multi(
                    inputs, fit_params_list, fit_type, fit_split_sign
                )
                if len(results) >= 3:
                    cal_fx_val, cal_fy_val, cal_fz_val = results[:3]
                elif len(results) >= 2:
                    cal_fx_val, cal_fy_val = results[:2]
                if cal_fx_val is not None and cal_fy_val is not None:
                    cal_angle_deg = pzt.compute_vector_angle(cal_fx_val, cal_fy_val)

            if pressure_start_t is None:
                pressure_start_t = press_item["t"]
            rel_time_ms = max(
                last_rel_ms,
                int(round((press_item["t"] - pressure_start_t) * 1000)),
            )
            last_rel_ms = rel_time_ms
            return {
                "press_item": press_item,
                "rel_ms": rel_time_ms,
                "base": base_sub_arr,
                "total": total_press_val,
                "angle": pzt_angle_deg,
                "cop_x": cop_curr_x,
                "cop_y": cop_curr_y,
                "origin_x": origin_x,
                "origin_y": origin_y,
                "dx": cop_delta_x_filt,
                "dy": cop_delta_y_filt,
                "state": cop_state,
                "contact": contact_init,
                "display_contact": contact_init_display,
                "refined": refined,
                "gradient": gradient_arr,
                "table_angle": (-pzt_angle_deg) % 360.0,
                "region_mask": region_mask,
                "regions": region_list,
                "centroid": centroid_xy,
                "cal_fx": cal_fx_val,
                "cal_fy": cal_fy_val,
                "cal_fz": cal_fz_val,
                "cal_angle": cal_angle_deg,
            }

        def write_snapshot(snapshot, force_item):
            """写一行 CSV；force_item=None 表示该压力帧没有匹配力帧。"""
            nonlocal row_count, force_fx_filt, force_fy_filt
            nonlocal force_fz_filt, force_angle_deg
            nonlocal first_saved_press_t, previous_saved_press_t

            press_timestamp = float(snapshot["press_item"]["t"])
            if first_saved_press_t is None:
                csv_rel_ms = 0.0
                csv_delta_ms = 0.0
            else:
                csv_rel_ms = max(
                    0.0,
                    round((press_timestamp - first_saved_press_t) * 1000.0, 6),
                )
                csv_delta_ms = max(
                    0.0,
                    round((press_timestamp - previous_saved_press_t) * 1000.0, 6),
                )

            if force_item is None:
                force_data = nan6
                force_ts = float("nan")
                row_fx = row_fy = row_fz = float("nan")
                row_angle = float("nan")
            else:
                force_data = force_item["data"]
                force_ts = force_item["t"]
                raw_fx, raw_fy, raw_fz = force_data[:3]
                buf_force_fx.append(raw_fx)
                buf_force_fy.append(raw_fy)
                buf_force_fz.append(raw_fz)
                force_fx_filt = float(np.median(buf_force_fx))
                force_fy_filt = float(np.median(buf_force_fy))
                force_fz_filt = float(np.median(buf_force_fz))
                force_angle_deg = pzt.compute_6Dforce_angle(
                    force_fx_filt, force_fy_filt
                )
                row_fx, row_fy, row_fz = (
                    force_fx_filt, force_fy_filt, force_fz_filt
                )
                row_angle = force_angle_deg

            csv_writer.writerow(build_csv_row(
                press_timestamp=press_timestamp,
                rel_ms=csv_rel_ms,
                delta_ms=csv_delta_ms,
                ch_data=snapshot["base"],
                force_data=force_data,
                force_timestamp=force_ts,
                delta_cop_x=snapshot["dx"],
                delta_cop_y=snapshot["dy"],
                delta_force_x=row_fx,
                delta_force_y=row_fy,
                delta_force_z=row_fz,
                adc_angle=snapshot["angle"],
                force_angle=row_angle,
                fx_cal=snapshot["cal_fx"],
                fy_cal=snapshot["cal_fy"],
                force_cal_angle=snapshot["cal_angle"],
                cop_state=snapshot["state"],
                adc_sum=snapshot["total"],
                valid=1 if snapshot["state"] > 0 else 0,
            ))
            csv_file_obj.flush()
            row_count += 1
            if first_saved_press_t is None:
                first_saved_press_t = press_timestamp
            previous_saved_press_t = press_timestamp

        def drain_pending(now):
            """按压力时间顺序一对一匹配力帧。"""
            nonlocal last_force_seq
            while pending_press:
                snapshot = pending_press[0]
                press_ts = snapshot["press_item"]["t"]
                force_item = match_closest(
                    buf_force,
                    press_ts,
                    MAIN_MAX_TIME_DIFF_S,
                    min_seq=last_force_seq,
                )
                if force_item is not None:
                    pending_press.popleft()
                    last_force_seq = force_item["seq"]
                    write_snapshot(snapshot, force_item)
                    continue
                if now - press_ts > MAIN_MAX_TIME_DIFF_S:
                    pending_press.popleft()
                    continue
                break

        def percentile_ms(values, percentile):
            if not values:
                return float("nan")
            return float(np.percentile(values, percentile) * 1000.0)

        def log_timing_stats(now):
            """每秒报告两路采集真实频率、延迟和错误累计值。"""
            nonlocal last_stats_log_t, last_stats_frames
            nonlocal last_force_stats_frames
            if now - last_stats_log_t < MAIN_TIMING_LOG_INTERVAL_S:
                return
            stats = sensor_press.get_timing_stats()
            elapsed = now - last_stats_log_t
            frame_count = stats["frames"]
            fps = (frame_count - last_stats_frames) / elapsed
            tx_p50 = percentile_ms(stats["tx_intervals_s"], 50)
            tx_p95 = percentile_ms(stats["tx_intervals_s"], 95)
            latency_p50 = percentile_ms(stats["latencies_s"], 50)
            latency_p95 = percentile_ms(stats["latencies_s"], 95)
            print(
                "⏱ 压力时序: "
                f"{fps:.1f} Hz, 请求间隔 P50/P95="
                f"{tx_p50:.2f}/{tx_p95:.2f} ms, 响应延迟 P50/P95="
                f"{latency_p50:.2f}/{latency_p95:.2f} ms, "
                f"超时={stats['response_timeouts']}, "
                f"CRC={stats['crc_errors']}, 状态={stats['status_errors']}, "
                f"队列丢帧={stats['queue_drops']}, "
                f"跳过周期={stats['schedule_skips']}"
            )
            if has_force:
                force_stats = sensor_force.get_timing_stats()
                force_frame_count = force_stats["frames"]
                force_fps = (force_frame_count - last_force_stats_frames) / elapsed
                force_tx_p50 = percentile_ms(force_stats["tx_intervals_s"], 50)
                force_tx_p95 = percentile_ms(force_stats["tx_intervals_s"], 95)
                force_latency_p50 = percentile_ms(force_stats["latencies_s"], 50)
                force_latency_p95 = percentile_ms(force_stats["latencies_s"], 95)
                print(
                    "⏱ 六维力时序: "
                    f"{force_fps:.1f} Hz, 请求间隔 P50/P95="
                    f"{force_tx_p50:.2f}/{force_tx_p95:.2f} ms, 响应延迟 P50/P95="
                    f"{force_latency_p50:.2f}/{force_latency_p95:.2f} ms, "
                    f"超时={force_stats['response_timeouts']}, "
                    f"帧头错误={force_stats['framing_errors']}, "
                    f"尾部错误={force_stats['tail_errors']}, "
                    f"读错={force_stats['serial_read_errors']}, "
                    f"写错={force_stats['serial_write_errors']}, "
                    f"队列丢帧={force_stats['queue_drops']}, "
                    f"跳过周期={force_stats['schedule_skips']}"
                )
            last_stats_log_t = now
            last_stats_frames = frame_count
            if has_force:
                last_force_stats_frames = force_frame_count

        while not g_main_stop_flag.is_set():
            loop_start_s = time.perf_counter()
            if thread_press.error is not None:
                raise RuntimeError(f"压力采集线程异常: {thread_press.error}")
            if thread_force is not None and thread_force.error is not None:
                raise RuntimeError(f"六维力采集线程异常: {thread_force.error}")
            new_items = buf_press.get_after(last_press_seq)
            latest_snapshot = None

            for press_item in new_items:
                last_press_seq = press_item["seq"]
                latest_snapshot = process_pressure(press_item)
                if has_force:
                    pending_press.append(latest_snapshot)
                else:
                    write_snapshot(latest_snapshot, None)

            if has_force:
                drain_pending(time.perf_counter())

            log_timing_stats(time.perf_counter())

            if latest_snapshot is not None:
                now = time.perf_counter()
                if now - last_plot_t >= plot_interval_s:
                    plot.set_data(
                        latest_snapshot["angle"], force_angle_deg,
                        latest_snapshot["base"], latest_snapshot["total"],
                        latest_snapshot["cop_x"], latest_snapshot["cop_y"],
                        latest_snapshot["origin_x"], latest_snapshot["origin_y"],
                        latest_snapshot["dx"], latest_snapshot["dy"],
                        force_fx_filt, force_fy_filt, force_fz_filt,
                        latest_snapshot["cal_fx"], latest_snapshot["cal_fy"],
                        latest_snapshot["cal_fz"],
                        cop_state=latest_snapshot["state"],
                        gradient=latest_snapshot["gradient"],
                        contact_init=latest_snapshot["display_contact"],
                        refined=latest_snapshot["refined"],
                        pzt_table_angle_deg=latest_snapshot["table_angle"],
                        region_mask=latest_snapshot["region_mask"],
                        regions=latest_snapshot["regions"],
                        centroid=latest_snapshot["centroid"],
                    )
                    if latest_snapshot["contact"]:
                        plot.append_full_data(
                            latest_snapshot["rel_ms"],
                            latest_snapshot["angle"], latest_snapshot["total"],
                            latest_snapshot["dx"], latest_snapshot["dy"],
                            force_angle_deg,
                            force_fz_filt, force_fx_filt, force_fy_filt,
                            latest_snapshot["cal_angle"],
                            latest_snapshot["cal_fx"], latest_snapshot["cal_fy"],
                            latest_snapshot["cal_fz"],
                        )
                    last_plot_t = now

            elapsed = time.perf_counter() - loop_start_s
            time.sleep(max(0.001, 1 / MAIN_TARGET_FPS - elapsed))

    finally:
        g_main_stop_flag.set()
        if thread_press is not None and thread_press.is_alive():
            thread_press.join(timeout=2)
        if thread_force is not None and thread_force.is_alive():
            thread_force.join(timeout=2)
        for task in rezero_threads:
            if task.is_alive():
                task.join(timeout=1)
        for sensor in (sensor_press, sensor_force):
            if sensor is not None:
                try:
                    sensor.close()
                except Exception:
                    pass
        if csv_file_obj is not None:
            csv_file_obj.close()
        if csv_path is not None:
            if row_count == 0 and os.path.exists(csv_path):
                os.remove(csv_path)
                print("⚠️ 无数据，CSV 已删除")
            elif row_count > 0:
                print(f"✅ CSV已关闭（{row_count} 行）")


def _data_thread_entry(plot, error_queue):
    try:
        data_loop(plot)
    except Exception as exc:
        error_queue.put(exc)
        g_main_stop_flag.set()


# ===================== 主函数 =====================
def main():
    g_main_stop_flag.clear()
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    plot = RealTimePlot()
    errors = queue.Queue()
    data_thread = threading.Thread(
        target=_data_thread_entry, args=(plot, errors), daemon=True
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
        plot.plot_full_analysis(MAIN_SAVE_DIR)


if __name__ == "__main__":
    main()
