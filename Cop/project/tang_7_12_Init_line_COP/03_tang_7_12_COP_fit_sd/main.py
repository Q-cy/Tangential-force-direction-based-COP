# file_name: main.py

import time
import os
from collections import deque
import numpy as np
import threading
from pyqtgraph.Qt import QtWidgets
import sys

from data import PressureSensor, SixAxisForceSensor, TimestampedBuffer, match_closest
from realtime import RealTimePlot
from table import auto_get_csv_path, init_csv_file, build_csv_row
from fit import load_coefs, apply_predict_multi
import tang_7_12_InitCOP_realtime_other_package as pzt
from tang_7_12_InitCOP_realtime_package_note import PZTSensorAngle

# ===================== 配置 =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SAVE_DIR = os.path.join(BASE_DIR,"../../../../../../../data/2.PZT_tangential/weight/test")  # 数据保存根目录
fit_coefs_path = os.path.join(BASE_DIR,"./fit_coefs.bin")   # fit 模型路径
MAIN_CAL_MODE = "fit"                       # 唯一支持的标定模式（calibrate.py 已移除）
MAIN_CAL_DIM = "3D"                         # "2D"=仅切向力(Fx,Fy), "3D"=三维力(Fz,Fx,Fy)
MAIN_REFINE_REZERO_FORCE = True             # True=COP二次精修后重新置零六维力

MAIN_TARGET_FPS = 100                      # sub thread 限速 (Hz), 防止 GIL 争抢让 GUI 卡
MAIN_READ_INTERVAL_S = 0.005               # 采集线程读取间隔(秒)，0.005=200Hz tick (sensor 帧间隔 ~12ms)
MAIN_PLOT_FPS = 60                         # plot set_data 限速 (Hz), 防止 GUI 卡顿
MAIN_MAX_TIME_DIFF_S = 0.015               # 压力-力传感器最大时间匹配差(秒)
g_main_stop_flag = threading.Event()       # 全局停止信号
g_main_plot = None                         # 绘图对象引用

# ===================== 采集线程 =====================
class PressureThread(threading.Thread):                   
    def __init__(self, sensor, buf):                      
        super().__init__(daemon=True)
        self.s = sensor
        self.buf = buf
    def run(self):
        while not g_main_stop_flag.is_set():
            ts = time.perf_counter()

            raw = self.s.read_data()
            if raw:
                try:
                    d = np.asarray(self.s.decode(raw), dtype=np.float64)
                    self.buf.append({"t":ts,"data":d})
                except:
                    pass
            time.sleep(MAIN_READ_INTERVAL_S)

class ForceThread(threading.Thread):
    def __init__(self, sensor, buf):
        super().__init__(daemon=True)
        self.s = sensor
        self.buf = buf
    def run(self):
        while not g_main_stop_flag.is_set():
            ts = time.perf_counter()
            d = self.s.read()
            if d:
                self.buf.append({"t":ts,"data":d})
            time.sleep(0.001)

# ===================== 数据循环 =====================
def data_loop():
    global g_main_plot
    # 自动获取CSV文件路径
    csv_path = auto_get_csv_path(MAIN_SAVE_DIR)
    # 初始化CSV文件（写入表头）
    csv_writer, csv_file_obj = init_csv_file(csv_path)

    has_press = True          # 是否有压力传感器
    has_force = True           # 是否有六维力传感器

    try:
        sensor_press = PressureSensor()
        buf_press = TimestampedBuffer(500)
        thread_press = PressureThread(sensor_press, buf_press)
        thread_press.start()
        print("✅ 压力传感器就绪")
    except Exception as e:
        has_press = False
        buf_press = None
        print(f"⚠️ 压力传感器未连接: {e}")

    try:
        sensor_force = SixAxisForceSensor()
        sensor_force.calibrate_zero()
        buf_force = TimestampedBuffer(500)
        thread_force = ForceThread(sensor_force, buf_force)
        thread_force.start()
        print("✅ 六维力传感器就绪")
    except Exception as e:
        has_force = False
        buf_force = None
        print(f"⚠️ 六维力传感器未连接: {e}")

    if not has_press and not has_force:
        print("❌ 无任何传感器，退出")
        return

    print("🎨 绘图已打开")
    start_time_s = time.perf_counter()

    # 加载 fit 标定模型
    fit_type = None
    fit_params_list = None
    fit_split_sign = False
    if os.path.exists(fit_coefs_path):
        try:
            fit_type, _, fit_params_list, fit_split_sign = load_coefs(fit_coefs_path)
            type_summary = ", ".join(f"{p[1]}{'(split)' if p[2] else ''}" for p in fit_params_list)
            print(f"📐 fit模型已加载: {fit_coefs_path} (outputs: {type_summary})")
        except Exception as e:
            print(f"⚠️ fit模型加载失败: {e}")
            fit_split_sign = False
    else:
        print("💡 未找到 fit 模型文件")

    median_filt_window = 5
    buf_cop_delta_x = deque(maxlen=median_filt_window)
    buf_cop_delta_y = deque(maxlen=median_filt_window)
    buf_force_fx = deque(maxlen=median_filt_window)
    buf_force_fy = deque(maxlen=median_filt_window)
    buf_force_fz = deque(maxlen=median_filt_window)

    cop_sensor = PZTSensorAngle()  # 每帧唯一推进 CoP 状态的入口

    _NAN6 = [float('nan')] * 6  # 力传感器占位
    _prev_refined = False       # COP精修状态（用于检测跳变）
    _prev_contact = False       # COP接触状态（用于检测力卸载）
    _last_plot_t = 0.0           # plot 节流: 上次 plot 更新时间
    PLOT_INTERVAL_S = 1.0 / MAIN_PLOT_FPS   # plot 限速间隔 (秒)

    while not g_main_stop_flag.is_set():
        loop_start_s = time.perf_counter()
        rel_time_ms = int((loop_start_s - start_time_s) * 1000)

        # ---- 采集压力数据 ----
        press_item = buf_press.get_latest() if has_press else None
        force_item = buf_force.get_latest() if has_force else None

        if press_item is None and force_item is None:
            time.sleep(0.001)
            continue

        # ---- F5 严格时间同步: 双传感器时以较新帧为锚点, 在另一缓冲找
        #      MAIN_MAX_TIME_DIFF_S 窗内最近帧; 匹配成功才写 CSV 行,
        #      保证行内 press/force 时间差有界。状态机/绘图仍用最新帧逐帧驱动。
        csv_press_item = press_item
        csv_force_item = force_item
        if has_press and has_force:
            if force_item["t"] >= press_item["t"]:
                csv_press_item = match_closest(buf_press, force_item["t"], MAIN_MAX_TIME_DIFF_S)
            else:
                csv_force_item = match_closest(buf_force, press_item["t"], MAIN_MAX_TIME_DIFF_S)
        write_row = (csv_press_item is not None) and (csv_force_item is not None)

        # ---- 计算 PZT / CoP ----
        if press_item is not None:
            base_sub_arr = np.array(press_item["data"])

            # 阈值学习（外部 API, 同时更新 _total_thresh 和 _pixel_thresh）
            frame2d = base_sub_arr.reshape(cop_sensor.rows, cop_sensor.cols)
            cop_sensor.dynamic_threshold(frame2d)

            pzt_angle_deg, cop_delta_x, cop_delta_y, cop_curr_x, cop_curr_y = cop_sensor.get_all(base_sub_arr)
            origin_x, origin_y = cop_sensor.get_origin()
            cop_state = cop_sensor.get_state()
            cop_base_x = cop_curr_x if origin_x is None else origin_x
            cop_base_y = cop_curr_y if origin_y is None else origin_y
            gradient_arr = cop_sensor.get_gradient(base_sub_arr)
            contact_init = cop_state > 0
            refined = cop_state == 2
            total_press_val = float(np.sum(press_item["data"]))
            pzt_table_angle_deg = (-pzt_angle_deg) % 360.0

            # per-region CoP / delta (与整帧同一 frame2d 对齐);
            # region_id 取稳定 id (F11), reset_region_origin 按足迹判定"已放手"(F7)
            region_results = cop_sensor._compute_region_delta_cop(frame2d)
            for region in region_results:
                region_id = region['id']
                cop_sensor.reset_region_origin(region_id, frame2d)
                print(f"  region {region_id}: cop={region['cop']}, "
                      f"delta={region['delta']}, "
                      f"area={region['area']}, "
                      f"total_p={region['total_pressure']:.1f}")

            print(f"frame {rel_time_ms}ms: angle={pzt_angle_deg:.1f}°, "
                  f"dx={cop_delta_x:+.2f}, dy={cop_delta_y:+.2f}, "
                  f"cop=({cop_curr_x:.2f},{cop_curr_y:.2f})")

            # COP精修完成后重新归零 Fx/Fy（Fz不变，10帧平均）
            if MAIN_REFINE_REZERO_FORCE and has_force:
                if refined and not _prev_refined:
                    def _rezero():
                        vals = []
                        for _ in range(10):
                            d = sensor_force.read()
                            if d:
                                vals.append(d)
                            time.sleep(0.001)
                        if vals:
                            avg = np.mean(vals, axis=0)
                            sensor_force.add_zero_bias(avg[0], avg[1])
                            print("🔄 COP精修完成，Fx/Fy已归零")
                    threading.Thread(target=_rezero, daemon=True).start()
            _prev_refined = refined

            # 力卸载后COP重置 → 六维力归零
            if has_force and _prev_contact and not contact_init:
                def _rezero_unload():
                    vals = []
                    for _ in range(10):
                        d = sensor_force.read()
                        if d:
                            vals.append(d)
                        time.sleep(0.001)
                    if vals:
                        avg = np.mean(vals, axis=0)
                        sensor_force.add_zero_bias(avg[0], avg[1])
                        print("🔄 力卸载，Fx/Fy已归零")
                threading.Thread(target=_rezero_unload, daemon=True).start()
            _prev_contact = contact_init

            buf_cop_delta_x.append(cop_delta_x)
            buf_cop_delta_y.append(cop_delta_y)
            cop_delta_x_filt = float(np.median(buf_cop_delta_x))
            cop_delta_y_filt = float(np.median(buf_cop_delta_y))
        else:
            base_sub_arr = np.zeros(84)
            cop_curr_x = cop_curr_y = cop_delta_x = cop_delta_y = cop_base_x = cop_base_y = float('nan')
            cop_delta_x_filt = cop_delta_y_filt = 0.0
            pzt_angle_deg = 0.0
            total_press_val = 0.0
            cop_state = 0
            contact_init = False
            refined = False
            gradient_arr = np.zeros((12, 7, 2), dtype=np.float32)

        # ---- 计算 Force ----
        # 滤波队列只接收匹配帧 (F5): 保证行内 force 各列 (raw+滤波+角度) 来自同一匹配帧序列
        if force_item is not None:
            if csv_force_item is not None:
                force_fx_val, force_fy_val, force_fz_val = csv_force_item["data"][:3]
                buf_force_fx.append(force_fx_val)
                buf_force_fy.append(force_fy_val)
                buf_force_fz.append(force_fz_val)
                force_fx_filt = np.median(buf_force_fx)
                force_fy_filt = np.median(buf_force_fy)
                force_fz_filt = np.median(buf_force_fz)
                force_angle_deg = pzt.compute_6Dforce_angle(force_fx_filt, force_fy_filt)
                force_data_out = csv_force_item["data"]
            else:
                # 无匹配帧: 力数据超窗, 保留上次滤波值, 本次不写行
                force_fx_val = force_fy_val = force_fz_val = float('nan')
                force_fx_filt = float(np.median(buf_force_fx)) if buf_force_fx else float('nan')
                force_fy_filt = float(np.median(buf_force_fy)) if buf_force_fy else float('nan')
                force_fz_filt = float(np.median(buf_force_fz)) if buf_force_fz else float('nan')
                force_angle_deg = float('nan')
                force_data_out = _NAN6
        else:
            force_fx_val = force_fy_val = force_fz_val = float('nan')
            force_fx_filt = force_fy_filt = force_fz_filt = float('nan')
            force_angle_deg = float('nan')
            force_data_out = _NAN6

        # ---- 标定（仅 fit 引擎）----
        cal_fx_val = cal_fy_val = cal_fz_val = cal_angle_deg = None
        if press_item is not None and fit_params_list is not None:
            if MAIN_CAL_DIM == "3D":
                x_inputs = [cop_delta_x_filt, cop_delta_y_filt, total_press_val]
            else:
                x_inputs = [cop_delta_x_filt, cop_delta_y_filt]
            results = apply_predict_multi(x_inputs, fit_params_list, fit_type, fit_split_sign)
            if len(results) >= 3:
                cal_fx_val, cal_fy_val, cal_fz_val = results[0], results[1], results[2]
            elif len(results) >= 2:
                cal_fx_val, cal_fy_val, cal_fz_val = results[0], results[1], None
            if cal_fx_val is not None:
                cal_angle_deg = pzt.compute_vector_angle(cal_fx_val, cal_fy_val)

        # ---- CSV ----
        # 行内 force 来自匹配帧 (F5), press 列来自最新帧 (状态机输出);
        # dt 列 = |press_t - 匹配 force_t|, 双传感器时 <= MAIN_MAX_TIME_DIFF_S
        press_ts = press_item["t"] if press_item is not None else None
        force_ts_now = csv_force_item["t"] if csv_force_item is not None else None

        if write_row:
            csv_row = build_csv_row(
                press_timestamp=press_ts if press_ts is not None else float('nan'),
                rel_ms=rel_time_ms,
                ch_data=press_item["data"] if press_item is not None else [0]*84,
                force_data=force_data_out,
                force_timestamp=force_ts_now if force_ts_now is not None else float('nan'),
                delta_cop_x=cop_delta_x_filt,
                delta_cop_y=cop_delta_y_filt,
                delta_force_x=force_fx_filt,
                delta_force_y=force_fy_filt,
                delta_force_z=force_fz_filt,
                adc_angle=pzt_angle_deg,
                force_angle=force_angle_deg,
                fx_cal=cal_fx_val,
                fy_cal=cal_fy_val,
                force_cal_angle=cal_angle_deg,
                cop_state=cop_state,
                adc_sum=total_press_val,
                valid=1 if cop_state > 0 else 0,
            )
            csv_writer.writerow(csv_row)

        # ---- 更新绘图 (限速到 MAIN_PLOT_FPS Hz) ----
        now = time.perf_counter()
        if now - _last_plot_t >= PLOT_INTERVAL_S:
            g_main_plot.set_data(
                pzt_angle_deg, force_angle_deg,
                base_sub_arr, total_press_val,
                cop_curr_x, cop_curr_y, cop_base_x, cop_base_y,
                cop_delta_x_filt, cop_delta_y_filt,
                force_fx_filt, force_fy_filt, force_fz_filt,
                cal_fx_val, cal_fy_val, cal_fz_val, cal_angle_deg,
                cop_state=cop_state,
                gradient=gradient_arr,
                contact_init=contact_init,
                refined=refined,
                pzt_table_angle_deg=pzt_table_angle_deg,
            )
            if contact_init:
                g_main_plot.append_full_data(
                    rel_time_ms,
                    pzt_angle_deg, total_press_val,
                    cop_delta_x_filt, cop_delta_y_filt,
                    force_angle_deg,
                    force_fz_filt, force_fx_filt, force_fy_filt,
                    cal_angle_deg, cal_fx_val, cal_fy_val, cal_fz_val)
            _last_plot_t = now

        elapsed = time.perf_counter() - loop_start_s
        time.sleep(max(0, 1/MAIN_TARGET_FPS - elapsed))   # 限速 100Hz, 让 Qt 主线程有 CPU 跑 GUI

    # F12 资源清理: 关闭串口 (close 内含线程 join, 由 try 兜底)
    if has_press:
        try:
            sensor_press.close()
        except Exception:
            pass
    if has_force:
        try:
            sensor_force.close()
        except Exception:
            pass

    csv_file_obj.close()
    row_count = sum(1 for _ in open(csv_path)) - 1
    if row_count <= 0:
        os.remove(csv_path)
        print("⚠️ 无有效数据，CSV 已删除")
    else:
        print(f"✅ CSV已关闭（{row_count} 行）")

# ===================== 主函数 =====================
def main():
    global g_main_plot
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    g_main_plot = RealTimePlot()
    data_thread = threading.Thread(target=data_loop)
    data_thread.start()

    app.exec()

    g_main_stop_flag.set()
    data_thread.join(timeout=2)
    g_main_plot.plot_full_analysis(MAIN_SAVE_DIR)

if __name__ == "__main__":
    main()