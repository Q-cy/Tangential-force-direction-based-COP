# file_name: main.py

import time
import os
from collections import deque
import numpy as np
import threading
from pyqtgraph.Qt import QtWidgets
import sys

import angle as angle
import COP as COP
import data as data
import table as table
import calibrate
import importlib

# ===================== 配置 =====================
MAIN_REALTIME_MODULE = "realtime"           # "realtime"=全显示, "realtime2"=仅压阻
MAIN_SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"  # 数据保存根目录

realtime = importlib.import_module(MAIN_REALTIME_MODULE)
MAIN_TARGET_FPS = 200                      # 目标采集帧率
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
                    d = self.s.decode(raw)
                    self.buf.append({"t":ts,"data":d})
                except:
                    pass
            time.sleep(0.001)

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
    csv_path = table.auto_get_csv_path(MAIN_SAVE_DIR)
    # 初始化CSV文件（写入表头）
    csv_writer, csv_file_obj = table.init_csv_file(csv_path)

    has_press = True          # 是否有压力传感器
    has_force = True           # 是否有六维力传感器

    try:
        sensor_press = data.PressureSensor()
        buf_press = data.TimestampedBuffer(500)
        thread_press = PressureThread(sensor_press, buf_press)
        thread_press.start()
        print("✅ 压力传感器就绪")
    except Exception as e:
        has_press = False
        buf_press = None
        print(f"⚠️ 压力传感器未连接: {e}")

    try:
        sensor_force = data.SixAxisForceSensor()
        sensor_force.calibrate_zero()
        buf_force = data.TimestampedBuffer(500)
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

    # 加载标定查找表
    cal_npz_path = os.path.join(MAIN_SAVE_DIR, "cal_lookup.npz")
    cal_lut_ready_flag = False
    cal_pts_arr = cal_fx_arr = cal_fy_arr = None
    if os.path.exists(cal_npz_path):
        try:
            cal_pts_arr, cal_fx_arr, cal_fy_arr = calibrate.load_lookup(cal_npz_path)
            cal_lut_ready_flag = True
            print(f"📐 标定查找表已加载: {cal_npz_path}")
        except Exception as e:
            print(f"⚠️ 标定查找表加载失败: {e}")
    else:
        print("💡 未找到标定文件")

    median_filt_window = 5
    buf_cop_delta_x = deque(maxlen=median_filt_window)
    buf_cop_delta_y = deque(maxlen=median_filt_window)
    buf_force_fx = deque(maxlen=median_filt_window)
    buf_force_fy = deque(maxlen=median_filt_window)
    buf_force_fz = deque(maxlen=median_filt_window)

    _NAN6 = [float('nan')] * 6  # 力传感器占位

    while not g_main_stop_flag.is_set():
        loop_start_s = time.perf_counter()
        rel_time_ms = int((loop_start_s - start_time_s) * 1000)

        # ---- 采集压力数据 ----
        press_item = buf_press.get_latest() if has_press else None
        force_item = None

        if press_item is not None and has_force:
            force_item = buf_force.find_closest(press_item["t"])
            if force_item is not None and abs(press_item["t"] - force_item["t"]) > MAIN_MAX_TIME_DIFF_S:
                force_item = None
        elif has_force:
            force_item = buf_force.get_latest()

        if press_item is None and force_item is None:
            time.sleep(0.001)
            continue

        # ---- 计算 PZT / CoP ----
        if press_item is not None:
            base_sub_arr = COP.subtract_baseline(press_item["data"])
            cop_res = COP.compute_pressure_direction(base_sub_arr)
            cop_curr_x, cop_curr_y = cop_res[0], cop_res[1]
            cop_delta_x, cop_delta_y = cop_res[6], cop_res[7]
            cop_base_x, cop_base_y = cop_res[8], cop_res[9]
            total_press_val = np.sum(press_item["data"])

            buf_cop_delta_x.append(cop_delta_x)
            buf_cop_delta_y.append(cop_delta_y)
            cop_delta_x_filt = np.median(buf_cop_delta_x)
            cop_delta_y_filt = np.median(buf_cop_delta_y)
            pzt_angle_deg, pzt_mag_val = angle.compute_PZT_angle(cop_delta_x_filt, cop_delta_y_filt)
        else:
            base_sub_arr = np.zeros(84)
            cop_curr_x = cop_curr_y = cop_delta_x = cop_delta_y = cop_base_x = cop_base_y = float('nan')
            cop_delta_x_filt = cop_delta_y_filt = 0.0
            pzt_angle_deg = pzt_mag_val = 0.0
            total_press_val = 0.0

        # ---- 计算 Force ----
        if force_item is not None:
            force_fx_val, force_fy_val, force_fz_val = force_item["data"][:3]
            buf_force_fx.append(force_fx_val)
            buf_force_fy.append(force_fy_val)
            buf_force_fz.append(force_fz_val)
            force_fx_filt = np.median(buf_force_fx)
            force_fy_filt = np.median(buf_force_fy)
            force_fz_filt = np.median(buf_force_fz)
            force_angle_deg, force_mag_val = angle.compute_6Dforce_angle(force_fx_filt, force_fy_filt)
            force_data_out = force_item["data"]
            force_ts_out = force_item["t"]
        else:
            force_fx_val = force_fy_val = force_fz_val = float('nan')
            force_fx_filt = force_fy_filt = force_fz_filt = float('nan')
            force_angle_deg = force_mag_val = float('nan')
            force_data_out = _NAN6
            force_ts_out = float('nan')

        # ---- 标定 ----
        if cal_lut_ready_flag and press_item is not None:
            cal_fx_val, cal_fy_val = calibrate.apply(cop_delta_x_filt, cop_delta_y_filt, cal_pts_arr, cal_fx_arr, cal_fy_arr)
            cal_angle_deg, cal_mag_val = angle.compute_vector_angle(cal_fx_val, cal_fy_val)
        else:
            cal_fx_val = cal_fy_val = cal_angle_deg = cal_mag_val = None

        # ---- CSV ----
        press_ts = press_item["t"] if press_item is not None else float('nan')
        csv_row = table.build_csv_row(
            press_timestamp=press_ts,
            rel_ms=rel_time_ms,
            ch_data=press_item["data"] if press_item is not None else [0]*84,
            force_data=force_data_out,
            force_timestamp=force_ts_out,
            delta_cop_x=cop_delta_x_filt,
            delta_cop_y=cop_delta_y_filt,
            delta_force_x=force_fx_filt,
            delta_force_y=force_fy_filt,
            delta_force_z=force_fz_filt,
            adc_angle=pzt_angle_deg,
            adc_mag=pzt_mag_val,
            force_angle=force_angle_deg,
            force_mag=force_mag_val,
            fx_cal=cal_fx_val,
            fy_cal=cal_fy_val,
            force_cal_mag=cal_mag_val,
            force_cal_angle=cal_angle_deg,
        )
        csv_writer.writerow(csv_row)

        # ---- 更新绘图 ----
        g_main_plot.set_data(
            pzt_angle_deg, pzt_mag_val, force_angle_deg, force_mag_val,
            base_sub_arr, total_press_val, force_mag_val,
            cop_curr_x, cop_curr_y, cop_base_x, cop_base_y,
            cop_delta_x_filt, cop_delta_y_filt,
            force_fx_filt, force_fy_filt, force_fz_filt,
            cal_fx_val, cal_fy_val, cal_angle_deg, cal_mag_val,
        )
        if COP.g_cop_contact_init_flag:
            g_main_plot.append_full_data(
                rel_time_ms,
                pzt_angle_deg, pzt_mag_val, total_press_val,
                cop_delta_x_filt, cop_delta_y_filt,
                force_angle_deg, force_mag_val,
                force_fz_filt, force_fx_filt, force_fy_filt,
                cal_angle_deg, cal_mag_val, cal_fx_val, cal_fy_val)

        elapsed = time.perf_counter() - loop_start_s
        time.sleep(max(0, 1/MAIN_TARGET_FPS - elapsed))

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

    g_main_plot = realtime.RealTimePlot()
    data_thread = threading.Thread(target=data_loop)
    data_thread.start()

    app.exec()

    g_main_stop_flag.set()
    data_thread.join(timeout=2)
    g_main_plot.plot_full_magnitude_curve(MAIN_SAVE_DIR)

if __name__ == "__main__":
    main()