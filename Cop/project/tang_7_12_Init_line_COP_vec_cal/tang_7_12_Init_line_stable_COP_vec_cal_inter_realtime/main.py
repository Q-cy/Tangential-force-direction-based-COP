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
import realtime as realtime
import table as table
import calibrate

# ===================== 配置 =====================
MAIN_SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"  # 数据保存根目录
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

    sensor_press = data.PressureSensor()         # 压力传感器对象
    sensor_force = data.SixAxisForceSensor()     # 六维力传感器对象
    sensor_force.calibrate_zero()
    print("✅ 传感器初始化完成")

    buf_press = data.TimestampedBuffer(500)      # 压力数据时间戳缓冲
    buf_force = data.TimestampedBuffer(500)      # 力数据时间戳缓冲

    thread_press = PressureThread(sensor_press, buf_press)
    thread_force = ForceThread(sensor_force, buf_force)
    thread_press.start()
    thread_force.start()

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
        print("💡 未找到标定文件。如需标定，先运行本程序采集CSV，然后执行：")
        print("   python /home/qcy/Project/code/Tangential/finger_tang_7_12/Cop/project/tang_7_12_Init_line_stable_COP_vec_cal_inter_realtime/calibrate.py 1")

    median_filt_window = 5  # 中值滤波窗口大小
    buf_cop_delta_x = deque(maxlen=median_filt_window)
    buf_cop_delta_y = deque(maxlen=median_filt_window)
    buf_force_fx = deque(maxlen=median_filt_window)
    buf_force_fy = deque(maxlen=median_filt_window)
    buf_force_fz = deque(maxlen=median_filt_window)

    while not g_main_stop_flag.is_set():
        loop_start_s = time.perf_counter()
        rel_time_ms = int((loop_start_s - start_time_s) * 1000)

        press_item = buf_press.get_latest()
        if not press_item:
            time.sleep(0.001)
            continue

        force_item = buf_force.find_closest(press_item["t"])
        if not force_item or abs(press_item["t"] - force_item["t"]) > MAIN_MAX_TIME_DIFF_S:
            time.sleep(0.001)
            continue

        # CoP 计算
        base_sub_arr = COP.subtract_baseline(press_item["data"])
        cop_res = COP.compute_pressure_direction(base_sub_arr)
        cop_curr_x, cop_curr_y = cop_res[0], cop_res[1]
        cop_delta_x, cop_delta_y = cop_res[6], cop_res[7]
        cop_base_x, cop_base_y = cop_res[8], cop_res[9]

        # 六维力传感器数据
        force_fx_val, force_fy_val, force_fz_val = force_item["data"][:3]

        # 中值滤波
        buf_cop_delta_x.append(cop_delta_x)
        buf_cop_delta_y.append(cop_delta_y)
        buf_force_fx.append(force_fx_val)
        buf_force_fy.append(force_fy_val)
        buf_force_fz.append(force_fz_val)
        cop_delta_x_filt = np.median(buf_cop_delta_x)
        cop_delta_y_filt = np.median(buf_cop_delta_y)
        force_fx_filt = np.median(buf_force_fx)
        force_fy_filt = np.median(buf_force_fy)
        force_fz_filt = np.median(buf_force_fz)
        total_press_val = np.sum(press_item["data"])

        # 角度和幅值
        pzt_angle_deg, pzt_mag_val = angle.compute_PZT_angle(cop_delta_x_filt, cop_delta_y_filt)
        force_angle_deg, force_mag_val = angle.compute_6Dforce_angle(force_fx_filt, force_fy_filt)

        # 标定
        if cal_lut_ready_flag:
            cal_fx_val, cal_fy_val = calibrate.apply(cop_delta_x_filt, cop_delta_y_filt, cal_pts_arr, cal_fx_arr, cal_fy_arr)
            cal_angle_deg, cal_mag_val = angle.compute_vector_angle(cal_fx_val, cal_fy_val)
        else:
            cal_fx_val, cal_fy_val, cal_angle_deg, cal_mag_val = None, None, None, None

        # CSV 行数据
        csv_row = table.build_csv_row(
            press_timestamp=press_item["t"],
            rel_ms=rel_time_ms,
            ch_data=press_item["data"],
            force_data=force_item["data"],
            force_timestamp=force_item["t"],
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

        # 写入CSV行
        csv_writer.writerow(csv_row)
        csv_file_obj.flush()  # 立即刷新到文件

        # 更新绘图数据
        g_main_plot.set_data(
            pzt_angle_deg, pzt_mag_val, force_angle_deg, force_mag_val,
            base_sub_arr, total_press_val, force_mag_val,
            cop_curr_x, cop_curr_y, cop_base_x, cop_base_y,
            cop_delta_x_filt, cop_delta_y_filt,
            force_fx_filt, force_fy_filt, force_fz_filt,
            cal_fx_val, cal_fy_val, cal_angle_deg, cal_mag_val,
        )
        # 追加全程数据
        if COP.g_cop_contact_init_flag:
                    g_main_plot.append_full_data(
                        rel_time_ms,
                        pzt_angle_deg, pzt_mag_val, total_press_val,
                        cop_delta_x_filt, cop_delta_y_filt,
                        force_angle_deg, force_mag_val,
                        force_fz_filt, force_fx_filt, force_fy_filt,
                        cal_angle_deg, cal_mag_val, cal_fx_val, cal_fy_val)

        # 控制采集频率
        elapsed = time.perf_counter() - loop_start_s
        time.sleep(max(0, 1/MAIN_TARGET_FPS - elapsed))

    # 关闭CSV文件
    csv_file_obj.close()
    print("✅ CSV文件已关闭")

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