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
MAIN_CAL_MODE = "fit"                       # "lookup"=最近邻查表, "discrete"=双线性插值, "fit"=拟合, "auto"=优先拟合回退查表
MAIN_CAL_DIM = "3D"                         # "2D"=仅切向力(Fx,Fy), "3D"=三维力(Fz,Fx,Fy)
MAIN_REFINE_REZERO_FORCE = True             # True=COP二次精修后重新置零六维力

realtime = importlib.import_module(MAIN_REALTIME_MODULE)
MAIN_TARGET_FPS = 100                      # 目标采集帧率
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

    if MAIN_CAL_MODE == "fit":
        import fit

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

    # 加载标定模型（查找表 + 拟合）
    cal_bin_path = os.path.join(MAIN_SAVE_DIR, "cal_lookup.bin")
    cal_fit_path = os.path.join(MAIN_SAVE_DIR, "cal_fit.bin")
    fit_coefs_path = "/home/qcy/Project/data/2.PZT_tangential/weight/png/fit_coefs.bin"
    cal_lut_ready_flag = False
    cal_fit_ready_flag = False
    cal_pts_arr = cal_fz_arr = cal_fx_arr = cal_fy_arr = None
    cal_coefs = None
    disc_dx_grid = disc_dy_grid = disc_fx_grid = disc_fy_grid = None
    fit_type = None
    fit_params_list = None
    fit_split_sign = False
    if os.path.exists(cal_bin_path):
        try:
            if MAIN_CAL_DIM == "3D":
                cal_pts_arr, cal_fz_arr, cal_fx_arr, cal_fy_arr = calibrate.load_lookup(cal_bin_path, dim="3D")
            else:
                cal_pts_arr, cal_fx_arr, cal_fy_arr = calibrate.load_lookup(cal_bin_path, dim="2D")
            cal_lut_ready_flag = True
            # discrete 模式：构建规则网格
            if MAIN_CAL_MODE == "discrete":
                disc_dx_grid, disc_dy_grid, disc_fx_grid, disc_fy_grid = calibrate.build_discrete_grid(cal_pts_arr, cal_fx_arr, cal_fy_arr)
            print(f"📐 查找表已加载: {cal_bin_path}")
        except Exception as e:
            print(f"⚠️ 查找表加载失败: {e}")
    if os.path.exists(cal_fit_path):
        try:
            cal_coefs = calibrate.load_fit_model(cal_fit_path, dim=MAIN_CAL_DIM)
            cal_fit_ready_flag = True
            print(f"📐 拟合模型已加载: {cal_fit_path}")
        except Exception as e:
            print(f"⚠️ 拟合模型加载失败: {e}")
    if MAIN_CAL_MODE == "fit" and os.path.exists(fit_coefs_path):
        try:
            fit_type, _, fit_params_list, fit_split_sign = fit.load_coefs(fit_coefs_path)
            cal_fit_ready_flag = True
            type_summary = ", ".join(f"{p[1]}{'(split)' if p[2] else ''}" for p in fit_params_list)
            print(f"📐 fit模型已加载: {fit_coefs_path} (outputs: {type_summary})")
        except Exception as e:
            print(f"⚠️ fit模型加载失败: {e}")
            fit_split_sign = False
    if not cal_lut_ready_flag and not cal_fit_ready_flag:
        print("💡 未找到标定文件")

    median_filt_window = 5
    buf_cop_delta_x = deque(maxlen=median_filt_window)
    buf_cop_delta_y = deque(maxlen=median_filt_window)
    buf_force_fx = deque(maxlen=median_filt_window)
    buf_force_fy = deque(maxlen=median_filt_window)
    buf_force_fz = deque(maxlen=median_filt_window)

    _NAN6 = [float('nan')] * 6  # 力传感器占位
    _prev_refined = False       # COP精修状态（用于检测跳变）
    _prev_contact = False       # COP接触状态（用于检测力卸载）

    while not g_main_stop_flag.is_set():
        loop_start_s = time.perf_counter()
        rel_time_ms = int((loop_start_s - start_time_s) * 1000)

        # ---- 采集压力数据 ----
        press_item = buf_press.get_latest() if has_press else None
        force_item = buf_force.get_latest() if has_force else None

        if press_item is None and force_item is None:
            time.sleep(0.001)
            continue

        # ---- 计算 PZT / CoP ----
        if press_item is not None:
            cop_res = COP.compute_pressure_direction(press_item["data"])
            base_sub_arr = np.array(press_item["data"])
            cop_curr_x, cop_curr_y = cop_res[0], cop_res[1]
            cop_delta_x, cop_delta_y = cop_res[6], cop_res[7]
            cop_base_x, cop_base_y = cop_res[8], cop_res[9]
            cop_state = cop_res[10]
            total_press_val = np.sum(press_item["data"])

            # COP精修完成后重新归零 Fx/Fy（Fz不变，10帧平均）
            if MAIN_REFINE_REZERO_FORCE and has_force:
                if COP.g_cop_post_refined_flag and not _prev_refined:
                    def _rezero():
                        vals = []
                        for _ in range(10):
                            d = sensor_force.read()
                            if d:
                                vals.append(d)
                            time.sleep(0.001)
                        if vals:
                            avg = np.mean(vals, axis=0)
                            sensor_force.zero_data[0] += avg[0]
                            sensor_force.zero_data[1] += avg[1]
                            print("🔄 COP精修完成，Fx/Fy已归零")
                    threading.Thread(target=_rezero, daemon=True).start()
            _prev_refined = COP.g_cop_post_refined_flag

            # 力卸载后COP重置 → 六维力归零
            if has_force and _prev_contact and not COP.g_cop_contact_init_flag:
                def _rezero_unload():
                    vals = []
                    for _ in range(10):
                        d = sensor_force.read()
                        if d:
                            vals.append(d)
                        time.sleep(0.001)
                    if vals:
                        avg = np.mean(vals, axis=0)
                        sensor_force.zero_data[0] += avg[0]
                        sensor_force.zero_data[1] += avg[1]
                        print("🔄 力卸载，Fx/Fy已归零")
                threading.Thread(target=_rezero_unload, daemon=True).start()
            _prev_contact = COP.g_cop_contact_init_flag

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
            cop_state = 0

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
        cal_fx_val = cal_fy_val = cal_fz_val = cal_angle_deg = cal_mag_val = None
        if press_item is not None:
            do_fit = MAIN_CAL_MODE == "fit" and cal_fit_ready_flag
            do_lut = MAIN_CAL_MODE == "lookup" and cal_lut_ready_flag
            do_discrete = MAIN_CAL_MODE == "discrete" and cal_lut_ready_flag
            do_auto = MAIN_CAL_MODE == "auto"

            if MAIN_CAL_DIM == "3D":
                query = [total_press_val, cop_delta_x_filt, cop_delta_y_filt]
                if do_fit and fit_params_list is not None:
                    # 3D: params order = [Fx←CoPX, Fy←CoPY, Fz←adc_sum]
                    x_inputs = [cop_delta_x_filt, cop_delta_y_filt, total_press_val]
                    results = fit.apply_predict_multi(x_inputs, fit_params_list, fit_type, fit_split_sign)
                    if len(results) >= 3:
                        cal_fx_val, cal_fy_val, cal_fz_val = results[0], results[1], results[2]
                    elif len(results) >= 2:
                        cal_fx_val, cal_fy_val = results[0], results[1]
                elif do_fit:
                    cal_fz_val, cal_fx_val, cal_fy_val = calibrate.apply_fit(query, cal_coefs, dim="3D")
                elif do_lut:
                    cal_fz_val, cal_fx_val, cal_fy_val = calibrate.apply(query, cal_pts_arr, cal_fx_arr, cal_fy_arr, fz_vals=cal_fz_arr)
                elif do_discrete:
                    cal_fx_val, cal_fy_val = calibrate.apply_discrete(cop_delta_x_filt, cop_delta_y_filt, disc_dx_grid, disc_dy_grid, disc_fx_grid, disc_fy_grid)
                elif do_auto:
                    if fit_params_list is not None:
                        x_inputs = [cop_delta_x_filt, cop_delta_y_filt, total_press_val]
                        results = fit.apply_predict_multi(x_inputs, fit_params_list, fit_type, fit_split_sign)
                        if len(results) >= 3:
                            cal_fx_val, cal_fy_val, cal_fz_val = results[0], results[1], results[2]
                        elif len(results) >= 2:
                            cal_fx_val, cal_fy_val = results[0], results[1]
                    elif cal_fit_ready_flag:
                        cal_fz_val, cal_fx_val, cal_fy_val = calibrate.apply_fit(query, cal_coefs, dim="3D")
                    elif cal_lut_ready_flag:
                        cal_fz_val, cal_fx_val, cal_fy_val = calibrate.apply(query, cal_pts_arr, cal_fx_arr, cal_fy_arr, fz_vals=cal_fz_arr)
                if cal_fx_val is not None:
                    cal_angle_deg, cal_mag_val = angle.compute_vector_angle(cal_fx_val, cal_fy_val)
            else:
                query = [cop_delta_x_filt, cop_delta_y_filt]
                if do_fit and fit_params_list is not None:
                    # 2D: Fx←CoPX, Fy←CoPY
                    x_inputs = [cop_delta_x_filt, cop_delta_y_filt]
                    results = fit.apply_predict_multi(x_inputs, fit_params_list, fit_type, fit_split_sign)
                    cal_fx_val, cal_fy_val = (results[0], results[1]) if len(results) >= 2 else (None, None)
                elif do_fit:
                    cal_fx_val, cal_fy_val = calibrate.apply_fit(query, cal_coefs, dim="2D")
                elif do_lut:
                    cal_fx_val, cal_fy_val = calibrate.apply(query, cal_pts_arr, cal_fx_arr, cal_fy_arr)
                elif do_discrete:
                    cal_fx_val, cal_fy_val = calibrate.apply_discrete(cop_delta_x_filt, cop_delta_y_filt, disc_dx_grid, disc_dy_grid, disc_fx_grid, disc_fy_grid)
                elif do_auto:
                    if fit_params_list is not None:
                        x_inputs = [cop_delta_x_filt, cop_delta_y_filt]
                        results = fit.apply_predict_multi(x_inputs, fit_params_list, fit_type, fit_split_sign)
                        cal_fx_val, cal_fy_val = (results[0], results[1]) if len(results) >= 2 else (None, None)
                    elif cal_fit_ready_flag:
                        cal_fx_val, cal_fy_val = calibrate.apply_fit(query, cal_coefs, dim="2D")
                    elif cal_lut_ready_flag:
                        cal_fx_val, cal_fy_val = calibrate.apply(query, cal_pts_arr, cal_fx_arr, cal_fy_arr)
                if cal_fx_val is not None:
                    cal_angle_deg, cal_mag_val = angle.compute_vector_angle(cal_fx_val, cal_fy_val)

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
            cop_state=cop_state,
            adc_sum=total_press_val,
        )
        csv_writer.writerow(csv_row)

        # ---- 更新绘图 ----
        g_main_plot.set_data(
            pzt_angle_deg, pzt_mag_val, force_angle_deg, force_mag_val,
            base_sub_arr, total_press_val, force_mag_val,
            cop_curr_x, cop_curr_y, cop_base_x, cop_base_y,
            cop_delta_x_filt, cop_delta_y_filt,
            force_fx_filt, force_fy_filt, force_fz_filt,
            cal_fx_val, cal_fy_val, cal_fz_val, cal_angle_deg, cal_mag_val,
            cop_state=cop_state,
        )
        if COP.g_cop_contact_init_flag:
            g_main_plot.append_full_data(
                rel_time_ms,
                pzt_angle_deg, pzt_mag_val, total_press_val,
                cop_delta_x_filt, cop_delta_y_filt,
                force_angle_deg, force_mag_val,
                force_fz_filt, force_fx_filt, force_fy_filt,
                cal_angle_deg, cal_mag_val, cal_fx_val, cal_fy_val, cal_fz_val)

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
