# file_name: replay.py
# CSV 回放模式：将录制的 CSV 数据模拟实时流输入 COP 处理流水线

import csv
import time
import os
import sys
import threading
from collections import deque
import numpy as np
from pyqtgraph.Qt import QtWidgets

import angle
import COP
import table
import calibrate
import realtime2

# ===================== 配置 =====================
MAIN_SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"
MAIN_CAL_MODE = "lookup"

g_main_stop_flag = threading.Event()
g_main_plot = None

_NAN6 = [float('nan')] * 6


def _load_csv_rows(path):
    """加载 CSV 文件，返回 [{"rel_ms": int, "ch_data": [84 ints]}, ...]"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header = [h.strip() for h in header]
        for row in reader:
            if not row:
                continue
            vals = [v.strip() for v in row]
            rel_ms = int(vals[1])
            ch_data = [int(float(vals[i])) for i in range(3, 87)]
            rows.append({"rel_ms": rel_ms, "ch_data": ch_data})
    return rows


def replay_loop(csv_path):
    global g_main_plot

    # 加载 CSV
    rows = _load_csv_rows(csv_path)
    print(f"📂 已加载 {len(rows)} 行数据: {csv_path}")

    # 初始化输出 CSV
    out_csv_path = table.auto_get_csv_path(MAIN_SAVE_DIR)
    csv_writer, csv_file_obj = table.init_csv_file(out_csv_path)

    # 加载标定模型
    cal_bin_path = os.path.join(MAIN_SAVE_DIR, "cal_lookup.bin")
    cal_fit_path = os.path.join(MAIN_SAVE_DIR, "cal_fit.bin")
    cal_lut_ready = False
    cal_fit_ready = False
    cal_pts_arr = cal_fx_arr = cal_fy_arr = None
    cal_coef_fx = cal_coef_fy = None
    if os.path.exists(cal_bin_path):
        try:
            cal_pts_arr, cal_fx_arr, cal_fy_arr = calibrate.load_lookup(cal_bin_path)
            cal_lut_ready = True
            print(f"📐 查找表已加载: {cal_bin_path}")
        except Exception as e:
            print(f"⚠️ 查找表加载失败: {e}")
    if os.path.exists(cal_fit_path):
        try:
            cal_coef_fx, cal_coef_fy = calibrate.load_fit_model(cal_fit_path)
            cal_fit_ready = True
            print(f"📐 拟合模型已加载: {cal_fit_path}")
        except Exception as e:
            print(f"⚠️ 拟合模型加载失败: {e}")
    if not cal_lut_ready and not cal_fit_ready:
        print("💡 未找到标定文件")

    # 中值滤波缓冲
    median_filt_window = 5
    buf_cop_delta_x = deque(maxlen=median_filt_window)
    buf_cop_delta_y = deque(maxlen=median_filt_window)

    start_time_s = time.perf_counter()
    prev_rel_ms = 0

    for i, row in enumerate(rows):
        if g_main_stop_flag.is_set():
            break

        # 按原始时间间隔休眠
        if i > 0:
            dt_ms = row["rel_ms"] - prev_rel_ms
            if dt_ms > 0:
                time.sleep(dt_ms / 1000.0)
        prev_rel_ms = row["rel_ms"]

        loop_start_s = time.perf_counter()
        rel_time_ms = row["rel_ms"]
        ch_data = row["ch_data"]

        # CoP 计算
        cop_res = COP.compute_pressure_direction(ch_data)
        base_sub_arr = np.array(ch_data)
        cop_curr_x, cop_curr_y = cop_res[0], cop_res[1]
        cop_delta_x, cop_delta_y = cop_res[6], cop_res[7]
        cop_base_x, cop_base_y = cop_res[8], cop_res[9]
        cop_state = cop_res[10]
        total_press_val = np.sum(base_sub_arr)

        buf_cop_delta_x.append(cop_delta_x)
        buf_cop_delta_y.append(cop_delta_y)
        cop_delta_x_filt = np.median(buf_cop_delta_x)
        cop_delta_y_filt = np.median(buf_cop_delta_y)
        pzt_angle_deg, pzt_mag_val = angle.compute_PZT_angle(cop_delta_x_filt, cop_delta_y_filt)

        # 标定
        if MAIN_CAL_MODE == "fit" and cal_fit_ready:
            cal_fx_val, cal_fy_val = calibrate.apply_fit(cop_delta_x_filt, cop_delta_y_filt, cal_coef_fx, cal_coef_fy)
            cal_angle_deg, cal_mag_val = angle.compute_vector_angle(cal_fx_val, cal_fy_val)
        elif MAIN_CAL_MODE == "lookup" and cal_lut_ready:
            cal_fx_val, cal_fy_val = calibrate.apply(cop_delta_x_filt, cop_delta_y_filt, cal_pts_arr, cal_fx_arr, cal_fy_arr)
            cal_angle_deg, cal_mag_val = angle.compute_vector_angle(cal_fx_val, cal_fy_val)
        else:
            cal_fx_val = cal_fy_val = cal_angle_deg = cal_mag_val = None

        # 写入 CSV
        press_ts = loop_start_s
        csv_row = table.build_csv_row(
            press_timestamp=press_ts,
            rel_ms=rel_time_ms,
            ch_data=ch_data,
            force_data=_NAN6,
            force_timestamp=float('nan'),
            delta_cop_x=cop_delta_x_filt,
            delta_cop_y=cop_delta_y_filt,
            delta_force_x=float('nan'),
            delta_force_y=float('nan'),
            delta_force_z=float('nan'),
            adc_angle=pzt_angle_deg,
            adc_mag=pzt_mag_val,
            force_angle=float('nan'),
            force_mag=float('nan'),
            fx_cal=cal_fx_val,
            fy_cal=cal_fy_val,
            force_cal_mag=cal_mag_val,
            force_cal_angle=cal_angle_deg,
            cop_state=cop_state,
            adc_sum=total_press_val,
        )
        csv_writer.writerow(csv_row)

        # 更新 GUI
        g_main_plot.set_data(
            pzt_angle_deg, pzt_mag_val, float('nan'), float('nan'),
            base_sub_arr, total_press_val, float('nan'),
            cop_curr_x, cop_curr_y, cop_base_x, cop_base_y,
            cop_delta_x_filt, cop_delta_y_filt,
            float('nan'), float('nan'), float('nan'),
            cal_fx_val, cal_fy_val, cal_angle_deg, cal_mag_val,
            cop_state=cop_state,
        )
        if COP.g_cop_contact_init_flag:
            g_main_plot.append_full_data(
                rel_time_ms,
                pzt_angle_deg, pzt_mag_val, total_press_val,
                cop_delta_x_filt, cop_delta_y_filt,
                float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'),
                cal_angle_deg, cal_mag_val, cal_fx_val, cal_fy_val)

    # 收尾
    csv_file_obj.close()
    row_count = sum(1 for _ in open(out_csv_path)) - 1
    if row_count <= 0:
        os.remove(out_csv_path)
        print("⚠️ 无有效数据，CSV 已删除")
    else:
        print(f"✅ CSV 已关闭（{row_count} 行）: {out_csv_path}")


def main():
    global g_main_plot

    if len(sys.argv) < 2:
        print("用法: python replay.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        sys.exit(1)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    g_main_plot = realtime2.RealTimePlot()
    data_thread = threading.Thread(target=replay_loop, args=(csv_path,))
    data_thread.start()

    app.exec()

    g_main_stop_flag.set()
    data_thread.join(timeout=2)
    g_main_plot.plot_full_magnitude_curve(MAIN_SAVE_DIR)


if __name__ == "__main__":
    main()
