import time
import os
import numpy as np
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 导入自定义模块
import angle as angle
import COP as COP  
import data as data
import realtime as realtime
import table as table 

# ===================== 配置 =====================
SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"
TARGET_FPS = 100
MAX_TIME_DIFF = 0.015
stop_event = threading.Event()
plot = None

# ===================== 采集线程 =====================
class PressureThread(threading.Thread):                   
    def __init__(self, sensor, buf):                      
        super().__init__(daemon=True)
        self.s = sensor                                   
        self.buf = buf
    def run(self):
        while not stop_event.is_set():
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
        while not stop_event.is_set():
            ts = time.perf_counter()
            d = self.s.read()
            if d:
                self.buf.append({"t":ts,"data":d})
            time.sleep(0.001)

# ===================== 数据循环 =====================
def data_loop():
    global plot
    # 自动获取CSV文件路径
    csv_path = table.auto_get_csv_path(SAVE_DIR)
    # 初始化CSV文件（写入表头）
    csv_writer, csv_file_obj = table.init_csv_file(csv_path)

    # 初始化传感器
    s_press = data.PressureSensor()
    s_force = data.SixAxisForceSensor()
    s_force.calibrate_zero()
    print("✅ 传感器初始化完成")

    # 初始化缓存
    buf_press = data.TimestampedBuffer(500)
    buf_force = data.TimestampedBuffer(500)

    # 启动采集线程
    t1 = PressureThread(s_press, buf_press)
    t2 = ForceThread(s_force, buf_force)
    t1.start()
    t2.start()

    print("🎨 绘图已打开")
    t0 = time.perf_counter()

    while not stop_event.is_set():
        now = time.perf_counter()
        rel_ms = int((now - t0) * 1000)  # 相对毫秒数
        
        # 获取最新压力传感器数据
        press_data_item = buf_press.get_latest()
        if not press_data_item:
            time.sleep(0.001)
            continue
        
        # 匹配最近的力传感器数据
        force_data_item = buf_force.find_closest(press_data_item["t"])
        if not force_data_item or abs(press_data_item["t"] - force_data_item["t"]) > MAX_TIME_DIFF:
            time.sleep(0.001)
            continue

        # 计算CoP和方向数据
        base = COP.subtract_baseline(press_data_item["data"])
        cop_res = COP.compute_pressure_direction(base)
        cx, cy = cop_res[0], cop_res[1]
        dx, dy = cop_res[6], cop_res[7]
        bx, by = cop_res[8], cop_res[9]

        # 解析力传感器数据
        fx, fy, fz, mx, my, mz = force_data_item["data"]
        
        # 计算角度和幅值
        adc_angle, adc_mag = angle.compute_PZT_angle(dx, dy)
        force_angle, force_mag = angle.compute_6Dforce_angle(fx, fy)
        
        # 构造CSV行数据（调用封装函数）
        csv_row = table.build_csv_row(
            press_timestamp=press_data_item["t"],
            rel_ms=rel_ms,
            ch_data=press_data_item["data"],
            force_data=force_data_item["data"],
            force_timestamp=force_data_item["t"],
            adc_angle=adc_angle,
            adc_mag=adc_mag,
            force_angle=force_angle,
            force_mag=force_mag
        )
        
        # 写入CSV行
        csv_writer.writerow(csv_row)
        csv_file_obj.flush()  # 立即刷新到文件
        
        # 更新绘图数据
        plot.set_data(
            adc_angle, adc_mag, force_angle, force_mag,
            base, np.sum(press_data_item["data"]), force_mag,
            cx, cy, bx, by, dx, dy
        )
        # 追加全程数据
        if COP.contact_initialized: 
                    plot.append_full_data(rel_ms, adc_mag, force_mag) 
        
        # 控制采集频率
        elapsed = time.perf_counter() - now
        time.sleep(max(0, 1/TARGET_FPS - elapsed))

    # 关闭CSV文件
    csv_file_obj.close()
    print("✅ CSV文件已关闭")

# ===================== 主函数 =====================
def main():
    global plot
    plot = realtime.RealTimePlot()
    # 启动数据采集线程
    data_thread = threading.Thread(target=data_loop)
    data_thread.start()
    plt.show()  # 阻塞直到关闭绘图窗口
    
    # 停止采集线程
    stop_event.set()
    data_thread.join(timeout=2)
    # 绘制全程静态图
    plot.plot_full_magnitude_curve(SAVE_DIR)

if __name__ == "__main__":
    main()
