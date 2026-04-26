import time
import csv
import os
import numpy as np
import threading
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import angle as angle
import COP as COP  # 仅导入，不整合逻辑
import data as data
import realtime as realtime

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
    os.makedirs(SAVE_DIR, exist_ok=True)

    s_press = data.PressureSensor()
    s_force = data.SixAxisForceSensor()
    s_force.calibrate_zero()
    print("✅ 传感器初始化完成")

    buf_press = data.TimestampedBuffer(500)
    buf_force = data.TimestampedBuffer(500)

    t1 = PressureThread(s_press, buf_press)
    t2 = ForceThread(s_force, buf_force)
    t1.start()
    t2.start()

    idx = 1
    while os.path.exists(f"{SAVE_DIR}/data_{idx}.csv"):
        idx +=1
    path = f"{SAVE_DIR}/data_{idx}.csv"

    with open(path,"w",encoding="utf-8",newline="") as csv_file_obj:
        w = csv.writer(csv_file_obj)
        w.writerow(["ms","cop_x","cop_y","dx","dy","Fx","Fy","Fz","Mx","My","Mz","pzt","force","err"])
        print(f"📂 保存：{path}")
        print("🎨 绘图已打开")
        t0 = time.perf_counter()

        while not stop_event.is_set():
            now = time.perf_counter()
            ms = int((now-t0)*1000)
            p = buf_press.get_latest()
            if not p:
                time.sleep(0.001)
                continue
            force_data_item = buf_force.find_closest(p["t"])
            if not force_data_item or abs(p["t"]-force_data_item["t"])>MAX_TIME_DIFF:
                time.sleep(0.001)
                continue

            # 仅调用COP模块的函数，不整合其逻辑
            base = COP.subtract_baseline(p["data"])
            res = COP.compute_pressure_direction(base)
            cx, cy = res[0], res[1]
            dx, dy = res[6], res[7]
            bx, by = res[8], res[9]

            fx,fy,fz,mx,my,mz = force_data_item["data"]
            pzt,_ = angle.compute_PZT_angle(dx,dy)
            force,_ = angle.compute_6Dforce_angle(fx,fy)
            err = angle.angle_difference(pzt,force)

            plot.set_data(pzt,np.hypot(dx,dy),force,np.hypot(fx,fy),base,np.sum(p["data"]),np.hypot(fx,fy),cx,cy,bx,by,dx,dy)
            w.writerow([ms,cx,cy,dx,dy,fx,fy,fz,mx,my,mz,pzt,force,err])
            csv_file_obj.flush()
            elapsed = time.perf_counter()-now
            time.sleep(max(0, 1/TARGET_FPS-elapsed))

# ===================== 主函数 =====================
def main():
    global plot
    plot = realtime.RealTimePlot()
    threading.Thread(target=data_loop, daemon=True).start()
    plt.show()
    stop_event.set()

if __name__ == "__main__":
    main()