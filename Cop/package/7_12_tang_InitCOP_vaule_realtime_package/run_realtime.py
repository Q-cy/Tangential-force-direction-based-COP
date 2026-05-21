"""
实时读取压力传感器数据，经 COP 算法计算后打印到终端。
用法：python run_realtime.py
退出：Ctrl+C
"""
import sys
import os
import time
import threading
# 将 data.py 所在目录加入 sys.path
DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "project",
    "tang_7_12_Init_line_COP_vec_cal",
    "tang_7_12_Init_line_stable_COP_vec_cal_inter_realtime",
)
sys.path.insert(0, os.path.abspath(DATA_DIR))

import data
import tang_7_12_InitCOP_vaule_realtime_package as cop


class PressureThread(threading.Thread):
    def __init__(self, sensor, buf):
        super().__init__(daemon=True)
        self.s = sensor
        self.buf = buf

    def run(self):
        while True:
            ts = time.perf_counter()
            raw = self.s.read_data()
            if raw:
                try:
                    d = self.s.decode(raw)
                    self.buf.append({"t": ts, "data": d})
                except Exception:
                    pass
            time.sleep(0.001)


def main():
    sensor = data.PressureSensor()
    buf = data.TimestampedBuffer(500)
    thread = PressureThread(sensor, buf)
    thread.start()
    print("压力传感器就绪，开始采集... (Ctrl+C 退出)\n")

    frame_cnt = 0
    try:
        while True:
            item = buf.get_latest()
            if item is None:
                time.sleep(0.001)
                continue

            frame_cnt += 1
            adc = item["data"]

            pzt_angle, magnitude, state = cop.get_pzt_angle(adc)

            print(f"frame={frame_cnt:04d}  Angle={pzt_angle:.1f}°  Mag={magnitude:.3f}  state={state}")

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\n已停止采集")


if __name__ == "__main__":
    main()
