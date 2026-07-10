"""01_…_Init_line_COP_vec_cal_realtime/run_realtime.py

参考 ../7_12_tang_InitCOP_vaule_realtime_package/run_realtime.py 的主循环结构，
但调本目录的 PZTSensorAngle.get_all(adc) (3-tuple 输出)。
"""

import sys
import os
import time
import threading

sys.path.insert(0, os.path.dirname(__file__))

import data                                                  # 串口 + 解包
import tang_7_12_InitCOP_realtime_package_note as cop        # 文档版（带中文 docstring）


class PressureThread(threading.Thread):
    """后台读帧：每秒 ~1000 次把解码后的 84 通道帧扔进时间戳缓冲。"""

    def __init__(self, sensor, buf):
        super().__init__(daemon=True)
        self._sensor = sensor
        self._buf = buf

    def run(self):
        while True:
            try:
                ts = time.perf_counter()
                raw = self._sensor.read_data()
                if raw is not None:
                    decoded = self._sensor.decode(raw)
                    self._buf.append({"t": ts, "data": decoded})
            except Exception:
                pass
            time.sleep(0.001)


def main():
    sensor = data.PressureSensor(port='/dev/ttyUSB2')       # 用户硬件在 /dev/ttyUSB2
    buf = data.TimestampedBuffer(500)
    pt = PressureThread(sensor, buf)
    pt.start()

    angle_estimator = cop.PZTSensorAngle()                  # 唯一实例，跨帧保留状态
    start_ts = time.perf_counter()                          # 用于 5s 一次性 reset
    reset_done = False

    print("压阻传感器就绪，开始采集... (Ctrl+C 退出)")

    frame_cnt = 0
    try:
        while True:
            item = buf.get_latest()
            if item is None:
                time.sleep(0.001)
                continue

            frame_cnt += 1
            adc = item["data"]                              # 84 长
            try:
                angle, dx, dy = angle_estimator.get_all(adc)  # 3-tuple (与 vaule 的 9-tuple 不同)
            except ValueError as e:
                print(f"frame {frame_cnt}: {e}")
                time.sleep(0.005)
                continue

            print(f"frame {frame_cnt:6d}  angle={angle:7.2f}°  "
                  f"dx={dx:+5.2f}  dy={dy:+5.2f}")

            if not reset_done and (time.perf_counter() - start_ts) >= 5.0:
                angle_estimator.reset_origin()
                reset_done = True
                print("[reset] 5s 触发 reset_origin()")

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\n已停止采集")


if __name__ == "__main__":
    main()
