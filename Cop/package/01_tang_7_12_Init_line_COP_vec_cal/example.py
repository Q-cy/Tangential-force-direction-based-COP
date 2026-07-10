from tang_7_12_InitCOP_realtime_package import PZTSensorAngle
import numpy as np

if __name__ == "__main__":
    # 传感器独立实例
    sensor_a = PZTSensorAngle()
    sensor_b = PZTSensorAngle()

    # 1) 阈值（喂入前 collect_frames 帧让 sensor 各自学动态阈值）
    for sensor in (sensor_a, sensor_b):
        for _ in range(sensor.collect_frames):
            idle = [np.random.randint(0, 50) for _ in range(sensor.rows * sensor.cols)]
            sensor.get_angle(idle)

    # 2) 同一接触帧，两个 sensor 用不同 API 处理：
    # sensor_a 用 get_angle 输出角度
    # sensor_b 用 get_all 输出角度、dx、dy
    contact = [np.random.randint(0, 1000) for _ in range(84)]
    a_angle = sensor_a.get_angle(contact)
    b_angle, b_dx, b_dy = sensor_b.get_all(contact)
    print(f"sensor_a (get_angle): angle={a_angle:.2f}°")
    print(f"sensor_b (get_all): angle={b_angle:.2f}° dx={b_dx:+.2f} dy={b_dy:+.2f}")

    # 3) 重置初始 COP
    sensor_a.reset_origin()
