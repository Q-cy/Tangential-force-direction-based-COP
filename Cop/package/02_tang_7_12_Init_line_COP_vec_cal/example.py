import os

import numpy as np
import tang_7_12_InitCOP_realtime_package_note as cop

# ===================== 示例:84 通道 ADC → (angle, Fx, Fy, Fz) =====================
if __name__ == "__main__":
    # 1. 输入 84 通道 ADC(实际使用时替换为真实传感器数据)
    adc_data = [np.random.randint(0, 1000) for _ in range(84)]

    # 2. 模型路径:相对 example.py 自身目录,任何 CWD 下都能找到
    _model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(_model_dir, "fit_coefs.bin")

    # 3. 一站式:得到 (angle, Fx, Fy, Fz)
    #    无模型时只算 angle
    try:
        angle, Fx, Fy, Fz = cop.compute_cop_data(adc_data, model_path)
        print(f"压阻传感器角度:{angle:.2f}°")
        print(f"Fx = {Fx:.4f} N")
        print(f"Fy = {Fy:.4f} N")
        print(f"Fz = {Fz:.4f} N")
    except FileNotFoundError:
        # 演示用:无真实模型时,只算 angle
        angle = cop.compute_cop_angle(adc_data)
        print(f"压阻传感器角度:{angle:.2f}°")
        print(f"(无拟合模型 {model_path},Fx/Fy/Fz 不可用)")
