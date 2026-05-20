import tang_7_12_InitCOP_realtime_package as cop
import numpy as np

# ===================== 示例使用 =====================
if __name__ == "__main__":
    # 模拟84个ADC原始数据（实际使用时替换为真实传感器数据）
    mock_adc_data = [np.random.randint(0, 1000) for _ in range(84)]
    
    # 获取压阻传感器角度
    angle = cop.get_pzt_angle(mock_adc_data)
    print(f"压阻传感器角度：{angle:.2f}°")
    
    # 如需重新校准（比如传感器归零）
    # reset_baseline()