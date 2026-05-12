import numpy as np

def compute_vector_angle(x: float, y: float) -> tuple[float, float]:    #->返回值类型是tuple元组（固定长度、不可修改）
    """
    计算向量(x,y)的角度(0~360°)和幅值
    """
    epsilon = 1e-8
    mag = np.hypot(x, y)                                   #np.hypot计算欧几里得范数（向量模长）
    angle = np.degrees(np.arctan2(y, x + epsilon))         #np.arctan2计算反正切函数（弧度），np.degrees将弧度转换为角度
    if angle < 0:
        angle += 360
    return angle, mag

def compute_6Dforce_angle(Fx: float, Fy: float) -> tuple[float, float]:
    """
    计算六维力传感器(Fx,Fy)的角度(0~360°)和幅值
    """
    return compute_vector_angle(Fx, Fy)

def compute_PZT_angle(Px: float, Py: float) -> tuple[float, float]:
    """
    计算压阻传感器(Px,Py)的角度(0~360°)和幅值
    """
    return compute_vector_angle(Px, -Py)

def angle_difference(a1: float, a2: float) -> float:
    """
    计算两个角度的最小差值(0~180°)
    """
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)