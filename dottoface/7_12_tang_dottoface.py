import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import display, HTML
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)
# np.set_printoptions(threshold=np.inf, linewidth=400, precision=2)

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = "all"
# 全局变量：保存上一帧数据，用于实时计算
prev_frame = None 
# ===================== 数组自动网格输出（你要的格式） =====================
def auto_print(data, title):
    print(f"\n{'='*60}")
    print(f"📌 {title} | 形状 = {data.shape}")
    vec = data[0]
    n = len(vec)
    
    if n == 84:
        rows, cols = 7, 12
    elif n == 77:
        rows, cols = 7, 11
    elif n == 72:
        rows, cols = 6, 12
    elif n == 66:
        rows, cols = 6, 11
    else:
        print(vec)
        return
    
    print(f"✅ 输出格式：{rows}行 × {cols}列")
    print("-" * (cols * 7))
    grid = vec.reshape(rows, cols)
    for i in range(rows):
        print(f"行{i+1:2d} | " + " ".join(f"{v:6.1f}" for v in grid[i]))

# 打印全部行的矩阵
def print_full_matrix(data, title, rows, cols):
    print(f"\n{'='*60}")
    print(f"📌 {title} | 全部行 | {rows}×{cols}")
    for i in range(data.shape[0]):
        print(f"\n--- 第 {i+1} 行 ---")
        grid = data[i].reshape(rows, cols)
        for r in range(rows):
            print(f"行{r+1:2d} | " + " ".join(f"{v:6.1f}" for v in grid[r]))
# ===================== 纯基线扣除（无峰值检测） =====================
global first_frame
first_frame = None

def subtract_baseline(current_frame):
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()

    # 第一帧作为基线
    if first_frame is None:
        first_frame = current_frame.copy()

    # 只做：当前帧 - 基线帧
    diff_frame = current_frame - first_frame
    
    return diff_frame
# ===================== 受压矩形范围 → 只返回矩形内所有点索引 =====================
def get_pressure_region_indices(diff_frame, threshold):
    """
    步骤：
    1. 找到所有 > threshold 的有效点
    2. 找到最左、最右、最上、最下 → 得到受压矩形
    3. 返回矩形内部所有点的索引列表
    """
    rows = 12
    cols = 7

    # 1. 找出所有值 > threshold 的索引
    valid_indices = []
    for idx in range(rows * cols):
        if diff_frame[idx] > threshold:
            valid_indices.append(idx)

    # 没有有效点，返回空列表
    if not valid_indices:
        return []

    # 2. 转成 (行, 列) 坐标
    coords = [(idx // cols, idx % cols) for idx in valid_indices]

    # 3. 找矩形边界
    min_col = min(c for r, c in coords)
    max_col = max(c for r, c in coords)
    min_row = min(r for r, c in coords)
    max_row = max(r for r, c in coords)

    # 4. 收集矩形范围内所有点的索引
    region_indices = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            idx = r * cols + c
            region_indices.append(idx)

    # 只返回矩形内所有点索引
    return region_indices
# ===================== 在受压区内计算梯度 =====================
def compute_gradient_in_region(frame, region_indices):
    """
    完全按照你给的 C++ VectorPlot::update_vectors() 逻辑
    只在 受压区域 内计算 sum_gx 和 sum_gy
    """
    frame = np.array(frame).reshape(12, 7)  # 转成矩阵
    sum_gx = 0.0
    sum_gy = 0.0
    width = 7
    height = 12

    # 只遍历受压区内的点
    for idx in region_indices:
        y = idx // 7
        x = idx % 7

        # ===================== 梯度公式 =====================
        # gx = 0.5*(右 - 左)
        left  = frame[y, x-1] if (x-1 >= 0) else 0.0
        right = frame[y, x+1] if (x+1 < width) else 0.0
        gx = 0.5 * (right - left)

        # gy = 0.5*(上 - 下)
        up   = frame[y-1, x] if (y-1 >= 0) else 0.0
        down = frame[y+1, x] if (y+1 < height) else 0.0
        gy = 0.5 * (up - down)

        sum_gx += gx
        sum_gy += gy

    # 归一化（和C++完全一样）
    dir_len = np.hypot(sum_gx, sum_gy)
    if dir_len < 1e-6:
        return 0.0, 1.0  # 防止除0
    
    dir_x = sum_gx / dir_len
    dir_y = sum_gy / dir_len

    return dir_x, dir_y
# -------------------- 计算ADC【一个总角度 + 一个总大小】 --------------------
def compute_gradient_angle_single(x, y):
    epsilon = 1e-8
    angle = np.degrees(np.arctan2(y, x + epsilon))
    if angle < 0:
        angle += 360
    magnitude = np.sqrt(x**2 + y**2)
    return angle, magnitude
# -------------------- 计算Force角度 + 大小 --------------------
def compute_force_angle(Fx, Fy):
    import numpy as np
    epsilon = 1e-8
    
    force_magnitude = np.sqrt(Fx**2 + Fy**2)
    force_angle = np.degrees(np.arctan2(Fy, Fx + epsilon))
    force_angle = np.where(force_angle < 0, force_angle + 360, force_angle)

    return force_angle, force_magnitude
# ================================== 【最终主程序】已修正 ==================================
# 1. 读取数据
df = pd.read_csv('/home/qcy/Project/data/2.PZT_tangential/weight/test/data_9.csv')
data_original = df.iloc[:, 2:86].values
force_Fx = df.iloc[:, 86].values
force_Fy = df.iloc[:, 87].values

# 2. 初始化存储
data_diff = np.zeros_like(data_original)
ADC_angle_list = []
ADC_magnitude_list = []
force_angle_list = []
force_magnitude_list = []

# 3. 主循环
for i in range(data_original.shape[0]):
    current_frame = data_original[i]
    diff_frame = subtract_baseline(current_frame)
    data_diff[i] = diff_frame

    # -------------------- 新逻辑：受压区 + 梯度 --------------------
    region_indices = get_pressure_region_indices(diff_frame, threshold=1000)
    dir_x, dir_y = compute_gradient_in_region(diff_frame, region_indices)

    # -------------------- 计算 ADC 角度 --------------------
    angle, magnitude = compute_gradient_angle_single(dir_x, dir_y)
    ADC_angle_list.append([angle])
    ADC_magnitude_list.append([magnitude])

    # -------------------- 计算力角度 --------------------
    fx = force_Fx[i]
    fy = force_Fy[i]
    f_angle, f_mag = compute_force_angle(fx, fy)
    force_angle_list.append(f_angle)
    force_magnitude_list.append(f_mag)

# 转数组
gradient_angle = np.array(ADC_angle_list)
gradient_magnitude = np.array(ADC_magnitude_list)
force_angle = np.array(force_angle_list)
force_magnitude = np.array(force_magnitude_list)

# ===================== 输出 =====================
print(f"\n✅ data_original shape: {data_original.shape}")
print(f"\n✅ data_diff 基线扣除 shape: {data_diff.shape}")
print(f"✅ gradient_angle ADC角度 shape: {gradient_angle.shape}")
print(f"✅ force_angle Force角度 shape: {force_angle.shape}")
# ===================== 打印输出 =====================
print(f"\n✅ data_original shape: {data_original.shape}")
print_full_matrix(data_original, "data_original", 12, 7)
print(f"\n✅ data_diff 基线扣除 shape: {data_diff.shape}")
print_full_matrix(data_diff, "data_diff 基线扣除", 12, 7)
print(f"✅ gradient_angle ADC角度 shape: {gradient_angle.shape}")
print_full_matrix(gradient_angle, "gradient_angle ADC角度", 1, 1)
print(f"✅ force_angle Force角度 shape: {force_angle.shape}")
print_full_matrix(force_angle, "force_angle Force角度", 1, 1)
# ===================== 拼接+保存（最终版：总角度1 + 总大小1） =====================
force_angle = np.array(force_angle_list).reshape(-1, 1)
force_mag   = np.array(force_magnitude_list).reshape(-1, 1)

gradient_angle_2d = np.array(ADC_angle_list)
gradient_mag_2d   = np.array(ADC_magnitude_list)

ADC_force_result = np.hstack([
    gradient_angle_2d,   
    gradient_mag_2d,     
    force_angle,        
    force_mag           
])

columns = ["ADC_angle", "ADC_mag", "Force_angle", "Force_mag"]
df_result = pd.DataFrame(ADC_force_result, columns=columns)

# ===================== 角度对比输出 =====================
print("  帧号   |   ADC梯度角度   |   六维力角度")
print("-" * 50)

for i in range(len(ADC_angle_list)):
    adc_a = ADC_angle_list[i][0]
    force_a = force_angle_list[i]
    print(f"{i:6d}   |      {adc_a:6.1f}°     |      {force_a:6.1f}°")
# ===================== 两张图：只有方向 + 方向+大小（ADC/力 独立分开归一化） =====================
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ===================== 字体设置（无乱码） =====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 关闭多余输出
try:
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "none"
except ImportError:
    pass

row_idx = 720

# ===================== 数据：总角度 + 总大小 =====================
adc_total_angle = ADC_force_result[row_idx, 0]    # ADC总角度
adc_total_mag   = ADC_force_result[row_idx, 1]    # ADC总大小
force_angle     = ADC_force_result[row_idx, 2]    # Force角度
force_mag       = ADC_force_result[row_idx, 3]    # Force大小

epsilon = 1e-8
fixed_arrow = 0.35


# ===================== 图1：只有方向（固定长度） =====================
fig1, axes1 = plt.subplots(10, 6, figsize=(12, 6), gridspec_kw={'width_ratios': [1,1,1,1,1, 0.8]})

# 清空左边
for r in range(10):
    for c in range(5):
        ax = axes1[r, c]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

# 中心画总ADC箭头（固定长度）
center_ax1 = axes1[4:6, 2:3][0, 0]
theta1 = np.deg2rad(adc_total_angle)
center_ax1.arrow(0.5, 0.5,
         0.4 * np.cos(theta1),
         0.4 * np.sin(theta1),
         head_width=0.15, head_length=0.15,
         fc='k', ec='k', linewidth=3, zorder=2)

# 右侧Force箭头（固定长度）
gs1 = GridSpec(10, 6, figure=fig1, width_ratios=[1,1,1,1,1,0.8])
ax_force1 = fig1.add_subplot(gs1[:, 5])
theta_f1 = np.deg2rad(force_angle)
ax_force1.arrow(0.5, 0.5,
          fixed_arrow * np.cos(theta_f1),
          fixed_arrow * np.sin(theta_f1),
          head_width=0.12, head_length=0.12,
          fc='r', ec='r', linewidth=2, zorder=2)

ax_force1.set_xlim(0, 1)
ax_force1.set_ylim(0, 1)
ax_force1.set_aspect('equal', adjustable='box')
ax_force1.axis('off')
ax_force1.set_title("Force", fontsize=12, pad=20)

# 隐藏多余
for r in range(10):
    axes1[r,5].remove()

fig1.suptitle(f"Only Direction (Fixed Length) - row {row_idx+1}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()


# ===================== 图2：方向+大小（✅ ADC 和 力 独立分开归一化） =====================
fig2, axes2 = plt.subplots(10, 6, figsize=(12, 6), gridspec_kw={'width_ratios': [1,1,1,1,1, 0.8]})

# 清空左边
for r in range(10):
    for c in range(5):
        ax = axes2[r, c]
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')

# 中心ADC箭头（独立归一化）
center_ax2 = axes2[4:6, 2:3][0, 0]
theta2 = np.deg2rad(adc_total_angle)

# --------------------------
# ✅ ADC 自己单独归一化
# --------------------------
max_adc = np.max(ADC_force_result[:, 1]) if np.max(ADC_force_result[:, 1]) > epsilon else 1.0
adc_arrow_len = 0.4 * (adc_total_mag / max_adc)

center_ax2.arrow(0.5, 0.5,
         adc_arrow_len * np.cos(theta2),
         adc_arrow_len * np.sin(theta2),
         head_width=0.15, head_length=0.15,
         fc='k', ec='k', linewidth=3, zorder=2)

# 右侧Force箭头（独立归一化）
gs2 = GridSpec(10, 6, figure=fig2, width_ratios=[1,1,1,1,1,0.8])
ax_force2 = fig2.add_subplot(gs2[:, 5])
theta_f2 = np.deg2rad(force_angle)

# --------------------------
# ✅ Force 自己单独归一化
# --------------------------
max_force = np.max(ADC_force_result[:, 3]) if np.max(ADC_force_result[:, 3]) > epsilon else 1.0
force_arrow_len = 0.35 * (force_mag / max_force)

ax_force2.arrow(0.5, 0.5,
          force_arrow_len * np.cos(theta_f2),
          force_arrow_len * np.sin(theta_f2),
          head_width=0.12, head_length=0.12,
          fc='r', ec='r', linewidth=2, zorder=2)

ax_force2.set_xlim(0, 1)
ax_force2.set_ylim(0, 1)
ax_force2.set_aspect('equal', adjustable='box')
ax_force2.axis('off')
ax_force2.set_title("Force", fontsize=12, pad=20)

# 隐藏多余
for r in range(10):
    axes2[r,5].remove()

fig2.suptitle(f"Direction + Magnitude (Variable Length) - row {row_idx+1}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()

# ===================== 【新增】图3：ADC角度 vs Force角度 全程对比折线图 =====================
plt.figure(figsize=(14, 6))
frames = np.arange(len(ADC_angle_list))
adc_angles = np.array(ADC_angle_list).flatten()
force_angles = np.array(force_angle_list)

# 绘制两条折线
plt.plot(frames, adc_angles, label='ADC 计算角度', color='#1f77b4', linewidth=2, alpha=0.8)
plt.plot(frames, force_angles, label='六维力 真实角度', color='#ff4b33', linewidth=2, alpha=0.8)

# 角度范围固定 0~360
plt.ylim(0, 360)
plt.yticks(np.arange(0, 361, 45))

# 标签与网格
plt.xlabel('帧序号', fontsize=12)
plt.ylabel('角度 (°)', fontsize=12)
plt.title('ADC 梯度角度 与 六维力角度 全程对比', fontsize=14, pad=15)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()