import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from IPython.display import display, HTML

# ===================== 全局变量和配置 =====================
# 传感器阵列尺寸
SENSOR_ROWS = 12
SENSOR_COLS = 7
CELL_AREA = 1.0 # 假设每个单元面积为1，用于CoP计算。如果需要实际力，需要知道真实面积。

# CoP基准点，用于计算偏移。第一次有效接触时设置。
# first_contact_CoP_x/y 用于记录第一次接触时的CoP，作为切向力为0的参考点
first_contact_CoP_x = None
first_contact_CoP_y = None
contact_initialized = False # 标记是否已经建立了基准CoP

# ===================== 数组自动网格输出 =====================
def auto_print(data, title):
    print(f"\n{'='*60}")
    print(f"📌 {title} | 形状 = {data.shape}")
    
    # 确保 data 是二维的，或者如果是1D，则取第一个元素
    if data.ndim == 1:
        vec = data
    else:
        # 如果data是(N,1)形状，取第一个元素，避免打印成[[]]
        if data.shape[1] == 1:
            vec = data[0]
        else:
            vec = data[0] # 否则取第一行

    n = len(vec)
    
    # 根据长度判断行和列，适配12x7=84
    if n == SENSOR_ROWS * SENSOR_COLS: # 84
        rows, cols = SENSOR_ROWS, SENSOR_COLS
    elif n == 77:
        rows, cols = 7, 11
    elif n == 72:
        rows, cols = 6, 12
    elif n == 66:
        rows, cols = 6, 11
    elif n == 1: # 用于角度或大小的单值数组
        print(f"✅ 输出格式：单个值 {vec[0]:.2f}")
        return
    else:
        print(f"✅ 未知或非标准格式，原始输出：{vec}")
        return
    
    print(f"✅ 输出格式：{rows}行 × {cols}列")
    print("-" * (cols * 7)) # 调整分隔线长度
    grid = vec.reshape(rows, cols)
    for i in range(rows):
        print(f"行{i+1:2d} | " + " ".join(f"{v:6.1f}" for v in grid[i]))

# 打印全部行的矩阵 (此函数在实际运行时被注释掉，所以不影响主要逻辑，但保留其定义)
def print_full_matrix(data, title, rows, cols):
    print(f"\n{'='*60}")
    print(f"📌 {title} | 全部行 | {rows}×{cols}")
    for i in range(data.shape[0]):
        print(f"\n--- 第 {i+1} 行 ---")
        if data.ndim == 1:
            grid = np.array([data[i]]).reshape(rows, cols)
        else:
            grid = data[i].reshape(rows, cols)
            
        for r in range(rows):
            print(f"行{r+1:2d} | " + " ".join(f"{v:6.1f}" for v in grid[r]))

# ===================== 纯基线扣除 =====================
global first_frame_baseline
first_frame_baseline = None

def subtract_baseline(current_frame):
    global first_frame_baseline
    current_frame = np.array(current_frame, dtype=np.float32).flatten()

    if first_frame_baseline is None:
        first_frame_baseline = current_frame.copy()

    diff_frame = current_frame - first_frame_baseline
    
    return diff_frame

# ===================== 计算压强重心 (CoP) 和总法向力 =====================
def calculate_CoP(pressure_frame, threshold=0.5):
    pressure_matrix = pressure_frame.reshape(SENSOR_ROWS, SENSOR_COLS)
    x_coords_matrix = np.tile(np.arange(SENSOR_COLS), (SENSOR_ROWS, 1))
    y_coords_matrix = np.repeat(np.arange(SENSOR_ROWS), SENSOR_COLS).reshape(SENSOR_ROWS, SENSOR_COLS)

    valid_pressure_mask = pressure_matrix > threshold
    
    if not np.any(valid_pressure_mask):
        return None, None, 0

    total_effective_pressure = np.sum(pressure_matrix[valid_pressure_mask])

    if total_effective_pressure == 0:
        return None, None, 0

    CoP_x = np.sum(pressure_matrix[valid_pressure_mask] * x_coords_matrix[valid_pressure_mask]) / total_effective_pressure
    CoP_y = np.sum(pressure_matrix[valid_pressure_mask] * y_coords_matrix[valid_pressure_mask]) / total_effective_pressure
    
    total_normal_force = total_effective_pressure * CELL_AREA

    return CoP_x, CoP_y, total_normal_force

# ===================== 计算Force角度 + 大小 =====================
def compute_force_angle(Fx, Fy):
    epsilon = 1e-8
    
    force_magnitude = np.sqrt(Fx**2 + Fy**2)
    force_angle = np.degrees(np.arctan2(Fy, Fx + epsilon))
    force_angle = np.where(force_angle < 0, force_angle + 360, force_angle)

    return force_angle, force_magnitude

# ================================== 最终完整主程序 ==================================
# 1. 读取数据
df = pd.read_csv('/home/qcy/Project/data/2.PZT_tangential/weight/test/data_8.csv')
data_original = df.iloc[:, 2:86].values

force_Fx = df.iloc[:, 86].values
force_Fy = df.iloc[:, 87].values

# 2. 初始化存储数组
data_diff = np.zeros_like(data_original)
current_CoP_x_list = []
current_CoP_y_list = []
total_normal_force_list = []
delta_CoP_x_list = []
delta_CoP_y_list = []
CoP_angle_list = []
CoP_magnitude_list = []
force_angle_list = []
force_magnitude_list = []

# 3. 主循环
for i in range(data_original.shape[0]):
    current_frame_data = data_original[i]
    diff_frame = subtract_baseline(current_frame_data)
    data_diff[i] = diff_frame

    current_CoP_x, current_CoP_y, total_normal_force = calculate_CoP(diff_frame, threshold=0.5)
    
    current_CoP_x_list.append(current_CoP_x if current_CoP_x is not None else np.nan)
    current_CoP_y_list.append(current_CoP_y if current_CoP_y is not None else np.nan)
    total_normal_force_list.append(total_normal_force)

    if current_CoP_x is not None and current_CoP_y is not None:
        if not contact_initialized:
            first_contact_CoP_x = current_CoP_x
            first_contact_CoP_y = current_CoP_y
            contact_initialized = True
            
            delta_CoP_x = 0.0
            delta_CoP_y = 0.0
        else:
            # calculate delta relative to the first contact CoP
            delta_CoP_x = current_CoP_x - first_contact_CoP_x # <<< Corrected: Use first_contact_CoP_x as base
            delta_CoP_y = current_CoP_y - first_contact_CoP_y # <<< Corrected: Use first_contact_CoP_y as base (Y-down positive)
    else:
        delta_CoP_x = np.nan
        delta_CoP_y = np.nan
        contact_initialized = False # If contact is lost, reset flag, wait for next valid contact to re-establish first_contact_CoP
        first_contact_CoP_x = None # Reset
        first_contact_CoP_y = None # Reset

    delta_CoP_x_list.append(delta_CoP_x)
    delta_CoP_y_list.append(delta_CoP_y)

    if not np.isnan(delta_CoP_x) and not np.isnan(delta_CoP_y):
        # np.arctan2 expects (Y, X) where Y is positive upwards.
        # Our delta_CoP_y is positive downwards (heatmap Y-axis).
        # So, to convert delta_CoP_y (Y-down positive) to Y-up positive for arctan2, we use -delta_CoP_y.
        angle_rad = np.arctan2(-delta_CoP_y, delta_CoP_x) 
        angle_deg = np.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        
        magnitude = np.sqrt(delta_CoP_x**2 + delta_CoP_y**2)
    else:
        angle_deg = np.nan
        magnitude = np.nan
    
    CoP_angle_list.append([angle_deg])
    CoP_magnitude_list.append([magnitude])

    fx = force_Fx[i]
    fy = force_Fy[i]
    f_angle, f_mag = compute_force_angle(fx, fy)
    force_angle_list.append(f_angle)
    force_magnitude_list.append(f_mag)

# Convert lists to numpy arrays
CoP_angle = np.array(CoP_angle_list)
CoP_magnitude = np.array(CoP_magnitude_list)
current_CoP_x = np.array(current_CoP_x_list).reshape(-1, 1)
current_CoP_y = np.array(current_CoP_y_list).reshape(-1, 1)
delta_CoP_x = np.array(delta_CoP_x_list).reshape(-1, 1)
delta_CoP_y = np.array(delta_CoP_y_list).reshape(-1, 1)
total_normal_force = np.array(total_normal_force_list).reshape(-1, 1)
force_angle_arr = np.array(force_angle_list).reshape(-1, 1)
force_magnitude_arr = np.array(force_magnitude_list).reshape(-1, 1)

# ===================== 打印输出 =====================
print(f"\n✅ data_original shape: {data_original.shape}")
print(f"\n✅ data_diff 基线扣除 shape: {data_diff.shape}")

print(f"\n✅ current_CoP_x shape: {current_CoP_x.shape}")
auto_print(current_CoP_x[0], "current_CoP_x (第一帧)") 

print(f"\n✅ current_CoP_y shape: {current_CoP_y.shape}")
auto_print(current_CoP_y[0], "current_CoP_y (第一帧)") 

print(f"\n✅ delta_CoP_x shape: {delta_CoP_x.shape}")
auto_print(delta_CoP_x[0], "delta_CoP_x (第一帧)") 

print(f"\n✅ delta_CoP_y shape: {delta_CoP_y.shape}")
auto_print(delta_CoP_y[0], "delta_CoP_y (第一帧)") 

print(f"\n✅ CoP_angle CoP偏移角度 shape: {CoP_angle.shape}")
auto_print(CoP_angle[0], "CoP_angle CoP偏移角度 (第一帧)") 

print(f"\n✅ CoP_magnitude CoP偏移大小 shape: {CoP_magnitude.shape}")
auto_print(CoP_magnitude[0], "CoP_magnitude CoP偏移大小 (第一帧)") 

print(f"✅ force_angle Force角度 shape: {force_angle_arr.shape}")
auto_print(force_angle_arr[0], "force_angle Force角度 (第一帧)") 

print(f"✅ force_magnitude Force大小 shape: {force_magnitude_arr.shape}")
auto_print(force_magnitude_arr[0], "force_magnitude Force大小 (第一帧)") 

# ===================== 拼接+保存 =====================
final_result = np.hstack([
    current_CoP_x,
    current_CoP_y,
    delta_CoP_x,
    delta_CoP_y,
    CoP_angle,
    CoP_magnitude,
    total_normal_force,
    force_angle_arr,
    force_magnitude_arr
])

columns = ["Current_CoP_x", "Current_CoP_y", "Delta_CoP_x", "Delta_CoP_y", 
           "CoP_angle", "CoP_magnitude", "Total_Normal_Force", "Force_angle", "Force_magnitude"]

df_result = pd.DataFrame(final_result, columns=columns)

output_csv_path = '/home/qcy/Project/data/2.PZT_tangential/weight/test/data_8_processed_cop.csv'
df_result.to_csv(output_csv_path, index=False)
print(f"\n✅ Processed results saved to {output_csv_path}")

# ===================== 可视化部分 =====================
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

try:
    from IPython.core.interactiveshell import InteractiveShell
    InteractiveShell.ast_node_interactivity = "none"
except ImportError:
    pass

# 选择要可视化的行索引
row_idx = 1503 # Corresponding to Row 990 in the image

# ===================== 数据获取 =====================
current_diff_frame_matrix = data_diff[row_idx].reshape(SENSOR_ROWS, SENSOR_COLS)

# 从df_result中获取CoP相关数据，确保显示的是实际计算值
curr_cop_x = df_result.loc[row_idx, "Current_CoP_x"]
curr_cop_y = df_result.loc[row_idx, "Current_CoP_y"]
delta_cop_x_val = df_result.loc[row_idx, "Delta_CoP_x"]
delta_cop_y_val = df_result.loc[row_idx, "Delta_CoP_y"] # This delta_cop_y_val is Y-down positive
cop_angle = df_result.loc[row_idx, "CoP_angle"]
cop_mag = df_result.loc[row_idx, "CoP_magnitude"]

# Corrected: Get force_angle and force_mag from df_result
force_angle = df_result.loc[row_idx, "Force_angle"] 
force_mag = df_result.loc[row_idx, "Force_magnitude"]

# base_plot_cop_x/y 作为蓝色叉号和紫色箭头的起点，来自第一次接触时的CoP
# It is crucial to get the first_contact_CoP_x/y that was *actually used* for delta_CoP_x/y calculation for this specific row_idx.
# If row_idx is early and contact_initialized was still False, first_contact_CoP_x/y would be None.
# So we fetch the actual first_contact_CoP_x/y from the point it was set in the loop.
# The `base_CoP_x` and `base_CoP_y` in the loop were actually `first_contact_CoP_x` and `first_contact_CoP_y`.
# To correctly retrieve the base_CoP for visualization (the blue cross)
# We need to find the first non-NaN CoP for this sequence of contact.
# A simpler approach for visualization is to reconstruct the base_CoP from current and delta.
if not np.isnan(curr_cop_x) and not np.isnan(delta_cop_x_val):
    base_plot_cop_x = curr_cop_x - delta_cop_x_val
    base_plot_cop_y = curr_cop_y - delta_cop_y_val
else:
    # Fallback if no valid CoP data for this frame
    base_plot_cop_x = SENSOR_COLS / 2
    base_plot_cop_y = SENSOR_ROWS / 2


epsilon = 1e-8

# --- 调试信息：检查 CoP 点和箭头终点是否一致 ---
print(f"\n--- DEBUG for visualization (row_idx {row_idx}) ---")
print(f"Base CoP (blue x): ({base_plot_cop_x:.2f}, {base_plot_cop_y:.2f})")
print(f"Current CoP (green o): ({curr_cop_x:.2f}, {curr_cop_y:.2f})")
print(f"Delta CoP (purple arrow components): dx={delta_cop_x_val:.2f}, dy={delta_cop_y_val:.2f}")

arrow_end_x_calculated = base_plot_cop_x + delta_cop_x_val
arrow_end_y_calculated = base_plot_cop_y + delta_cop_y_val
print(f"Purple arrow's calculated end point: ({arrow_end_x_calculated:.2f}, {arrow_end_y_calculated:.2f})")

if abs(arrow_end_x_calculated - curr_cop_x) < epsilon and abs(arrow_end_y_calculated - curr_cop_y) < epsilon:
    print("✅ Verified: Green dot and purple arrow endpoint COINCIDE.")
else:
    print("❌ WARNING: Green dot and purple arrow endpoint do NOT coincide. This indicates a data inconsistency.")
print("-------------------------------------------------")
# --- 调试信息结束 ---


# ===================== 图1：压力分布 + CoP位置 + CoP偏移向量 =====================
fig1 = plt.figure(figsize=(15, 8))
gs1 = GridSpec(1, 2, width_ratios=[2, 1], figure=fig1)

# Left: Pressure distribution heatmap and CoP points
ax_left = fig1.add_subplot(gs1[0, 0])
im = ax_left.imshow(current_diff_frame_matrix, cmap='hot', interpolation='nearest', origin='upper',
                    extent=[-0.5, SENSOR_COLS-0.5, SENSOR_ROWS-0.5, -0.5]) 
fig1.colorbar(im, ax=ax_left, fraction=0.046, pad=0.04)

ax_left.set_title(f"Pressure Distribution (Row {row_idx+1})", fontsize=14)
ax_left.set_xlabel("Column Index", fontsize=12)
ax_left.set_ylabel("Row Index (Y increases downwards)", fontsize=12) 
ax_left.set_xticks(np.arange(SENSOR_COLS))
ax_left.set_yticks(np.arange(SENSOR_ROWS))
ax_left.grid(True, linestyle='--', alpha=0.7)

# Plot Current CoP (green circle)
if not np.isnan(curr_cop_x) and not np.isnan(curr_cop_y):
    ax_left.plot(curr_cop_x, curr_cop_y, 'go', markersize=10, label=f'Current CoP ({curr_cop_x:.1f}, {curr_cop_y:.1f})')
    
    # Plot Base CoP (blue cross) and Offset Arrow (purple)
    if not np.isnan(base_plot_cop_x) and not np.isnan(base_plot_cop_y):
        ax_left.plot(base_plot_cop_x, base_plot_cop_y, 'bx', markersize=10, label=f'Base CoP ({base_plot_cop_x:.1f}, {base_plot_cop_y:.1f})')
        
        if not np.isnan(delta_cop_x_val) and not np.isnan(delta_cop_y_val) and (abs(delta_cop_x_val) > epsilon or abs(delta_cop_y_val) > epsilon):
            # For ax.arrow on a Y-down positive axis (origin='upper'), dx and dy should directly correspond to the change in those coordinates.
            # Since delta_cop_y_val is already Y-down positive, we pass it directly.
            ax_left.arrow(base_plot_cop_x, base_plot_cop_y, 
                          delta_cop_x_val,        # X-component (no change)
                          delta_cop_y_val,        # Y-component (Pass directly for Y-down positive axis)
                          head_width=0.3, head_length=0.3, fc='purple', ec='purple', linewidth=2, zorder=3,
                          label=f'CoP Offset (dx={delta_cop_x_val:.1f}, dy={delta_cop_y_val:.1f}) (Y-down positive visual)') 
    ax_left.legend()


# Right: CoP Offset Direction vs Force Direction (fixed length)
ax_right = fig1.add_subplot(gs1[0, 1])
ax_right.set_xlim(-1, 1)
ax_right.set_ylim(-1, 1)
ax_right.set_aspect('equal', adjustable='box')
ax_right.axis('off') # Hide axes for a cleaner vector field
ax_right.set_title("CoP Offset Direction vs Force Direction (Y-up positive)", fontsize=14) 

# Plot CoP Offset Vector (black arrow, fixed length, from origin)
if not np.isnan(cop_angle):
    theta_cop = np.deg2rad(cop_angle)
    ax_right.arrow(0, 0,
              0.7 * np.cos(theta_cop), # X-component (Y-up positive)
              0.7 * np.sin(theta_cop), # Y-component (Y-up positive)
              head_width=0.1, head_length=0.1,
              fc='k', ec='k', linewidth=3, zorder=2, label=f'CoP Angle: {cop_angle:.1f}°')

# Plot actual Force Vector (red arrow, fixed length, from origin)
if not np.isnan(force_angle):
    theta_force = np.deg2rad(force_angle)
    ax_right.arrow(0, 0,
              0.6 * np.cos(theta_force),
              0.6 * np.sin(theta_force),
              head_width=0.1, head_length=0.1,
              fc='r', ec='r', linewidth=2, zorder=2, label=f'Force Angle: {force_angle:.1f}°')
ax_right.legend(loc='upper right')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# ===================== Fig 2: CoP Offset Magnitude vs Force Magnitude (Normalized) =====================
fig2, ax = plt.subplots(figsize=(10, 6))

max_cop_mag = np.nanmax(df_result["CoP_magnitude"]) 
if np.isnan(max_cop_mag) or max_cop_mag < epsilon: max_cop_mag = 1.0

max_force_mag = np.nanmax(df_result["Force_magnitude"])
if np.isnan(max_force_mag) or max_force_mag < epsilon: max_force_mag = 1.0

ax.plot(df_result["CoP_magnitude"] / max_cop_mag, label="CoP Offset Magnitude (Normalized)", color='blue', alpha=0.7)
ax.plot(df_result["Force_magnitude"] / max_force_mag, label="Force Magnitude (Normalized)", color='red', alpha=0.7)

ax.set_title("CoP Offset Magnitude vs Force Magnitude Over Time (Normalized)", fontsize=14)
ax.set_xlabel("Frame Index", fontsize=12)
ax.set_ylabel("Magnitude (Normalized)", fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# ===================== Fig 3: CoP Offset Angle vs Force Angle (Trend) =====================
fig3, ax = plt.subplots(figsize=(10, 6))
ax.plot(df_result["CoP_angle"], label="CoP Offset Angle", color='blue', alpha=0.7)
ax.plot(df_result["Force_angle"], label="Force Angle", color='red', alpha=0.7)

ax.set_title("CoP Offset Angle vs Force Angle Over Time", fontsize=14)
ax.set_xlabel("Frame Index", fontsize=12)
ax.set_ylabel("Angle (Degrees)", fontsize=12)
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()