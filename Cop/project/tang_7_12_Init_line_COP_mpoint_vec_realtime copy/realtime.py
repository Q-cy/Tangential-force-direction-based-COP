import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from collections import deque
import threading
import COP as COP # 引入COP模块

# Plotting constants
PLOT_INTERVAL_MS = 100
ERROR_PLOT_LEN = 100
MAG_PLOT_LEN = 100

class RealTimePlot:
    """
    实时绘图类，使用 Matplotlib 绘制传感器数据。
    集成了多个子图，以 GridSpec 布局实现。
    支持多点CoP绘制。
    """
    def __init__(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

        self.rows, self.cols = COP.SENSOR_ROWS, COP.SENSOR_COLS # 从COP模块获取行列数

        # 初始化ROI范围变量（修复未定义错误）
        self.final_r1 = 0
        self.final_r2 = self.rows - 1
        self.final_c1 = 0
        self.final_c2 = self.cols - 1

        # epsilon 用于避免绘制极小的箭头
        self.epsilon = 0.01 

        # ===================== 全程数据存储列表 =====================
        self.full_time_list = []          # 存储全程时间戳 (ms)
        self.full_adc_mag_list = []       # 存储全程 CoP 偏移幅值 (主CoP)
        self.full_force_mag_list = []     # 存储全程力传感器幅值

        self.fig = plt.figure(figsize=(16, 12))
        self.build_layout()

        self.ani = FuncAnimation(self.fig, self.update_all, interval=PLOT_INTERVAL_MS, cache_frame_data=False)
        self.lock = threading.Lock() # 用于保护绘图数据

        self.init_history()

        # 初始化默认数据，确保不会出现 None 值
        # 这些变量现在存储“主CoP”的数据
        self.adc_angle = 0.0
        self.adc_mag = 0.0
        self.force_angle = 0.0
        self.force_mag = 0.0
        self.raw_fx = 0.0 # 主CoP对应的力传感器分量
        self.raw_fy = 0.0 # 主CoP对应的力传感器分量

        self.diff_frame = np.zeros((self.rows, self.cols)) # 用于显示压力分布
        self.total_pressure_sum = 0.0

        # 新增：存储所有CoP的数据列表
        self.current_cops_data = [] # 存储从COP.py获取的CoP列表

    def build_layout(self):
        gs_outer = GridSpec(4, 2, width_ratios=[1, 1], height_ratios=[6, 2, 2, 1], hspace=0.3, wspace=0.3)
        gs_arrows = gs_outer[0, 0].subgridspec(1, 2, wspace=0.3)

        self.ax1 = plt.subplot(gs_arrows[0, 0])
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal')
        self.ax1.axis('off')
        self.ax1.set_title("Direction Arrows (CoP Offset & Force)", fontsize=10)

        self.ax2 = plt.subplot(gs_arrows[0, 1])
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_aspect('equal')
        self.ax2.axis('off')
        self.ax2.set_title("Magnitude Arrows (CoP Offset & Force)", fontsize=10)

        gs_adc_components = gs_outer[1, 0].subgridspec(1, 2, hspace=0.4)
        self.ax_adc_dx = plt.subplot(gs_adc_components[0, 0])
        self.ax_adc_dx.set_title("PZT_Fx Magnitude (CoP Offset X)", fontsize=10)
        self.ax_adc_dx.grid(True, alpha=0.3)
        self.line_adc_dx, = self.ax_adc_dx.plot([], [], 'b-', linewidth=1.5)

        self.ax_adc_dy = plt.subplot(gs_adc_components[0, 1])
        self.ax_adc_dy.set_title("PZT_Fy Magnitude (CoP Offset Y)", fontsize=10)
        self.ax_adc_dy.grid(True, alpha=0.3)
        self.line_adc_dy, = self.ax_adc_dy.plot([], [], 'c-', linewidth=1.5)

        gs_force_components = gs_outer[2, 0].subgridspec(1, 2, hspace=0.4)
        self.ax_force_fx = plt.subplot(gs_force_components[0, 0])
        self.ax_force_fx.set_title("Force_Fx Magnitude", fontsize=10)
        self.ax_force_fx.grid(True, alpha=0.3)
        self.line_force_fx, = self.ax_force_fx.plot([], [], 'r-', linewidth=1.5)

        self.ax_force_fy = plt.subplot(gs_force_components[0, 1])
        self.ax_force_fy.set_title("Force_Fy Magnitude", fontsize=10)
        self.ax_force_fy.grid(True, alpha=0.3)
        self.line_force_fy, = self.ax_force_fy.plot([], [], 'm-', linewidth=1.5)

        self.ax4 = plt.subplot(gs_outer[3, 0])
        self.ax4.set_title("Angle Error between CoP Offset and Force (Main CoP)", fontsize=10)
        self.ax4.set_ylim(0, 180)
        self.ax4.grid(True, alpha=0.3)
        self.error_line, = self.ax4.plot([], [], 'g-o', linewidth=1.5, markersize=2)

        gs_right = gs_outer[:, 1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
        self.ax5 = self.fig.add_subplot(gs_right[0, 0])
        self.ax5.set_title("Pressure Table", fontsize=10)
        self.ax5.axis('off')

        self.ax6 = self.fig.add_subplot(gs_right[0, 1])
        self.ax6.set_title("Gradient Arrows", fontsize=10)
        self.ax6.axis('off')

    def init_history(self):
        self.angle_error_history = deque(maxlen=ERROR_PLOT_LEN)
        self.adc_dx_history = deque(maxlen=MAG_PLOT_LEN)
        self.adc_dy_history = deque(maxlen=MAG_PLOT_LEN)
        self.force_fx_history = deque(maxlen=MAG_PLOT_LEN)
        self.force_fy_history = deque(maxlen=MAG_PLOT_LEN)
        self.adc_mag_history = deque(maxlen=MAG_PLOT_LEN) 
        self.raw_force_mag_history = deque(maxlen=MAG_PLOT_LEN)

    def set_data(self, main_adc_angle, main_adc_mag, main_force_angle, main_force_mag, 
                 diff_frame, total_pressure_sum, raw_fx, raw_fy, list_of_cop_data): # list_of_cop_data 是新增的

        with self.lock:
            # 更新主CoP数据
            self.adc_angle = main_adc_angle
            self.adc_mag = main_adc_mag
            self.force_angle = main_force_angle
            self.force_mag = main_force_mag
            self.raw_fx = raw_fx
            self.raw_fy = raw_fy

            # 更新用于压力图的数据
            self.diff_frame = diff_frame.reshape(self.rows, self.cols)
            self.total_pressure_sum = total_pressure_sum

            # 更新所有CoP的数据列表
            self.current_cops_data = list_of_cop_data

            # 计算角度误差
            diff = abs(self.adc_angle - self.force_angle)
            error = min(diff, 360 - diff)
            self.angle_error_history.append(error)

            # 更新历史数据 (主CoP)
            self.adc_mag_history.append(self.adc_mag)
            self.raw_force_mag_history.append(self.force_mag) # 这里用force_mag而不是force_total_mag，因为main.py里已计算好
            self.adc_dx_history.append(self.current_cops_data[0]['delta_cop_x'] if self.current_cops_data else 0.0)
            self.adc_dy_history.append(self.current_cops_data[0]['delta_cop_y'] if self.current_cops_data else 0.0)
            self.force_fx_history.append(self.raw_fx)
            self.force_fy_history.append(self.raw_fy)

    def append_full_data(self, current_ms, adc_mag, force_mag):
        """
        单独的函数：向全程列表追加数据 (主CoP)
        """
        with self.lock:
            self.full_time_list.append(current_ms)
            self.full_adc_mag_list.append(adc_mag)
            self.full_force_mag_list.append(force_mag)

    def update_all(self, frame):
        self.update_direction_arrows()
        self.update_magnitude_arrows()
        self.update_adc_components() 
        self.update_force_components() 
        self.update_error()
        self.update_pressure_table()
        self.update_gradient_table()
        return []

    def update_direction_arrows(self):
        with self.lock:
            a = self.adc_angle
            f = self.force_angle
        self.ax1.clear()
        self.ax1.set_xlim(0, 1)
        self.ax1.set_ylim(0, 1)
        self.ax1.set_aspect('equal')
        self.ax1.axis('off')
        self.ax1.set_title("Direction Arrows (CoP Offset & Force)")
        self.ax1.arrow(0.5, 0.5, 0.4*np.cos(np.radians(a)), 0.4*np.sin(np.radians(a)),
                       head_width=0.12, fc='k', ec='k', lw=2.5)
        self.ax1.arrow(0.5, 0.5, 0.35*np.cos(np.radians(f)), 0.35*np.sin(np.radians(f)),
                       head_width=0.1, fc='r', ec='r', lw=2)

    def update_magnitude_arrows(self):
        with self.lock:
            a = self.adc_angle
            m = self.adc_mag
            fa = self.force_angle
            fm = self.force_mag

        self.ax2.clear()
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.set_aspect('equal')
        self.ax2.axis('off')
        self.ax2.set_title("Magnitude Arrows (CoP Offset & Force)", fontsize=10)

        # 黑色箭头（CoP Offset）逻辑
        th_adc = np.deg2rad(a)
        max_adc_length = 0.45
        adc_normalize_denominator = 5.0 # 根据最大理论CoP偏移幅值12.5附近调整
        l_adc = (m / adc_normalize_denominator) * max_adc_length if m > self.epsilon else 0.0
        l_adc = min(l_adc, max_adc_length)

        if l_adc > 0.02:
            head_width_adc = max(0.06, l_adc * 0.15)
            head_length_adc = max(0.04, l_adc * 0.1)
            self.ax2.arrow(0.5, 0.5, l_adc*np.cos(th_adc), l_adc*np.sin(th_adc),
                        head_width=head_width_adc, head_length=head_length_adc,
                        fc='k', ec='k', lw=2.5, length_includes_head=True,
                        alpha=1.0, joinstyle='round', capstyle='round')
        elif l_adc > self.epsilon:
            self.ax2.plot([0.5, 0.5 + l_adc*np.cos(th_adc)],
                        [0.5, 0.5 + l_adc*np.sin(th_adc)],
                        'k-', lw=2.5, alpha=1.0)
        self.ax2.text(0.5, 0.1, f"CoP Offset: {m:.2f}", ha='center', va='center', fontsize=8, color='black')

        # 红色箭头（Force）
        th_force = np.deg2rad(fa)
        max_force_length = 0.4
        l_force = (abs(fm) / 20.0) * max_force_length if abs(fm) > 0.0 else 0.0
        l_force = min(l_force, max_force_length)

        if l_force > 0.02:
            head_width = max(0.06, l_force * 0.15)
            head_length = max(0.04, l_force * 0.1)
            self.ax2.arrow(0.5, 0.5, l_force*np.cos(th_force), l_force*np.sin(th_force),
                          head_width=head_width, head_length=head_length,
                          fc='red', ec='darkred', lw=3.5, length_includes_head=True,
                          alpha=1.0, joinstyle='round', capstyle='round')
        else:
            self.ax2.plot([0.5, 0.5 + l_force*np.cos(th_force)],
                          [0.5, 0.5 + l_force*np.sin(th_force)],
                          'r-', lw=3.5, alpha=1.0)
        self.ax2.text(0.5, 0.9, f"Force: {fm:.1f}", ha='center', va='center', fontsize=8, color='red')

    def update_adc_components(self):
        # PZT_Fx (delta_cop_x)
        if len(self.adc_dx_history) > 0:
            xs = list(range(len(self.adc_dx_history)))
            ys = list(self.adc_dx_history)
            self.line_adc_dx.set_data(xs, ys)
            self.ax_adc_dx.set_xlim(0, len(xs))
            min_y_dx, max_y_dx = min(ys) * 0.95, max(ys) * 1.05
            if min_y_dx == max_y_dx:
                self.ax_adc_dx.set_ylim(min_y_dx - 1, max_y_dx + 1)
            else:
                self.ax_adc_dx.set_ylim(min_y_dx, max_y_dx)
        
        # PZT_Fy (delta_cop_y)
        if len(self.adc_dy_history) > 0:
            xs = list(range(len(self.adc_dy_history)))
            ys = list(self.adc_dy_history)
            self.line_adc_dy.set_data(xs, ys)
            self.ax_adc_dy.set_xlim(0, len(xs))
            min_y_dy, max_y_dy = min(ys) * 0.95, max(ys) * 1.05
            if min_y_dy == max_y_dy:
                self.ax_adc_dy.set_ylim(min_y_dy - 1, max_y_dy + 1)
            else:
                self.ax_adc_dy.set_ylim(min_y_dy, max_y_dy)

    def update_force_components(self):
        # Force_Fx
        if len(self.force_fx_history) > 0:
            xs = list(range(len(self.force_fx_history)))
            ys = list(self.force_fx_history)
            self.line_force_fx.set_data(xs, ys)
            self.ax_force_fx.set_xlim(0, len(xs))
            min_y_fx, max_y_fx = min(ys) * 0.95, max(ys) * 1.05
            if min_y_fx == max_y_fx:
                self.ax_force_fx.set_ylim(min_y_fx - 1, max_y_fx + 1)
            else:
                self.ax_force_fx.set_ylim(min_y_fx, max_y_fx)
        
        # Force_Fy
        if len(self.force_fy_history) > 0:
            xs = list(range(len(self.force_fy_history)))
            ys = list(self.force_fy_history)
            self.line_force_fy.set_data(xs, ys)
            self.ax_force_fy.set_xlim(0, len(xs))
            min_y_fy, max_y_fy = min(ys) * 0.95, max(ys) * 1.05
            if min_y_fy == max_y_fy:
                self.ax_force_fy.set_ylim(min_y_fy - 1, max_y_fy + 1)
            else:
                self.ax_force_fy.set_ylim(min_y_fy, max_y_fy)

    def update_error(self):
        if len(self.angle_error_history) > 0:
            xs = list(range(len(self.angle_error_history)))
            ys = list(self.angle_error_history)
            self.error_line.set_data(xs, ys)
            self.ax4.set_xlim(0, len(xs))

    def update_pressure_table(self):
        """
        更新压力表 (ax5)，显示初始CoP、动态CoP和CoP偏移向量。
        现在支持显示多个CoP。
        """
        with self.lock:
            data = self.diff_frame.copy()
            cops_to_plot = self.current_cops_data # 获取所有CoP数据

        self.ax5.clear()
        self.ax5.set_title("Pressure Table", fontsize=10)
        self.ax5.axis('off')
        nrows, ncols = self.rows, self.cols
        self.ax5.set_xlim(-0.5, ncols - 0.5)
        self.ax5.set_ylim(nrows - 0.5, -0.5) # Y轴反向，索引大的在上方
        self.ax5.set_aspect('equal')

        vmax = np.max(data) if np.max(data) != 0 else 1
        norm = data / vmax
        cmap = plt.colormaps['Reds']
        colors = cmap(norm)
        cell_text = [[f"{v:.0f}" for v in row] for row in data]

        self.table_plot = self.ax5.table(
            cellText=cell_text,
            cellColours=colors,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        self.table_plot.auto_set_font_size(False)
        self.table_plot.set_fontsize(8)

        for i in range(nrows):
            for j in range(ncols):
                cell = self.table_plot[(i, j)]
                cell.set_height(1/nrows)
                cell.set_width(1/ncols)

        # 绘制所有检测到的CoP
        for i, cop_info in enumerate(cops_to_plot):
            cop_x_plot = cop_info['cop_y'] # 注意这里x和y是反的，因为COP.py中x是列，y是行
            cop_y_plot = cop_info['cop_x']
            base_cop_x = cop_info['initial_cop_y']
            base_cop_y = cop_info['initial_cop_x']
            delta_cop_x = cop_info['delta_cop_y']
            delta_cop_y = cop_info['delta_cop_x'] # 这里的delta_cop_y是负的，因为y轴反向

            color = plt.cm.get_cmap('Set1', len(cops_to_plot))(i) # 为每个CoP分配不同颜色
            
            # 绘制初始CoP
            if cop_info['is_initialized']:
                self.ax5.plot(base_cop_x, base_cop_y, 'x', markersize=10, color=color, label=f'Initial CoP {cop_info["id"]}')
            
            # 绘制当前CoP
            self.ax5.scatter(cop_x_plot, cop_y_plot, s=150, facecolors='none', edgecolors=color, linewidths=2, label=f'Current CoP {cop_info["id"]}')

            # 绘制CoP偏移向量
            if cop_info['is_initialized'] and np.hypot(delta_cop_x, delta_cop_y) > 0.05:
                # delta_cop_y 为负值，因为绘制时Y轴向上为正，但我们的传感器Y轴向下
                self.ax5.arrow(base_cop_x, base_cop_y, delta_cop_x, -delta_cop_y, 
                               head_width=0.3, head_length=0.3, fc=color, ec=color, linewidth=2, alpha=0.7)

        # self.ax5.legend(fontsize=8, loc='upper left') # 太多CoP时legend可能不好看

    def update_gradient_table(self):
        """
        更新梯度箭头图 (ax6)，显示每个点的梯度。
        现在支持显示多个CoP。
        """
        with self.lock:
            with COP.grad_table_lock: 
                data = COP.grad_table_data.copy()
            cops_to_plot = self.current_cops_data

        self.ax6.clear()
        self.ax6.set_title("Gradient Arrows (gx, gy) 12×7", fontsize=10)
        nrows, ncols = self.rows, self.cols
        self.ax6.set_xlim(-0.5, ncols - 0.5)
        self.ax6.set_ylim(nrows - 0.5, -0.5)
        self.ax6.set_aspect('equal')
        self.ax6.axis('off')

        # 网格
        for r_grid in range(nrows + 1):
            self.ax6.axhline(r_grid - 0.5, color='black', linestyle='-', linewidth=0.5, zorder=0)
        for c_grid in range(ncols + 1):
            self.ax6.axvline(c_grid - 0.5, color='black', linestyle='-', linewidth=0.5, zorder=0)

        # 梯度箭头
        for r in range(nrows):
            for c in range(ncols):
                gx, gy = data[r, c, 0], data[r, c, 1]
                mag = np.hypot(gx, gy)

                if mag > 1.0: # 阈值 1.0
                    gx_norm = gx / mag
                    gy_norm = gy / mag

                    self.ax6.quiver(c, r, gx_norm, gy_norm, # x=col, y=row
                                    color='k',
                                    scale=2.5,
                                    width=0.02,
                                    headwidth=6,
                                    headlength=8,
                                    headaxislength=7,
                                    angles='xy',
                                    scale_units='xy',
                                    zorder=5)

        # 绘制所有检测到的CoP
        for i, cop_info in enumerate(cops_to_plot):
            cop_x_plot = cop_info['cop_y'] # CoP的x是列索引，对应绘图的x
            cop_y_plot = cop_info['cop_x'] # CoP的y是行索引，对应绘图的y
            base_cop_x = cop_info['initial_cop_y']
            base_cop_y = cop_info['initial_cop_x']
            delta_cop_x = cop_info['delta_cop_y']
            delta_cop_y = cop_info['delta_cop_x'] # 这里的delta_cop_y是负的，因为y轴反向

            color = plt.cm.get_cmap('Set1', len(cops_to_plot))(i)

            # 绘制初始CoP
            if cop_info['is_initialized']:
                self.ax6.plot(base_cop_x, base_cop_y, 'x', markersize=10, color=color, zorder=10)
            
            # 绘制当前CoP
            self.ax6.scatter(cop_x_plot, cop_y_plot, s=150, facecolors='none', edgecolors=color, linewidths=2, zorder=10)

            # 绘制CoP偏移向量
            if cop_info['is_initialized'] and np.hypot(delta_cop_x, delta_cop_y) > 0.05:
                self.ax6.arrow(base_cop_x, base_cop_y, delta_cop_x, -delta_cop_y,
                               head_width=0.3, head_length=0.3, fc=color, ec=color, linewidth=2, zorder=10)

        # self.ax6.legend(loc='upper left', fontsize=8) # 太多CoP时legend可能不好看

    # ==================== 程序结束绘制全程静态图 ====================
    def plot_full_magnitude_curve(self, save_dir):
        """
        在程序结束后绘制全程的ADC(CoP偏移)和力传感器幅值曲线。
        :param save_dir: 图片保存路径
        """
        import os
        
        # 如果没有数据，直接返回
        if len(self.full_time_list) == 0:
            print("⚠️ 没有采集到全程数据，跳过绘图。")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 绘制 CoP 偏移幅值
        if len(self.full_adc_mag_list) > 0:
            ax1.plot(self.full_time_list, self.full_adc_mag_list, 'b-', linewidth=1.5, label='CoP Offset Magnitude (Main CoP)')
            ax1.set_title("CoP Offset Magnitude Over Time (Main CoP)", fontsize=14)
            ax1.set_xlabel("Time (ms)", fontsize=12)
            ax1.set_ylabel("CoP Offset Magnitude (Units)", fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=12)
            ax1_min_y, ax1_max_y = (min(self.full_adc_mag_list) * 0.9 if len(self.full_adc_mag_list) > 0 else 0), \
                                  (max(self.full_adc_mag_list) * 1.1 if max(self.full_adc_mag_list) > 0 else 1)
            ax1.set_ylim(ax1_min_y if ax1_min_y != ax1_max_y else ax1_min_y - 1,
                         ax1_max_y if ax1_min_y != ax1_max_y else ax1_max_y + 1)


        # 绘制力传感器幅值
        if len(self.full_force_mag_list) > 0:
            ax2.plot(self.full_time_list, self.full_force_mag_list, 'r-', linewidth=1.5, label='Force Magnitude')
            ax2.set_title("Force Magnitude Over Time", fontsize=14)
            ax2.set_xlabel("Time (ms)", fontsize=12)
            ax2.set_ylabel("Force Magnitude (N)", fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=12)
            ax2_min_y, ax2_max_y = (min(self.full_force_mag_list) * 0.9 if len(self.full_force_mag_list) > 0 else 0), \
                                  (max(self.full_force_mag_list) * 1.1 if max(self.full_force_mag_list) > 0 else 1)
            ax2.set_ylim(ax2_min_y if ax2_min_y != ax2_max_y else ax2_min_y - 1,
                         ax2_max_y if ax2_min_y != ax2_max_y else ax2_max_y + 1)

        plt.tight_layout()
        save_path = os.path.join(save_dir, "full_magnitude_curve_cop.png")
        plt.savefig(save_path, dpi=300)
        print(f"📊 全程幅值曲线已保存至：{save_path}")
        plt.close(fig) # 关闭静态图，避免与实时图冲突

    def show(self):
        plt.tight_layout()
        plt.show()
