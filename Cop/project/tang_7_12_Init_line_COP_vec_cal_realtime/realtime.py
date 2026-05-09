import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from collections import deque
import threading
import COP as COP


# Plotting constants
PLOT_INTERVAL_MS = 100
ERROR_PLOT_LEN = 100
MAG_PLOT_LEN = 100


class RealTimePlot:
    """
    实时绘图类，使用 Matplotlib 绘制传感器数据。
    集成了多个子图，以 GridSpec 布局实现。
    """
    def __init__(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        self.rows, self.cols = 12, 7

        # 初始化ROI范围变量（修复未定义错误）
        self.final_r1 = 0
        self.final_r2 = self.rows - 1
        self.final_c1 = 0
        self.final_c2 = self.cols - 1

        # epsilon 用于避免绘制极小的箭头
        self.epsilon = 0.01 

        # ===================== 全程数据存储列表 =====================
        self.full_time_list = []          # 存储全程时间戳 (ms)
        self.full_adc_mag_list = []       # 存储全程 CoP 偏移幅值
        self.full_force_mag_list = []     # 存储全程力传感器幅值
        self.full_cal_mag_list = []       # 存储全程标定力幅值
        self.full_fx_list = []            # 全程实测 Fx
        self.full_fy_list = []            # 全程实测 Fy
        self.full_fx_cal_list = []        # 全程标定 Fx
        self.full_fy_cal_list = []        # 全程标定 Fy

        self.fig = plt.figure(figsize=(16, 12))
        self.build_layout()

        self.ani = FuncAnimation(self.fig, self.update_all, interval=PLOT_INTERVAL_MS, cache_frame_data=False)
        self.lock = threading.Lock()

        self.init_history()

        # 初始化默认数据，确保不会出现 None 值
        self.adc_angle = 0.0
        self.adc_mag = 0.0
        self.force_angle = 0.0
        self.force_mag = 0.0
        self.diff_frame = np.zeros((12, 7))

        self.cop_x = 0.0
        self.cop_y = 0.0
        self.base_cop_x = 0.0
        self.base_cop_y = 0.0
        self.delta_cop_x = 0.0
        self.delta_cop_y = 0.0
        self.raw_fx = 0.0 # 新增
        self.raw_fy = 0.0 # 新增
        self.fx_cal = None   # 标定力 X
        self.fy_cal = None   # 标定力 Y
        self.cal_angle = None
        self.cal_mag = None

        # 图右上方 Force/Cal 信息文本（大字）
        self.force_info_text = self.fig.text(0.52, 0.97, "", transform=self.fig.transFigure,
                                              fontsize=12, color='red', va='top', ha='left', weight='bold')
        self.cal_info_text = self.fig.text(0.78, 0.97, "", transform=self.fig.transFigure,
                                            fontsize=12, color='green', va='top', ha='left', weight='bold')

    def build_layout(self):
        # 调整GridSpec以容纳更多图表
        # 将原始的 [6, 1, 1, 1] height_ratios 调整为 [6, 1, 1, 1, 1, 1] 使得 ax3a, ax3b, ax4 可以分别再拆分
        # 实际操作是，将 ax3a, ax3b 的位置变成 subgridspec
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

        # ========== 修改点1：更新子图标题为 ADC Mag ==========
        # 为 ADC Fx 和 Fy 创建子图
        gs_adc_components = gs_outer[1, 0].subgridspec(1, 2, hspace=0.4)          # 主网格中选择第 2 行、第 1 列的单元格,并且划分 1 行 2 列个子网格
        self.ax_adc_dx = plt.subplot(gs_adc_components[0, 0])
        self.ax_adc_dx.set_title("PZT_Fx Magnitude (CoP Offset X)", fontsize=10)
        self.ax_adc_dx.grid(True, alpha=0.3)
        self.line_adc_dx, = self.ax_adc_dx.plot([], [], 'b-', linewidth=1.5)

        self.ax_adc_dy = plt.subplot(gs_adc_components[0, 1])
        self.ax_adc_dy.set_title("PZT_Fy Magnitude (CoP Offset Y)", fontsize=10)
        self.ax_adc_dy.grid(True, alpha=0.3)
        self.line_adc_dy, = self.ax_adc_dy.plot([], [], 'c-', linewidth=1.5)

        # 为 Force Fx 和 Fy 创建子图
        gs_force_components = gs_outer[2, 0].subgridspec(1, 2, hspace=0.4)
        self.ax_force_fx = plt.subplot(gs_force_components[0, 0])
        self.ax_force_fx.set_title("Force_Fx Magnitude", fontsize=10)
        self.ax_force_fx.grid(True, alpha=0.3)
        self.line_force_fx, = self.ax_force_fx.plot([], [], 'r-', linewidth=1.5)
        self.line_force_fx_cal, = self.ax_force_fx.plot([], [], 'g--', linewidth=1.5, alpha=0.8)

        self.ax_force_fy = plt.subplot(gs_force_components[0, 1])
        self.ax_force_fy.set_title("Force_Fy Magnitude", fontsize=10)
        self.ax_force_fy.grid(True, alpha=0.3)
        self.line_force_fy, = self.ax_force_fy.plot([], [], 'm-', linewidth=1.5)
        self.line_force_fy_cal, = self.ax_force_fy.plot([], [], 'g--', linewidth=1.5, alpha=0.8)

        # Angle Error 保持不变，但更新标题
        self.ax4 = plt.subplot(gs_outer[3, 0])
        self.ax4.set_title("Angle Error between CoP Offset and Force", fontsize=10)
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

    # ========== 修改点2：初始化历史队列，将raw_adc_sum_history改为adc_mag_history ==========
    def init_history(self):
        self.angle_error_history = deque(maxlen=ERROR_PLOT_LEN)
        # ADC (CoP Offset) 分量历史
        self.adc_dx_history = deque(maxlen=MAG_PLOT_LEN)
        self.adc_dy_history = deque(maxlen=MAG_PLOT_LEN)
        # Force 分量历史
        self.force_fx_history = deque(maxlen=MAG_PLOT_LEN)
        self.force_fy_history = deque(maxlen=MAG_PLOT_LEN)

        # 保留原有的总幅值历史，用于结束时的全程曲线绘制
        self.adc_mag_history = deque(maxlen=MAG_PLOT_LEN)
        self.raw_force_mag_history = deque(maxlen=MAG_PLOT_LEN)

        # 标定力分量历史
        self.force_fx_cal_history = deque(maxlen=MAG_PLOT_LEN)
        self.force_fy_cal_history = deque(maxlen=MAG_PLOT_LEN)


    def set_data(self, adc_angle, adc_mag, force_angle, force_mag, diff_frame, total_pressure_sum, force_total_mag,
                 cop_x, cop_y, base_cop_x, base_cop_y, delta_cop_x, delta_cop_y, raw_fx, raw_fy,
                 fx_cal=None, fy_cal=None, cal_angle=None, cal_mag=None):
        with self.lock:
            self.adc_angle = adc_angle
            self.adc_mag = adc_mag
            self.force_angle = force_angle
            self.force_mag = force_mag
            self.diff_frame = diff_frame.reshape(self.rows, self.cols)
            self.cop_x = cop_x
            self.cop_y = cop_y
            self.base_cop_x = base_cop_x
            self.base_cop_y = base_cop_y
            self.delta_cop_x = delta_cop_x
            self.delta_cop_y = delta_cop_y
            self.raw_fx = raw_fx
            self.raw_fy = raw_fy
            self.fx_cal = fx_cal
            self.fy_cal = fy_cal
            self.cal_angle = cal_angle
            self.cal_mag = cal_mag

            diff = abs(adc_angle - force_angle)
            error = min(diff, 360 - diff)
            self.angle_error_history.append(error)

            # ========== adc_mag到历史队列（不再使用raw_adc_sum） ==========
            self.adc_mag_history.append(adc_mag) # 用于绘制全程总幅值曲线
            self.raw_force_mag_history.append(force_total_mag)

            # 追加 CoP 偏移分量 (PZT Fx/Fy)
            self.adc_dx_history.append(delta_cop_x)
            self.adc_dy_history.append(delta_cop_y)

            # 追加力传感器分量 (Force Fx/Fy)
            self.force_fx_history.append(raw_fx)
            self.force_fy_history.append(raw_fy)

            # 追加标定力分量
            if fx_cal is not None:
                self.force_fx_cal_history.append(fx_cal)
                self.force_fy_cal_history.append(fy_cal)


    def append_full_data(self, current_ms, adc_mag, force_mag, cal_mag=None,
                          fx=None, fy=None, fx_cal=None, fy_cal=None):
        """
        单独的函数：向全程列表追加数据
        """
        with self.lock:
            self.full_time_list.append(current_ms)
            self.full_adc_mag_list.append(adc_mag)
            self.full_force_mag_list.append(force_mag)
            if cal_mag is not None:
                self.full_cal_mag_list.append(cal_mag)
            if fx is not None:
                self.full_fx_list.append(fx)
                self.full_fy_list.append(fy)
                self.full_fx_cal_list.append(fx_cal if fx_cal is not None else float('nan'))
                self.full_fy_cal_list.append(fy_cal if fy_cal is not None else float('nan'))


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

        # ========== 黑色箭头（CoP Offset）逻辑 ==========
        th_adc = np.deg2rad(a)
        max_adc_length = 0.45
        
        # 调整归一化分母。假设CoP偏移的幅值m最大约为15（12x7的网格中，sqrt(6^2+11^2)约等于12.5）。
        # 如果m/adc_normalize_denominator过小，箭头会一直很短。
        # 如果m/adc_normalize_denominator过大，箭头会很快达到max_adc_length。
        adc_normalize_denominator = 5.0 # 根据最大理论CoP偏移幅值12.5附近调整，比如15.0，这样当m接近15时箭头接近max_adc_length

        l_adc = (m / adc_normalize_denominator) * max_adc_length if m > self.epsilon else 0.0
        l_adc = min(l_adc, max_adc_length) # 确保箭头不会超出绘图区域

        # 动态调整箭头样式，借鉴红色箭头的逻辑
        if l_adc > 0.02:  # 长度足够时显示带箭头的箭头
            head_width_adc = max(0.06, l_adc * 0.15) # 头部宽度至少0.06，且随箭头长度线性增长
            head_length_adc = max(0.04, l_adc * 0.1) # 头部长度至少0.04，且随箭头长度线性增长
            self.ax2.arrow(0.5, 0.5, l_adc*np.cos(th_adc), l_adc*np.sin(th_adc),
                        head_width=head_width_adc, head_length=head_length_adc,
                        fc='k', ec='k', lw=2.5, length_includes_head=True,
                        alpha=1.0, joinstyle='round', capstyle='round')
        elif l_adc > self.epsilon:  # 长度过小时显示纯线段
            self.ax2.plot([0.5, 0.5 + l_adc*np.cos(th_adc)],
                        [0.5, 0.5 + l_adc*np.sin(th_adc)],
                        'k-', lw=2.5, alpha=1.0)
        # 如果 l_adc <= self.epsilon，则不绘制任何东西

        # ========== 红色箭头（Force）==========
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

        self.force_info_text.set_text(f"Force: (Fx={self.raw_fx:.2f}, Fy={self.raw_fy:.2f}) N")

        # ========== 绿色箭头（标定力）==========
        if self.cal_mag is not None and self.cal_angle is not None:
            th_cal = np.deg2rad(self.cal_angle)
            max_cal_length = 0.4
            l_cal = (self.cal_mag / 20.0) * max_cal_length if self.cal_mag > 0.0 else 0.0
            l_cal = min(l_cal, max_cal_length)

            if l_cal > 0.02:
                head_width = max(0.06, l_cal * 0.15)
                head_length = max(0.04, l_cal * 0.1)
                self.ax2.arrow(0.5, 0.5, l_cal*np.cos(th_cal), l_cal*np.sin(th_cal),
                              head_width=head_width, head_length=head_length,
                              fc='green', ec='darkgreen', lw=3.5, length_includes_head=True,
                              alpha=0.8, joinstyle='round', capstyle='round')
            elif l_cal > 0.0:
                self.ax2.plot([0.5, 0.5 + l_cal*np.cos(th_cal)],
                              [0.5, 0.5 + l_cal*np.sin(th_cal)],
                              'g-', lw=3.5, alpha=0.8)

            self.cal_info_text.set_text(f"Cal: (Fx={self.fx_cal:.2f}, Fy={self.fy_cal:.2f}) N")
        else:
            self.cal_info_text.set_text("")


    # ========== 更新绘图逻辑，读取adc_mag_history ==========
    def update_adc_components(self):
        # PZT_Fx (delta_cop_x)
        if len(self.adc_dx_history) > 0:
            xs = list(range(len(self.adc_dx_history)))
            ys = list(self.adc_dx_history)
            self.line_adc_dx.set_data(xs, ys)
            self.ax_adc_dx.set_xlim(0, len(xs))
            # 动态调整Y轴范围，避免数据被截断，且考虑零值情况
            min_y_dx, max_y_dx = min(ys) * 0.95, max(ys) * 1.05
            if min_y_dx == max_y_dx: # 如果所有值都相同
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
            all_ys = list(ys)
            if len(self.force_fx_cal_history) > 0:
                cal_xs = list(range(len(self.force_fx_cal_history)))
                cal_ys = list(self.force_fx_cal_history)
                self.line_force_fx_cal.set_data(cal_xs, cal_ys)
                all_ys.extend(cal_ys)
            min_y_fx, max_y_fx = min(all_ys) * 0.95, max(all_ys) * 1.05
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
            all_ys = list(ys)
            if len(self.force_fy_cal_history) > 0:
                cal_xs = list(range(len(self.force_fy_cal_history)))
                cal_ys = list(self.force_fy_cal_history)
                self.line_force_fy_cal.set_data(cal_xs, cal_ys)
                all_ys.extend(cal_ys)
            min_y_fy, max_y_fy = min(all_ys) * 0.95, max(all_ys) * 1.05
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
        """
        with self.lock:
            data = self.diff_frame.copy()
            cop_x_plot = self.cop_x
            cop_y_plot = self.cop_y
            r1, r2, c1, c2 = self.final_r1, self.final_r2, self.final_c1, self.final_c2
            delta_cop_x = self.delta_cop_x
            delta_cop_y = self.delta_cop_y
            base_cop_x = self.base_cop_x
            base_cop_y = self.base_cop_y

        self.ax5.clear()
        self.ax5.set_title("Pressure Table", fontsize=10)
        self.ax5.axis('off')
        nrows, ncols = self.rows, self.cols
        self.ax5.set_xlim(-0.5, ncols - 0.5)
        self.ax5.set_ylim(nrows - 0.5, -0.5)
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

        # 初始CoP 与 CoP 偏移向量
        if not np.isnan(base_cop_x) and not np.isnan(base_cop_y):
            self.ax5.plot(base_cop_x, base_cop_y, 'bx', markersize=10, label='Initial CoP')
        self.ax5.scatter(cop_x_plot, cop_y_plot, s=150, color='green', label='Current CoP')
        if np.hypot(delta_cop_x, delta_cop_y) > 0.05:
            self.ax5.arrow(base_cop_x, base_cop_y, delta_cop_x, -delta_cop_y,                          #-delta_cop_y是因为y坐标轴是索引大的在上(self.ax5.set_ylim(nrows - 0.5, -0.5))
                           head_width=0.3, head_length=0.3, fc='purple', ec='purple', linewidth=2)

        self.ax5.legend(fontsize=8)


    def update_gradient_table(self):
        """
        更新梯度箭头图 (ax6)，显示每个点的梯度。
        """
        with self.lock:
            # 第一步：加锁读取梯度数据（避免脏读）
            # 注意：这里需要修改为 COP.grad_table_lock 和 COP.grad_table_data
            with COP.grad_table_lock: 
                data = COP.grad_table_data.copy()
            # 第二步：读取其他变量（原逻辑不变）
            cop_x_plot = self.cop_x
            cop_y_plot = self.cop_y
            r1, r2, c1, c2 = self.final_r1, self.final_r2, self.final_c1, self.final_c2
            delta_cop_x = self.delta_cop_x
            delta_cop_y = self.delta_cop_y
            base_cop_x = self.base_cop_x
            base_cop_y = self.base_cop_y

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

                    self.ax6.quiver(c, r, gx_norm, gy_norm,
                                    color='k',
                                    scale=2.5,
                                    width=0.02,
                                    headwidth=6,
                                    headlength=8,
                                    headaxislength=7,
                                    angles='xy',
                                    scale_units='xy',
                                    zorder=5)

        # CoP 标记
        if not np.isnan(cop_x_plot) and not np.isnan(cop_y_plot):
            self.ax6.scatter(cop_x_plot, cop_y_plot, s=150, color='green', marker='o', zorder=10)

        if not np.isnan(base_cop_x) and not np.isnan(base_cop_y):
            self.ax6.plot(base_cop_x, base_cop_y, 'bx', markersize=10, zorder=10)

        if np.hypot(delta_cop_x, delta_cop_y) > 0.05:
            self.ax6.arrow(base_cop_x, base_cop_y, delta_cop_x, -delta_cop_y,
                           head_width=0.3, head_length=0.3, fc='purple', ec='purple', linewidth=2)

        # ROI 框（与压力表可对齐）
        rect_x = c1 - 0.5
        rect_y = r1 - 0.5
        rect_width = (c2 - c1 + 1)
        rect_height = (r2 - r1 + 1)
        roi_rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                 linewidth=3, edgecolor='blue', facecolor='none', linestyle='--', zorder=5)
        self.ax6.add_patch(roi_rect)

        self.ax6.legend(loc='upper left', fontsize=8)

    # ==================== 程序结束绘制全程综合图 ====================
    def plot_full_magnitude_curve(self, save_dir):
        """
        程序结束后绘制 6 面板综合图：
        (1) CoP Offset Magnitude, (2) Force Magnitude (实测+标定),
        (3) Fx (实测+标定), (4) Fy (实测+标定),
        (5) Error Fx + RMS, (6) Error Fy + RMS
        """
        import os

        if len(self.full_time_list) == 0:
            print("⚠️ 没有采集到全程数据，跳过绘图。")
            return

        has_cal = len(self.full_fx_cal_list) == len(self.full_time_list)
        has_fx = len(self.full_fx_list) == len(self.full_time_list)

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        (ax1, ax2), (ax3, ax4), (ax5, ax6) = axes
        t = self.full_time_list

        # --- (1) CoP Offset Magnitude ---
        if len(self.full_adc_mag_list) > 0:
            ax1.plot(t, self.full_adc_mag_list, 'b-', linewidth=1.0)
            ax1.set_title("CoP Offset Magnitude", fontsize=11)
            ax1.set_ylabel("Magnitude", fontsize=9)
            ax1.grid(True, alpha=0.3)

        # --- (2) Force Magnitude (实测 + 标定) ---
        if len(self.full_force_mag_list) > 0:
            ax2.plot(t, self.full_force_mag_list, 'r-', linewidth=1.0, label='Measured')
            if has_cal and len(self.full_cal_mag_list) == len(t):
                ax2.plot(t, self.full_cal_mag_list, 'g--', linewidth=1.0, label='Calibrated')
            ax2.set_title("Force Magnitude", fontsize=11)
            ax2.set_ylabel("Force (N)", fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=8)

        # --- (3) Fx (实测 + 标定) ---
        if has_fx:
            ax3.plot(t, self.full_fx_list, 'r-', linewidth=1.0, label='Fx measured')
            if has_cal:
                ax3.plot(t, self.full_fx_cal_list, 'g--', linewidth=1.0, label='Fx calibrated')
            ax3.set_title("Fx: Measured vs Calibrated", fontsize=11)
            ax3.set_ylabel("Force (N)", fontsize=9)
            ax3.grid(True, alpha=0.3)
            ax3.legend(fontsize=8)

        # --- (4) Fy (实测 + 标定) ---
        if has_fx:
            ax4.plot(t, self.full_fy_list, 'm-', linewidth=1.0, label='Fy measured')
            if has_cal:
                ax4.plot(t, self.full_fy_cal_list, 'c--', linewidth=1.0, label='Fy calibrated')
            ax4.set_title("Fy: Measured vs Calibrated", fontsize=11)
            ax4.set_ylabel("Force (N)", fontsize=9)
            ax4.grid(True, alpha=0.3)
            ax4.legend(fontsize=8)

        # --- (5) Error Fx ---
        if has_fx and has_cal:
            fx_err = [fx - fxc for fx, fxc in zip(self.full_fx_list, self.full_fx_cal_list)]
            rms_fx = np.sqrt(np.mean(np.array(fx_err)**2))
            ax5.plot(t, fx_err, 'r-', linewidth=1.0)
            ax5.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax5.set_title(f"Error Fx (RMS={rms_fx:.3f} N)", fontsize=11)
            ax5.set_ylabel("Error (N)", fontsize=9)
            ax5.grid(True, alpha=0.3)
            ax5.text(0.02, 0.95, f"RMS={rms_fx:.3f} N", transform=ax5.transAxes,
                     fontsize=10, va='top', color='red')

        # --- (6) Error Fy ---
        if has_fx and has_cal:
            fy_err = [fy - fyc for fy, fyc in zip(self.full_fy_list, self.full_fy_cal_list)]
            rms_fy = np.sqrt(np.mean(np.array(fy_err)**2))
            ax6.plot(t, fy_err, 'm-', linewidth=1.0)
            ax6.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax6.set_title(f"Error Fy (RMS={rms_fy:.3f} N)", fontsize=11)
            ax6.set_ylabel("Error (N)", fontsize=9)
            ax6.grid(True, alpha=0.3)
            ax6.text(0.02, 0.95, f"RMS={rms_fy:.3f} N", transform=ax6.transAxes,
                     fontsize=10, va='top', color='red')

        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_xlabel("Time (ms)", fontsize=9)

        plt.tight_layout()
        idx = 1
        while os.path.exists(os.path.join(save_dir, f"full_analysis_cop_{idx}.png")):
            idx += 1
        save_path = os.path.join(save_dir, f"full_analysis_cop_{idx}.png")
        plt.savefig(save_path, dpi=300)
        print(f"📊 全程综合分析图已保存至：{save_path}")
        plt.close(fig)


    def show(self):
        plt.tight_layout()
        plt.show()
