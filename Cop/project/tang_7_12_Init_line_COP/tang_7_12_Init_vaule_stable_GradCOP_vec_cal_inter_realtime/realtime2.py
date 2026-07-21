"""pyqtgraph 实时绘图 — GPU 渲染, 100fps"""
import numpy as np
from collections import deque
import threading
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import COP as COP

pg.setConfigOptions(antialias=True, background='w', foreground='k')

PLOT_TIMER_INTERVAL_MS = 10    # 绘图定时器刷新间隔(毫秒)
PLOT_MAG_HISTORY_LEN = 100     # 幅值历史缓冲区长度

def _yrange(data, pad=0.1):
    if len(data) < 2: return -1, 1
    mn, mx = min(data), max(data)
    r = mx - mn if mx != mn else 1
    return mn - r * pad, mx + r * pad


class CellGridItem(pg.GraphicsObject):
    """84 个独立色块 + 数值文字，复现 matplotlib table 效果"""
    def __init__(self, rows=12, cols=7):
        pg.GraphicsObject.__init__(self)
        self.rows, self.cols = rows, cols
        self.data = np.zeros((rows, cols))
        self.vmax = 1.0

    def set_data(self, data, vmax):
        self.data = data
        self.vmax = max(vmax, 1)
        self.update()

    def paint(self, p, opt, widget):
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        w = self.cols
        h = self.rows
        # 画色块
        for r in range(h):
            for c in range(w):
                v = self.data[r, c]
                t = v / self.vmax
                brush = self._brush(t)
                p.fillRect(QtCore.QRectF(c - 0.5, r - 0.5, 1, 1), brush)
        # 画网格线（有限线段，cosmetic pen 保证 1px 等宽）
        pen = QtGui.QPen(QtGui.QColor(128, 128, 128))
        pen.setCosmetic(True)
        p.setPen(pen)
        # 竖线
        for c in range(w + 1):
            x = c - 0.5
            p.drawLine(QtCore.QPointF(x, -0.5), QtCore.QPointF(x, h - 0.5))
        # 横线
        for r in range(h + 1):
            y = r - 0.5
            p.drawLine(QtCore.QPointF(-0.5, y), QtCore.QPointF(w - 0.5, y))

    def boundingRect(self):
        return QtCore.QRectF(-0.5, -0.5, self.cols, self.rows)

    @staticmethod
    def _brush(t):
        """白→浅红→红→深红，纯红色系"""
        t = max(0, min(1, t))
        pts = [(0.00, 255, 255, 255),   # 白
               (0.25, 255, 150, 150),   # 浅红
               (0.55, 255, 30, 30),     # 红
               (0.80, 180, 0, 0),       # 深红
               (1.00, 80, 0, 0)]        # 暗红
        for i in range(len(pts) - 1):
            t0, r0, g0, b0 = pts[i]
            t1, r1, g1, b1 = pts[i + 1]
            if t <= t1:
                s = (t - t0) / (t1 - t0)
                r = int(r0 + (r1 - r0) * s)
                g = int(g0 + (g1 - g0) * s)
                b = int(b0 + (b1 - b0) * s)
                return QtGui.QBrush(QtGui.QColor(r, g, b))
        return QtGui.QBrush(QtGui.QColor(80, 0, 0))


class GridLinesItem(pg.GraphicsObject):
    """纯网格线，避免 addLine 在 ViewBox 边界裁剪导致外圈视觉偏大"""
    def __init__(self, rows=12, cols=7):
        pg.GraphicsObject.__init__(self)
        self.rows, self.cols = rows, cols

    def paint(self, p, opt, widget):
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        pen = QtGui.QPen(QtGui.QColor(128, 128, 128))
        pen.setCosmetic(True)
        p.setPen(pen)
        for c in range(self.cols + 1):
            x = c - 0.5
            p.drawLine(QtCore.QPointF(x, -0.5), QtCore.QPointF(x, self.rows - 0.5))
        for r in range(self.rows + 1):
            y = r - 0.5
            p.drawLine(QtCore.QPointF(-0.5, y), QtCore.QPointF(self.cols - 0.5, y))

    def boundingRect(self):
        return QtCore.QRectF(-0.5, -0.5, self.cols, self.rows)


class RealTimePlot:
    def __init__(self):
        self.rows, self.cols = 12, 7
        self.lock = threading.Lock()
        self._fps_times = deque(maxlen=30)
        self._heat_vmax = 500.0   # 热力图色阶下限

        # === 全程存储 ===
        self.full_time_list = []
        self.full_adc_angle_list, self.full_adc_mag_list = [], []
        self.full_total_pressure_list = []
        self.full_adc_dx_list, self.full_adc_dy_list = [], []

        self.init_defaults()
        self.init_history()
        self.build_layout()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)
        self.timer.start(PLOT_TIMER_INTERVAL_MS)

    def init_defaults(self):
        self._pzt_angle_deg = 0.0           # PZT方向角度(度)
        self._pzt_mag_val = 0.0             # PZT幅值
        self._press_table_arr = np.zeros((12, 7))  # 压力表数据(12×7)
        self._cop_curr_x = 0.0              # 当前CoP X
        self._cop_curr_y = 0.0              # 当前CoP Y
        self._cop_base_x = 0.0              # 初始CoP X
        self._cop_base_y = 0.0              # 初始CoP Y
        self._cop_delta_x = 0.0             # CoP偏移X
        self._cop_delta_y = 0.0             # CoP偏移Y
        self._lcc_cx = float('nan')         # 最大连通域中心X
        self._lcc_cy = float('nan')         # 最大连通域中心Y
        self._skew_x = float('nan')         # 不对称性X
        self._skew_y = float('nan')         # 不对称性Y
        self._skew_pt_x = float('nan')      # 不对称偏移点X
        self._skew_pt_y = float('nan')      # 不对称偏移点Y
        self._angle_init_to_curr = float('nan')  # 初始→当前COP角度
        self._angle_skew = float('nan')          # 不对称方向角度
        self._lcc_min_r = self._lcc_max_r = float('nan')
        self._lcc_min_c = self._lcc_max_c = float('nan')
        self._total_press_val = 0.0         # 总压力值
        self._cop_state = 0                 # 接触状态
        self._pre_init_trail_x = []         # 初始COP确定前的轨迹X
        self._pre_init_trail_y = []         # 初始COP确定前的轨迹Y
        self._prev_init_flag = False        # 前一帧的初始COP确定标志，用于检测reset

    def init_history(self):
        hist_len = PLOT_MAG_HISTORY_LEN
        self.pzt_fz_history = deque(maxlen=hist_len)
        self.adc_dx_history = deque(maxlen=hist_len)
        self.adc_dy_history = deque(maxlen=hist_len)
        self.adc_mag_history = deque(maxlen=hist_len)

    # ===== 手工箭头工具 =====
    def _make_arrow_parts(self, plot):
        """在 plot 上创建箭头杆+三角头，返回 (shaft, head_L, head_R) 三条 PlotDataItem"""
        shaft = plot.plot([], [], pen=pg.mkPen('k', width=3))
        hL = plot.plot([], [], pen=pg.mkPen('k', width=2))
        hR = plot.plot([], [], pen=pg.mkPen('k', width=2))
        return shaft, hL, hR

    def _update_arrow(self, parts, angle_deg, length, color, origin=(0.0, 0.0)):
        """更新箭头：angle_deg=0=右, 90=上；尾部固定在 origin"""
        shaft, hL, hR = parts
        pen = pg.mkPen(color, width=3)
        shaft.setPen(pen); hL.setPen(pen); hR.setPen(pen)
        if length < 0.005:
            shaft.setData([], [])
            hL.setData([], []); hR.setData([], [])
            return
        rad = np.radians(angle_deg)
        dx = np.cos(rad) * length; dy = np.sin(rad) * length
        ox, oy = origin
        tip_x = ox + dx; tip_y = oy + dy
        shaft.setData([ox, tip_x], [oy, tip_y])

        # 箭头尖三角形：两条边
        head_len = min(length * 0.35, 0.12)
        back_angle = rad + np.pi
        aL = back_angle + np.radians(30)
        aR = back_angle - np.radians(30)
        hL.setData([tip_x, tip_x + np.cos(aL) * head_len], [tip_y, tip_y + np.sin(aL) * head_len])
        hR.setData([tip_x, tip_x + np.cos(aR) * head_len], [tip_y, tip_y + np.sin(aR) * head_len])

    # ===== 布局 =====
    def build_layout(self):
        self.win = pg.GraphicsLayoutWidget(title="RealTime")
        self.win.resize(1900, 1050)
        def _style_plot(p, title):
            p.setTitle(title, size='11pt', bold=True)

        # --- 左列 (col 0-1): PZT=红, Force=蓝 ---
        # row 0: PZT_z 跨两列
        p = self.win.addPlot(row=0, col=0, colspan=2, title="PZT_z")
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis('left').setWidth(45); p.getAxis('bottom').setHeight(28)
        _style_plot(p, "PZT_z")
        self.p_pzt_fz = p
        self._c_pzt_fz = p.plot(pen=pg.mkPen('r', width=3))
        self._t_pzt_fz = pg.TextItem("", color='r', anchor=(1, 1))
        p.addItem(self._t_pzt_fz)

        # row 1: PZT_x 跨两列
        p = self.win.addPlot(row=1, col=0, colspan=2, title="PZT_x")
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis('left').setWidth(45); p.getAxis('bottom').setHeight(28)
        _style_plot(p, "PZT_x")
        self.p_pzt_fx = p
        self._c_pzt_fx = p.plot(pen=pg.mkPen('r', width=3))
        self._t_pzt_fx = pg.TextItem("", color='r', anchor=(1, 1))
        p.addItem(self._t_pzt_fx)

        # row 2: PZT_y 跨两列
        p = self.win.addPlot(row=2, col=0, colspan=2, title="PZT_y")
        p.showGrid(x=True, y=True, alpha=0.3)
        p.getAxis('left').setWidth(45); p.getAxis('bottom').setHeight(28)
        _style_plot(p, "PZT_y")
        self.p_pzt_fy = p
        self._c_pzt_fy = p.plot(pen=pg.mkPen('r', width=3))
        self._t_pzt_fy = pg.TextItem("", color='r', anchor=(1, 1))
        p.addItem(self._t_pzt_fy)

        # --- 右列上方: Direction + Magnitude (col 2-3) ---
        self.p_dir = self.win.addPlot(row=0, col=2, title="Direction")
        self.p_dir.hideAxis('left'); self.p_dir.hideAxis('bottom')
        self.p_dir.setXRange(-1.2, 1.2); self.p_dir.setYRange(-1.2, 1.2); self.p_dir.setAspectLocked()
        self._dir_pzt = self._make_arrow_parts(self.p_dir)
        self._update_arrow(self._dir_pzt, 0, 0.45, 'r')
        self._dir_txt_pzt = pg.TextItem("", anchor=(0, 1))
        self.p_dir.addItem(self._dir_txt_pzt)
        self._dir_init_to_curr = self._make_arrow_parts(self.p_dir)
        self._dir_txt_init_to_curr = pg.TextItem("", anchor=(0, 1))
        self.p_dir.addItem(self._dir_txt_init_to_curr)
        self._dir_skew = self._make_arrow_parts(self.p_dir)
        self._dir_txt_skew = pg.TextItem("", anchor=(0, 1))
        self.p_dir.addItem(self._dir_txt_skew)

        self.p_mag = self.win.addPlot(row=0, col=3, title="Magnitude")
        self.p_mag.hideAxis('left'); self.p_mag.hideAxis('bottom')
        self.p_mag.setXRange(-0.8, 0.8); self.p_mag.setYRange(-0.8, 0.8); self.p_mag.setAspectLocked()
        self._mag_pzt = self._make_arrow_parts(self.p_mag)
        self._update_arrow(self._mag_pzt, 0, 0.10, 'r')
        self._mag_txt_pzt = pg.TextItem("", anchor=(0, 1))
        self.p_mag.addItem(self._mag_txt_pzt)

        # --- 右列下方: Pressure Table + Gradient (row 1-3, col 2-3) ---
        self.p_table = self.win.addPlot(row=1, col=2, rowspan=3, title="Pressure Table")
        self.p_table.hideAxis('left'); self.p_table.hideAxis('bottom')
        self.p_table.setAspectLocked(); self.p_table.invertY(True)
        self.p_table.setXRange(-0.5, 6.5); self.p_table.setYRange(-0.5, 11.5)
        self.p_table.getViewBox().setBackgroundColor('w')
        self.p_table.getViewBox().setBorder(pg.mkPen(width=0))
        # CellGridItem — 84 个独立色块 + 网格线（网格线在 paint() 中绘制）
        self._cell_grid = CellGridItem(12, 7)
        self.p_table.addItem(self._cell_grid)
        # 数值文字
        self._cell_txts = []
        for r in range(12):
            row_t = []
            for c in range(7):
                t = pg.TextItem("", color='k', anchor=(0.5, 0.5))
                self.p_table.addItem(t)
                t.setPos(c, r)
                row_t.append(t)
            self._cell_txts.append(row_t)
        # 初始COP确定前轨迹（橙色小点，持久显示）
        self._pre_init_trail = pg.ScatterPlotItem()
        self.p_table.addItem(self._pre_init_trail)
        # CoP 标记
        self._cop_dots = pg.ScatterPlotItem()
        self.p_table.addItem(self._cop_dots)
        self._cop_arr, self._cop_hL, self._cop_hR = self._make_arrow_parts(self.p_table)
        # LCC 矩形边框（4条线）
        self._lcc_rect_lines = [self.p_table.plot([], [], pen=pg.mkPen('#CCAA00', width=2)) for _ in range(4)]
        # LCC 中心 + 不对称偏移点
        self._lcc_dot = pg.ScatterPlotItem()
        self.p_table.addItem(self._lcc_dot)
        self._skew_pt_dot = pg.ScatterPlotItem()
        self.p_table.addItem(self._skew_pt_dot)
        # 初始COP → 当前COP 箭头
        self._init_to_curr_arrow = list(self._make_arrow_parts(self.p_table))
        # 图例
        self._legend = pg.TextItem("", anchor=(0, 0))
        self.p_table.addItem(self._legend)
        self._legend.setPos(0, -1.2)

        self.p_grad = self.win.addPlot(row=1, col=3, rowspan=3, title="Gradient Arrows")
        self.p_grad.hideAxis('left'); self.p_grad.hideAxis('bottom')
        self.p_grad.setAspectLocked(); self.p_grad.invertY(True)
        self.p_grad.setXRange(-0.5, 6.5); self.p_grad.setYRange(-0.5, 11.5)
        self.p_grad.getViewBox().setBackgroundColor('w')
        self.p_grad.getViewBox().setBorder(pg.mkPen(width=0))
        self._grid_lines = GridLinesItem(12, 7)
        self.p_grad.addItem(self._grid_lines)
        self._g_lines = []
        self._g_heads = []
        for _ in range(84):
            ln = self.p_grad.plot([0, 0], [0, 0], pen=pg.mkPen('k', width=1.5))
            self._g_lines.append(ln)
            dot = pg.ScatterPlotItem()
            self.p_grad.addItem(dot)
            self._g_heads.append(dot)


        self.win.show()

    # ===== 数据接口 =====
    def set_data(self, pzt_angle_deg, pzt_mag_val, force_angle_deg, force_mag_val,
                 press_table_arr, total_press_val, force_total_mag,
                 cop_curr_x, cop_curr_y, cop_base_x, cop_base_y, cop_delta_x, cop_delta_y,
                 force_fx_val, force_fy_val, force_fz_val,
                 cal_fx_val=None, cal_fy_val=None, cal_angle_deg=None, cal_mag_val=None,
                 cop_state=0, lcc_cx=None, lcc_cy=None,
                 skew_x=None, skew_y=None,
                 skew_pt_x=None, skew_pt_y=None,
                 angle_init_to_curr=None, angle_skew=None,
                 lcc_min_r=None, lcc_max_r=None,
                 lcc_min_c=None, lcc_max_c=None):
        with self.lock:
            self._pzt_angle_deg = pzt_angle_deg
            self._pzt_mag_val = pzt_mag_val
            self._press_table_arr = press_table_arr.reshape(self.rows, self.cols)
            self._cop_state = cop_state
            self._cop_curr_x = cop_curr_x
            self._cop_curr_y = cop_curr_y
            self._cop_base_x = cop_base_x
            self._cop_base_y = cop_base_y
            self._cop_delta_x = cop_delta_x
            self._cop_delta_y = cop_delta_y
            self._lcc_cx = lcc_cx if lcc_cx is not None else float('nan')
            self._lcc_cy = lcc_cy if lcc_cy is not None else float('nan')
            self._skew_x = skew_x if skew_x is not None else float('nan')
            self._skew_y = skew_y if skew_y is not None else float('nan')
            self._skew_pt_x = skew_pt_x if skew_pt_x is not None else float('nan')
            self._skew_pt_y = skew_pt_y if skew_pt_y is not None else float('nan')
            self._angle_init_to_curr = angle_init_to_curr if angle_init_to_curr is not None else float('nan')
            self._angle_skew = angle_skew if angle_skew is not None else float('nan')
            self._lcc_min_r = lcc_min_r if lcc_min_r is not None else float('nan')
            self._lcc_max_r = lcc_max_r if lcc_max_r is not None else float('nan')
            self._lcc_min_c = lcc_min_c if lcc_min_c is not None else float('nan')
            self._lcc_max_c = lcc_max_c if lcc_max_c is not None else float('nan')
            self._total_press_val = total_press_val

            # 初始COP确定前：累积COP轨迹；检测reset时清空
            if not COP.g_cop_contact_init_flag:
                if self._prev_init_flag:  # True→False：reset 发生
                    self._pre_init_trail_x.clear()
                    self._pre_init_trail_y.clear()
                if not np.isnan(cop_curr_x):
                    self._pre_init_trail_x.append(cop_curr_x)
                    self._pre_init_trail_y.append(cop_curr_y)
            self._prev_init_flag = COP.g_cop_contact_init_flag

            self.adc_mag_history.append(pzt_mag_val)
            self.pzt_fz_history.append(total_press_val)
            self.adc_dx_history.append(cop_delta_x)
            self.adc_dy_history.append(cop_delta_y)

    def append_full_data(self, rel_time_ms,
                          pzt_angle_deg, pzt_mag_val, total_press_val,
                          cop_delta_x_filt, cop_delta_y_filt,
                          force_angle_deg, force_mag_val,
                          force_fz_filt, force_fx_filt, force_fy_filt,
                          cal_angle_deg=None, cal_mag_val=None, cal_fx_val=None, cal_fy_val=None):
        with self.lock:
            self.full_time_list.append(rel_time_ms)
            self.full_adc_angle_list.append(pzt_angle_deg)
            self.full_adc_mag_list.append(pzt_mag_val)
            self.full_total_pressure_list.append(total_press_val)
            self.full_adc_dx_list.append(cop_delta_x_filt)
            self.full_adc_dy_list.append(cop_delta_y_filt)

    # ===== 更新 =====
    def update_all(self):
        t0 = time.perf_counter()
        with self.lock:
            pzt_angle_deg = self._pzt_angle_deg
            pzt_mag_val = self._pzt_mag_val
            pzt_fz_hist = list(self.pzt_fz_history)
            cop_dx_hist = list(self.adc_dx_history); cop_dy_hist = list(self.adc_dy_history)
            press_table_arr = self._press_table_arr.copy()
            cop_curr_x = self._cop_curr_x; cop_curr_y = self._cop_curr_y
            cop_base_x = self._cop_base_x; cop_base_y = self._cop_base_y
            cop_delta_x = self._cop_delta_x; cop_delta_y = self._cop_delta_y
            lcc_cx = self._lcc_cx; lcc_cy = self._lcc_cy
            skew_x = self._skew_x; skew_y = self._skew_y
            skew_pt_x = self._skew_pt_x; skew_pt_y = self._skew_pt_y
            angle_init_to_curr = self._angle_init_to_curr
            angle_skew = self._angle_skew
            lcc_min_r = self._lcc_min_r; lcc_max_r = self._lcc_max_r
            lcc_min_c = self._lcc_min_c; lcc_max_c = self._lcc_max_c
            cop_state = self._cop_state
            with COP.g_cop_grad_table_lock:
                grad_arr = COP.g_cop_grad_table_arr.copy()

        # 状态显示
        _state_names = {0: "未接触", 1: "粗略测量", 2: "精细测量"}
        self.win.setWindowTitle(f"RealTime2 — {_state_names.get(cop_state, '?')}")

        # Direction: PZT=red
        fs = self._font_size(12)
        self._update_arrow(self._dir_pzt, pzt_angle_deg, 0.45, 'r')
        self._dir_txt_pzt.setHtml(self._html(f'PZT_Angle: {pzt_angle_deg:.1f}°', 'red', fs))
        self._dir_txt_pzt.setPos(0.75, 1.15)

        # Direction: 初始COP→当前COP = blue
        if not np.isnan(angle_init_to_curr):
            self._update_arrow(self._dir_init_to_curr, angle_init_to_curr, 0.40, 'b')
            self._dir_txt_init_to_curr.setHtml(self._html(f'Init→Curr: {angle_init_to_curr:.1f}°', 'blue', fs))
        else:
            self._update_arrow(self._dir_init_to_curr, 0, 0.0, 'b')
            self._dir_txt_init_to_curr.setHtml("")
        self._dir_txt_init_to_curr.setPos(0.75, 0.95)

        # Direction: 不对称方向 = orange
        if not np.isnan(angle_skew):
            self._update_arrow(self._dir_skew, angle_skew, 0.40, 'orange')
            self._dir_txt_skew.setHtml(self._html(f'Skew: {angle_skew:.1f}°', 'orange', fs))
        else:
            self._update_arrow(self._dir_skew, 0, 0.0, 'orange')
            self._dir_txt_skew.setHtml("")
        self._dir_txt_skew.setPos(0.75, 0.75)

        # Magnitude
        pzt_mag_len = max(min((pzt_mag_val / 5.0) * 0.65, 0.65), 0.01)
        self._update_arrow(self._mag_pzt, pzt_angle_deg, pzt_mag_len, 'r')
        self._mag_txt_pzt.setHtml(self._html(f'PZT_Mag: {pzt_mag_val:.1f}', 'red', fs))
        self._mag_txt_pzt.setPos(0.35, 0.75)

        # Time-series
        self._u1(self._c_pzt_fz, self.p_pzt_fz, pzt_fz_hist, self._t_pzt_fz, "PZT_z", fs=fs)
        self._u1(self._c_pzt_fx, self.p_pzt_fx, cop_dx_hist, self._t_pzt_fx, "PZT_x", fs=fs)
        self._u1(self._c_pzt_fy, self.p_pzt_fy, cop_dy_hist, self._t_pzt_fy, "PZT_y", fs=fs)

        # 初始COP确定前轨迹（始终显示）
        if self._pre_init_trail_x:
            self._pre_init_trail.setData(
                x=self._pre_init_trail_x, y=self._pre_init_trail_y,
                brush=(255, 165, 0), size=6
            )

        # Pressure table + CoP + Gradient
        if COP.g_cop_contact_init_flag:
            cell_vmax = max(np.max(press_table_arr), self._heat_vmax)
            self._cell_grid.set_data(press_table_arr, cell_vmax)
            for row_idx in range(12):
                for col_idx in range(7):
                    cell_val = press_table_arr[row_idx, col_idx]
                    self._cell_txts[row_idx][col_idx].setText(f"{cell_val:.0f}" if cell_val > 0 else "")
            # CoP dots + arrow (绿色=COP, 蓝色x=初始COP)
            spots = [{'pos': (cop_curr_x, cop_curr_y), 'brush': 'g', 'size': 12, 'symbol': 'o'}]
            if not np.isnan(cop_base_x) and not np.isnan(cop_base_y):
                spots.append({'pos': (cop_base_x, cop_base_y), 'brush': 'b', 'symbol': 'x', 'size': 15})
            self._cop_dots.setData(spots=spots)
            if not np.isnan(cop_base_x) and not np.isnan(cop_base_y) and np.hypot(cop_delta_x, cop_delta_y) > 0.05:
                self._update_arrow((self._cop_arr, self._cop_hL, self._cop_hR),
                                   np.degrees(np.arctan2(-cop_delta_y, cop_delta_x)) if abs(cop_delta_x) + abs(cop_delta_y) > 1e-6 else 0,
                                   np.hypot(cop_delta_x, cop_delta_y), 'r', (cop_base_x, cop_base_y))
            else:
                self._cop_arr.setData([], [])
                self._cop_hL.setData([], [])
                self._cop_hR.setData([], [])

            # LCC 中心（黄色方块）
            if not np.isnan(lcc_cx) and not np.isnan(lcc_cy):
                self._lcc_dot.setData(x=[lcc_cx], y=[lcc_cy], brush='y', symbol='s', size=14)
            else:
                self._lcc_dot.setData(x=[], y=[])

            # LCC 矩形边框
            has_rect = not (np.isnan(lcc_min_r) or np.isnan(lcc_max_r) or
                           np.isnan(lcc_min_c) or np.isnan(lcc_max_c))
            if has_rect:
                x0, x1 = lcc_min_c - 0.5, lcc_max_c + 0.5
                y0, y1 = lcc_min_r - 0.5, lcc_max_r + 0.5
                self._lcc_rect_lines[0].setData([x0, x1], [y0, y0])  # top
                self._lcc_rect_lines[1].setData([x0, x1], [y1, y1])  # bottom
                self._lcc_rect_lines[2].setData([x0, x0], [y0, y1])  # left
                self._lcc_rect_lines[3].setData([x1, x1], [y0, y1])  # right
            else:
                for ln in self._lcc_rect_lines:
                    ln.setData([], [])

            # 不对称偏移点（橙色圆点）
            has_skew = not (np.isnan(skew_pt_x) or np.isnan(skew_pt_y))
            if has_skew:
                self._skew_pt_dot.setData(x=[skew_pt_x], y=[skew_pt_y], brush='orange', symbol='o', size=12)
            else:
                self._skew_pt_dot.setData(x=[], y=[])

            # 初始COP → 当前COP 箭头（蓝色）
            has_cop = not (np.isnan(cop_curr_x) or np.isnan(cop_curr_y))
            has_base = not (np.isnan(cop_base_x) or np.isnan(cop_base_y))
            if has_cop and has_base:
                dx, dy = cop_curr_x - cop_base_x, cop_curr_y - cop_base_y
                dist = np.hypot(dx, dy)
                if dist > 0.03:
                    ang = np.degrees(np.arctan2(dy, dx))
                    self._update_arrow(tuple(self._init_to_curr_arrow), ang, dist, 'b', (cop_base_x, cop_base_y))
                else:
                    for p in self._init_to_curr_arrow: p.setData([], [])
            else:
                for p in self._init_to_curr_arrow: p.setData([], [])

            # 角度文字和颜色图例
            a1_str = f"{angle_init_to_curr:.1f}" if not np.isnan(angle_init_to_curr) else "--"
            a2_str = f"{angle_skew:.1f}" if not np.isnan(angle_skew) else "--"
            legend_html = (
                f'<span style="color:black;font-size:10pt;font-weight:bold">Legend:</span><br>'
                f'<span style="color:blue">✚ InitCOP</span>  '
                f'<span style="color:green">● CurrCOP</span><br>'
                f'<span style="color:#CCAA00">■ LCC</span>  '
                f'<span style="color:orange">● SkewPt</span><br>'
                f'<span style="color:blue">→ Init→Curr: {a1_str}°</span><br>'
                f'<span style="color:orange">→ Skew: {a2_str}°</span>'
            )
            self._legend.setHtml(legend_html)

            # Gradient arrows
            for grad_idx, (grad_ln, grad_dot) in enumerate(zip(self._g_lines, self._g_heads)):
                grad_row, grad_col = divmod(grad_idx, 7)
                grad_x, grad_y = grad_arr[grad_row, grad_col, 0], grad_arr[grad_row, grad_col, 1]
                grad_mag = np.hypot(grad_x, grad_y)
                if grad_mag > 1.0:
                    arrow_dx = -grad_x / grad_mag * 0.3
                    arrow_dy = grad_y / grad_mag * 0.3
                    tip_x = grad_col + arrow_dx
                    tip_y = grad_row + arrow_dy
                    grad_ln.setData([grad_col, tip_x], [grad_row, tip_y])
                    grad_dot.setData(x=[tip_x], y=[tip_y], brush='k', size=4)
                else:
                    grad_ln.setData([], [])
                    grad_dot.setData(x=[], y=[])
        else:
            # 初始COP未确定：正常显示压力表，用黄色标记当前COP
            cell_vmax = max(np.max(press_table_arr), self._heat_vmax)
            self._cell_grid.set_data(press_table_arr, cell_vmax)
            for row_idx in range(12):
                for col_idx in range(7):
                    cell_val = press_table_arr[row_idx, col_idx]
                    self._cell_txts[row_idx][col_idx].setText(f"{cell_val:.0f}" if cell_val > 0 else "")
            self._cop_dots.setData(spots=[{'pos': (cop_curr_x, cop_curr_y), 'brush': 'y', 'size': 12}])
            self._cop_arr.setData([], [])
            self._cop_hL.setData([], [])
            self._cop_hR.setData([], [])
            self._lcc_dot.setData(x=[], y=[])
            self._skew_pt_dot.setData(x=[], y=[])
            for ln in self._lcc_rect_lines: ln.setData([], [])
            for p in self._init_to_curr_arrow: p.setData([], [])
            self._legend.setHtml("")
            for grad_ln, grad_dot in zip(self._g_lines, self._g_heads):
                grad_ln.setData([], [])
                grad_dot.setData(x=[], y=[])

        # FPS
    @staticmethod
    def _html(text, color, size=16):
        return f'<span style="color:{color};font-size:{size}pt;font-weight:bold">{text}</span>'

    def _font_size(self, base=16):
        w = self.win.width()
        return max(int(base * w / 1900), 7)

    def _u1(self, curve, plot, data, txt, label, color='red', fs=16):
        if data:
            xs = list(range(len(data)))
            curve.setData(xs, data)
            plot.setXRange(0, max(len(xs) - 1, 1))
            lo, hi = _yrange(data)
            plot.setYRange(lo, hi, padding=0)
            span = hi - lo if hi != lo else 1
            txt.setHtml(self._html(f'{label}={data[-1]:.2f}', color, fs))
            txt.setPos(int(max(len(xs) - 1, 1) * 1), hi - span * 0.12)

    # ===== 全程静态图 (matplotlib Agg) =====
    def plot_full_magnitude_curve(self, save_dir):
        import os; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        if len(self.full_time_list) == 0: print("⚠️ 无数据"); return
        t = self.full_time_list
        fig, axes = plt.subplots(5, 1, figsize=(14, 18))
        a1, a2, a3, a4, a5 = axes
        def _p(ax, d, c, lbl):
            if d and len(d) == len(t): ax.plot(t, d, c, linewidth=1.0, label=lbl)
        _p(a1, self.full_adc_angle_list, 'b-', 'PZT Angle'); a1.set_title("PZT Angle"); a1.grid(True, alpha=0.3)
        _p(a2, self.full_adc_mag_list, 'b-', 'PZT Mag'); a2.set_title("PZT Mag"); a2.grid(True, alpha=0.3)
        _p(a3, self.full_total_pressure_list, 'b-', 'PZT Fz'); a3.set_title("PZT Fz"); a3.grid(True, alpha=0.3)
        _p(a4, self.full_adc_dx_list, 'b-', 'PZT Fx'); a4.set_title("PZT Fx"); a4.grid(True, alpha=0.3)
        _p(a5, self.full_adc_dy_list, 'c-', 'PZT Fy'); a5.set_title("PZT Fy"); a5.grid(True, alpha=0.3)
        for ax in axes: ax.set_xlabel("Time (ms)", fontsize=9)
        plt.tight_layout()
        idx = 1
        while os.path.exists(os.path.join(save_dir, f"full_analysis_cop_{idx}.png")): idx += 1
        sp = os.path.join(save_dir, f"full_analysis_cop_{idx}.png")
        plt.savefig(sp, dpi=300); print(f"📊 已保存：{sp}"); plt.close(fig)
