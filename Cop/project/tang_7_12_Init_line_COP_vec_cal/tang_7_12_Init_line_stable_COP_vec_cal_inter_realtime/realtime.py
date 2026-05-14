"""pyqtgraph 实时绘图 — GPU 渲染, 100fps"""
import numpy as np
from collections import deque
import threading
import time
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import COP as COP

pg.setConfigOptions(antialias=True, background='w', foreground='k')

PLOT_INTERVAL_MS = 10
ERROR_PLOT_LEN = 100
MAG_PLOT_LEN = 100

HAS_CAL = True

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
        self.full_force_angle_list, self.full_force_mag_list = [], []
        self.full_fz_list, self.full_fx_list, self.full_fy_list = [], [], []
        self.full_cal_angle_list, self.full_cal_mag_list = [], []
        self.full_fx_cal_list, self.full_fy_cal_list = [], []

        self.init_defaults()
        self.init_history()
        self.build_layout()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)
        self.timer.start(PLOT_INTERVAL_MS)

    def init_defaults(self):
        self.adc_angle = 0.0; self.adc_mag = 0.0
        self.force_angle = 0.0; self.force_mag = 0.0
        self.diff_frame = np.zeros((12, 7))
        self.cop_x = 0.0; self.cop_y = 0.0
        self.base_cop_x = 0.0; self.base_cop_y = 0.0
        self.delta_cop_x = 0.0; self.delta_cop_y = 0.0
        self.raw_fx = 0.0; self.raw_fy = 0.0; self.raw_fz = 0.0
        self.total_pressure = 0.0
        self.fx_cal = None; self.fy_cal = None
        self.cal_angle = None; self.cal_mag = None

    def init_history(self):
        L = MAG_PLOT_LEN; eL = ERROR_PLOT_LEN
        self.angle_error_history = deque(maxlen=eL)
        self.pzt_fz_history = deque(maxlen=L)
        self.adc_dx_history = deque(maxlen=L); self.adc_dy_history = deque(maxlen=L)
        self.force_fz_history = deque(maxlen=L)
        self.force_fx_history = deque(maxlen=L); self.force_fy_history = deque(maxlen=L)
        self.adc_mag_history = deque(maxlen=L); self.raw_force_mag_history = deque(maxlen=L)
        self.force_fx_cal_history = deque(maxlen=L); self.force_fy_cal_history = deque(maxlen=L)

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
        for r, (pzt_n, frc_n) in enumerate([("PZT_Fz", "Force_Fz"), ("PZT_Fx", "Force_Fx"), ("PZT_Fy", "Force_Fy")]):
            p = self.win.addPlot(row=r, col=0, title=pzt_n)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis('left').setWidth(45); p.getAxis('bottom').setHeight(28)
            _style_plot(p, pzt_n)
            setattr(self, f"p_pzt_{['fz','fx','fy'][r]}", p)
            pc = 'r'
            c = p.plot(pen=pg.mkPen(pc, width=2))
            setattr(self, f"_c_pzt_{['fz','fx','fy'][r]}", c)
            t = pg.TextItem("", color=pc, anchor=(1, 1))
            p.addItem(t)
            setattr(self, f"_t_pzt_{['fz','fx','fy'][r]}", t)

            p2 = self.win.addPlot(row=r, col=1, title=frc_n)
            p2.showGrid(x=True, y=True, alpha=0.3)
            p2.getAxis('left').setWidth(45); p2.getAxis('bottom').setHeight(28)
            _style_plot(p2, frc_n)
            setattr(self, f"p_frc_{['fz','fx','fy'][r]}", p2)
            c2 = p2.plot(pen=pg.mkPen('b', width=2))
            setattr(self, f"_c_frc_{['fz','fx','fy'][r]}", c2)
            t2 = pg.TextItem("", color='b', anchor=(1, 1))
            p2.addItem(t2)
            setattr(self, f"_t_frc_{['fz','fx','fy'][r]}", t2)
            # 红色文字：Fz=PZT_Fz, Fx/Fy=Cal
            t2r = pg.TextItem("", color='r', anchor=(1, 0))
            p2.addItem(t2r)
            setattr(self, f"_t_frc_{['fz','fx','fy'][r]}_r", t2r)
            if r > 0:  # Fx/Fy have cal line
                c2c = p2.plot(pen=pg.mkPen('r', width=2, style=QtCore.Qt.DashLine))
                setattr(self, f"_c_frc_{['fx','fy'][r-1]}_cal", c2c)

        # Angle Error
        self.p_err = self.win.addPlot(row=3, col=0, colspan=2, title="Angle Error")
        self.p_err.showGrid(x=True, y=True, alpha=0.3)
        self.p_err.setYRange(0, 180)
        self.p_err.getAxis('left').setWidth(45); self.p_err.getAxis('bottom').setHeight(28)
        _style_plot(self.p_err, "Angle Error")
        self._c_err = self.p_err.plot(pen=pg.mkPen('g', width=2))
        self._t_err = pg.TextItem("", color='g', anchor=(1, 1))
        self.p_err.addItem(self._t_err)

        # --- 右列上方: Direction + Magnitude (col 2-3) ---
        self.p_dir = self.win.addPlot(row=0, col=2, title="Direction")
        self.p_dir.hideAxis('left'); self.p_dir.hideAxis('bottom')
        self.p_dir.setXRange(-1.2, 1.2); self.p_dir.setYRange(-1.2, 1.2); self.p_dir.setAspectLocked()
        self._dir_pzt = self._make_arrow_parts(self.p_dir)
        self._dir_frc = self._make_arrow_parts(self.p_dir)
        self._update_arrow(self._dir_pzt, 0, 0.45, 'r')
        self._update_arrow(self._dir_frc, 0, 0.40, 'b')

        self.p_mag = self.win.addPlot(row=0, col=3, title="Magnitude")
        self.p_mag.hideAxis('left'); self.p_mag.hideAxis('bottom')
        self.p_mag.setXRange(-0.8, 0.8); self.p_mag.setYRange(-0.8, 0.8); self.p_mag.setAspectLocked()
        self._mag_pzt = self._make_arrow_parts(self.p_mag)
        self._mag_frc = self._make_arrow_parts(self.p_mag)
        self._update_arrow(self._mag_pzt, 0, 0.10, 'r')
        self._update_arrow(self._mag_frc, 0, 0.10, 'b')

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
        # CoP 标记
        self._cop_dots = pg.ScatterPlotItem()
        self.p_table.addItem(self._cop_dots)
        self._cop_arr, self._cop_hL, self._cop_hR = self._make_arrow_parts(self.p_table)

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

    @staticmethod
    def _hot_color(t):
        """t: 0→1, 返回 (R,G,B) 白→红"""
        t = max(0, min(1, t))
        if t < 0.15:
            return (255, 255, int(255 * (1 - t / 0.15)))
        elif t < 0.35:
            s = (t - 0.15) / 0.2
            return (255, int(255 * (1 - s)), 0)
        elif t < 0.65:
            s = (t - 0.35) / 0.3
            return (255, int(55 * (1 - s)), 0)
        else:
            s = (t - 0.65) / 0.35
            return (int(255 * (1 - s * 0.7)), 0, 0)

    @staticmethod
    def _hot_lut():
        """白→黄→橙→红→深红 (0=白)"""
        lut = np.zeros((256, 4), dtype=np.uint8)
        for i in range(256):
            t = i / 255.0
            if t < 0.15:  # white→yellow
                lut[i, 0] = 255; lut[i, 1] = 255; lut[i, 2] = 255 - int(255 * t / 0.15)
            elif t < 0.35:  # yellow→orange
                s = (t - 0.15) / 0.2
                lut[i, 0] = 255; lut[i, 1] = 255 - int(200 * s); lut[i, 2] = 0
            elif t < 0.65:  # orange→red
                s = (t - 0.35) / 0.3
                lut[i, 0] = 255; lut[i, 1] = 55 - int(55 * s); lut[i, 2] = 0
            else:  # red→dark red
                s = (t - 0.65) / 0.35
                lut[i, 0] = 255 - int(180 * s); lut[i, 1] = 0; lut[i, 2] = 0
            lut[i, 3] = 255
        return lut

    # ===== 数据接口 =====
    def set_data(self, adc_angle, adc_mag, force_angle, force_mag, diff_frame, total_pressure_sum, force_total_mag,
                 cop_x, cop_y, base_cop_x, base_cop_y, delta_cop_x, delta_cop_y, raw_fx, raw_fy, raw_fz,
                 fx_cal=None, fy_cal=None, cal_angle=None, cal_mag=None):
        with self.lock:
            self.adc_angle = adc_angle; self.adc_mag = adc_mag
            self.force_angle = force_angle; self.force_mag = force_mag
            self.diff_frame = diff_frame.reshape(self.rows, self.cols)
            self.cop_x = cop_x; self.cop_y = cop_y
            self.base_cop_x = base_cop_x; self.base_cop_y = base_cop_y
            self.delta_cop_x = delta_cop_x; self.delta_cop_y = delta_cop_y
            self.raw_fx = raw_fx; self.raw_fy = raw_fy; self.raw_fz = raw_fz
            self.total_pressure = total_pressure_sum
            self.fx_cal = fx_cal; self.fy_cal = fy_cal
            self.cal_angle = cal_angle; self.cal_mag = cal_mag

            err = min(abs(adc_angle - force_angle), 360 - abs(adc_angle - force_angle))
            self.angle_error_history.append(err)
            self.adc_mag_history.append(adc_mag); self.raw_force_mag_history.append(force_total_mag)
            self.pzt_fz_history.append(total_pressure_sum)
            self.adc_dx_history.append(delta_cop_x); self.adc_dy_history.append(delta_cop_y)
            self.force_fz_history.append(raw_fz)
            self.force_fx_history.append(raw_fx); self.force_fy_history.append(raw_fy)
            if fx_cal is not None:
                self.force_fx_cal_history.append(fx_cal)
                self.force_fy_cal_history.append(fy_cal)

    def append_full_data(self, current_ms,
                          adc_angle, adc_mag, total_pressure, dx_f, dy_f,
                          force_angle, force_mag, fz, fx, fy,
                          cal_angle=None, cal_mag=None, fx_cal=None, fy_cal=None):
        with self.lock:
            self.full_time_list.append(current_ms)
            self.full_adc_angle_list.append(adc_angle); self.full_adc_mag_list.append(adc_mag)
            self.full_total_pressure_list.append(total_pressure)
            self.full_adc_dx_list.append(dx_f); self.full_adc_dy_list.append(dy_f)
            self.full_force_angle_list.append(force_angle); self.full_force_mag_list.append(force_mag)
            self.full_fz_list.append(fz); self.full_fx_list.append(fx); self.full_fy_list.append(fy)
            if cal_mag is not None:
                self.full_cal_angle_list.append(cal_angle); self.full_cal_mag_list.append(cal_mag)
                self.full_fx_cal_list.append(fx_cal if fx_cal is not None else float('nan'))
                self.full_fy_cal_list.append(fy_cal if fy_cal is not None else float('nan'))

    # ===== 更新 =====
    def update_all(self):
        t0 = time.perf_counter()
        with self.lock:
            aa, am = self.adc_angle, self.adc_mag
            fa, fm = self.force_angle, self.force_mag
            ca, cm = self.cal_angle, self.cal_mag
            pz_h = list(self.pzt_fz_history)
            dx_h, dy_h = list(self.adc_dx_history), list(self.adc_dy_history)
            fz_h = list(self.force_fz_history)
            fx_h, fy_h = list(self.force_fx_history), list(self.force_fy_history)
            fc_h = list(self.force_fx_cal_history)
            fcy_h = list(self.force_fy_cal_history)
            err_h = list(self.angle_error_history)
            table = self.diff_frame.copy()
            cx_p, cy_p = self.cop_x, self.cop_y
            bx_p, by_p = self.base_cop_x, self.base_cop_y
            ddx, ddy = self.delta_cop_x, self.delta_cop_y
            fx_c, fy_c = self.fx_cal, self.fy_cal
            with COP.grad_table_lock:
                grad = COP.grad_table_data.copy()

        # Direction: PZT=red + Force=blue, tail at origin
        self._update_arrow(self._dir_pzt, aa, 0.45, 'r')
        self._update_arrow(self._dir_frc, fa, 0.40, 'b')

        # Magnitude: proportional length（最小 0.01，保持初始可见）
        la = max(min((am / 5.0) * 0.65, 0.65), 0.01)
        self._update_arrow(self._mag_pzt, aa, la, 'r')
        lf = max(min((abs(fm) / 20.0) * 0.65, 0.65), 0.01)
        self._update_arrow(self._mag_frc, fa, lf, 'b')

        # Time-series
        self._u1(self._c_pzt_fz, self.p_pzt_fz, pz_h, self._t_pzt_fz, "PZT_Fz")
        self._u1(self._c_pzt_fx, self.p_pzt_fx, dx_h, self._t_pzt_fx, "PZT_Fx")
        self._u1(self._c_pzt_fy, self.p_pzt_fy, dy_h, self._t_pzt_fy, "PZT_Fy")
        self._u1(self._c_frc_fz, self.p_frc_fz, fz_h, self._t_frc_fz, "Fz",
                 pzt_val=pz_h[-1] if pz_h else 0, pzt_label="PZT_Fz", txt_r=self._t_frc_fz_r)
        self._u2(self._c_frc_fx, self._c_frc_fx_cal, self.p_frc_fx, fx_h, fc_h, self._t_frc_fx, "Fx",
                 txt_r=self._t_frc_fx_r)
        self._u2(self._c_frc_fy, self._c_frc_fy_cal, self.p_frc_fy, fy_h, fcy_h, self._t_frc_fy, "Fy",
                 txt_r=self._t_frc_fy_r)
        if err_h:
            xs = list(range(len(err_h)))
            self._c_err.setData(xs, err_h)
            self.p_err.setXRange(0, max(len(xs) - 1, 1))
            self._t_err.setText(f'{err_h[-1]:.1f}°')
            hi = self.p_err.viewRange()[1][1]
            self._t_err.setPos(max(len(xs) - 1, 1), hi)

        # Pressure table: 同 matplotlib table，vmax = max(每帧最大值, 下限)
        vmax = max(np.max(table), self._heat_vmax)
        self._cell_grid.set_data(table, vmax)
        for r in range(12):
            for c in range(7):
                v = table[r, c]
                self._cell_txts[r][c].setText(f"{v:.0f}" if v > 0 else "")
        # CoP dots + arrow
        spots = [{'pos': (cx_p, cy_p), 'brush': 'g', 'size': 12}]
        if not np.isnan(bx_p) and not np.isnan(by_p):
            spots.append({'pos': (bx_p, by_p), 'brush': 'b', 'symbol': 'x', 'size': 15})
        self._cop_dots.setData(spots=spots)
        if not np.isnan(bx_p) and not np.isnan(by_p) and np.hypot(ddx, ddy) > 0.05:
            self._update_arrow((self._cop_arr, self._cop_hL, self._cop_hR),
                               np.degrees(np.arctan2(-ddy, ddx)) if abs(ddx) + abs(ddy) > 1e-6 else 0,
                               np.hypot(ddx, ddy), 'r', (bx_p, by_p))
        else:
            self._cop_arr.setData([], [])
            self._cop_hL.setData([], [])
            self._cop_hR.setData([], [])

        # Gradient arrows
        for i, (ln, dot) in enumerate(zip(self._g_lines, self._g_heads)):
            r, c = divmod(i, 7)
            gx, gy = grad[r, c, 0], grad[r, c, 1]
            m = np.hypot(gx, gy)
            if m > 1.0:
                dx = -gx / m * 0.3; dy = gy / m * 0.3  # 反方向
                tip_x = c + dx; tip_y = r + dy
                ln.setData([c, tip_x], [r, tip_y])
                dot.setData(x=[tip_x], y=[tip_y], brush='k', size=4)
            else:
                ln.setData([], [])
                dot.setData(x=[], y=[])

        # FPS
    def _u1(self, curve, plot, data, txt, label, pzt_val=None, pzt_label=None, txt_r=None):
        if data:
            xs = list(range(len(data)))
            curve.setData(xs, data)
            plot.setXRange(0, max(len(xs) - 1, 1))
            lo, hi = _yrange(data)
            plot.setYRange(lo, hi)
            if txt_r and pzt_val is not None:
                txt.setText(f'True_{label}={data[-1]:.2f}')
                txt.setPos(max(len(xs) - 1, 1), hi - (hi - lo) * 0.05)
                txt_r.setText(f'{pzt_label}={pzt_val:.2f}')
                txt_r.setPos(max(len(xs) - 1, 1), hi - (hi - lo) * 0.15)
            else:
                txt.setText(f'{label}={data[-1]:.2f}')
                txt.setPos(max(len(xs) - 1, 1), hi - (hi - lo) * 0.12)

    def _u2(self, c1, c2, plot, d1, d2, txt, label, txt_r=None):
        if d1:
            xs = list(range(len(d1)))
            c1.setData(xs, d1)
            all_y = list(d1)
            if len(d2) == len(d1):
                c2.setData(xs, d2); all_y.extend(d2)
            plot.setXRange(0, max(len(xs) - 1, 1))
            lo, hi = _yrange(all_y); plot.setYRange(lo, hi)
            val = d2[-1] if len(d2) == len(d1) else 0
            txt.setText(f'True_{label}={d1[-1]:.2f}')
            txt.setPos(max(len(xs) - 1, 1), hi - (hi - lo) * 0.05)
            if txt_r:
                txt_r.setText(f'Cal_{label}={val:.2f}')
                txt_r.setPos(max(len(xs) - 1, 1), hi - (hi - lo) * 0.15)

    # ===== 全程静态图 (matplotlib Agg) =====
    def plot_full_magnitude_curve(self, save_dir):
        import os; import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        if len(self.full_time_list) == 0: print("⚠️ 无数据"); return
        has_cal = len(self.full_cal_mag_list) == len(self.full_time_list)
        t = self.full_time_list
        fig, axes = plt.subplots(5, 2, figsize=(18, 24))
        (aL1, aR1), (aL2, aR2), (aL3, aR3), (aL4, aR4), (aL5, aR5) = axes
        def _p(ax, d, c, lbl):
            if d and len(d) == len(t): ax.plot(t, d, c, linewidth=1.0, label=lbl)
        _p(aL1, self.full_adc_angle_list, 'b-', 'PZT Angle'); aL1.set_title("PZT Angle"); aL1.grid(True, alpha=0.3)
        _p(aL2, self.full_adc_mag_list, 'b-', 'PZT Mag'); aL2.set_title("PZT Mag"); aL2.grid(True, alpha=0.3)
        _p(aL3, self.full_total_pressure_list, 'b-', 'PZT Fz'); aL3.set_title("PZT Fz"); aL3.grid(True, alpha=0.3)
        _p(aL4, self.full_adc_dx_list, 'b-', 'PZT Fx'); aL4.set_title("PZT Fx"); aL4.grid(True, alpha=0.3)
        _p(aL5, self.full_adc_dy_list, 'c-', 'PZT Fy'); aL5.set_title("PZT Fy"); aL5.grid(True, alpha=0.3)
        _p(aR1, self.full_force_angle_list, 'r-', 'Measured')
        if has_cal: _p(aR1, self.full_cal_angle_list, 'g--', 'Calibrated')
        aR1.set_title("Angle: Meas vs Cal"); aR1.grid(True, alpha=0.3)
        if has_cal: aR1.legend(fontsize=8)
        _p(aR2, self.full_force_mag_list, 'r-', 'Measured')
        if has_cal: _p(aR2, self.full_cal_mag_list, 'g--', 'Calibrated')
        aR2.set_title("Mag: Meas vs Cal"); aR2.grid(True, alpha=0.3)
        if has_cal: aR2.legend(fontsize=8)
        _p(aR3, self.full_fz_list, 'r-', 'Fz'); aR3.set_title("Fz: Measured"); aR3.grid(True, alpha=0.3)
        _p(aR4, self.full_fx_list, 'r-', 'Measured')
        if has_cal: _p(aR4, self.full_fx_cal_list, 'g--', 'Calibrated')
        aR4.set_title("Fx: Meas vs Cal"); aR4.grid(True, alpha=0.3)
        if has_cal: aR4.legend(fontsize=8)
        _p(aR5, self.full_fy_list, 'm-', 'Measured')
        if has_cal: _p(aR5, self.full_fy_cal_list, 'c--', 'Calibrated')
        aR5.set_title("Fy: Meas vs Cal"); aR5.grid(True, alpha=0.3)
        if has_cal: aR5.legend(fontsize=8)
        for row in axes:
            for ax in row: ax.set_xlabel("Time (ms)", fontsize=9)
        plt.tight_layout()
        idx = 1
        while os.path.exists(os.path.join(save_dir, f"full_analysis_cop_{idx}.png")): idx += 1
        sp = os.path.join(save_dir, f"full_analysis_cop_{idx}.png")
        plt.savefig(sp, dpi=300); print(f"📊 已保存：{sp}"); plt.close(fig)
