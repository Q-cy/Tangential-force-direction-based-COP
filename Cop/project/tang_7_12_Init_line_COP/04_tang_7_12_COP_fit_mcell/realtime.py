"""pyqtgraph 实时绘图 — GPU 渲染, 100fps"""
import numpy as np
from collections import deque
import threading
import time
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

pg.setConfigOptions(antialias=True, background='w', foreground='k')

PLOT_TIMER_INTERVAL_MS = 10    # 绘图定时器刷新间隔(毫秒)
PLOT_ERR_HISTORY_LEN = 100     # 角度误差历史缓冲区长度
PLOT_HISTORY_LEN = 100         # 时序曲线历史缓冲区长度

MAX_REGION_ARROWS = 8          # per-region 角度箭头最大套数（与 REGION_PALETTE 等长）

REGION_PALETTE = [(0, 102, 255), (0, 204, 51), (255, 128, 0), (153, 0, 255),
                  (0, 204, 204), (255, 204, 0), (255, 0, 153), (255, 61, 61)]

def _yrange(data, pad=0.1):
    clean = [v for v in data if v == v]  # filter NaN
    if len(clean) < 2: return -1, 1
    mn, mx = min(clean), max(clean)
    r = mx - mn if mx != mn else 1
    return mn - r * pad, mx + r * pad


class CellGridItem(pg.GraphicsObject):
    """84 个独立色块 + 数值文字，复现 matplotlib table 效果"""
    def __init__(self, rows=12, cols=7):
        pg.GraphicsObject.__init__(self)
        self.rows, self.cols = rows, cols
        self.data = np.zeros((rows, cols))
        self.vmax = 1.0
        self.region_mask = np.zeros((rows, cols), dtype=np.int32)
        self.region_palette = []
        self.region_frames = {}

    def set_data(self, data, vmax):
        self.data = data
        self.vmax = max(vmax, 1)
        self.update()

    def set_regions(self, region_mask, palette):
        self.region_mask = region_mask
        self.region_palette = palette
        # 外框线段: region 与背景 0 / 阵列边缘 / 其他 region 相邻的边, 每个 region 一条闭合轮廓;
        # 公共边双方都描, paint 后画覆盖 → 接缝显示为后出现 region 的颜色
        h, w = region_mask.shape
        padded = np.pad(region_mask, 1, constant_values=0)
        frames = {}
        for r in range(h):
            for c in range(w):
                k = region_mask[r, c]
                if k <= 0:
                    continue
                if padded[r, c + 1] != k:      # 上
                    frames.setdefault(k, []).append(((c - 0.5, r - 0.5), (c + 0.5, r - 0.5)))
                if padded[r + 2, c + 1] != k:  # 下
                    frames.setdefault(k, []).append(((c - 0.5, r + 0.5), (c + 0.5, r + 0.5)))
                if padded[r + 1, c] != k:      # 左
                    frames.setdefault(k, []).append(((c - 0.5, r - 0.5), (c - 0.5, r + 0.5)))
                if padded[r + 1, c + 2] != k:  # 右
                    frames.setdefault(k, []).append(((c + 0.5, r - 0.5), (c + 0.5, r + 0.5)))
        self.region_frames = frames
        self.update()

    def paint(self, p, opt, widget):
        p.setRenderHint(p.RenderHint.Antialiasing, False)
        w = self.cols
        h = self.rows
        # 画色块（数据 Y = r，与视图一致）
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
        # region 外框线: 后画, 2px 彩色覆盖 1px 灰网格线
        pen = QtGui.QPen()
        pen.setCosmetic(True)
        pen.setWidth(4)
        for rid, seg_list in self.region_frames.items():
            pr, pgc, pb = self.region_palette[(rid - 1) % len(self.region_palette)]
            pen.setColor(QtGui.QColor(pr, pgc, pb))
            p.setPen(pen)
            for (x1, y1), (x2, y2) in seg_list:
                p.drawLine(QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2))

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
        self._heat_vmax = 500.0   # 热力图色阶下限

        # === 全程存储 ===
        self.full_time_list = []
        self.full_adc_angle_list = []
        self.full_total_pressure_list = []
        self.full_adc_dx_list, self.full_adc_dy_list = [], []
        self.full_force_angle_list = []
        self.full_fz_list, self.full_fx_list, self.full_fy_list = [], [], []
        self.full_cal_angle_list = []
        self.full_fx_cal_list, self.full_fy_cal_list = [], []
        self.full_fz_cal_list = []

        self.init_defaults()
        self.init_history()
        self.build_layout()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_all)
        self.timer.start(PLOT_TIMER_INTERVAL_MS)

    def init_defaults(self):
        self._pzt_angle_deg = 0.0           # PZT方向角度(度)
        self._force_angle_deg = 0.0         # 六维力方向角度(度)
        self._press_table_arr = np.zeros((12, 7))  # 压力表数据(12×7)
        self._cop_curr_x = 0.0              # 当前CoP X
        self._cop_curr_y = 0.0              # 当前CoP Y
        self._cop_base_x = 0.0              # 初始CoP X
        self._cop_base_y = 0.0              # 初始CoP Y
        self._cop_delta_x = 0.0             # CoP偏移X
        self._cop_delta_y = 0.0             # CoP偏移Y
        self._force_fx_val = 0.0            # 力传感器Fx
        self._force_fy_val = 0.0            # 力传感器Fy
        self._cal_fx_val = None             # 标定力Fx
        self._cal_fy_val = None             # 标定力Fy
        self._cop_state = 0                 # CoP状态(0=未接触,1=粗略,2=精细)
        self._gradient_arr = np.zeros((12, 7, 2), dtype=np.float32)  # 压力梯度(每帧由 main 传入)
        self._contact_init = False
        self._pzt_table_angle_deg = None     # Pressure Table 专用角度（invertY 视图）
        self._region_mask = np.zeros((12, 7), dtype=np.int32)   # per-region 着色掩码
        self._regions = []                   # per-region 数据（cop/delta/id, 供点+箭头显示）
        self._centroid_xy = None             # 整帧形心（不加权, 品红菱形显示）

    def init_history(self):
        hist_len = PLOT_HISTORY_LEN
        err_len = PLOT_ERR_HISTORY_LEN
        self.angle_error_history = deque(maxlen=err_len)
        self.pzt_fz_history = deque(maxlen=hist_len)
        self.adc_dx_history = deque(maxlen=hist_len)
        self.adc_dy_history = deque(maxlen=hist_len)
        self.force_fz_history = deque(maxlen=hist_len)
        self.force_fx_history = deque(maxlen=hist_len)
        self.force_fy_history = deque(maxlen=hist_len)
        self.force_fx_cal_history = deque(maxlen=hist_len)
        self.force_fz_cal_history = deque(maxlen=hist_len)
        self.force_fy_cal_history = deque(maxlen=hist_len)

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
        for r, (pzt_n, frc_n) in enumerate([("PZT_z", "Force_Fz"), ("PZT_x", "Force_Fx"), ("PZT_y", "Force_Fy")]):
            p = self.win.addPlot(row=r, col=0, title=pzt_n)
            p.showGrid(x=True, y=True, alpha=0.3)
            p.getAxis('bottom').setHeight(28)
            _style_plot(p, pzt_n)
            setattr(self, f"p_pzt_{['fz','fx','fy'][r]}", p)
            pc = 'r'
            c = p.plot(pen=pg.mkPen(pc, width=3))
            setattr(self, f"_c_pzt_{['fz','fx','fy'][r]}", c)
            t = pg.TextItem("", anchor=(1, 1))
            p.addItem(t)
            setattr(self, f"_t_pzt_{['fz','fx','fy'][r]}", t)

            p2 = self.win.addPlot(row=r, col=1, title=frc_n)
            p2.showGrid(x=True, y=True, alpha=0.3)
            p2.getAxis('bottom').setHeight(28)
            _style_plot(p2, frc_n)
            setattr(self, f"p_frc_{['fz','fx','fy'][r]}", p2)
            c2 = p2.plot(pen=pg.mkPen('b', width=3))
            setattr(self, f"_c_frc_{['fz','fx','fy'][r]}", c2)
            t2 = pg.TextItem("", anchor=(1, 1))
            p2.addItem(t2)
            setattr(self, f"_t_frc_{['fz','fx','fy'][r]}", t2)
            # 红色文字：Fz=PZT_Fz, Fx/Fy=Cal
            t2r = pg.TextItem("", anchor=(1, 1))
            p2.addItem(t2r)
            setattr(self, f"_t_frc_{['fz','fx','fy'][r]}_r", t2r)
            # All rows have cal line
            c2c = p2.plot(pen=pg.mkPen('r', width=3, style=QtCore.Qt.DashLine))
            setattr(self, f"_c_frc_{['fz','fx','fy'][r]}_cal", c2c)

        # Angle Error
        self.p_err = self.win.addPlot(row=3, col=0, colspan=2, title="Angle Error")
        self.p_err.showGrid(x=True, y=True, alpha=0.3)
        self.p_err.setYRange(0, 180)
        self.p_err.getAxis('bottom').setHeight(28)
        _style_plot(self.p_err, "Angle Error")
        self._c_err = self.p_err.plot(pen=pg.mkPen('g', width=3))
        self._t_err = pg.TextItem("", anchor=(0, 1))
        self.p_err.addItem(self._t_err)

        # --- 右列上方: Direction + Magnitude (col 2-3) ---
        self.p_dir = self.win.addPlot(row=0, col=2, title="Direction")
        self.p_dir.hideAxis('left'); self.p_dir.hideAxis('bottom')
        self.p_dir.setXRange(-1.2, 1.2); self.p_dir.setYRange(-1.2, 1.2); self.p_dir.setAspectLocked()
        self._dir_pzt = self._make_arrow_parts(self.p_dir)
        self._dir_frc = self._make_arrow_parts(self.p_dir)
        self._update_arrow(self._dir_pzt, 0, 0.45, 'r')
        self._update_arrow(self._dir_frc, 0, 0.40, 'b')
        self._dir_txt_pzt = pg.TextItem("", anchor=(0, 1))
        self.p_dir.addItem(self._dir_txt_pzt)
        self._dir_txt_frc = pg.TextItem("", anchor=(0, 1))
        self.p_dir.addItem(self._dir_txt_frc)

        self.p_mag = self.win.addPlot(row=0, col=3, title="Pressure Snapshot")
        self.p_mag.hideAxis('left'); self.p_mag.hideAxis('bottom')
        self.p_mag.setXRange(-0.8, 0.8); self.p_mag.setYRange(-0.8, 0.8); self.p_mag.setAspectLocked()
        self._mag_pzt = self._make_arrow_parts(self.p_mag)
        self._mag_frc = self._make_arrow_parts(self.p_mag)
        self._update_arrow(self._mag_pzt, 0, 0.10, 'r')
        self._update_arrow(self._mag_frc, 0, 0.10, 'b')
        self._mag_txt_pzt = pg.TextItem("", anchor=(0, 1))
        self.p_mag.addItem(self._mag_txt_pzt)
        self._mag_txt_frc = pg.TextItem("", anchor=(0, 1))
        self.p_mag.addItem(self._mag_txt_frc)

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
        self._centroid_dot = pg.ScatterPlotItem()   # 整帧形心（不加权, 品红菱形）
        self.p_table.addItem(self._centroid_dot)
        self._cop_arr, self._cop_hL, self._cop_hR = self._make_arrow_parts(self.p_table)
        # per-region CoP 点 + 角度箭头（region 外框色, 与整帧同款显示）
        self._region_cop_dots = pg.ScatterPlotItem()
        self.p_table.addItem(self._region_cop_dots)
        self._region_base_dots = pg.ScatterPlotItem()
        self.p_table.addItem(self._region_base_dots)
        self._region_arrows = [self._make_arrow_parts(self.p_table) for _ in range(MAX_REGION_ARROWS)]

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
        self._g_txts = []
        for _ in range(84):
            t = pg.TextItem("", color='k', anchor=(0.5, 0.5))
            self.p_grad.addItem(t)
            self._g_txts.append(t)
        self._grad_cop_dots = pg.ScatterPlotItem()
        self.p_grad.addItem(self._grad_cop_dots)


        # 强制左两列等宽
        half_w = self.win.width() // 2 // 2
        for r in range(3):
            getattr(self, f"p_pzt_{['fz','fx','fy'][r]}").setPreferredWidth(half_w)
            getattr(self, f"p_frc_{['fz','fx','fy'][r]}").setPreferredWidth(half_w)
        self.p_err.setPreferredWidth(self.win.width() // 2)
        self.win.show()

    # ===== 数据接口 =====
    def set_data(self, pzt_angle_deg, force_angle_deg,
                 press_table_arr, total_press_val,
                 cop_curr_x, cop_curr_y, cop_base_x, cop_base_y, cop_delta_x, cop_delta_y,
                 force_fx_val, force_fy_val, force_fz_val,
                 cal_fx_val=None, cal_fy_val=None, cal_fz_val=None,
                 cop_state=0,
                 gradient=None,
                 contact_init=False,
                 refined=False,
                 pzt_table_angle_deg=None,
                 region_mask=None,
                 regions=None,
                 centroid=None):
        with self.lock:
            self._pzt_angle_deg = pzt_angle_deg
            self._force_angle_deg = force_angle_deg
            self._press_table_arr = press_table_arr.reshape(self.rows, self.cols)
            self._cop_curr_x = cop_curr_x
            self._cop_curr_y = cop_curr_y
            self._cop_base_x = cop_curr_x if cop_base_x is None else cop_base_x
            self._cop_base_y = cop_curr_y if cop_base_y is None else cop_base_y
            self._cop_delta_x = cop_delta_x
            self._cop_delta_y = cop_delta_y
            self._force_fx_val = force_fx_val
            self._force_fy_val = force_fy_val
            self._cal_fx_val = cal_fx_val
            self._cal_fy_val = cal_fy_val
            self._cop_state = cop_state
            self._contact_init = contact_init
            self._pzt_table_angle_deg = pzt_table_angle_deg
            if gradient is not None:
                self._gradient_arr = np.asarray(gradient, dtype=np.float32)
            self._region_mask = (np.zeros((self.rows, self.cols), dtype=np.int32)
                                 if region_mask is None
                                 else np.asarray(region_mask, dtype=np.int32))
            self._regions = regions or []
            self._centroid_xy = centroid

            angle_err = min(abs(pzt_angle_deg - force_angle_deg),
                           360 - abs(pzt_angle_deg - force_angle_deg))
            self.angle_error_history.append(angle_err)
            self.pzt_fz_history.append(total_press_val)
            self.adc_dx_history.append(cop_delta_x)
            self.adc_dy_history.append(cop_delta_y)
            self.force_fz_history.append(force_fz_val)
            self.force_fx_history.append(force_fx_val)
            self.force_fy_history.append(force_fy_val)
            if cal_fx_val is not None:
                self.force_fx_cal_history.append(cal_fx_val)
                self.force_fy_cal_history.append(cal_fy_val)
            if cal_fz_val is not None:
                self.force_fz_cal_history.append(cal_fz_val)

    def append_full_data(self, rel_time_ms,
                          pzt_angle_deg, total_press_val,
                          cop_delta_x_filt, cop_delta_y_filt,
                          force_angle_deg,
                          force_fz_filt, force_fx_filt, force_fy_filt,
                          cal_angle_deg=None, cal_fx_val=None, cal_fy_val=None, cal_fz_val=None):
        with self.lock:
            self.full_time_list.append(rel_time_ms)
            self.full_adc_angle_list.append(pzt_angle_deg)
            self.full_total_pressure_list.append(total_press_val)
            self.full_adc_dx_list.append(cop_delta_x_filt)
            self.full_adc_dy_list.append(cop_delta_y_filt)
            self.full_force_angle_list.append(force_angle_deg)
            self.full_fz_list.append(force_fz_filt)
            self.full_fx_list.append(force_fx_filt)
            self.full_fy_list.append(force_fy_filt)
            if cal_fx_val is not None:
                self.full_cal_angle_list.append(cal_angle_deg)
                self.full_fx_cal_list.append(cal_fx_val)
                self.full_fy_cal_list.append(cal_fy_val)
            if cal_fz_val is not None:
                self.full_fz_cal_list.append(cal_fz_val)

    # ===== 更新 =====
    def update_all(self):
        t0 = time.perf_counter()
        with self.lock:
            pzt_angle_deg = self._pzt_angle_deg
            force_angle_deg = self._force_angle_deg
            pzt_fz_hist = list(self.pzt_fz_history)
            cop_dx_hist = list(self.adc_dx_history); cop_dy_hist = list(self.adc_dy_history)
            force_fz_hist = list(self.force_fz_history)
            force_fx_hist = list(self.force_fx_history); force_fy_hist = list(self.force_fy_history)
            cal_fx_hist = list(self.force_fx_cal_history)
            cal_fy_hist = list(self.force_fy_cal_history)
            cal_fz_hist = list(self.force_fz_cal_history)
            err_hist = list(self.angle_error_history)
            press_table_arr = self._press_table_arr.copy()
            cop_curr_x = self._cop_curr_x; cop_curr_y = self._cop_curr_y
            cop_base_x = self._cop_base_x; cop_base_y = self._cop_base_y
            cop_delta_x = self._cop_delta_x; cop_delta_y = self._cop_delta_y
            cal_fx_val = self._cal_fx_val; cal_fy_val = self._cal_fy_val
            force_fx_val = self._force_fx_val; force_fy_val = self._force_fy_val
            pzt_table_angle_deg = self._pzt_table_angle_deg
            cop_state = self._cop_state
            contact_init = self._contact_init
            grad_arr = self._gradient_arr.copy()
            regions = list(self._regions)

        # 状态显示
        _state_names = {0: "未接触", 1: "粗略测量", 2: "精细测量"}
        self.win.setWindowTitle(f"RealTime — {_state_names.get(cop_state, '?')}")

        # 初始 CoP 未确定时冻结蓝色箭头（与红色一致）
        _fa = force_angle_deg if contact_init else 0.0

        # Direction: PZT=red + Force=blue
        fs = self._font_size(12)
        self._update_arrow(self._dir_pzt, pzt_angle_deg, 0.45, 'r')
        self._update_arrow(self._dir_frc, _fa, 0.40, 'b')
        self._dir_txt_pzt.setHtml(self._html(f'PZT_Angle: {pzt_angle_deg:.1f}°', 'red', fs))
        self._dir_txt_pzt.setPos(0.75, 1.15)
        self._dir_txt_frc.setHtml(self._html(f'Force_Angle: {_fa:.1f}°', 'blue', fs))
        self._dir_txt_frc.setPos(0.75, 0.95)

        # Pressure Table + Gradient 占据原 Magnitude 区域
        # 此处只保留箭头几何，不持久化 magnitude 数据
        pzt_arrow_len = min(max(np.hypot(cop_delta_x, cop_delta_y) * 0.5, 0.0), 0.65)
        self._update_arrow(self._mag_pzt, pzt_angle_deg, pzt_arrow_len, 'r')
        _force_fx = 0.0 if (force_fx_val is None or np.isnan(force_fx_val)) else force_fx_val
        _force_fy = 0.0 if (force_fy_val is None or np.isnan(force_fy_val)) else force_fy_val
        force_arrow_len = min(max(np.hypot(_force_fx, _force_fy) * 0.05, 0.0), 0.65)
        self._update_arrow(self._mag_frc, _fa, force_arrow_len, 'b')
        self._mag_txt_pzt.setHtml(self._html(f'PZT_Angle: {pzt_angle_deg:.1f}°', 'red', fs))
        self._mag_txt_pzt.setPos(0.35, 0.75)
        self._mag_txt_frc.setHtml(self._html(f'Force_Angle: {_fa:.1f}°', 'blue', fs))
        self._mag_txt_frc.setPos(0.35, 0.62)

        # Time-series
        self._u1(self._c_pzt_fz, self.p_pzt_fz, pzt_fz_hist, self._t_pzt_fz, "PZT_z", fs=fs)
        self._u1(self._c_pzt_fx, self.p_pzt_fx, cop_dx_hist, self._t_pzt_fx, "PZT_x", fs=fs)
        self._u1(self._c_pzt_fy, self.p_pzt_fy, cop_dy_hist, self._t_pzt_fy, "PZT_y", fs=fs)
        self._u2(self._c_frc_fz, self._c_frc_fz_cal, self.p_frc_fz, force_fz_hist, cal_fz_hist,
                 self._t_frc_fz, "Fz", txt_r=self._t_frc_fz_r, fs=fs)
        self._u2(self._c_frc_fx, self._c_frc_fx_cal, self.p_frc_fx, force_fx_hist, cal_fx_hist,
                 self._t_frc_fx, "Fx", txt_r=self._t_frc_fx_r, fs=fs)
        self._u2(self._c_frc_fy, self._c_frc_fy_cal, self.p_frc_fy, force_fy_hist, cal_fy_hist,
                 self._t_frc_fy, "Fy", txt_r=self._t_frc_fy_r, fs=fs)
        if err_hist:
            x_vals = list(range(len(err_hist)))
            self._c_err.setData(x_vals, err_hist)
            self.p_err.setXRange(0, max(len(x_vals) - 1, 1))
            self._t_err.setHtml(self._html(f'Error: {err_hist[-1]:.1f}°', 'green', fs))
            self._t_err.setPos(int(max(len(x_vals) - 1, 1) * 0.85), 180 - 180 * 0.12)

        # Pressure table + CoP + Gradient：仅在初始 CoP 确定后显示
        if contact_init:
            cell_vmax = max(np.max(press_table_arr), self._heat_vmax)
            self._cell_grid.set_data(press_table_arr, cell_vmax)
            self._cell_grid.set_regions(self._region_mask, REGION_PALETTE)
            for row_idx in range(12):
                for col_idx in range(7):
                    cell_val = press_table_arr[row_idx, col_idx]
                    self._cell_txts[row_idx][col_idx].setText(f"{cell_val:.0f}" if cell_val > 0 else "")
            # CoP dots + arrow (region 模式下 cop_curr 为 NaN, 只显示 region 自己的点)
            if not np.isnan(cop_curr_x) and not np.isnan(cop_curr_y):
                spots = [{'pos': (cop_curr_x, cop_curr_y), 'brush': 'g', 'size': 12}]
                if not np.isnan(cop_base_x) and not np.isnan(cop_base_y):
                    spots.append({'pos': (cop_base_x, cop_base_y), 'brush': 'b', 'symbol': 'x', 'size': 15})
            else:
                spots = []
            self._cop_dots.setData(spots=spots)
            # 整帧形心（不加权, 与压力无关）
            if self._centroid_xy is not None:
                self._centroid_dot.setData(spots=[{'pos': self._centroid_xy, 'brush': 'm', 'symbol': 'd', 'size': 14}])
            else:
                self._centroid_dot.setData(spots=[])
            if not np.isnan(cop_base_x) and not np.isnan(cop_base_y) and np.hypot(cop_delta_x, cop_delta_y) > 0.05:
                table_angle = pzt_table_angle_deg if pzt_table_angle_deg is not None else pzt_angle_deg
                self._update_arrow((self._cop_arr, self._cop_hL, self._cop_hR),
                                   table_angle,
                                   np.hypot(cop_delta_x, cop_delta_y), 'r', (cop_base_x, cop_base_y))
            else:
                self._cop_arr.setData([], [])
                self._cop_hL.setData([], [])
                self._cop_hR.setData([], [])

            # per-region CoP 点 + 角度箭头（region 外框色, 与整帧同款; 外框线由 CellGridItem 保留）
            cop_spots, base_spots = [], []
            for i, reg in enumerate(regions):
                cx, cy = reg['cop']
                dx, dy = reg['delta']
                bx, by = cx - dx, cy - dy
                cop_spots.append({'pos': (cx, cy), 'brush': 'g', 'size': 12})
                base_spots.append({'pos': (bx, by), 'brush': 'b', 'symbol': 'x', 'size': 15})
                if i < MAX_REGION_ARROWS:
                    if np.hypot(dx, dy) > 0.05:
                        angle = np.degrees(np.arctan2(dy, dx)) % 360.0
                        pr, pgc, pb = REGION_PALETTE[(reg['id'] - 1) % len(REGION_PALETTE)]
                        self._update_arrow(self._region_arrows[i], angle, np.hypot(dx, dy),
                                           (pr, pgc, pb), (bx, by))
                    else:
                        for part in self._region_arrows[i]:
                            part.setData([], [])
            self._region_cop_dots.setData(spots=cop_spots)
            self._region_base_dots.setData(spots=base_spots)

            # Gradient arrows
            if not np.isnan(cop_curr_x) and not np.isnan(cop_curr_y):
                grad_spots = [{'pos': (cop_curr_x, cop_curr_y), 'brush': 'g', 'size': 12}]
                if not np.isnan(cop_base_x) and not np.isnan(cop_base_y):
                    grad_spots.append({'pos': (cop_base_x, cop_base_y), 'brush': 'b', 'symbol': 'x', 'size': 15})
            else:
                grad_spots = []
            self._grad_cop_dots.setData(spots=grad_spots)
            for grad_idx, (grad_ln, grad_dot) in enumerate(zip(self._g_lines, self._g_heads)):
                grad_row, grad_col = divmod(grad_idx, 7)
                grad_x, grad_y = grad_arr[grad_row, grad_col, 0], grad_arr[grad_row, grad_col, 1]
                grad_norm = np.hypot(grad_x, grad_y)
                if grad_norm > 1.0:
                    arrow_dx = -grad_x / grad_norm * 0.3
                    arrow_dy = grad_y / grad_norm * 0.3
                    tip_x = grad_col + arrow_dx
                    tip_y = grad_row + arrow_dy
                    grad_ln.setData([grad_col, tip_x], [grad_row, tip_y])
                    grad_dot.setData(x=[tip_x], y=[tip_y], brush='k', size=4)
                    self._g_txts[grad_idx].setText(f"{grad_norm:.0f}")
                    self._g_txts[grad_idx].setPos(grad_col, grad_row)
                else:
                    grad_ln.setData([], [])
                    grad_dot.setData(x=[], y=[])
                    self._g_txts[grad_idx].setText("")
        else:
            # CoP 未确定：清空两张表
            self._cell_grid.set_data(np.zeros((12, 7)), 1.0)
            self._cell_grid.set_regions(np.zeros((12, 7), dtype=np.int32), [])
            for row_idx in range(12):
                for col_idx in range(7):
                    self._cell_txts[row_idx][col_idx].setText("")
            self._cop_dots.setData(spots=[])
            self._centroid_dot.setData(spots=[])
            self._cop_arr.setData([], [])
            self._cop_hL.setData([], [])
            self._cop_hR.setData([], [])
            self._region_cop_dots.setData(spots=[])
            self._region_base_dots.setData(spots=[])
            for part_set in self._region_arrows:
                for part in part_set:
                    part.setData([], [])
            for grad_ln, grad_dot in zip(self._g_lines, self._g_heads):
                grad_ln.setData([], [])
                grad_dot.setData(x=[], y=[])
            for t in self._g_txts:
                t.setText("")
            self._grad_cop_dots.setData(spots=[])

    @staticmethod
    def _html(text, color, size=16):
        return f'<span style="color:{color};font-size:{size}pt;font-weight:bold">{text}</span>'

    def _font_size(self, base=16):
        """根据窗口宽度动态计算字号"""
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

    def _u2(self, c1, c2, plot, d1, d2, txt, label, color='blue', txt_r=None, fs=16):
        if d1:
            xs = list(range(len(d1)))
            c1.setData(xs, d1)
            all_y = list(d1)
            if len(d2) == len(d1):
                c2.setData(xs, d2); all_y.extend(d2)
            plot.setXRange(0, max(len(xs) - 1, 1))
            lo, hi = _yrange(all_y); plot.setYRange(lo, hi, padding=0)
            span = hi - lo if hi != lo else 1
            val = d2[-1] if len(d2) == len(d1) else 0
            txt.setHtml(self._html(f'True_{label}={d1[-1]:.2f}', color, fs))
            txt.setPos(int(max(len(xs) - 1, 1) * 1), hi - span * 0.12)
            if txt_r:
                txt_r.setHtml(self._html(f'Cal_{label}={val:.2f}', 'red', fs))
                txt_r.setPos(int(max(len(xs) - 1, 1) * 1), hi - span * 0.19)

    # ===== 全程静态图 (matplotlib Agg, 缺则 skip) =====
    def plot_full_analysis(self, save_dir):
        if len(self.full_time_list) == 0: print("⚠️ 无数据"); return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("⚠️ matplotlib 未安装, 跳过 PNG 出图 (CSV 数据已保留)")
            return
        has_cal = self.full_cal_angle_list and len(self.full_cal_angle_list) == len(self.full_time_list)
        t = self.full_time_list
        fig, axes = plt.subplots(4, 2, figsize=(18, 20))
        (aL1, aR1), (aL2, aR2), (aL3, aR3), (aL4, aR4) = axes
        def _p(ax, d, c, lbl):
            if d and len(d) == len(t): ax.plot(t, d, c, linewidth=1.0, label=lbl)
        _p(aL1, self.full_adc_angle_list, 'b-', 'PZT Angle'); aL1.set_title("PZT Angle"); aL1.grid(True, alpha=0.3)
        _p(aL2, self.full_total_pressure_list, 'b-', 'PZT Fz'); aL2.set_title("PZT Fz"); aL2.grid(True, alpha=0.3)
        _p(aL3, self.full_adc_dx_list, 'b-', 'PZT Fx'); aL3.set_title("PZT Fx"); aL3.grid(True, alpha=0.3)
        _p(aL4, self.full_adc_dy_list, 'c-', 'PZT Fy'); aL4.set_title("PZT Fy"); aL4.grid(True, alpha=0.3)
        _p(aR1, self.full_force_angle_list, 'r-', 'Measured')
        if has_cal: _p(aR1, self.full_cal_angle_list, 'g--', 'Calibrated')
        aR1.set_title("Angle: Meas vs Cal"); aR1.grid(True, alpha=0.3)
        if has_cal: aR1.legend(fontsize=8)
        _p(aR2, self.full_fz_list, 'r-', 'Fz'); aR2.set_title("Fz: Measured"); aR2.grid(True, alpha=0.3)
        has_cal_fz = self.full_fz_cal_list and len(self.full_fz_cal_list) == len(t)
        if has_cal_fz: _p(aR2, self.full_fz_cal_list, 'g--', 'Fz_cal')
        if has_cal_fz: aR2.legend(fontsize=8)
        _p(aR3, self.full_fx_list, 'r-', 'Measured')
        if has_cal: _p(aR3, self.full_fx_cal_list, 'g--', 'Calibrated')
        aR3.set_title("Fx: Meas vs Cal"); aR3.grid(True, alpha=0.3)
        if has_cal: aR3.legend(fontsize=8)
        _p(aR4, self.full_fy_list, 'm-', 'Measured')
        if has_cal: _p(aR4, self.full_fy_cal_list, 'c--', 'Calibrated')
        aR4.set_title("Fy: Meas vs Cal"); aR4.grid(True, alpha=0.3)
        if has_cal: aR4.legend(fontsize=8)
        for row in axes:
            for ax in row: ax.set_xlabel("Time (ms)", fontsize=9)
        plt.tight_layout()
        idx = 1
        while os.path.exists(os.path.join(save_dir, f"full_analysis_cop_{idx}.png")): idx += 1
        sp = os.path.join(save_dir, f"full_analysis_cop_{idx}.png")
        plt.savefig(sp, dpi=300); print(f"📊 已保存：{sp}"); plt.close(fig)
