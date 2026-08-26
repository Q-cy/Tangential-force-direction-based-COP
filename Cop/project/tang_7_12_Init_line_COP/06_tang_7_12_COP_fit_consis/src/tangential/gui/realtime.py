"""PyQtGraph 实时绘图组件。

本模块只负责把父进程传入的压力、方向、梯度、CoP 和六维力状态绘制到
Qt 窗口；串口读取、采样节拍、状态机和 CSV 写入由上层采集会话负责。
阵列坐标统一使用 ``x=列``、``y=行``，单元中心分别位于 ``(c, r)``，
并在表格和梯度视图中使用 ``invertY(True)`` 适配屏幕坐标。模块导入
需要 PyQtGraph/PyQt5，基础 ``tangential`` API 不应直接导入本模块。
"""
import numpy as np
from collections import deque
import threading
import time
import os
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui

from ..config import GuiConfig

pg.setConfigOptions(antialias=True, background='w', foreground='k')

def _yrange(data, pad=0.1):
    """根据有限数据计算带边距的纵轴范围。

    Args:
        data: 可迭代的数值序列；NaN 会被过滤。
        pad: 在数据跨度两侧附加的相对边距，默认 ``0.1``。

    Returns:
        ``(low, high)`` 浮点元组。有效值少于两个时返回 ``(-1, 1)``；
        所有值相等时使用跨度 1，避免 Qt 轴范围退化为零。
    """
    clean = [v for v in data if v == v]  # filter NaN
    if len(clean) < 2: return -1, 1
    mn, mx = min(clean), max(clean)
    r = mx - mn if mx != mn else 1
    return mn - r * pad, mx + r * pad


class CellGridItem(pg.GraphicsObject):
    """绘制 12×7 压力色块、网格和区域轮廓的 PyQtGraph 图元。

    坐标中 ``x`` 对应列、``y`` 对应行；每个单元以 ``(c, r)`` 为中心，
    边界位于半整数位置。数据和区域掩码由 :meth:`set_data` 与
    :meth:`set_regions` 更新，绘制请求由 Qt/pyqtgraph 自动触发。
    """
    def __init__(self, rows=12, cols=7):
        """创建阵列图元并初始化零数据。

        Args:
            rows: 阵列行数，默认 12。
            cols: 阵列列数，默认 7。

        Returns:
            ``None``。初始化 ``GraphicsObject``、数据数组、色阶和区域状态。
        """
        pg.GraphicsObject.__init__(self)
        self.rows, self.cols = rows, cols
        self.data = np.zeros((rows, cols))
        self.vmax = 1.0
        self.region_mask = np.zeros((rows, cols), dtype=np.int32)
        self.region_palette = []
        self.region_frames = {}

    def set_data(self, data, vmax):
        """替换压力矩阵并请求重绘。

        Args:
            data: 形状应为 ``(rows, cols)`` 的 ADC/压力数组。
            vmax: 热力图最大色阶；小于 1 时按 1 处理。

        Returns:
            ``None``。

        Side Effects:
            更新内部数组和色阶，并调用 ``update()`` 让 Qt 安排重绘。
        """
        self.data = data
        self.vmax = max(vmax, 1)
        self.update()

    def set_regions(self, region_mask, palette):
        """设置区域掩码并重新计算区域边框线段。

        Args:
            region_mask: 与压力矩阵同形状的整数掩码；0 表示背景，正整数
                表示区域编号。
            palette: ``(R, G, B)`` 颜色序列，供区域编号循环取色。

        Returns:
            ``None``。

        Side Effects:
            覆盖 ``region_frames``，清除旧区域轮廓的逻辑状态，并请求 Qt
            重绘。公共边会由后绘制的区域颜色覆盖。
        """
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
        """在 Qt 提供的绘图上下文中绘制色块、网格和区域边框。

        Args:
            p: Qt 的 ``QPainter``，坐标系与图元局部坐标一致。
            opt: Qt/pyqtgraph 的 ``StyleOptionGraphicsItem``；当前仅为接口参数。
            widget: 发起绘制的 Qt widget；当前仅为接口参数。

        Returns:
            ``None``。绘图结果直接写入 Qt painter，不返回像素数据。

        Side Effects:
            修改 painter 的抗锯齿、画笔和画刷状态；pyqtgraph 调用方负责
            painter 生命周期。
        """
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
        """返回图元的局部包围矩形。

        Returns:
            ``QRectF(-0.5, -0.5, cols, rows)``，覆盖所有单元边界，供
            Qt 进行更新区域和视图裁剪。
        """
        return QtCore.QRectF(-0.5, -0.5, self.cols, self.rows)

    @staticmethod
    def _brush(t):
        """把归一化压力值映射为白到深红的 Qt 画刷。

        Args:
            t: 归一化色阶值；会裁剪到 ``[0, 1]``。

        Returns:
            ``QBrush``，由相邻颜色控制点线性插值得到；异常范围不会抛出，
            而是按端点颜色处理。
        """
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
    """只绘制 12×7 阵列网格线的 PyQtGraph 图元。

    网格线位于半整数边界，独立于色块绘制，用于避免 ``addLine`` 在
    ``ViewBox`` 边界裁剪时造成外圈视觉尺寸偏差。
    """
    def __init__(self, rows=12, cols=7):
        """创建网格图元。

        Args:
            rows: 网格行数，默认 12。
            cols: 网格列数，默认 7。

        Returns:
            ``None``。
        """
        pg.GraphicsObject.__init__(self)
        self.rows, self.cols = rows, cols

    def paint(self, p, opt, widget):
        """在 Qt painter 中绘制所有横向和纵向网格线。

        Args:
            p: Qt ``QPainter``。
            opt: Qt/pyqtgraph 绘制选项，当前不读取。
            widget: 发起绘制的 Qt widget，当前不读取。

        Returns:
            ``None``；线条直接绘制到 Qt 场景。
        """
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
        """返回覆盖整个阵列网格的局部矩形。

        Returns:
            ``QRectF(-0.5, -0.5, cols, rows)``。
        """
        return QtCore.QRectF(-0.5, -0.5, self.cols, self.rows)


class RealTimePlot:
    """管理实时采集窗口、缓存曲线和 12×7 阵列可视化。

    对外通过 :meth:`set_data` 接收最新状态，通过
    :meth:`append_full_data` 保存静态分析所需历史数据；内部 Qt 定时器以
    ``PLOT_TIMER_INTERVAL_MS`` 刷新图元。类不读取串口，也不决定采样频率。
    所有 Qt 图元、窗口和历史列表都由实例拥有，调用 :meth:`plot_full_analysis`
    时才按需导入 Matplotlib。
    """
    def __init__(self, config: GuiConfig | None = None):
        """创建实时窗口、绘图图元、定时器和历史缓存。

        Returns:
            ``None``。构造过程会创建并显示 Qt 窗口，并启动定时刷新器。

        Side Effects:
            分配大量 PyQtGraph 图元；若 Qt 应用未初始化或 GUI 依赖缺失，
            可能抛出 Qt/PyQtGraph 相关异常。
        """
        self.config = (config or GuiConfig()).validate()
        self._base_window_title = self.config.window_title
        self.rows, self.cols = 12, 7
        self.lock = threading.Lock()
        self._heat_vmax = self.config.heat_vmax

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
        self.timer.start(self.config.timer_interval_ms)

    def set_status(self, status: str | None = None) -> None:
        """更新窗口状态，同时保留配置中的传感器标签。

        Args:
            status: 可选状态文本，例如 ``未接触`` 或 ``数据线程异常``；
                ``None`` 表示恢复为基础窗口标题。

        Returns:
            None: 只更新当前 Qt 窗口标题。

        Side Effects:
            调用 Qt ``setWindowTitle``；不会修改采集数据、曲线或传感器状态。
        """
        title = self._base_window_title
        if status:
            title = f"{title} — {status}"
        self.win.setWindowTitle(title)

    def init_defaults(self):
        """初始化最新一帧的方向、压力、CoP、力和区域显示状态。

        Returns:
            ``None``。

        Side Effects:
            重置所有 ``_`` 前缀的显示状态字段；不创建 Qt 图元，也不启动
            定时器。通常只在构造阶段调用。
        """
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
        self._motion_state = 0              # 滑移状态(0=NO CONTACT,1=STICK,2=SLIP)
        self._is_slipping = False
        self._slip_motion_distance = 0.0
        self._slip_confidence = 0.0
        self._angle_vector_magnitude = 0.0  # 当前角度实际使用的向量模长(cell)
        self._gradient_arr = np.zeros((12, 7, 2), dtype=np.float32)  # 压力梯度(每帧由 main 传入)
        self._contact_init = False
        self._pzt_table_angle_deg = None     # Pressure Table 专用角度（invertY 视图）
        self._region_mask = np.zeros((12, 7), dtype=np.int32)   # per-region 着色掩码
        self._regions = []                   # per-region 数据（cop/delta/id, 供点+箭头显示）
        self._centroid_xy = None             # 整帧形心（不加权, 品红菱形显示）

    def init_history(self):
        """创建固定长度的时序和角度误差 deque。

        Returns:
            ``None``。

        Side Effects:
            清空并替换所有实时历史缓存；长度分别由
            ``PLOT_HISTORY_LEN`` 与 ``PLOT_ERR_HISTORY_LEN`` 控制。
        """
        hist_len = self.config.history_size
        err_len = self.config.error_history_size
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
        """在指定 plot 中创建一支由三条曲线组成的手工箭头。

        Args:
            plot: PyQtGraph ``PlotItem``，箭头的三个 ``PlotDataItem`` 将加入
                其中。

        Returns:
            ``(shaft, head_L, head_R)`` 元组，分别为箭杆和三角箭头两条边。

        Side Effects:
            向 ``plot`` 添加三条空曲线；后续可由 :meth:`_update_arrow` 更新。
        """
        shaft = plot.plot([], [], pen=pg.mkPen('k', width=3))
        hL = plot.plot([], [], pen=pg.mkPen('k', width=2))
        hR = plot.plot([], [], pen=pg.mkPen('k', width=2))
        return shaft, hL, hR

    def _update_arrow(self, parts, angle_deg, length, color, origin=(0.0, 0.0)):
        """按方向、长度和起点更新手工箭头的三条曲线。

        Args:
            parts: ``(shaft, head_L, head_R)`` 三个 ``PlotDataItem``。
            angle_deg: 角度（度）；0° 指向 +X 右方，90° 指向 +Y 上方。
            length: 箭头长度，使用 plot 的数据坐标单位；小于 0.005 时隐藏。
            color: Qt/pyqtgraph 可接受的画笔颜色。
            origin: 箭尾 ``(x, y)`` 数据坐标，默认 ``(0, 0)``。

        Returns:
            ``None``。

        Side Effects:
            更新三条曲线的画笔和顶点；短箭头会清空曲线数据而不抛出异常。
        """
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
        """创建所有实时曲线、方向箭头、压力表和梯度图元。

        Returns:
            ``None``。

        Side Effects:
            创建并显示 ``GraphicsLayoutWidget``，加入 Qt/pyqtgraph 图元，
            保存它们的句柄到实例属性，并建立阵列坐标范围。该方法不读取
            传感器数据；调用失败通常表示 Qt 应用或 GUI 依赖未就绪。
        """
        self.win = pg.GraphicsLayoutWidget(title=self._base_window_title)
        self.win.resize(self.config.window_width, self.config.window_height)
        def _style_plot(p, title):
            """统一设置单个 PlotItem 的标题样式。

            Args:
                p: 要设置的 PyQtGraph ``PlotItem``。
                title: 标题文本；当前同时作为 ``setTitle`` 的标题。

            Returns:
                ``None``；只改变 plot 的标题外观。
            """
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
        self._region_arrows = [
            self._make_arrow_parts(self.p_table)
            for _ in range(self.config.max_region_arrows)
        ]

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
                 centroid=None,
                 motion_state=0,
                 is_slipping=False,
                 slip_motion_distance=0.0,
                 slip_confidence=0.0,
                 angle_vector_magnitude=None):
        """提交一帧实时显示状态，并更新曲线历史缓存。

        Args:
            pzt_angle_deg: PZT 压力方向角，单位为度。
            force_angle_deg: 六维力方向角，单位为度。
            press_table_arr: 84 个压力值，可 reshape 为 ``(12, 7)``。
            total_press_val: 当前帧压力总和。
            cop_curr_x, cop_curr_y: 当前 CoP 的阵列坐标。
            cop_base_x, cop_base_y: 接触基准 CoP；``None`` 时使用当前 CoP。
            cop_delta_x, cop_delta_y: 当前 CoP 相对基准的位移。
            force_fx_val, force_fy_val, force_fz_val: 原始六维力分量。
            cal_fx_val, cal_fy_val, cal_fz_val: 可选标定力分量。
            cop_state: CoP 状态编号，通常 0/1/2 分别表示未接触、粗略和精细。
            gradient: 可选 ``(12, 7, 2)`` 梯度数组。
            contact_init: 是否已经锁定初始接触基准。
            refined: 是否已完成精修；保留该接口参数供上层调用。
            pzt_table_angle_deg: 压力表坐标专用方向角，可选。
            region_mask: 可选 ``(12, 7)`` 区域编号掩码。
            regions: 可选区域字典序列，每项包含 ``id``、``cop`` 和 ``delta``。
            centroid: 可选整帧形心 ``(x, y)``。
            motion_state: 滑移状态枚举值或整数，0/1/2 为未接触/静摩擦/滑移。
            is_slipping: 当前是否处于滑移状态。
            slip_motion_distance: 短窗 CoP 位移，单位为 cell。
            slip_confidence: 压力斑块平移确认置信度，范围通常为 0..1。
            angle_vector_magnitude: 当前 ``pzt_angle_deg`` 实际使用的方向向量
                模长，单位为 cell。STICK 时为静态 CoP delta 模长，SLIP 时为
                EMA 滑移向量模长；仅控制 Pressure Snapshot 红色箭头长度。
                ``None`` 仅为直接调用 ``set_data`` 的旧代码提供兼容回退，
                此时使用 ``hypot(cop_delta_x, cop_delta_y)``；完整采集会话会
                始终显式传入处理器生成的模长。

        Returns:
            ``None``。

        Side Effects:
            在锁内替换下一次绘图所需状态，并向固定长度历史 deque 追加
            当前帧。方法不直接绘制、不启动串口，也不改变采集时间戳。
            ``refined`` 当前仅为兼容调用方的语义参数，未参与绘图分支。
        """
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
            self._motion_state = int(motion_state)
            self._is_slipping = bool(is_slipping)
            self._slip_motion_distance = float(slip_motion_distance)
            self._slip_confidence = float(slip_confidence)
            self._angle_vector_magnitude = (
                float(np.hypot(cop_delta_x, cop_delta_y))
                if angle_vector_magnitude is None
                else float(angle_vector_magnitude)
            )

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
        """追加一帧用于结束后静态分析的完整历史数据。

        Args:
            rel_time_ms: 相对首帧时间，单位为毫秒。
            pzt_angle_deg: PZT 方向角（度）。
            total_press_val: 压力总和。
            cop_delta_x_filt, cop_delta_y_filt: 滤波后的 CoP 位移。
            force_angle_deg: 原始六维力方向角（度）。
            force_fz_filt, force_fx_filt, force_fy_filt: 滤波后的原始力分量。
            cal_angle_deg: 可选标定方向角（度）。
            cal_fx_val, cal_fy_val, cal_fz_val: 可选标定力分量；提供时分别
                追加对应标定历史。

        Returns:
            ``None``。

        Side Effects:
            在锁内追加到实例的 ``full_*`` 列表；这些列表供
            :meth:`plot_full_analysis` 使用，不会写入 CSV。
        """
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
        """从线程安全状态刷新全部 Qt 图元。

        Returns:
            ``None``。

        Side Effects:
            读取并复制锁内状态，更新窗口标题、时序曲线、方向箭头、CoP
            标记、区域轮廓、梯度箭头和压力文字。该方法通常由 Qt
            ``QTimer`` 调用，绘图刷新可能触发 pyqtgraph 的场景更新，但不
            读取串口、不写 CSV，也不改变采样调度。

        Raises:
            可能传播 Qt/pyqtgraph 图元或输入数组形状相关异常；调用方应
            在 Qt 主线程中执行。
        """
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
            motion_state = self._motion_state
            is_slipping = self._is_slipping
            slip_motion_distance = self._slip_motion_distance
            slip_confidence = self._slip_confidence
            angle_vector_magnitude = self._angle_vector_magnitude
            contact_init = self._contact_init
            grad_arr = self._gradient_arr.copy()
            regions = list(self._regions)
            region_mask = self._region_mask.copy()

        # 状态显示
        _state_names = {0: "未接触", 1: "粗略测量", 2: "精细测量"}
        _motion_names = {0: "NO CONTACT", 1: "STICK", 2: "SLIP"}
        cop_status = _state_names.get(cop_state, '?')
        motion_status = _motion_names.get(motion_state, '?')
        self.set_status(
            f"{cop_status} | {motion_status} | "
            f"d={slip_motion_distance:.3f} c={slip_confidence:.3f}"
        )

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

        # Pressure Snapshot：方向沿用 pzt_angle，红箭头长度使用处理器实际
        # 角度向量模长（STICK=静态CoP delta，SLIP=EMA滑移向量）。
        pzt_arrow_len = min(max(angle_vector_magnitude * 0.5, 0.0), 0.65)
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
            self._cell_grid.set_regions(region_mask, self.config.region_palette)
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
                if i < self.config.max_region_arrows:
                    if np.hypot(dx, dy) > 0.05:
                        angle = np.degrees(np.arctan2(dy, dx)) % 360.0
                        palette = self.config.region_palette
                        pr, pgc, pb = palette[(reg['id'] - 1) % len(palette)]
                        self._update_arrow(self._region_arrows[i], angle, np.hypot(dx, dy),
                                           (pr, pgc, pb), (bx, by))
                    else:
                        for part in self._region_arrows[i]:
                            part.setData([], [])
            self._region_cop_dots.setData(spots=cop_spots)
            self._region_base_dots.setData(spots=base_spots)
            # region 数量减少时清理上一帧多余箭头，避免残影。
            for i in range(
                min(len(regions), self.config.max_region_arrows),
                self.config.max_region_arrows,
            ):
                for part in self._region_arrows[i]:
                    part.setData([], [])

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
        """生成实时图中文字使用的粗体 HTML span。

        Args:
            text: 要显示的文本。
            color: CSS/Qt 可识别的颜色字符串。
            size: 字号，单位为 pt。

        Returns:
            可传给 ``pyqtgraph.TextItem.setHtml`` 的 HTML 字符串。
        """
        return f'<span style="color:{color};font-size:{size}pt;font-weight:bold">{text}</span>'

    def _font_size(self, base=16):
        """按窗口当前宽度缩放实时文字字号。

        Args:
            base: 在 1900 像素参考宽度下的字号，单位为 pt。

        Returns:
            不小于 7 的整数字号；窗口越宽，字号按宽度比例增加。
        """
        w = self.win.width()
        return max(int(base * w / 1900), 7)

    def _u1(self, curve, plot, data, txt, label, color='red', fs=16):
        """更新一条单序列时序曲线及其末值标签。

        Args:
            curve: 要写入 ``setData(x, y)`` 的 PlotDataItem。
            plot: 用于设置 X/Y 范围的 PlotItem。
            data: 按时间顺序排列的数值序列。
            txt: 显示末值的 ``TextItem``。
            label: 标签文本。
            color: 标签颜色。
            fs: 标签字号（pt）。

        Returns:
            ``None``。空数据时保持当前图元状态不变。

        Side Effects:
            更新曲线数据、坐标范围和文字标签。
        """
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
        """更新原始/标定双序列曲线及两种末值标签。

        Args:
            c1: 原始值 PlotDataItem。
            c2: 标定值 PlotDataItem。
            plot: 用于设置坐标范围的 PlotItem。
            d1: 原始值序列。
            d2: 标定值序列；只有长度与 ``d1`` 相同时才绘制。
            txt: 原始值 TextItem。
            label: 曲线标签。
            color: 原始值标签颜色。
            txt_r: 可选标定值 TextItem。
            fs: 标签字号（pt）。

        Returns:
            ``None``。空原始序列时不改变已有图元。

        Side Effects:
            更新原始和可能存在的标定曲线、坐标范围及标签文字。
        """
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
        """将实例累计历史绘制为 PNG 静态分析图。

        Args:
            save_dir: PNG 输出目录；文件名自动选择为
                ``full_analysis_cop_<n>.png``，避免覆盖已有文件。

        Returns:
            成功保存或无数据/缺少 Matplotlib 时返回 ``None``。无数据时打印
            提示；缺少可选 Matplotlib 时跳过出图但保留内存数据。

        Side Effects:
            按需导入 Matplotlib，创建并关闭一个 4×2 图形，并在
            ``save_dir`` 写入 PNG；不会修改采集列表或 CSV。

        Raises:
            输出目录不可写、数据结构不一致或 Matplotlib 绘图失败时可能
            传播对应异常。
        """
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
            """仅在序列与时间轴等长时绘制一条静态分析曲线。

            Args:
                ax: Matplotlib ``Axes``。
                d: 要绘制的数值序列。
                c: Matplotlib 线型/颜色格式字符串。
                lbl: 图例标签。

            Returns:
                ``None``；长度不匹配或序列为空时不绘制。
            """
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
