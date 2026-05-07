# ================================================================
# 修复版：支持多指多COP独立检测、不粘连、不合并重心
# 单点/双点/多点均可独立显示各自COP与方向箭头
# ================================================================
import numpy as np
from collections import deque
import threading
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import csv
import os
import serial
import serial.tools.list_ports
import struct
import cv2

# ---------------- 全局参数调优（适配多指分离） ----------------
PRESSURE_THRESHOLD = 15
MIN_REGION_PRESSURE = 80
MAX_MATCH_DIST = 3.5
MAX_TRACKED_REGIONS = 10
COP_STABILITY_FRAMES = 5
SENSOR_ROWS = 12
SENSOR_COLS = 7
LINE_DIST_THRESHOLD = 0.1
DIR_DOT_THRESHOLD = 0.7
TOTAL_PRESSURE_LOW_THRESHOLD = 500

# ---------------- angle工具函数 ----------------
def compute_vector_angle(x: float, y: float) -> tuple[float, float]:
    epsilon = 1e-8
    mag = np.hypot(x, y)
    angle = np.degrees(np.arctan2(y, x + epsilon))
    if angle < 0:
        angle += 360
    return angle, mag

def compute_6Dforce_angle(Fx: float, Fy: float) -> tuple[float, float]:
    return compute_vector_angle(Fx, Fy)

def compute_PZT_angle(Px: float, Py: float) -> tuple[float, float]:
    return compute_vector_angle(Px, Py)

def angle_difference(a1: float, a2: float) -> float:
    diff = abs(a1 - a2)
    return min(diff, 360 - diff)

# ---------------- 压力传感器 ----------------
BAUDRATE_PRESS = 921600
BAUDRATE_FORCE = 460860

class PressureSensor:
    def __init__(self):
        self.ser = None
        self.port = None
        self.last = None
        self.auto_find_port()

    def auto_find_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p, _, _ in ports:
            if p == "/dev/ttyUSB0":
                continue
            try:
                self.ser = serial.Serial(p, BAUDRATE_PRESS, timeout=0.01)
                self.port = p
                time.sleep(0.1)
                self.ser.reset_input_buffer()
                return
            except:
                continue
        raise Exception("未找到压力传感器")

    def reconnect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.auto_find_port()

    def read_data(self):
        if not self.ser or not self.ser.is_open:
            return None
        try:
            cmd = [0x55,0xAA,9,0,0x34,0,0xFB,0,0x1C,0,0,0xA8,0,0x35]
            self.ser.write(bytearray(cmd))
            time.sleep(0.005)
            resp = self.ser.read(256)
            idx = resp.find(b'\xaa\x55')
            if idx == -1 or len(resp[idx:]) < 182:
                return None
            return resp[idx+14:idx+14+168]
        except Exception as e:
            return None

    def decode(self, raw):
        arr = []
        for i in range(0, 168, 2):
            arr.append(struct.unpack("<H", raw[i:i+2])[0])
        out = []
        for i in range(12):
            out.extend(arr[i*7:(i+1)*7])
        self.last = out.copy()
        return out

# ---------------- 六维力传感器 ----------------
class SixAxisForceSensor:
    def __init__(self):
        self.ser = None
        self.port = "/dev/ttyUSB0"
        self.zero_data = [0.0]*6
        self.open_port()

    def open_port(self):
        try:
            self.ser = serial.Serial(self.port, BAUDRATE_FORCE, timeout=0.05)
            time.sleep(0.1)
            self.ser.reset_input_buffer()
        except:
            self.ser = None

    def reconnect(self):
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except:
            pass
        time.sleep(0.2)
        self.open_port()

    def read(self):
        if not self.ser or not self.ser.is_open:
            return None
        try:
            self.ser.write(b'\x49\xAA\x0D\x0A')
            time.sleep(0.005)
            resp = self.ser.read(28)
            if len(resp) != 28 or resp[:2] != b'\x49\xAA':
                return None
            Fx = struct.unpack('<f', resp[2:6])[0]
            Fy = struct.unpack('<f', resp[6:10])[0]
            Fz = struct.unpack('<f', resp[10:14])[0]
            Mx = struct.unpack('<f', resp[14:18])[0]
            My = struct.unpack('<f', resp[18:22])[0]
            Mz = struct.unpack('<f', resp[22:26])[0]
            Fx *= 9.8; Fy *= 9.8; Fz *= 9.8
            Mx *= 9.8; My *= 9.8; Mz *= 9.8
            Fx -= self.zero_data[0]; Fy -= self.zero_data[1]; Fz -= self.zero_data[2]
            Mx -= self.zero_data[3]; My -= self.zero_data[4]; Mz -= self.zero_data[5]
            return [round(v, 2) for v in [Fx, Fy, Fz, Mx, My, Mz]]
        except Exception as e:
            return None

    def calibrate_zero(self):
        vals = []
        for _ in range(20):
            d = self.read()
            if d:
                vals.append(d)
            time.sleep(0.05)
        if len(vals) >= 5:
            self.zero_data = np.mean(np.array(vals), axis=0).tolist()

# ---------------- 时间戳缓冲区 ----------------
class TimestampedBuffer:
    def __init__(self, maxlen=500):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()

    def append(self, item):
        with self.lock:
            self.buf.append(item)

    def get_latest(self):
        with self.lock:
            return self.buf[-1] if self.buf else None

    def find_closest(self, ts):
        with self.lock:
            best = None
            best_dt = 1e9
            for item in self.buf:
                dt = abs(item["t"] - ts)
                if dt < best_dt:
                    best_dt = dt
                    best = item
            return best

# ---------------- COP多区域跟踪核心 ----------------
first_frame = None
first_frame_lock = threading.Lock()
grad_table_data = np.zeros((12, 7, 2))
grad_table_lock = threading.Lock()

class RegionTracker:
    def __init__(self, region_id):
        self.region_id = region_id
        self.first_contact_cop_x = None
        self.first_contact_cop_y = None
        self.contact_initialized = False
        self.initial_cop_x_buffer = deque(maxlen=COP_STABILITY_FRAMES)
        self.initial_cop_y_buffer = deque(maxlen=COP_STABILITY_FRAMES)
        self.total_pressure_low_counter = 0
        self.last_cop_x = 0.0
        self.last_cop_y = 0.0

tracked_regions = {}
_next_region_id = 0

def _label_connected_components(binary_mask):
    h, w = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    regions = []
    dirs = [(-1,-1),(-1,0),(-1,1),
            (0,-1),        (0,1),
            (1,-1),(1,0),(1,1)]
    for y in range(h):
        for x in range(w):
            if binary_mask[y,x] and not visited[y,x]:
                q = [(y,x)]
                visited[y,x] = True
                reg = []
                while q:
                    cy, cx = q.pop(0)
                    reg.append((cy, cx))
                    for dy, dx in dirs:
                        ny, nx = cy+dy, cx+dx
                        if 0<=ny<h and 0<=nx<w and binary_mask[ny,nx] and not visited[ny,nx]:
                            visited[ny,nx] = True
                            q.append((ny,nx))
                regions.append(reg)
    return regions

def _match_regions(candidates):
    global tracked_regions, _next_region_id
    matched = []
    used = set()
    # 匹配已有跟踪区域
    for cand in candidates:
        cx, cy = cand["cop_x"], cand["cop_y"]
        best_id, best_dist = None, 999
        for rid, rt in tracked_regions.items():
            if rid in used:
                continue
            d = np.hypot(cx - rt.last_cop_x, cy - rt.last_cop_y)
            if d < MAX_MATCH_DIST and d < best_dist:
                best_dist = d
                best_id = rid
        if best_id is not None:
            used.add(best_id)
            cand["region_id"] = best_id
            matched.append(cand)
        else:
            # 新触点直接新建ID，不限制数量
            new_id = _next_region_id
            _next_region_id += 1
            tracked_regions[new_id] = RegionTracker(new_id)
            cand["region_id"] = new_id
            matched.append(cand)
    # 保留当前匹配到的区域，清除消失的旧区域
    new_tr = {k:v for k,v in tracked_regions.items() if k in [m["region_id"] for m in matched]}
    tracked_regions.clear()
    tracked_regions.update(new_tr)
    return matched

def _compute_region_cop(region, frame):
    xs, ys, ps = [], [], []
    for (y,x) in region:
        v = frame[y,x]
        xs.append(x)
        ys.append(y)
        ps.append(v)
    total = sum(ps)
    if total < MIN_REGION_PRESSURE:
        return None
    cx = sum(x*p for x,p in zip(xs,ps)) / total
    cy = sum(y*p for y,p in zip(ys,ps)) / total
    return {"cop_x":cx, "cop_y":cy, "total_pressure":total}

def _update_init(rt, cx, cy):
    if not rt.contact_initialized:
        rt.initial_cop_x_buffer.append(cx)
        rt.initial_cop_y_buffer.append(cy)
        if len(rt.initial_cop_x_buffer) >= COP_STABILITY_FRAMES:
            rt.first_contact_cop_x = rt.initial_cop_x_buffer[0]
            rt.first_contact_cop_y = rt.initial_cop_y_buffer[0]
            rt.contact_initialized = True

def subtract_baseline(current_frame):
    global first_frame
    curr = np.array(current_frame, dtype=np.float32).flatten()
    with first_frame_lock:
        if first_frame is None:
            first_frame = curr.copy()
    diff = curr - first_frame
    return np.clip(diff, 0, None)

def reset_all_regions():
    global tracked_regions, _next_region_id, grad_table_data
    tracked_regions.clear()
    _next_region_id = 0
    with grad_table_lock:
        grad_table_data.fill(0)

def compute_pressure_direction(baseline_subtracted_frame):
    frame2d = np.array(baseline_subtracted_frame).reshape(SENSOR_ROWS, SENSOR_COLS)
    # 二值化
    binary = frame2d >= PRESSURE_THRESHOLD
    # 形态学腐蚀：分离近距离双触点，防止粘连成一块
    kernel = np.ones((2,2), np.uint8)
    binary = cv2.erode(binary.astype(np.uint8), kernel, iterations=1)
    # 连通域检测
    regions = _label_connected_components(binary)

    # 梯度计算
    grad = np.zeros_like(grad_table_data)
    for y in range(SENSOR_ROWS):
        for x in range(SENSOR_COLS):
            val = frame2d[y,x]
            l = frame2d[y,x-1] if x>0 else val
            r = frame2d[y,x+1] if x<SENSOR_COLS-1 else val
            u = frame2d[y-1,x] if y>0 else val
            d = frame2d[y+1,x] if y<SENSOR_ROWS-1 else val
            grad[y,x] = (r-l, u-d)
    with grad_table_lock:
        grad_table_data[:] = grad

    total = np.sum(frame2d)
    if total < TOTAL_PRESSURE_LOW_THRESHOLD:
        reset_all_regions()

    # 计算每个连通域独立COP
    candidates = []
    for reg in regions:
        res = _compute_region_cop(reg, frame2d)
        if res:
            candidates.append(res)

    # 不排序不丢弃，全部保留
    matched = _match_regions(candidates)

    out = []
    for idx, cand in enumerate(matched):
        rid = cand["region_id"]
        rt = tracked_regions[rid]
        cx, cy = cand["cop_x"], cand["cop_y"]
        tp = cand["total_pressure"]
        _update_init(rt, cx, cy)
        rt.last_cop_x, rt.last_cop_y = cx, cy

        dx = cx - rt.first_contact_cop_x if rt.contact_initialized else 0
        dy = rt.first_contact_cop_y - cy if rt.contact_initialized else 0
        bx = rt.first_contact_cop_x if rt.contact_initialized else cx
        by = rt.first_contact_cop_y if rt.contact_initialized else cy

        out.append({
            "cop_x":cx, "cop_y":cy,
            "delta_cop_x":dx, "delta_cop_y":dy,
            "base_cop_x":bx, "base_cop_y":by,
            "total_pressure":tp,
            "region_id":rid,
            "is_initialized":rt.contact_initialized,
            "index":idx
        })

    return {
        "regions": out,
        "grid_info": (0, SENSOR_ROWS-1, 0, SENSOR_COLS-1),
        "is_contact": len(out)>0
    }

# ---------------- CSV保存 ----------------
N_COP_CSV = 5
CSV_HEADER = [
    "timestamp", "rel_ms",
    *[f"ch{i}" for i in range(1,85)],
    "Fx","Fy","Fz","Mx","My","Mz",
    "press_t","force_t","dt",
    "delta_CoP_X","delta_CoP_Y",
    "delta_Force_X","delta_Force_Y",
    "ADC_angle","ADC_mag","Force_angle","Force_mag"
]

for i in range(2, N_COP_CSV+1):
    p = f"COP{i}"
    CSV_HEADER += [f"{p}_x",f"{p}_y",f"{p}_dx",f"{p}_dy",f"{p}_total",f"{p}_angle",f"{p}_mag"]

def auto_get_csv_path(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    idx=1
    while os.path.exists(f"{save_dir}/data_{idx}.csv"):idx+=1
    return f"{save_dir}/data_{idx}.csv"

def init_csv_file(path):
    f = open(path,"w",encoding="utf-8",newline="")
    w = csv.writer(f)
    w.writerow(CSV_HEADER)
    return w,f

def build_csv_row(pts, rel_ms, ch, force, fts, dcx, dcy, dfx, dfy,
                  adc_a, adc_m, f_a, f_m, regions, angle_func):
    row = [pts*1000, rel_ms, *ch, *force, pts, fts, abs(pts-fts),
           dcx, dcy, dfx, dfy, adc_a, adc_m, f_a, f_m]
    for i in range(1, N_COP_CSV):
        if i < len(regions):
            r = regions[i]
            a,m = angle_func(r["delta_cop_x"], r["delta_cop_y"])
            row += [r["cop_x"],r["cop_y"],r["delta_cop_x"],r["delta_cop_y"],
                    r["total_pressure"],a,m]
        else:
            row += [0]*7
    return row

# ---------------- 实时绘图（无警告+多COP显示） ----------------
COLORS = ['green','cyan','orange','yellow','magenta','brown','gray','purple','red','blue']
PLOT_INTERVAL_MS = 100
ERROR_PLOT_LEN = 100
MAG_PLOT_LEN = 100

class RealTimePlot:
    def __init__(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        self.rows, self.cols = SENSOR_ROWS, SENSOR_COLS
        self.epsilon = 0.01
        self.full_time_list = []
        self.full_adc_mag_list = []
        self.full_force_mag_list = []

        # 关闭自动布局冲突，消除警告
        self.fig = plt.figure(figsize=(16,12), constrained_layout=False)
        self.build_layout()
        self.ani = FuncAnimation(self.fig, self.update_all, interval=PLOT_INTERVAL_MS, cache_frame_data=False)
        self.lock = threading.Lock()
        self.init_history()
        self.adc_angle=0; self.adc_mag=0; self.force_angle=0; self.force_mag=0
        self.diff_frame = np.zeros((12,7))
        self.regions = []
        self.raw_fx=0; self.raw_fy=0

    def build_layout(self):
        gs_outer = GridSpec(4,2, width_ratios=[1,1], height_ratios=[6,2,2,1], hspace=0.3, wspace=0.3)
        gs_arrows = gs_outer[0,0].subgridspec(1,2,wspace=0.3)

        self.ax1 = plt.subplot(gs_arrows[0,0])
        self.ax1.axis('off')

        self.ax2 = plt.subplot(gs_arrows[0,1])
        self.ax2.axis('off')

        gs_adc = gs_outer[1,0].subgridspec(1,2)
        self.ax_adc_dx = plt.subplot(gs_adc[0,0])
        self.line_adc_dx, = self.ax_adc_dx.plot([],[])

        self.ax_adc_dy = plt.subplot(gs_adc[0,1])
        self.line_adc_dy, = self.ax_adc_dy.plot([],[])

        gs_force = gs_outer[2,0].subgridspec(1,2)
        self.ax_force_fx = plt.subplot(gs_force[0,0])
        self.line_force_fx, = self.ax_force_fx.plot([],[])

        self.ax_force_fy = plt.subplot(gs_force[0,1])
        self.line_force_fy, = self.ax_force_fy.plot([],[])

        self.ax4 = plt.subplot(gs_outer[3,0])
        self.error_line, = self.ax4.plot([],[])

        gs_right = gs_outer[:,1].subgridspec(1,2)
        self.ax5 = plt.subplot(gs_right[0,0])
        self.ax5.axis('off')

        self.ax6 = plt.subplot(gs_right[0,1])
        self.ax6.axis('off')

    def init_history(self):
        self.angle_error_history = deque(maxlen=ERROR_PLOT_LEN)
        self.adc_dx_history = deque(maxlen=MAG_PLOT_LEN)
        self.adc_dy_history = deque(maxlen=MAG_PLOT_LEN)
        self.force_fx_history = deque(maxlen=MAG_PLOT_LEN)
        self.force_fy_history = deque(maxlen=MAG_PLOT_LEN)
        self.adc_mag_history = deque(maxlen=MAG_PLOT_LEN)
        self.raw_force_mag_history = deque(maxlen=MAG_PLOT_LEN)

    def set_data(self, adc_angle, adc_mag, force_angle, force_mag, diff_frame, regions, fx, fy):
        with self.lock:
            self.adc_angle=adc_angle
            self.adc_mag=adc_mag
            self.force_angle=force_angle
            self.force_mag=force_mag
            self.diff_frame = diff_frame.reshape(12,7)
            self.regions = regions
            self.raw_fx=fx
            self.raw_fy=fy

            err = min(abs(adc_angle-force_angle), 360-abs(adc_angle-force_angle))
            self.angle_error_history.append(err)

            if regions:
                self.adc_dx_history.append(regions[0]["delta_cop_x"])
                self.adc_dy_history.append(regions[0]["delta_cop_y"])

            self.force_fx_history.append(fx)
            self.force_fy_history.append(fy)

    def append_full_data(self, ms, a, f):
        with self.lock:
            self.full_time_list.append(ms)
            self.full_adc_mag_list.append(a)
            self.full_force_mag_list.append(f)

    def update_all(self, frame):
        self.update_direction()
        self.update_magnitude()
        self.update_adc()
        self.update_force()
        self.update_error()
        self.update_pressure_table()
        self.update_gradient()
        return []

    def update_direction(self):
        with self.lock: a,f = self.adc_angle, self.force_angle
        self.ax1.clear()
        self.ax1.set_xlim(0,1)
        self.ax1.set_ylim(0,1)
        self.ax1.axis('off')
        self.ax1.arrow(0.5,0.5, 0.4*np.cos(np.radians(a)),0.4*np.sin(np.radians(a)), head_width=0.12, lw=2.5)
        self.ax1.arrow(0.5,0.5, 0.35*np.cos(np.radians(f)),0.35*np.sin(np.radians(f)), head_width=0.1, color='r', lw=2)

    def update_magnitude(self):
        with self.lock: a,m,fa,fm = self.adc_angle,self.adc_mag,self.force_angle,self.force_mag
        self.ax2.clear()
        self.ax2.set_xlim(0,1)
        self.ax2.set_ylim(0,1)
        self.ax2.axis('off')
        if m>0.1:
            l = min(m/5*0.45, 0.45)
            self.ax2.arrow(0.5,0.5, l*np.cos(np.radians(a)), l*np.sin(np.radians(a)), head_width=0.1, lw=2.5)
        if fm>0:
            l = min(abs(fm)/20*0.4, 0.4)
            self.ax2.arrow(0.5,0.5, l*np.cos(np.radians(fa)), l*np.sin(np.radians(fa)), head_width=0.1, color='r', lw=2.5)

    def update_adc(self):
        if self.adc_dx_history:
            x=list(range(len(self.adc_dx_history)))
            y=list(self.adc_dx_history)
            self.line_adc_dx.set_data(x,y)
            self.ax_adc_dx.set_xlim(0,len(x))
            self.ax_adc_dx.relim()
            self.ax_adc_dx.autoscale_view()
        if self.adc_dy_history:
            x=list(range(len(self.adc_dy_history)))
            y=list(self.adc_dy_history)
            self.line_adc_dy.set_data(x,y)
            self.ax_adc_dy.set_xlim(0,len(x))
            self.ax_adc_dy.relim()
            self.ax_adc_dy.autoscale_view()

    def update_force(self):
        if self.force_fx_history:
            x=list(range(len(self.force_fx_history)))
            y=list(self.force_fx_history)
            self.line_force_fx.set_data(x,y)
            self.ax_force_fx.set_xlim(0,len(x))
            self.ax_force_fx.relim()
            self.ax_force_fx.autoscale_view()
        if self.force_fy_history:
            x=list(range(len(self.force_fy_history)))
            y=list(self.force_fy_history)
            self.line_force_fy.set_data(x,y)
            self.ax_force_fy.set_xlim(0,len(x))
            self.ax_force_fy.relim()
            self.ax_force_fy.autoscale_view()

    def update_error(self):
        if self.angle_error_history:
            x=list(range(len(self.angle_error_history)))
            y=list(self.angle_error_history)
            self.error_line.set_data(x,y)
            self.ax4.set_xlim(0,len(x))
            self.ax4.relim()
            self.ax4.autoscale_view()

    def update_pressure_table(self):
        with self.lock: data=self.diff_frame.copy(); regions=self.regions
        self.ax5.clear()
        self.ax5.axis('off')
        self.ax5.set_xlim(-0.5,6.5)
        self.ax5.set_ylim(11.5,-0.5)

        vmax = np.max(data) if np.max(data)!=0 else 1
        t = self.ax5.table(
            cellText=[[f"{v:.0f}" for v in r] for r in data],
            cellColours=plt.cm.Reds(data/vmax),
            cellLoc='center',
            bbox=[0,0,1,1]
        )
        t.set_fontsize(8)

        # 绘制所有COP点 + 各自方向箭头
        for i,r in enumerate(regions):
            c = COLORS[i % len(COLORS)]
            s = 200 if i == 0 else 120
            self.ax5.scatter(r["cop_x"], r["cop_y"], s=s, color=c, edgecolor='k')
            if r["is_initialized"] and np.hypot(r["delta_cop_x"], r["delta_cop_y"])>0.1:
                self.ax5.arrow(
                    r["base_cop_x"], r["base_cop_y"],
                    r["delta_cop_x"], -r["delta_cop_y"],
                    head_width=0.3, color=c, lw=2
                )

    def update_gradient(self):
        with grad_table_lock: g=grad_table_data.copy()
        self.ax6.clear()
        self.ax6.axis('off')
        self.ax6.set_xlim(-0.5,6.5)
        self.ax6.set_ylim(11.5,-0.5)

        for y in range(12):
            for x in range(7):
                gx,gy = g[y,x]
                if np.hypot(gx,gy)>1:
                    self.ax6.quiver(x,y,gx/np.hypot(gx,gy),gy/np.hypot(gx,gy), scale=3)

        with self.lock:
            for i,r in enumerate(self.regions):
                c = COLORS[i % len(COLORS)]
                self.ax6.scatter(r["cop_x"], r["cop_y"], color=c, s=100)

    def plot_full_magnitude_curve(self, d):
        if not self.full_time_list: return
        plt.figure(figsize=(12,8))
        plt.plot(self.full_time_list, self.full_adc_mag_list, label='COP')
        plt.plot(self.full_time_list, self.full_force_mag_list, label='Force')
        plt.legend()
        plt.savefig(os.path.join(d,"full.png"), dpi=300)
        plt.close()

    def show(self):
        plt.show()

# ---------------- 主程序入口 ----------------
SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"
stop_event = threading.Event()
plot = None
TARGET_FPS = 100
MAX_TIME_DIFF = 0.015

class PressureThread(threading.Thread):
    def __init__(self, s, b):
        super().__init__(daemon=True)
        self.s = s
        self.b = b
    def run(self):
        while not stop_event.is_set():
            t = time.perf_counter()
            r = self.s.read_data()
            if r:
                try:
                    self.b.append({"t":t,"data":self.s.decode(r)})
                except:
                    pass
            time.sleep(0.001)

class ForceThread(threading.Thread):
    def __init__(self, s, b):
        super().__init__(daemon=True)
        self.s = s
        self.b = b
    def run(self):
        while not stop_event.is_set():
            t = time.perf_counter()
            d = self.s.read()
            if d:
                self.b.append({"t":t,"data":d})
            time.sleep(0.001)

def data_loop():
    global plot
    csv_p = auto_get_csv_path(SAVE_DIR)
    w,f = init_csv_file(csv_p)
    s_p = PressureSensor()
    s_f = SixAxisForceSensor()
    s_f.calibrate_zero()
    bp = TimestampedBuffer(500)
    bf = TimestampedBuffer(500)

    PressureThread(s_p, bp).start()
    ForceThread(s_f, bf).start()

    t0 = time.perf_counter()

    while not stop_event.is_set():
        now = time.perf_counter()
        ms = int((now-t0)*1000)
        p = bp.get_latest()
        if not p:
            continue
        fr = bf.find_closest(p["t"])
        if not fr or abs(p["t"]-fr["t"])>MAX_TIME_DIFF:
            continue

        base = subtract_baseline(p["data"])
        res = compute_pressure_direction(base)
        regions = res["regions"]

        fx,fy,fz,mx,my,mz = fr["data"]
        aa,am = compute_PZT_angle(regions[0]["delta_cop_x"], regions[0]["delta_cop_y"]) if regions else (0,0)
        fa,fm = compute_6Dforce_angle(fx,fy)

        row = build_csv_row(
            p["t"],ms,p["data"],fr["data"],fr["t"],
            regions[0]["delta_cop_x"] if regions else 0,
            regions[0]["delta_cop_y"] if regions else 0,
            fx,fy,aa,am,fa,fm,regions,compute_PZT_angle
        )
        w.writerow(row)
        f.flush()

        plot.set_data(aa,am,fa,fm,base,regions,fx,fy)
        if regions and regions[0]["is_initialized"]:
            plot.append_full_data(ms,am,fm)

        elapsed = time.perf_counter() - now
        sleep_time = max(0, 1.0/TARGET_FPS - elapsed)
        time.sleep(sleep_time)

    f.close()

def main():
    global plot
    plot = RealTimePlot()
    threading.Thread(target=data_loop, daemon=True).start()
    plot.show()
    stop_event.set()
    plot.plot_full_magnitude_curve(SAVE_DIR)

if __name__ == "__main__":
    main()