import serial
import serial.tools.list_ports
import time
import struct
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from collections import deque
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from skimage.measure import label, regionprops

# ==================== Config ====================
BAUDRATE_PRESS = 921600
BAUDRATE_FORCE = 460800
SAVE_DIR = "/home/qcy/Project/data/2.PZT_tangential/weight/test"
TARGET_HZ = 100.0
PLOT_INTERVAL_MS = 100
MAX_SYNC_DT = 0.015
PRESS_BUFFER_SIZE = 500
FORCE_BUFFER_SIZE = 200

THRESHOLD_MULTIPLIER = 1
MIN_THRESHOLD_FLOOR = 500

DIR_SMOOTH_ALPHA = 0.15
ERROR_PLOT_LEN = 100
MAG_PLOT_LEN = 100

COP_STABILITY_FRAMES_REQUIRED = 5
COP_STABILITY_TOLERANCE = 0.1
ROI_EXPANSION_MARGIN = 1

# ==================== 二阶矩变化量全局变量 ====================
last_u20 = 0.0
last_u02 = 0.0
last_u11 = 0.0
smooth_dir_x = 0.0
smooth_dir_y = 0.0
MIN_DELTA_MAG = 0.005

# ==================== Global Variables ====================
first_frame = None
first_frame_lock = threading.Lock()

first_contact_CoP_x = None
first_contact_CoP_y = None
contact_initialized = False

cop_line_check_buffer = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)
current_line_segment_p1 = None
current_line_segment_p2 = None
cop_line_stability_active = False
line_equation_coeffs = None

adc_filtered_dir = None
grad_table_data = np.zeros((12, 7, 2))

angle_error_history = deque(maxlen=ERROR_PLOT_LEN)
adc_mag_history = deque(maxlen=MAG_PLOT_LEN)
force_mag_history = deque(maxlen=MAG_PLOT_LEN)
frame_count_history = deque(maxlen=MAG_PLOT_LEN)

raw_adc_sum_history = deque(maxlen=MAG_PLOT_LEN)
raw_force_mag_history = deque(maxlen=MAG_PLOT_LEN)

full_time_list = []
full_adc_mag_list = []
full_force_mag_list = []

# ==================== Baseline Subtraction ====================
def subtract_baseline(current_frame):
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()
    with first_frame_lock:
        if first_frame is None:
            first_frame = current_frame.copy()
    diff = current_frame - first_frame
    diff = np.clip(diff, 0, None)
    return diff

# ==================== 二阶矩变化量方向计算（你要的最终版）====================
def compute_gradient_in_region(frame):
    global adc_filtered_dir, grad_table_data
    global first_contact_CoP_x, first_contact_CoP_y, contact_initialized
    global cop_line_check_buffer, current_line_segment_p1, current_line_segment_p2, \
           cop_line_stability_active, line_equation_coeffs
    global last_u20, last_u02, last_u11, smooth_dir_x, smooth_dir_y

    rows, cols = 12, 7
    frame_flat = np.array(frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)

    positive_pressures = frame_flat[frame_flat > 0]
    if len(positive_pressures) > 0:
        current_avg_adc = np.mean(positive_pressures)
        calculated_threshold = max(MIN_THRESHOLD_FLOOR, current_avg_adc * THRESHOLD_MULTIPLIER)
    else:
        calculated_threshold = MIN_THRESHOLD_FLOOR
    threshold = calculated_threshold

    default_r1, default_r2 = 0, rows - 1
    default_c1, default_c2 = 0, cols - 1
    default_return_values = (0.0, 0.0, 0.0, 0.0, 0.0,
                             default_r1, default_r2, default_c1, default_c2,
                             0.0, 0.0, 0.0, 0.0)

    mask = frame2d > threshold
    total = np.sum(frame2d[mask])
    if total < 1e-3:
        adc_filtered_dir = None
        contact_initialized = False
        first_contact_CoP_x = None
        first_contact_CoP_y = None
        cop_line_check_buffer.clear()
        current_line_segment_p1 = None
        current_line_segment_p2 = None
        cop_line_stability_active = False
        line_equation_coeffs = None
        grad_table_data = np.zeros((rows, cols, 2), dtype=np.float32)
        last_u20 = last_u02 = last_u11 = 0.0
        smooth_dir_x = smooth_dir_y = 0.0
        return default_return_values

    x = np.tile(np.arange(cols), (rows, 1))
    y = np.repeat(np.arange(rows), cols).reshape(rows, cols)
    p = frame2d

    # ====================== 核心：二阶矩变化量方向 ======================
    u20 = np.sum(p[mask] * x[mask]**2) / total
    u02 = np.sum(p[mask] * y[mask]**2) / total
    u11 = np.sum(p[mask] * x[mask] * y[mask]) / total

    du20 = u20 - last_u20
    du02 = u02 - last_u02
    du11 = u11 - last_u11

    dir_x = du11
    dir_y = 0.5 * (du20 - du02)

    mag = np.hypot(dir_x, dir_y)
    if mag < MIN_DELTA_MAG:
        dir_x, dir_y = 0.0, 0.0
    else:
        dir_x /= mag
        dir_y /= mag

    smooth_dir_x = DIR_SMOOTH_ALPHA * smooth_dir_x + (1 - DIR_SMOOTH_ALPHA) * dir_x
    smooth_dir_y = DIR_SMOOTH_ALPHA * smooth_dir_y + (1 - DIR_SMOOTH_ALPHA) * dir_y
    n = np.hypot(smooth_dir_x, smooth_dir_y)
    if n > 1e-6:
        smooth_dir_x /= n
        smooth_dir_y /= n

    last_u20 = u20
    last_u02 = u02
    last_u11 = u11
    # ==================================================================

    fx, fy = smooth_dir_x, -smooth_dir_y
    vec_mag = mag

    cop_x = np.sum(p[mask] * x[mask]) / total
    cop_y = np.sum(p[mask] * y[mask]) / total

    binary_image = (frame2d > threshold).astype(int)
    labeled_image = label(binary_image)
    regions = regionprops(labeled_image)
    r1, r2, c1, c2 = default_r1, default_r2, default_c1, default_c2
    target_region = None

    cop_r = int(round(cop_y))
    cop_c = int(round(cop_x))
    cop_r = np.clip(cop_r, 0, rows-1)
    cop_c = np.clip(cop_c, 0, cols-1)
    if labeled_image[cop_r, cop_c] != 0:
        lab = labeled_image[cop_r, cop_c]
        for r in regions:
            if r.label == lab:
                target_region = r
                break

    if target_region is None:
        max_area = 0
        for r in regions:
            if r.area > max_area:
                max_area = r.area
                target_region = r

    if target_region is not None:
        min_r, min_c, max_r_ex, max_c_ex = target_region.bbox
        max_r_in = max_r_ex - 1
        max_c_in = max_c_ex - 1
        r1 = max(0, min_r - ROI_EXPANSION_MARGIN)
        c1 = max(0, min_c - ROI_EXPANSION_MARGIN)
        r2 = min(rows-1, max_r_in + ROI_EXPANSION_MARGIN)
        c2 = min(cols-1, max_c_in + ROI_EXPANSION_MARGIN)

    gy, gx = np.gradient(frame2d.astype(float))
    grad_table_data[...,0] = gx
    grad_table_data[...,1] = gy

    delta_CoP_x = 0.0
    delta_CoP_y = 0.0
    base_CoP_x_for_plot = cop_x
    base_CoP_y_for_plot = cop_y

    adc_filtered_dir = (fx, fy) if mag > MIN_DELTA_MAG else None

    return (fx, fy, vec_mag, cop_x, cop_y,
            r1, r2, c1, c2,
            delta_CoP_x, delta_CoP_y,
            base_CoP_x_for_plot, base_CoP_y_for_plot)

# ==================== Angle Calculation ====================
def compute_gradient_angle_single(x, y):
    epsilon = 1e-8
    angle = np.degrees(np.arctan2(y, x + epsilon))
    if angle < 0:
        angle += 360
    mag = np.hypot(x, y)
    return angle, mag

def compute_force_angle(Fx, Fy):
    epsilon = 1e-8
    mag = np.hypot(Fx, Fy)
    if mag < 1e-8:
        return 0.0, 0.0
    angle = np.degrees(np.arctan2(Fy, Fx + epsilon))
    if angle < 0:
        angle += 360
    return angle, mag

# ==================== 6-axis Force Sensor ====================
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
            if len(resp)!=28 or resp[:2]!=b'\x49\xAA': return None
            Fx = struct.unpack('<f', resp[2:6])[0]
            Fy = struct.unpack('<f', resp[6:10])[0]
            Fz = struct.unpack('<f', resp[10:14])[0]
            Mx = struct.unpack('<f', resp[14:18])[0]
            My = struct.unpack('<f', resp[18:22])[0]
            Mz = struct.unpack('<f', resp[22:26])[0]
            Fx *=9.8; Fy*=9.8; Fz*=9.8; Mx*=9.8; My*=9.8; Mz*=9.8
            Fx-=self.zero_data[0]; Fy-=self.zero_data[1]; Fz-=self.zero_data[2]
            Mx-=self.zero_data[3]; My-=self.zero_data[4]; Mz-=self.zero_data[5]
            return [round(v,2) for v in [Fx,Fy,Fz,Mx,My,Mz]]
        except:
            return None

    def calibrate_zero(self):
        vals = []
        for _ in range(20):
            d = self.read()
            if d: vals.append(d)
            time.sleep(0.05)
        if len(vals)>=5:
            self.zero_data = np.mean(np.array(vals), axis=0).tolist()

# ==================== Pressure Sensor ====================
class PressureSensor:
    def __init__(self):
        self.ser = None
        self.port = None
        self.auto_find_port()

    def auto_find_port(self):
        ports = list(serial.tools.list_ports.comports())
        for p,_,_ in ports:
            if p == "/dev/ttyUSB0": continue
            try:
                self.ser = serial.Serial(p, BAUDRATE_PRESS, timeout=0.01)
                self.port = p
                time.sleep(0.1)
                self.ser.reset_input_buffer()
                return
            except: continue
        raise Exception("Pressure sensor not found")

    def reconnect(self):
        try:
            if self.ser and self.ser.is_open: self.ser.close()
        except: pass
        time.sleep(0.2)
        self.auto_find_port()

    def read_data(self):
        if not self.ser or not self.ser.is_open: return None
        try:
            cmd = [0x55,0xAA,9,0,0x34,0,0xFB,0,0x1C,0,0,0xA8,0,0x35]
            self.ser.write(bytearray(cmd))
            time.sleep(0.005)
            resp = self.ser.read(256)
            idx = resp.find(b'\xaa\x55')
            if idx == -1 or len(resp[idx:])<182: return None
            return resp[idx+14:idx+14+168]
        except: return None

    def decode(self, raw):
        arr = []
        for i in range(0,168,2):
            arr.append(struct.unpack("<H", raw[i:i+2])[0])
        out = []
        for i in range(12):
            out.extend(arr[i*7:(i+1)*7])
        return out

# ==================== Buffer ====================
class TimestampedBuffer:
    def __init__(self, maxlen=300):
        self.buf = deque(maxlen=maxlen)
        self.lock = threading.Lock()
    def append(self, item):
        with self.lock: self.buf.append(item)
    def get_latest(self):
        with self.lock: return self.buf[-1] if self.buf else None
    def find_closest(self, ts):
        with self.lock:
            best = None
            best_dt = 1e9
            for item in self.buf:
                dt = abs(item["t"]-ts)
                if dt < best_dt:
                    best_dt = dt
                    best = item
            return best

# ==================== Threads ====================
class PressureReaderThread(threading.Thread):
    def __init__(self, sensor, buf, stop_event):
        super().__init__(daemon=True)
        self.sensor = sensor
        self.buf = buf
        self.stop = stop_event
        self.fail = 0
    def run(self):
        while not self.stop.is_set():
            t = time.perf_counter()
            raw = self.sensor.read_data()
            if raw is None:
                self.fail +=1
                if self.fail >=30:
                    self.sensor.reconnect()
                    self.fail=0
                time.sleep(0.002)
                continue
            try:
                data = self.sensor.decode(raw)
                self.buf.append({"t":t, "data":data})
                self.fail=0
            except: time.sleep(0.001)

class ForceReaderThread(threading.Thread):
    def __init__(self, sensor, buf, stop_event):
        super().__init__(daemon=True)
        self.sensor = sensor
        self.buf = buf
        self.stop = stop_event
        self.fail=0
    def run(self):
        while not self.stop.is_set():
            t = time.perf_counter()
            data = self.sensor.read()
            if data is None:
                self.fail +=1
                if self.fail >=30:
                    self.sensor.reconnect()
                    self.fail=0
                time.sleep(0.002)
                continue
            self.buf.append({"t":t, "data":data})
            self.fail=0

# ==================== Plot ====================
class RealTimePlot:
    def __init__(self):
        plt.rcParams['font.family'] = 'DejaVu Sans'
        self.rows = 12
        self.cols = 7
        self.fig = plt.figure(figsize=(16,12))
        gs_outer = GridSpec(4,2,width_ratios=[1,1],height_ratios=[6,1,1,1],hspace=0.2,wspace=0.3)
        gs_arrows = gs_outer[0,0].subgridspec(1,2,wspace=0.3)
        self.ax1 = plt.subplot(gs_arrows[0,0])
        self.ax1.set_xlim(0,1); self.ax1.set_ylim(0,1); self.ax1.set_aspect('equal'); self.ax1.axis('off')
        self.ax2 = plt.subplot(gs_arrows[0,1])
        self.ax2.set_xlim(0,1); self.ax2.set_ylim(0,1); self.ax2.set_aspect('equal'); self.ax2.axis('off')
        self.ax3a = plt.subplot(gs_outer[1,0])
        self.ax3b = plt.subplot(gs_outer[2,0])
        self.ax4 = plt.subplot(gs_outer[3,0])
        gs_right = gs_outer[:,1].subgridspec(1,2,wspace=0.3)
        self.ax5 = self.fig.add_subplot(gs_right[0,0])
        self.ax6 = self.fig.add_subplot(gs_right[0,1])
        self.adc_angle=0; self.adc_mag=0; self.force_angle=0; self.force_mag=0
        self.raw_adc_sum=0; self.raw_force_mag=0
        self.table_data=np.zeros((12,7))
        self.cop_x=0; self.cop_y=0
        self.r1=0;self.r2=11;self.c1=0;self.c2=6
        self.delta_CoP_x=0;self.delta_CoP_y=0
        self.base_CoP_x_for_plot=0;self.base_CoP_y_for_plot=0
        self.lock=threading.Lock()
        self.fixed_arrow=0.35
        self.frame_counter=0
        self.ani=FuncAnimation(self.fig,self.update_all,interval=PLOT_INTERVAL_MS,cache_frame_data=False)

    def set_data(self,adc_a,adc_m,f_a,f_m,diff_frame,raw_adc_sum,raw_force_mag,cop_x,cop_y,r1,r2,c1,c2,delta_CoP_x_val,delta_CoP_y_val,base_cop_x_plot,base_cop_y_plot):
        with self.lock:
            self.adc_angle=adc_a; self.adc_mag=adc_m
            self.force_angle=f_a; self.force_mag=f_m
            self.raw_adc_sum=raw_adc_sum; self.raw_force_mag=raw_force_mag
            self.table_data=diff_frame.reshape(12,7)
            self.cop_x=cop_x; self.cop_y=cop_y
            self.r1=r1;self.r2=r2;self.c1=c1;self.c2=c2
            self.delta_CoP_x=delta_CoP_x_val; self.delta_CoP_y=delta_CoP_y_val
            self.base_CoP_x_for_plot=base_cop_x_plot; self.base_CoP_y_for_plot=base_cop_y_plot
            self.frame_counter+=1
            diff=abs(adc_a-f_a)
            error=min(diff,360-diff)
            angle_error_history.append(error)
            adc_mag_history.append(adc_m)
            force_mag_history.append(f_m)
            frame_count_history.append(self.frame_counter)
            raw_adc_sum_history.append(raw_adc_sum)
            raw_force_mag_history.append(raw_force_mag)

    def update_all(self,frame):
        self.update_direction_arrows()
        self.update_magnitude_arrows()
        self.update_raw_adc_sum()
        self.update_raw_force_mag()
        self.update_angle_error()
        self.update_pressure_table()
        self.update_gradient_table()
        return []

    def update_direction_arrows(self):
        with self.lock: a=self.adc_angle; f=self.force_angle
        self.ax1.clear(); self.ax1.set_xlim(0,1); self.ax1.set_ylim(0,1); self.ax1.set_aspect('equal'); self.ax1.axis('off')
        th=np.deg2rad(a)
        self.ax1.arrow(0.5,0.5,0.4*np.cos(th),0.4*np.sin(th),head_width=0.12,fc='k',ec='k',lw=2.5)
        self.ax1.text(0.5,0.1,f"Moment Change: {a:.1f}°",ha='center',fontsize=8)
        thf=np.deg2rad(f)
        self.ax1.arrow(0.5,0.5,0.35*np.cos(thf),0.35*np.sin(thf),head_width=0.1,fc='r',ec='r',lw=2)
        self.ax1.text(0.5,0.9,f"Force: {f:.1f}°",ha='center',fontsize=8,color='red')

    def update_magnitude_arrows(self):
        with self.lock: a=self.adc_angle;m=self.adc_mag;fa=self.force_angle;fm=self.force_mag
        self.ax2.clear(); self.ax2.set_xlim(0,1); self.ax2.set_ylim(0,1); self.ax2.set_aspect('equal'); self.ax2.axis('off')
        th=np.deg2rad(a)
        l=min(m/5*0.45,0.45) if m>0.01 else 0
        self.ax2.arrow(0.5,0.5,l*np.cos(th),l*np.sin(th),head_width=0.12,fc='k',ec='k',lw=2.5)
        self.ax2.text(0.5,0.1,f"Mag: {m:.2f}",ha='center',fontsize=8)
        thf=np.deg2rad(fa)
        lf=min(abs(fm)/20*0.4,0.4) if abs(fm)>0.5 else 0
        if lf>0.02:
            self.ax2.arrow(0.5,0.5,lf*np.cos(thf),lf*np.sin(thf),head_width=0.1,fc='r',ec='r',lw=3)
        self.ax2.text(0.5,0.9,f"Force: {fm:.1f}N",ha='center',fontsize=8,color='red')

    def update_raw_adc_sum(self):
        if raw_adc_sum_history:
            xs=range(len(raw_adc_sum_history))
            ys=list(raw_adc_sum_history)
            self.ax3a.clear()
            self.ax3a.set_title("Raw ADC Sum")
            self.ax3a.grid(True,alpha=0.3)
            self.ax3a.plot(xs,ys,'b-',linewidth=1.5)

    def update_raw_force_mag(self):
        if raw_force_mag_history:
            xs=range(len(raw_force_mag_history))
            ys=list(raw_force_mag_history)
            self.ax3b.clear()
            self.ax3b.set_title("Raw Force Mag")
            self.ax3b.grid(True,alpha=0.3)
            self.ax3b.plot(xs,ys,'r-',linewidth=1.5)

    def update_angle_error(self):
        if angle_error_history:
            xs=range(len(angle_error_history))
            ys=list(angle_error_history)
            self.ax4.clear()
            self.ax4.set_title("Angle Error")
            self.ax4.set_ylim(0,180)
            self.ax4.grid(True,alpha=0.3)
            self.ax4.plot(xs,ys,'g-',linewidth=1.5)

    def update_pressure_table(self):
        with self.lock:
            data=self.table_data.copy()
            cx=self.cop_x; cy=self.cop_y
            r1=self.r1;r2=self.r2;c1=self.c1;c2=self.c2
            bcx=self.base_CoP_x_for_plot; bcy=self.base_CoP_y_for_plot
        self.ax5.clear(); self.ax5.set_title("Pressure Array")
        self.ax5.set_xlim(-0.5,6.5); self.ax5.set_ylim(11.5,-0.5); self.ax5.set_aspect('equal')
        vmax=np.max(data) if np.max(data)!=0 else 1
        norm=data/vmax
        cmap=plt.colormaps['Reds']
        colors=cmap(norm)
        cell=[[f"{v:.0f}" for v in row] for row in data]
        tbl=self.ax5.table(cellText=cell,cellColours=colors,cellLoc='center',bbox=[0,0,1,1])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        self.ax5.scatter(cx,cy,s=100,color='green')
        rect=Rectangle((c1-0.5,r1-0.5),c2-c1+1,r2-r1+1,ec='blue',fc='none',lw=2)
        self.ax5.add_patch(rect)

    def update_gradient_table(self):
        with self.lock: data=grad_table_data.copy()
        self.ax6.clear(); self.ax6.set_title("Gradient")
        self.ax6.set_xlim(-0.5,6.5); self.ax6.set_ylim(11.5,-0.5); self.ax6.set_aspect('equal')
        for r in range(12):
            for c in range(7):
                gx,gy=data[r,c]
                mg=np.hypot(gx,gy)
                if mg>1:
                    self.ax6.quiver(c,r,gx/mg,gy/mg,color='k',scale=3,width=0.015)

    def show(self):
        plt.tight_layout()
        plt.show()

# ==================== Collector ====================
class DataCollector:
    def __init__(self,force_sensor,press_sensor,csv_path):
        self.force_sensor=force_sensor
        self.press_sensor=press_sensor
        self.csv_path=csv_path
        self.running=True
        self.stop_event=threading.Event()
        self.press_buf=TimestampedBuffer(PRESS_BUFFER_SIZE)
        self.force_buf=TimestampedBuffer(FORCE_BUFFER_SIZE)
        self.plot=None
        self.start_time=None

    def set_plot(self,p):
        self.plot=p

    def start(self):
        self.press_thread=PressureReaderThread(self.press_sensor,self.press_buf,self.stop_event)
        self.force_thread=ForceReaderThread(self.force_sensor,self.force_buf,self.stop_event)
        self.press_thread.start()
        self.force_thread.start()
        threading.Thread(target=self.run_collect,daemon=True).start()

    def run_collect(self):
        with open(self.csv_path,'w',newline='') as f:
            w=csv.writer(f)
            w.writerow(["ts","ms"]+[f"ch{i}" for i in range(84)]+["Fx","Fy","Fz","Mx","My","Mz","pt","ft","dt","adca","adcm","fa","fm","cx","cy","r1","r2","c1","c2"])
            period=1/TARGET_HZ
            while self.running:
                t0=time.perf_counter()
                if not self.start_time: self.start_time=t0
                ms=int((t0-self.start_time)*1000)
                p=self.press_buf.get_latest()
                if not p: continue
                f=self.force_buf.find_closest(p["t"])
                if not f or abs(p["t"]-f["t"])>MAX_SYNC_DT: continue
                pd=p["data"]
                fd=f["data"]
                s=np.sum(pd)
                df=subtract_baseline(pd)
                fx,fy,mg,cx,cy,r1,r2,c1,c2,dx,dy,bcx,bcy=compute_gradient_in_region(df)
                aa,_=compute_gradient_angle_single(fx,fy)
                fa,fm=compute_force_angle(fd[0],fd[1])
                full_time_list.append(ms)
                full_adc_mag_list.append(mg)
                full_force_mag_list.append(fm)
                if self.plot:
                    self.plot.set_data(aa,mg,fa,fm,df,s,np.hypot(fd[0],fd[1]),cx,cy,r1,r2,c1,c2,dx,dy,bcx,bcy)
                w.writerow([time.strftime("%Y%m%d%H%M%S"),ms]+pd+fd+[round(p["t"],6),round(f["t"],6),round(abs(p["t"]-f["t"])*1000,3),aa,mg,fa,fm,cx,cy,r1,r2,c1,c2])
                f.flush()
                sl=max(0,period-(time.perf_counter()-t0))
                time.sleep(sl)

    def stop(self):
        self.running=False
        self.stop_event.set()
        time.sleep(0.3)

# ==================== Plot Full ====================
def plot_full_magnitude_curve():
    plt.figure(figsize=(10,6))
    plt.subplot(211)
    plt.plot(full_time_list,full_adc_mag_list,'b-',label='Moment Change Mag')
    plt.title("Moment Change")
    plt.grid(True)
    plt.subplot(212)
    plt.plot(full_time_list,full_force_mag_list,'r-',label='Force Mag')
    plt.title("Force")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR,"full_curve.png"))
    plt.show()

# ==================== MAIN ====================
if __name__=="__main__":
    os.makedirs(SAVE_DIR,exist_ok=True)
    i=1
    while os.path.exists(os.path.join(SAVE_DIR,f"data_{i}.csv")):i+=1
    path=os.path.join(SAVE_DIR,f"data_{i}.csv")
    fs=SixAxisForceSensor()
    ps=PressureSensor()
    fs.calibrate_zero()
    plot=RealTimePlot()
    coll=DataCollector(fs,ps,path)
    coll.set_plot(plot)
    coll.start()
    print("✅ 二阶矩变化量方向已运行，方向完全跟随滑动")
    plot.show()
    coll.stop()
    plot_full_magnitude_curve()