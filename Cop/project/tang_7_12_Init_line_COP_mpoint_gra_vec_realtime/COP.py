"""
CoP 压力中心计算核心模块
功能：基线减除、多区域CoP计算、初始稳定点判断、方向向量滤波
"""

import heapq
import numpy as np
from collections import deque
import threading


# ===================== 算法参数 =====================
COP_STABILITY_FRAMES_REQUIRED = 5       # 初始稳定帧数
TOTAL_PRESSURE_LOW_THRESHOLD = 500      # 全局低压判定阈值
SENSOR_ROWS = 12                        # 传感器阵列行数
SENSOR_COLS = 7                         # 传感器阵列列数

# ===================== 多区域检测参数 =====================
PEAK_MIN_HEIGHT = 1000                 # 局部极大值最小高度（找种子点）
REGION_GROW_THRESHOLD = 200            # 区域生长最小压力（低于此值的 cell 不归属任何区域）
MIN_REGION_PRESSURE = 500              # 区域最小总压力
MAX_MATCH_DIST = 2.0                   # 帧间匹配最大距离
SHALLOW_VALLEY_RATIO = 0.35            # 谷底压力/较低峰 > 此值 → 浅谷,进入梯度判断
MIN_SEP_GRADIENT = 800                 # 浅谷中若最大梯度 > 此值 → 真边缘 → 保留独立 COP

# ===================== 直线方向稳定判断参数 =====================
LINE_DIST_THRESHOLD = 0.1               # 点到直线最大允许距离 (CoP单位)
DIR_DOT_THRESHOLD = 0.7                 # 方向一致性最小点积 cos(夹角)


# ===================== 线程安全全局状态 =====================
first_frame = None                      # 第一帧基线
first_frame_lock = threading.Lock()     # 线程锁

tracked_regions = {}                    # region_id → RegionTracker
tracked_regions_lock = threading.Lock()
next_region_id = 0                      # 全局自增 ID

total_pressure_low_counter = 0          # 全局压力低于阈值计数器

adc_filtered_dir = None                 # 滤波后的方向向量（保留兼容）
grad_table_data = np.zeros((12, 7, 2))  # 梯度表（用于绘图）
grad_table_lock = threading.Lock()


# ===================== 峰间采样 =====================
def _sample_line(arr2d, y1, x1, y2, x2):
    """沿两点连线采样，返回路径上所有值。"""
    steps = max(abs(y2 - y1), abs(x2 - x1))
    if steps == 0:
        return [arr2d[y1, x1]]
    vals = []
    for i in range(steps + 1):
        t = i / steps
        y = int(round(y1 + t * (y2 - y1)))
        x = int(round(x1 + t * (x2 - x1)))
        if 0 <= y < arr2d.shape[0] and 0 <= x < arr2d.shape[1]:
            vals.append(arr2d[y, x])
    return vals


def _valley_between(arr2d, y1, x1, y2, x2):
    """两点连线上的最小值。"""
    vals = _sample_line(arr2d, y1, x1, y2, x2)
    return min(vals) if vals else 0.0


# ===================== 峰值+区域生长分割 =====================
def _segment_by_peaks(frame2d):
    """
    基于局部极大值的区域生长分割（watershed-by-propagation）。
    1. 找所有 4-邻域局部极大值（高于 PEAK_MIN_HEIGHT）作为种子
    2. 从各种子同时 BFS，压力值高的 cell 优先处理
    3. 即使弹性体在两峰间产生压力桥接，每个 cell 也会归入"先到达的峰"
    返回 (labeled_array, num_regions)。
    """
    rows, cols = frame2d.shape

    # 找局部极大值（4-邻域内没有比它更高的）
    peaks = []
    for y in range(rows):
        for x in range(cols):
            val = frame2d[y, x]
            if val < PEAK_MIN_HEIGHT:
                continue
            is_peak = True
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    if frame2d[ny, nx] > val:
                        is_peak = False
                        break
            if is_peak:
                # 检查是否与已找到的峰相邻（相邻峰只保留更高的）
                peaks.append((y, x, val))

    if not peaks:
        return np.zeros((rows, cols), dtype=np.int32), 0

    # 按高度降序排列
    peaks.sort(key=lambda p: p[2], reverse=True)

    # 计算梯度幅值（分水岭地形 + 种子过滤依据）
    grad_mag = np.zeros((rows, cols), dtype=np.float32)
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            gx = float(frame2d[y, x + 1]) - float(frame2d[y, x - 1])
            gy = float(frame2d[y + 1, x]) - float(frame2d[y - 1, x])
            grad_mag[y, x] = np.sqrt(gx * gx + gy * gy)

    # 种子过滤：浅谷 + 谷底低梯度 → 噪声 → 合并
    filtered_peaks = []
    for py, px, pval in peaks:
        is_subpeak = False
        for fpy, fpx, fval in filtered_peaks:
            valley = _valley_between(frame2d, py, px, fpy, fpx)
            if valley / pval > SHALLOW_VALLEY_RATIO:
                # 定位谷底 cell（沿线压力最低点），查其梯度
                steps = max(abs(py - fpy), abs(px - fpx))
                vy, vx = py, px
                if steps > 0:
                    for i in range(steps + 1):
                        t = i / steps
                        y = int(round(py + t * (fpy - py)))
                        x = int(round(px + t * (fpx - px)))
                        if 0 <= y < rows and 0 <= x < cols:
                            if frame2d[y, x] < valley * 1.01:  # 严格匹配谷底
                                vy, vx = y, x
                if grad_mag[vy, vx] < MIN_SEP_GRADIENT:
                    is_subpeak = True
                    break
        if not is_subpeak:
            filtered_peaks.append((py, px, pval))

    labeled = np.zeros((rows, cols), dtype=np.int32)

    # 分水岭涨水：(梯度幅值, y, x, label)，低梯度先淹没
    heap = []

    for label, (py, px, _) in enumerate(filtered_peaks, 1):
        labeled[py, px] = label
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if labeled[ny, nx] == 0 and frame2d[ny, nx] >= REGION_GROW_THRESHOLD:
                    heapq.heappush(heap, (grad_mag[ny, nx], ny, nx, label))

    # 低梯度（平坦区）先淹 → 高梯度（边缘/窄谷）后淹 → 分水线
    while heap:
        gval, y, x, label = heapq.heappop(heap)
        if labeled[y, x] != 0:
            continue
        labeled[y, x] = label
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if labeled[ny, nx] == 0 and frame2d[ny, nx] >= REGION_GROW_THRESHOLD:
                    heapq.heappush(heap, (grad_mag[ny, nx], ny, nx, label))

    return labeled, len(filtered_peaks)


# ===================== 区域边界提取 =====================
def _extract_region_boundaries(labeled):
    """提取每个区域的边界 cell 坐标列表，用于可视化描边。"""
    rows, cols = labeled.shape
    boundaries = {}
    for y in range(rows):
        for x in range(cols):
            lbl = labeled[y, x]
            if lbl == 0:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= rows or nx < 0 or nx >= cols or labeled[ny, nx] != lbl:
                    boundaries.setdefault(lbl, []).append((y, x))
                    break
    return boundaries


# ===================== 区域追踪类 =====================
class RegionTracker:
    __slots__ = ('region_id', 'first_contact_cop_x', 'first_contact_cop_y',
                 'contact_initialized', 'initial_cop_x_buffer', 'initial_cop_y_buffer',
                 'total_pressure_low_counter', 'last_cop_x', 'last_cop_y',
                 'frames_since_seen')

    def __init__(self, region_id):
        self.region_id = region_id
        self.first_contact_cop_x = None
        self.first_contact_cop_y = None
        self.contact_initialized = False
        self.initial_cop_x_buffer = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)
        self.initial_cop_y_buffer = deque(maxlen=COP_STABILITY_FRAMES_REQUIRED)
        self.total_pressure_low_counter = 0
        self.last_cop_x = None
        self.last_cop_y = None
        self.frames_since_seen = 0

    def reset(self):
        self.first_contact_cop_x = None
        self.first_contact_cop_y = None
        self.contact_initialized = False
        self.initial_cop_x_buffer.clear()
        self.initial_cop_y_buffer.clear()
        self.total_pressure_low_counter = 0


# ===================== 帧间区域匹配 =====================
def _match_regions(detected_list, max_dist=MAX_MATCH_DIST):
    """
    贪心最近邻匹配：已追踪区域 ↔ 当前帧检测区域。
    detected_list: [{'cop_x','cop_y','total_pressure'}, ...]
    返回: (matched_pairs, unmatched_det_idx, unmatched_tracked_ids)
    """
    global tracked_regions

    matched = []
    unmatched_det = set(range(len(detected_list)))

    for rid, tracker in list(tracked_regions.items()):
        if tracker.last_cop_x is None:
            continue
        best_idx = None
        best_dist = max_dist
        for i in unmatched_det:
            d = np.hypot(detected_list[i]['cop_x'] - tracker.last_cop_x,
                         detected_list[i]['cop_y'] - tracker.last_cop_y)
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            matched.append((best_idx, rid))
            unmatched_det.discard(best_idx)

    matched_tracked_ids = {m[1] for m in matched}
    unmatched_tracked = [rid for rid in tracked_regions
                         if rid not in matched_tracked_ids]

    return matched, list(unmatched_det), unmatched_tracked


# ===================== 基线减除 =====================
def set_baseline(frame):
    """手动设置基线（多帧平均后调用），替代自动捕获的第一帧"""
    global first_frame
    frame_arr = np.array(frame, dtype=np.float32).flatten()
    with first_frame_lock:
        first_frame = frame_arr.copy()


def subtract_baseline(current_frame):
    """用基线减去背景；若未手动校准则自动捕获第一帧"""
    global first_frame
    current_frame = np.array(current_frame, dtype=np.float32).flatten()

    with first_frame_lock:
        if first_frame is None:
            first_frame = current_frame.copy()

    diff = current_frame - first_frame
    return np.clip(diff, 0, None)


# ===================== 重置所有区域 =====================
def reset_all_regions():
    global tracked_regions, total_pressure_low_counter, adc_filtered_dir

    with tracked_regions_lock:
        tracked_regions.clear()

    total_pressure_low_counter = 0
    adc_filtered_dir = None

    with grad_table_lock:
        grad_table_data.fill(0)

# 保留旧名兼容
reset_cop_state = reset_all_regions


# ===================== 区域稳定性判断 =====================
def _check_region_stability(tracker, cop_x, cop_y):
    """对单个区域的 CoP 序列做直线稳定性判断。返回 True 表示已稳定初始化。"""
    tracker.initial_cop_x_buffer.append(cop_x)
    tracker.initial_cop_y_buffer.append(cop_y)

    is_stable = True

    if len(tracker.initial_cop_x_buffer) >= 2:
        p0x, p0y = tracker.initial_cop_x_buffer[0], tracker.initial_cop_y_buffer[0]
        p1x, p1y = tracker.initial_cop_x_buffer[1], tracker.initial_cop_y_buffer[1]

        dir_ref_x = p1x - p0x
        dir_ref_y = p1y - p0y
        dir_ref_len = np.hypot(dir_ref_x, dir_ref_y)

        if dir_ref_len < 1e-4:
            is_stable = False
        else:
            norm_dir_ref_x = dir_ref_x / dir_ref_len
            norm_dir_ref_y = dir_ref_y / dir_ref_len

            for i in range(2, len(tracker.initial_cop_x_buffer)):
                cx_ = tracker.initial_cop_x_buffer[i]
                cy_ = tracker.initial_cop_y_buffer[i]
                px = tracker.initial_cop_x_buffer[i - 1]
                py = tracker.initial_cop_y_buffer[i - 1]

                cross = abs((p1x - p0x) * (cy_ - p0y) - (p1y - p0y) * (cx_ - p0x))
                line_dist = cross / dir_ref_len
                if line_dist > LINE_DIST_THRESHOLD:
                    is_stable = False
                    break

                seg_x = cx_ - px
                seg_y = cy_ - py
                seg_len = np.hypot(seg_x, seg_y)
                if seg_len > 1e-4:
                    dot = norm_dir_ref_x * (seg_x / seg_len) + norm_dir_ref_y * (seg_y / seg_len)
                    if dot < DIR_DOT_THRESHOLD:
                        is_stable = False
                        break

    if not is_stable:
        tracker.initial_cop_x_buffer.clear()
        tracker.initial_cop_y_buffer.clear()
        tracker.initial_cop_x_buffer.append(cop_x)
        tracker.initial_cop_y_buffer.append(cop_y)
        return False

    if len(tracker.initial_cop_x_buffer) == COP_STABILITY_FRAMES_REQUIRED:
        tracker.first_contact_cop_x = tracker.initial_cop_x_buffer[0]
        tracker.first_contact_cop_y = tracker.initial_cop_y_buffer[0]
        tracker.contact_initialized = True
        tracker.initial_cop_x_buffer.clear()
        tracker.initial_cop_y_buffer.clear()
        return True

    return False


# ===================== 核心CoP计算（多区域）=====================
def compute_pressure_direction(baseline_subtracted_frame):
    """
    输入：基线减除后的84通道压力数据
    输出：dict {
        'regions': [{cop_x, cop_y, delta_cop_x, delta_cop_y,
                     base_cop_x, base_cop_y, total_pressure,
                     region_id, is_initialized, index}, ...]
        'grid_info': (min_y, max_y, min_x, max_x),
        'is_contact': bool,
    }
    regions 按 total_pressure 降序排列，index 为排序后序号(0=主COP)
    """
    global tracked_regions, next_region_id
    global total_pressure_low_counter
    global adc_filtered_dir, grad_table_data

    rows, cols = SENSOR_ROWS, SENSOR_COLS
    frame_flat = np.asarray(baseline_subtracted_frame, dtype=np.float32).flatten()
    frame2d = frame_flat.reshape(rows, cols)

    # ---- 计算梯度（用于可视化）----
    grad = np.zeros((rows, cols, 2), dtype=np.float32)
    for y in range(rows):
        for x in range(cols):
            val = frame2d[y, x]
            left = frame2d[y, x - 1] if x - 1 >= 0 else val
            right = frame2d[y, x + 1] if x + 1 < cols else val
            up = frame2d[y - 1, x] if y - 1 >= 0 else val
            down = frame2d[y + 1, x] if y + 1 < rows else val
            grad[y, x] = (right - left, up - down)
    with grad_table_lock:
        grad_table_data[:] = grad[:]

    # ---- 峰值+区域生长分割 ----
    labeled, num_labels = _segment_by_peaks(frame2d)
    region_boundaries = _extract_region_boundaries(labeled)

    x_grid = np.tile(np.arange(cols), (rows, 1))
    y_grid = np.repeat(np.arange(rows), cols).reshape(rows, cols)

    detected_regions = []
    for lbl in range(1, num_labels + 1):
        mask = (labeled == lbl)
        region_pressure = np.sum(frame2d[mask])
        if region_pressure < MIN_REGION_PRESSURE:
            continue
        cop_x_val = np.sum(frame2d[mask] * x_grid[mask]) / region_pressure
        cop_y_val = np.sum(frame2d[mask] * y_grid[mask]) / region_pressure
        detected_regions.append({
            'cop_x': cop_x_val,
            'cop_y': cop_y_val,
            'total_pressure': region_pressure,
            'mask': mask,
            'boundary_cells': region_boundaries.get(lbl, []),
        })

    # ---- 全局接触判断 ----
    global_total = np.sum(frame2d)
    if len(detected_regions) == 0 or global_total < TOTAL_PRESSURE_LOW_THRESHOLD:
        total_pressure_low_counter += 1
    else:
        total_pressure_low_counter = 0

    if total_pressure_low_counter >= COP_STABILITY_FRAMES_REQUIRED:
        reset_all_regions()

    # ---- 空帧快速返回 ----
    if global_total == 0 or len(detected_regions) == 0:
        return {
            'regions': [],
            'grid_info': (0, rows - 1, 0, cols - 1),
            'is_contact': False,
        }

    # ---- 帧间匹配 ----
    matched_pairs, unmatched_det, unmatched_tracked = _match_regions(detected_regions)

    # 移除失配超时的追踪区域
    with tracked_regions_lock:
        for rid in unmatched_tracked:
            tracker = tracked_regions.get(rid)
            if tracker:
                tracker.frames_since_seen += 1
        # 移除超过 3 帧未见的区域
        stale = [rid for rid, t in tracked_regions.items() if t.frames_since_seen > 3]
        for rid in stale:
            del tracked_regions[rid]

    # ---- 构建 (det_idx, rid) 对：已匹配 + 新创建 ----
    process_pairs = []  # [(det_idx, rid), ...]

    # 已匹配的
    for det_idx, rid in matched_pairs:
        process_pairs.append((det_idx, rid))

    # 为新检测区域创建追踪器并加入处理队列
    for i in unmatched_det:
        with tracked_regions_lock:
            rid = next_region_id
            next_region_id += 1
            tracked_regions[rid] = RegionTracker(rid)
        process_pairs.append((i, rid))

    # ---- 更新追踪器、计算稳定性与偏移 ----
    region_results = []

    for det_idx, rid in process_pairs:
        det = detected_regions[det_idx]
        with tracked_regions_lock:
            tracker = tracked_regions.get(rid)
        if tracker is None:
            continue

        cop_x_val = det['cop_x']
        cop_y_val = det['cop_y']

        tracker.last_cop_x = cop_x_val
        tracker.last_cop_y = cop_y_val
        tracker.frames_since_seen = 0

        # 稳定性判断
        if not tracker.contact_initialized:
            _check_region_stability(tracker, cop_x_val, cop_y_val)

        # 计算偏移量
        if tracker.contact_initialized:
            dx = cop_x_val - tracker.first_contact_cop_x
            dy = tracker.first_contact_cop_y - cop_y_val
            bx = tracker.first_contact_cop_x
            by = tracker.first_contact_cop_y
        else:
            dx = 0.0
            dy = 0.0
            bx = cop_x_val
            by = cop_y_val

        region_results.append({
            'cop_x': cop_x_val,
            'cop_y': cop_y_val,
            'delta_cop_x': dx,
            'delta_cop_y': dy,
            'base_cop_x': bx,
            'base_cop_y': by,
            'total_pressure': det['total_pressure'],
            'region_id': rid,
            'is_initialized': tracker.contact_initialized,
            'boundary_cells': det.get('boundary_cells', []),
        })

    # ---- 按 total_pressure 降序排列 ----
    region_results.sort(key=lambda r: r['total_pressure'], reverse=True)
    for idx, r in enumerate(region_results):
        r['index'] = idx

    return {
        'regions': region_results,
        'grid_info': (0, rows - 1, 0, cols - 1),
        'is_contact': True,
    }
