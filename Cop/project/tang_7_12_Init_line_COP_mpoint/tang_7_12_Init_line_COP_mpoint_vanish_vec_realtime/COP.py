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

    # 过滤掉与更高峰相邻的次高峰
    filtered_peaks = []
    for py, px, pval in peaks:
        too_close = False
        for fpy, fpx, _ in filtered_peaks:
            if abs(py - fpy) + abs(px - fpx) <= 1:  # 4-邻域
                too_close = True
                break
        if not too_close:
            filtered_peaks.append((py, px, pval))

    labeled = np.zeros((rows, cols), dtype=np.int32)

    # 优先队列：(-pressure, y, x, label)，高压力先处理
    heap = []

    for label, (py, px, _) in enumerate(filtered_peaks, 1):
        labeled[py, px] = label
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = py + dy, px + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if labeled[ny, nx] == 0 and frame2d[ny, nx] >= REGION_GROW_THRESHOLD:
                    heapq.heappush(heap, (-frame2d[ny, nx], ny, nx, label))

    # 优先处理压力值高的 cell → 波谷处自然形成边界
    while heap:
        neg_val, y, x, label = heapq.heappop(heap)
        if labeled[y, x] != 0:
            continue
        labeled[y, x] = label
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < rows and 0 <= nx < cols:
                if labeled[ny, nx] == 0 and frame2d[ny, nx] >= REGION_GROW_THRESHOLD:
                    heapq.heappush(heap, (-frame2d[ny, nx], ny, nx, label))

    return labeled, len(filtered_peaks)


# ===================== 邻接区域合并 =====================
MERGE_RATIO = 0.6   # 边界压力 ÷ min(两峰) > 此值 的 cell 达到一定数量时合并
MERGE_MIN_CELLS = 2  # 边界中至少 N 个 cell 超过阈值才触发合并
NARROW_NECK_MAX = 6  # 边界总 cell 数 ≤ 此值 → 窄颈（两圈相切）→ 不合并
NARROW_NECK_MIN_CELLS = 4  # 两区域 cell 数都 ≥ 此值 → 才触发窄颈判断


def _merge_adjacent_regions(labeled, frame2d):
    """
    合并应属同一接触的相邻区域。
    除压力判断外，增加窄颈判断：
    - 边界总 cell 数 ≤ NARROW_NECK_MAX → 窄颈（两圆相切）→ 保留为独立 COP
    - 边界宽 + 高压力 → 材料噪声 → 合并
    """
    n = labeled.max()
    if n <= 1:
        return labeled, n

    rows, cols = labeled.shape

    peak_vals = {}
    cell_counts = {}
    for lbl in range(1, n + 1):
        mask = labeled == lbl
        peak_vals[lbl] = frame2d[mask].max()
        cell_counts[lbl] = np.sum(mask)

    high_count = {}   # (min_lbl, max_lbl) → 高值 cell 计数
    total_count = {}  # (min_lbl, max_lbl) → 边界总 cell 计数
    for y in range(rows):
        for x in range(cols):
            l1 = labeled[y, x]
            if l1 == 0:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < rows and 0 <= nx < cols:
                    l2 = labeled[ny, nx]
                    if l2 != 0 and l2 != l1:
                        pair = (min(l1, l2), max(l1, l2))
                        boundary_p = min(frame2d[y, x], frame2d[ny, nx])
                        min_peak = min(peak_vals[l1], peak_vals[l2])
                        total_count[pair] = total_count.get(pair, 0) + 1
                        if min_peak > 0 and boundary_p / min_peak > MERGE_RATIO:
                            high_count[pair] = high_count.get(pair, 0) + 1

    # 高值达标 → 候选合并
    # 窄颈反制：仅当两区域都足够大（非噪声）且边界窄 → 不合并
    pairs_to_merge = set()
    for pair, cnt in high_count.items():
        if cnt < MERGE_MIN_CELLS:
            continue
        if total_count.get(pair, 0) <= NARROW_NECK_MAX:
            l1, l2 = pair
            if cell_counts.get(l1, 0) >= NARROW_NECK_MIN_CELLS and \
               cell_counts.get(l2, 0) >= NARROW_NECK_MIN_CELLS:
                continue  # 窄颈 + 两区域都大 → 独立接触 → 不合并
        pairs_to_merge.add(pair)

    if not pairs_to_merge:
        return labeled, n

    # Union-Find
    parent = {i: i for i in range(1, n + 1)}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in pairs_to_merge:
        union(a, b)

    # 重新映射标签
    new_label = {}
    next_lbl = 1
    for lbl in range(1, n + 1):
        root = find(lbl)
        if root not in new_label:
            new_label[root] = next_lbl
            next_lbl += 1

    merged = np.zeros_like(labeled)
    for y in range(rows):
        for x in range(cols):
            lbl = labeled[y, x]
            if lbl > 0:
                merged[y, x] = new_label[find(lbl)]

    return merged, len(new_label)


# ===================== 区域追踪类 =====================
MERGE_SEARCH_DIST = 3.0              # 合并搜索范围（比 MAX_MATCH_DIST 大）
MAX_MERGED_FRAMES = 30               # 合并状态下最长存活帧数


class RegionTracker:
    __slots__ = ('region_id', 'first_contact_cop_x', 'first_contact_cop_y',
                 'contact_initialized', 'initial_cop_x_buffer', 'initial_cop_y_buffer',
                 'total_pressure_low_counter', 'last_cop_x', 'last_cop_y',
                 'frames_since_seen', 'prev_cop_x', 'prev_cop_y',
                 'merged_with', 'frames_merged')

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
        self.prev_cop_x = None
        self.prev_cop_y = None
        self.merged_with = set()
        self.frames_merged = 0

    def reset(self):
        self.first_contact_cop_x = None
        self.first_contact_cop_y = None
        self.contact_initialized = False
        self.initial_cop_x_buffer.clear()
        self.initial_cop_y_buffer.clear()
        self.total_pressure_low_counter = 0
        self.prev_cop_x = None
        self.prev_cop_y = None
        self.merged_with.clear()
        self.frames_merged = 0


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
    # 合并压阻材料不均匀产生的伪峰
    labeled, num_labels = _merge_adjacent_regions(labeled, frame2d)

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

    # 移除失配超时的追踪区域（合并状态超时更长）
    with tracked_regions_lock:
        for rid in unmatched_tracked:
            tracker = tracked_regions.get(rid)
            if tracker:
                tracker.frames_since_seen += 1
        stale = [rid for rid, t in tracked_regions.items()
                 if t.frames_since_seen > (MAX_MERGED_FRAMES if t.merged_with else 3)]
        for rid in stale:
            del tracked_regions[rid]

    # ---- 检测合并：未匹配追踪器是否与某已匹配追踪器合并到同区域 ----
    merge_results = []
    for rid in unmatched_tracked:
        tracker = tracked_regions.get(rid)
        if tracker is None or tracker.last_cop_x is None:
            continue
        best_det = None
        best_dist = MERGE_SEARCH_DIST
        for i, det in enumerate(detected_regions):
            d = np.hypot(det['cop_x'] - tracker.last_cop_x,
                         det['cop_y'] - tracker.last_cop_y)
            if d < best_dist:
                best_dist = d
                best_det = i
        if best_det is None:
            continue
        other_rid = None
        for (di, rid2) in matched_pairs:
            if di == best_det:
                other_rid = rid2
                break
        if other_rid is not None:
            with tracked_regions_lock:
                t_other = tracked_regions.get(other_rid)
                if t_other:
                    tracker.merged_with.add(other_rid)
                    t_other.merged_with.add(rid)
            if tracker.prev_cop_x is not None:
                pred_x = tracker.last_cop_x + (tracker.last_cop_x - tracker.prev_cop_x)
                pred_y = tracker.last_cop_y + (tracker.last_cop_y - tracker.prev_cop_y)
            else:
                pred_x = tracker.last_cop_x
                pred_y = tracker.last_cop_y
            merge_results.append((pred_x, pred_y, rid))

    # ---- 构建 (det_idx, rid) 对：已匹配 + 新创建 ----
    process_pairs = []
    for det_idx, rid in matched_pairs:
        process_pairs.append((det_idx, rid))
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

        if tracker.last_cop_x is not None:
            tracker.prev_cop_x = tracker.last_cop_x
            tracker.prev_cop_y = tracker.last_cop_y
        tracker.last_cop_x = cop_x_val
        tracker.last_cop_y = cop_y_val
        tracker.frames_since_seen = 0
        tracker.frames_merged = 0
        tracker.merged_with.clear()

        if not tracker.contact_initialized:
            _check_region_stability(tracker, cop_x_val, cop_y_val)

        if tracker.contact_initialized:
            dx = cop_x_val - tracker.first_contact_cop_x
            dy = tracker.first_contact_cop_y - cop_y_val
            bx = tracker.first_contact_cop_x
            by = tracker.first_contact_cop_y
        else:
            dx = 0.0; dy = 0.0
            bx = cop_x_val; by = cop_y_val

        region_results.append({
            'cop_x': cop_x_val, 'cop_y': cop_y_val,
            'delta_cop_x': dx, 'delta_cop_y': dy,
            'base_cop_x': bx, 'base_cop_y': by,
            'total_pressure': det['total_pressure'],
            'region_id': rid,
            'is_initialized': tracker.contact_initialized,
        })

    # ---- 为合并追踪器生成预测结果 ----
    for pred_x, pred_y, rid in merge_results:
        with tracked_regions_lock:
            tracker = tracked_regions.get(rid)
        if tracker is None:
            continue
        tracker.frames_merged += 1
        if tracker.contact_initialized:
            dx = pred_x - tracker.first_contact_cop_x
            dy = tracker.first_contact_cop_y - pred_y
            bx = tracker.first_contact_cop_x
            by = tracker.first_contact_cop_y
        else:
            dx = 0.0; dy = 0.0
            bx = pred_x; by = pred_y
        region_results.append({
            'cop_x': pred_x, 'cop_y': pred_y,
            'delta_cop_x': dx, 'delta_cop_y': dy,
            'base_cop_x': bx, 'base_cop_y': by,
            'total_pressure': 0.0,
            'region_id': rid,
            'is_initialized': tracker.contact_initialized,
            'is_predicted': True,
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
