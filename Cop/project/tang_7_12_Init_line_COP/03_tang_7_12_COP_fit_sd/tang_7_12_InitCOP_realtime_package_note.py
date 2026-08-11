"""压阻传感器 CoP 角度估计（PRSensorAngle 类，支持多 sensor 实例）"""

import heapq
import numpy as np
import threading
from collections import deque
from scipy.ndimage import label, generate_binary_structure


class PRSensorAngle:
    """压阻传感器：12×7（默认）PZT 阵列的 CoP 角度估计"""

    def __init__(self, rows: int = 12, cols: int = 7,
                 total_threshold_factor: float = 3, pixel_threshold_factor: float = 5,
                 collect_frames: int = 10,
                 stability_frames: int = 5,
                 reset_at_frame: int = 0,
                 refine_cnt: int = 10,
                 refine_distance: float = 0.1,
                 merge_ratio: float = 0.8, merge_min_cells: int = 2,
                 region_match_dist: float = 10.0,
                 region_min_area: int = 4,
                 region_peak_ratio: float = 0.6, region_peak_dist: int = 20):
        """
        构造一个 PRSensorAngle 角度估计实例。

        :param rows: 传感器阵列行数（默认 12）。
        :param cols: 传感器阵列列数（默认 7），输入 ADC 序列长度 = rows * cols。
        :param total_threshold_factor: 动态低压阈值倍数：thresh = total_threshold_factor × mean(前 collect_frames 帧总压力)。
                                值越大，越难被判定为"低压"。
        :param pixel_threshold_factor: 逐像素动态阈值倍数：thresh = pixel_threshold_factor × 逐像素背景平均。
                                值越大，接触判定越严格。
        :param collect_frames: 学习动态阈值用的样本窗口大小（默认 20 帧）。
                              0 = 不学阈值, _total_thresh 直接 = 0（首帧非零压力锁 origin）。
        :param stability_frames: 阈值确定后，连续多少帧低压自动 reset_origin（默认 5）。
        :param reset_at_frame: 在第 N 帧自动调一次 reset_origin() 后重新锁新 origin；
                                0 = 禁用自动 reset（默认）。
        :param refine_cnt: origin 锁定后，连续多少帧 CoP 稳定触发"二次精修"
                                        （调用 reset_origin() 让下帧自然重新锁）。默认 10。
                                        0 = 禁用精修。
        :param refine_distance: 判定"稳定"的 CoP 与候选点欧氏距离阈值（cells，默认 0.1）。
                                      0 = 禁用精修。
        :param merge_ratio: 浅谷合并阈值：边界 cell 压力 ÷ min(两 region 峰值) > 此值视为浅谷（默认 0.8）。
        :param merge_min_cells: 浅谷合并最少满足 cell 数：边界中满足条件的 cell 数 ≥ 此值才合并（默认 2）。
        :param region_match_dist: region 帧间追踪质心最大匹配距离（cells，默认 10.0）：质心距离 ≤ 此值继承稳定 id。
        :param region_min_area: region 最小面积（cell 数，默认 4）：分割/追踪/合并统一过滤 < 此值的碎片 region。
        :param region_peak_ratio: 附属峰判据：峰高 < 此比例 × 更高峰（默认 0.6）且距离近 → 不成种子。
        :param region_peak_dist: 附属峰判据：与更高峰距离（曼哈顿, cells）< 此值（默认 5）且高度低 → 不成种子。
        """
        self.rows = rows
        self.cols = cols
        self.total_threshold_factor = total_threshold_factor
        self.pixel_threshold_factor = pixel_threshold_factor
        self.collect_frames = collect_frames
        self.stability_frames = stability_frames
        self._reset_at_frame = reset_at_frame
        self._frame_count = 0

        # 二次精修（post-refine）参数与状态
        # _refine_enabled 派生标志在构造时算一次, 运行时不再改
        self._refine_cnt = refine_cnt
        self._refine_distance = refine_distance
        self._refine_enabled = (refine_cnt > 0) and (refine_distance > 0)
        # 浅谷合并参数（_compute_region 用）
        self._merge_ratio = merge_ratio
        self._merge_min_cells = merge_min_cells
        # regions 帧间追踪: 质心匹配 + COP 落点判据（滑动继承）, stale 3 帧缓冲
        self._region_match_dist = region_match_dist
        # regions 最小面积: _compute_region 分割统一过滤
        self._region_min_area = region_min_area
        # 附属峰判据: 高度 < ratio×更高峰 且 距离 < dist → 不成种子
        self._region_peak_ratio = region_peak_ratio
        self._region_peak_dist = region_peak_dist
        # 候选点/稳定计数/已精修标志 —— 跟随 origin 生命周期
        self._refine_cand_x = None
        self._refine_cand_y = None
        self._refine_curr = 0
        self._refined = False
        # threshold相关
        self._pressure_history = deque(maxlen=collect_frames)
        self._pixel_avg_buffer = None
        self._pixel_cnt = 0
        self._total_thresh = None
        self._pixel_thresh = None

        self._lock = threading.Lock()

        self._origin_x = None
        self._origin_y = None
        self._contact_init = False
        self._low_counter = 0

        # per-region 完整状态: {region_id: {...}}；每个 region 独立锁定 origin + 二次精修状态
        self._region_states: dict[int, dict] = {}

    # ---------- 公共 API ----------

    def get_all(self, adc_data) -> tuple[float, float, float, float, float]:
        """
        输入 rows*cols 个 ADC 值，一次性输出 (angle, dx, dy, cop_x, cop_y)。

        :param adc_data: list/np.array，长度为 rows*cols 的 ADC 原始数据
        :return: (angle, dx, dy, cop_x, cop_y)
            · angle:     PZT 角度（0~360°）
            · dx:        CoP X 方向位移（列方向，cells）
            · dy:        CoP Y 方向位移（行方向，cells）
            · cop_x/y:   当前帧 CoP（rows*cols cell 坐标）
        :raises ValueError: ADC 数据长度不等于 rows*cols 时抛出
        """
        expected = self.rows * self.cols
        if len(adc_data) != expected:
            raise ValueError(f"ADC数据长度必须为{expected}")

        dx, dy = self._compute_delta_cop(adc_data)
        angle = self._compute_cop_angle(dx, dy)
        cop_x, cop_y = self.get_cop(adc_data)
        return angle, dx, dy, cop_x, cop_y

    def get_cop(self, adc_data) -> tuple[float, float]:
        """计算当前帧 CoP (cop_x, cop_y)，不影响 origin/精修状态。

        :param adc_data: list/np.array，长度为 rows*cols 的 ADC 原始数据
        :return: (cop_x, cop_y)，cell 单位；总压力为 0 时返回 (0.0, 0.0)
        :raises ValueError: ADC 数据长度不等于 rows*cols 时抛出
        """
        expected = self.rows * self.cols
        if len(adc_data) != expected:
            raise ValueError(f"ADC数据长度必须为{expected}")

        frame2d = np.asarray(adc_data, dtype=np.float32).reshape(self.rows, self.cols)
        total_pressure = float(np.sum(frame2d))
        if total_pressure == 0:
            return 0.0, 0.0
        cop_x, cop_y = self._compute_cop(frame2d, total_pressure)
        return cop_x, cop_y

    def get_gradient(self, adc_data) -> np.ndarray:
        """便捷接口：计算当前帧的 2D 梯度。等价于 compute_gradient(adc_data)。"""
        gradient = self._compute_gradient(adc_data)
        return gradient
    
    def get_origin(self) -> tuple[float | None, float | None]:
        """返回首次接触 origin：(origin_x, origin_y)；未锁时均为 None。"""
        return self._origin_x, self._origin_y

    def get_state(self) -> int:
        """返回 CoP 状态：0=未接触, 1=已接触/粗略, 2=已精修。"""
        if self._refined:
            return 2
        if self._contact_init:
            return 1
        return 0

    def reset_origin(self) -> None:
        """清掉首次接触 origin 与低压计数，同时清掉二次精修状态（候选点、稳定计数、已精修标志）；
        阈值（若已确定）保留。
        注意: 不清 _region_states (per-region 状态独立管理, 由 reset_region_origin 处理)。
        """
        self._origin_x = None
        self._origin_y = None
        self._contact_init = False
        self._low_counter = 0
        self._refine_cand_x = None
        self._refine_cand_y = None
        self._refine_curr = 0
        self._refined = False

    def _get_region_state(self, region_id: int) -> dict:
        """获取 region 的状态字典；首次访问时建空状态。

        每个 region 状态: {origin_x, origin_y, contact_init, refine_cand_x,
                          refine_cand_y, refine_curr, refined, coords,
                          last_cop_x, last_cop_y, last_coords, frames_since_seen}
        """
        s = self._region_states.get(region_id)
        if s is None:
            s = {
                'origin_x': None, 'origin_y': None,
                'contact_init': False,
                'refine_cand_x': None, 'refine_cand_y': None,
                'refine_curr': 0,
                'refined': False,
                'coords': None,
                'last_cop_x': None, 'last_cop_y': None,   # 最近一次出现 COP（质心匹配用）
                'last_coords': None,                      # 最近一次出现足迹（COP 落点判据用）
                'frames_since_seen': 0,                   # 未匹配帧数（stale 缓冲）
            }
            self._region_states[region_id] = s
        return s

    def dynamic_threshold(self, frame2d: np.ndarray) -> None:
        """外部 API: 同时更新两类阈值（总压 + 逐像素）。main 流程每帧调用一次。
        内部委托:
          - self._dynamic_total_threshold(total_pressure)  → 更新 _total_thresh
          - self._dynamic_pixel_threshold(frame2d)         → 更新 _pixel_thresh
        """
        total_pressure = float(np.sum(frame2d))
        self._dynamic_total_threshold(total_pressure)
        self._dynamic_pixel_threshold(frame2d)

    # ---------- 内部算法 ----------
    def _dynamic_total_threshold(self, total_pressure: float) -> None:
        with self._lock:
            # collect_frames=0: 跳过历史累积, _total_thresh 直接 = 0
            # (低压判定 `total_pressure < 0` 永不触发, 几乎所有帧都视为"非低")
            if self.collect_frames <= 0:
                if self._total_thresh is None:
                    self._total_thresh = 0
                return
            if self._total_thresh is None:
                self._pressure_history.append(total_pressure)
                if len(self._pressure_history) >= self.collect_frames:
                    self._total_thresh = self.total_threshold_factor * float(np.mean(self._pressure_history))
                    if self._total_thresh <= 0:
                        self._total_thresh = 10

    def _dynamic_pixel_threshold(self, frame2d: np.ndarray) -> None:
        with self._lock:
            if self.collect_frames <= 0:
                if self._pixel_thresh is None:
                    self._pixel_thresh = 0.0
                return
            if self._pixel_thresh is None:
                self._pixel_cnt += 1
                if self._pixel_avg_buffer is None:
                    # 强制 float64: 后续累加右侧 (frame2d - buffer) / int → float64
                    # 若 buffer 是 int64, 会因 same_kind 规则崩溃
                    self._pixel_avg_buffer = frame2d.astype(np.float64)
                else:
                    self._pixel_avg_buffer += (frame2d - self._pixel_avg_buffer) / self._pixel_cnt
                if self._pixel_cnt >= self.collect_frames:
                    self._pixel_thresh = self.pixel_threshold_factor * self._pixel_avg_buffer
                    self._pixel_thresh = np.where(self._pixel_thresh <= 0, 10.0, self._pixel_thresh)

    @staticmethod
    def _compute_cop_angle(px: float, py: float) -> float:
        angle = PRSensorAngle._compute_angle(px, -py)
        return angle

    @staticmethod
    def _compute_angle(x: float, y: float) -> float:
        epsilon = 1e-8
        angle = np.degrees(np.arctan2(y, x + epsilon))
        if angle < 0:
            angle += 360
        return angle

    def _compute_cop(self, frame2d: np.ndarray, total_pressure: float) -> tuple[float, float]:
        """计算 2D 帧的 CoP (X, Y) — 压力加权中心, 几何意义.

        输入:  (rows × cols) 帧 + 总压力 total_pressure
        输出:  (cop_x, cop_y) 浮动位置, 已是 cell 单位
        """
        x_grid = np.tile(np.arange(self.cols), (self.rows, 1))
        y_grid = np.repeat(np.arange(self.rows), self.cols).reshape(self.rows, self.cols)
        cop_x = float(np.sum(frame2d * x_grid) / total_pressure)
        cop_y = float(np.sum(frame2d * y_grid) / total_pressure)
        return cop_x, cop_y

    def _compute_centroid(self, frame2d: np.ndarray) -> tuple[float, float] | None:
        """整帧接触区域形心（不加权）: mask 内所有 cell 等权平均坐标。

        与 CoP 的区别: CoP 以压力值为权重, 形心每个 cell 权重 = 1,
        压力数值完全不参与计算 —— 斜向按压（压力分布不对称但接触几何
        不变）时形心位置不变。mask 与 _compute_region_BFS 同款
        (5×_pixel_thresh)。无接触返回 None。

        :param frame2d: (rows × cols) 2D 压力帧
        :return: (centroid_x, centroid_y) cell 单位, x = 列索引, y = 行索引
        """
        threshold = self._pixel_thresh if self._pixel_thresh is not None else 10.0
        mask = frame2d > 3 * threshold
        if not mask.any():
            return None
        rs, cs = np.where(mask)
        return float(cs.mean()), float(rs.mean())

    def _compute_delta_cop(self, raw_frame) -> tuple[float, float]:
        self._frame_count += 1
        if self._reset_at_frame > 0 and self._frame_count == self._reset_at_frame:
            self.reset_origin()

        rows, cols = self.rows, self.cols
        frame_flat = np.asarray(raw_frame, dtype=np.float32).flatten()
        frame2d = frame_flat.reshape(rows, cols)

        total_pressure = float(np.sum(frame2d))

        # thresh 未确定不进入低压分支
        if self._total_thresh is not None:
            if total_pressure < self._total_thresh:
                self._low_counter += 1
            else:
                self._low_counter = 0

            if self._low_counter >= self.stability_frames:
                self.reset_origin()
                return 0.0, 0.0

        if total_pressure == 0:
            return 0.0, 0.0

        # CoP 加权均值
        cop_x, cop_y = self._compute_cop(frame2d, total_pressure)

        delta_x = 0.0
        delta_y = 0.0
        if not self._contact_init:
            # 等待阈值已定 + 当前压力 > 阈值 才锁 origin(防闲置帧"先入为主"污染基准)
            if self._total_thresh is not None and total_pressure > self._total_thresh:
                # 真首次接触: 同时初始化 origin 与二次精修候选
                self._origin_x = cop_x
                self._origin_y = cop_y
                self._contact_init = True
                if self._refine_enabled:
                    # origin 锁定时同步把候选点设到这一帧的 CoP, 稳定计数从 1 开始
                    self._refine_cand_x = cop_x
                    self._refine_cand_y = cop_y
                    self._refine_curr = 1
            return 0.0, 0.0

        delta_x = cop_x - self._origin_x
        delta_y = cop_y - self._origin_y

        # 二次精修: 寻找连续 N 帧 CoP 几乎不动的"稳定点"
        if self._refine_enabled and not self._refined:
            if self._refine_cand_x is None:
                # 防御性: 首帧(origin 刚锁时已 init)理论上不会到这里, 写在以防 manual 重置
                self._refine_cand_x = cop_x
                self._refine_cand_y = cop_y
                self._refine_curr = 1
            else:
                # 欧氏距离
                dist = float(np.hypot(cop_x - self._refine_cand_x,
                                       cop_y - self._refine_cand_y))
                if dist <= self._refine_distance:
                    # 离候选点近 → 累计稳定计数
                    self._refine_curr += 1
                else:
                    # 离候选点远 → 重置计数
                    self._refine_cand_x = cop_x
                    self._refine_cand_y = cop_y
                    self._refine_curr = 1

            if self._refine_curr >= self._refine_cnt:
                # 候选点作为新 origin
                self._origin_x = self._refine_cand_x
                self._origin_y = self._refine_cand_y
                self._refined = True

        return delta_x, delta_y

    def _compute_gradient(self, adc_data) -> np.ndarray:
        """计算压力帧的 2D 梯度（中心差分）

        :param adc_data: list/np.array，长度为 rows*cols 的 ADC 原始数据
        :return: np.ndarray，shape (rows, cols, 2) — 最后一维是 (grad_x, grad_y)
            · grad_x = 右 - 左（列方向）
            · grad_y = 上 - 下（行方向）
            · 边界单元使用单侧差分（缺失邻居用中心值代替，等价于中心差分退化）
        :raises ValueError: ADC 数据长度不等于 rows*cols 时抛出
        """
        expected = self.rows * self.cols
        if len(adc_data) != expected:
            raise ValueError(f"ADC数据长度必须为{expected}")

        frame2d = np.asarray(adc_data, dtype=np.float32).reshape(self.rows, self.cols)

        grad_x = np.zeros((self.rows, self.cols), dtype=np.float32)
        grad_y = np.zeros((self.rows, self.cols), dtype=np.float32)

        if self.cols > 1:
            grad_x[:, 1:-1] = frame2d[:, 2:] - frame2d[:, :-2]      # 内部中心差分
            grad_x[:, 0] = frame2d[:, 1] - frame2d[:, 0]            # 左边界
            grad_x[:, -1] = frame2d[:, -1] - frame2d[:, -2]         # 右边界
        if self.rows > 1:
            grad_y[1:-1, :] = frame2d[:-2, :] - frame2d[2:, :]      # 内部: up - down
            grad_y[0, :] = frame2d[0, :] - frame2d[1, :]            # 上边界: center - down
            grad_y[-1, :] = frame2d[-2, :] - frame2d[-1, :]         # 下边界: up - center

        return np.stack([grad_x, grad_y], axis=-1)

    def _merge_shallow_regions(self, region_mask, frame2d):
        """合并浅谷相邻 region
        输入： region_mask 存储 cell 属于哪个 region 的 array
        输出： 
        边界 cell 压力 ≥ _merge_ratio × min(两 region 峰值) 的数量 ≥ _merge_min_cells
        视为浅谷（材料噪声）→ 合并；否则为深谷（独立接触）→ 保留。
        返回重编号为连续 1..M 的 region_mask。
        """
        n = int(region_mask.max())          # region_mask 为 int array
        if n <= 1:
            return region_mask
        rows, cols = region_mask.shape

        peak_vals = {}
        for lbl in range(1, n + 1):
            peak_vals[lbl] = float(frame2d[region_mask == lbl].max())    #同 region 最大的cell

        high_count = {}
        for y in range(rows):
            for x in range(cols):
                l1 = region_mask[y, x]
                if l1 == 0:
                    continue
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < rows and 0 <= nx < cols:
                        l2 = region_mask[ny, nx]
                        if l2 != 0 and l2 != l1:
                            pair = (min(l1, l2), max(l1, l2))
                            min_peak = min(peak_vals[l1], peak_vals[l2])
                            if min_peak > 0 and min(frame2d[y, x], frame2d[ny, nx]) / min_peak > self._merge_ratio:
                                high_count[pair] = high_count.get(pair, 0) + 1

        pairs_to_merge = [pair for pair, cnt in high_count.items() if cnt >= self._merge_min_cells]
        if not pairs_to_merge:
            return region_mask

        parent = list(range(n + 1))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # 弱者至多并入一次（弱峰被吸收 → 每个 region 最终只有一个峰）:
        # 防止谷 strip（自己成 basin 的浅谷）同时并入两侧 region 把独立接触连通;
        # 强 region 可吸收多个弱碎片（星形合并不受影响）。
        merged_weak = set()
        for (a, b), cnt in sorted(high_count.items(), key=lambda kv: -kv[1]):
            if cnt < self._merge_min_cells:
                continue
            weak, strong = (a, b) if peak_vals[a] < peak_vals[b] else (b, a)
            if weak in merged_weak:
                continue
            union(weak, strong)
            merged_weak.add(weak)

        new_label = {}
        next_lbl = 1
        for lbl in range(1, n + 1):
            root = find(lbl)
            if root not in new_label:
                new_label[root] = next_lbl
                next_lbl += 1

        merged = np.zeros_like(region_mask)
        for y in range(rows):
            for x in range(cols):
                lbl = region_mask[y, x]
                if lbl > 0:
                    merged[y, x] = new_label[find(lbl)]
        return merged

    def _grow_from_peaks(self, markers, frame2d, mask):
        """
        输入：markers 峰 ID 矩阵；frame2d ADC 矩阵；mask boolarray
        输出：labeled 存储 cell 属于哪个 region 的 array
        BFS 峰生长: 种子先标号, 优先队列按 -压力
        （高压力 cell 先被占领 → region 从峰一圈圈扩散, 波谷最后长到 → 天然边界）。
        4 邻扩散, 只长入 mask 内且未标号的 cell。"""
        rows, cols = self.rows, self.cols
        labeled = np.zeros((rows, cols), dtype=np.int32)
        heap = []
        for r in range(rows):
            for c in range(cols):
                lbl = markers[r, c]
                if lbl > 0:
                    labeled[r, c] = lbl         # 备份 markers[r, c]
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):           # 4‑邻域偏移
                        nr, nc = r + dr, c + dc
                         # 条件校验：坐标在图像范围内 && 邻居尚未分配标签 && 属于有效接触mask区域
                        if 0 <= nr < rows and 0 <= nc < cols and labeled[nr, nc] == 0 and mask[nr, nc]:
                            heapq.heappush(heap, (-frame2d[nr, nc], nr, nc, lbl))
                            # (-frame2d[nr, nc], # 第0项：负的传感器压力值（堆排序关键字）
                            # nr,                # 第1项：待处理像素 行号y
                            # nc,                # 第2项：待处理像素 列号x
                            # lbl)               # 第3项：这个像素将来归属哪一个种子区域ID
        while heap:
            _, y, x, lbl = heapq.heappop(heap)
            if labeled[y, x] != 0:          # 备份 markers[r, c]，像素已经被其他更高压力的区域抢先占领
                continue
            labeled[y, x] = lbl
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dr, x + dc
                if 0 <= ny < rows and 0 <= nx < cols and labeled[ny, nx] == 0 and mask[ny, nx]:
                    heapq.heappush(heap, (-frame2d[ny, nx], ny, nx, lbl))
        return labeled

    def _compute_region_BFS(self, frame2d: np.ndarray) -> list[dict]:
        """用 BFS 峰生长识别压力帧中的多指接触域

        算法：先在 mask 内找压力局部极大值作为种子，再从种子按压力高优先 4 邻扩散
        （高压力 cell 先被占领 → region 从峰一圈圈生长, 波谷自然成边界, 不绕过包围）。
        面积 < self._region_min_area 的碎片 region 统一过滤（__init__ 参数 region_min_area）。

        :param frame2d: (rows, cols) 2D 压力帧
        :return: list[dict]，按 area 降序，每个 dict 包含:
            · 'coords':         list[(row, col), ...] — 该域所有 cell 坐标
            · 'area':           int，区域内 cell 数（== len(coords)）
            · 'bbox':           (cmin, rmin, cmax, rmax) — 最小外接矩形（可能有空洞）
        """
        threshold = self._pixel_thresh if self._pixel_thresh is not None else 10.0
        mask = frame2d > 3 * threshold
        if not mask.any():
            return []

        rows, cols = self.rows, self.cols

        # 1) 在 mask 内找"严格峰"作为种子: 严格高于所有 8 邻
        # generate_binary_structure(rank, connectivity), rank=2：二维结构（图像是 2D）， connectivity=2：二维下代表8‑邻域， struct_3x3 返回 3x3 boolarray
        struct_3x3 = generate_binary_structure(2, 2)               
        # maximum_filter 的 footprint 含中心, `>` 恒 False, 用 pad+shift 求 8 邻最大（不含中心）
        padded = np.pad(frame2d, 1, mode='constant', constant_values=-np.inf)  # padding = 1,填充 -∞
        max_nb = np.full_like(frame2d, -np.inf)                                # 全 -∞ 矩阵
        for dr in (-1, 0, 1):                                                  # dr、dc 遍历3×3邻域的9个偏移
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                # 取出相对于原图偏移(dr,dc)的那一层，和max_nb逐元素取最大值，padded.shape = (rows+2, cols+2)
                max_nb = np.maximum(max_nb, padded[1 + dr:1 + dr + rows, 1 + dc:1 + dc + cols])
        # 遍历完 dr, dc 后 max_nb 存储的是相同位置 cell 的8邻域中的最大值， local_max 是局部极大值
        local_max = (max_nb < 1 * frame2d) & mask

        if local_max.any():
            # 附属峰过滤: 高度 < ratio×更高峰 且 距离 < dist → 不成种子（同一接触的起伏）
            # np.where 返回行数组和列数组，zip* 并行遍历两个可迭代对象，按位置索引一一配对，也就是变成坐标
            cand = [(r, c, float(frame2d[r, c])) for r, c in zip(*np.where(local_max))]
            cand.sort(key=lambda t: -t[2])                        # 对 cand 三元组列表里每一个三元组t，对于-t[2]，默认升序
            kept = []
            for r, c, v in cand:
                dominated = False
                for kr, kc, kv in kept:                           # 已保留的更高峰
                    if (v < self._region_peak_ratio * kv
                            and abs(r - kr) + abs(c - kc) < self._region_peak_dist):
                        dominated = True
                        break
                if not dominated:
                    kept.append((r, c, v))
            markers = np.zeros((rows, cols), dtype=np.int32)
            # enumerate(kept, 1)：从1开始计数，i为种子编号；_表示忽略压力v，只取坐标r,c
            for i, (r, c, _) in enumerate(kept, 1):
                markers[r, c] = i
        else:
            # 兜底: 无峰 → 在活动区质心处造一个种子, 整片作为单区域
            rs, cs = np.where(mask)
            weights = frame2d[rs, cs]
            total_w = float(weights.sum())
            if total_w > 0:
                cy = float((rs * weights).sum() / total_w)
                cx = float((cs * weights).sum() / total_w)
            else:
                cy = float(rs.mean()); cx = float(cs.mean())
            markers = np.zeros((rows, cols), dtype=np.int32)
            markers[int(round(cy)), int(round(cx))] = 1

        # 2) BFS 峰生长: 压力高优先 4 邻扩散, 只长入 mask 内 cell
        region_grid = self._grow_from_peaks(markers, frame2d, mask)

        # 2.5) 浅谷合并: 相邻 region 边界压力 ≥ merge_ratio×较低峰 的 cell 数 ≥ merge_min_cells
        #      则合并（浅谷=材料噪声, 深谷=独立接触）。只在 mask 内合并, 背景不参与。
        region_grid = self._merge_shallow_regions(region_grid, frame2d)

        # 3) 提取每域几何信息（8 邻连通分量拆分, 保证每 region 连通）
        regions_info = []
        for id in np.unique(region_grid):                         # region_grid[i,j] = (0 or k)(背景像素 or 归属于第 k 号种子的区域)
            if id == 0:
                continue                                          # 0 = 背景
            basin = region_grid == id
            comps, ncomp = label(basin, structure=struct_3x3)     # 8 邻连通分量
            for ci in range(1, ncomp + 1):
                region_mask = comps == ci                         # 第 id 域的第 ci 个连通分量
                area = int(region_mask.sum())
                if area < self._region_min_area:
                    continue
                rs, cs = np.where(region_mask)                    # rs, cs分别是行和列索引，都是一维ndarray，(rs, cs)不用交叉组合
                coords = list(zip(rs.tolist(), cs.tolist()))      # region_mask 的 True 索引展开，zip两两配对
                regions_info.append({
                    'area':  area,
                    'bbox':  (int(cs.min()), int(rs.min()),
                            int(cs.max()), int(rs.max())),      # 外接矩形（可能有洞）
                    'coords': coords,
                })

        regions_info.sort(key=lambda d: d['area'], reverse=True)
        return regions_info

    def _compute_region_cop(self, frame2d: np.ndarray) -> list[dict]:
        """识别压力帧中的多指接触域，并计算每个域的压力加权中心 (CoP)。

        流程: 先调用 self._compute_region_BFS(frame2d) 取得每个 region 的几何信息,
            再对每个 region 在 frame2d 上算压力加权中心。

        :param frame2d: (rows, cols) 2D 压力帧
        :return: list[dict]，按 area 降序，每个 dict 包含:
            · 'coords':         list[(row, col), ...] — 该域所有 cell 坐标
            · 'area':           int，区域内 cell 数（== len(coords)）
            · 'bbox':           (cmin, rmin, cmax, rmax) — 最小外接矩形（可能有空洞）
            · 'cop':            (cop_x, cop_y) — 该域压力加权中心
                                ⚠️ x = 列索引, y = 行索引
            · 'centroid':       (cx, cy) — 该域等权形心（每个 cell 权重=1, 压力不参与）
            · 'total_pressure': float，区域内 cell 总压力

        退化: coords 非空但 total_pressure == 0 时, cop 退化为几何中心 (rs.mean(), cs.mean())

        用法:
            for region in p._compute_region_cop(frame2d):
                print(region['cop'], region['area'], region['total_pressure'])
        """
        regions_info = self._compute_region_BFS(frame2d)
        for region in regions_info:
            coords = region['coords']
            rs = np.array([c[0] for c in coords])
            cs = np.array([c[1] for c in coords])
            pressure_values = frame2d[rs, cs]
            total_p = float(pressure_values.sum())

            if total_p > 0:
                cop_x = float((cs * pressure_values).sum() / total_p)
                cop_y = float((rs * pressure_values).sum() / total_p)
            else:
                cop_x = float(cs.mean())
                cop_y = float(rs.mean())

            region['cop'] = (cop_x, cop_y)
            region['centroid'] = (float(cs.mean()), float(rs.mean()))  # 等权形心, 压力不参与
            region['total_pressure'] = total_p

        return regions_info

    def _compute_region_delta_cop(self, frame2d: np.ndarray) -> list[dict]:
            """识别压力帧中的多指接触域，并计算每个域的 (delta_x, delta_y)。

            每个 region 独立锁定 origin（独立状态机）：
            - region_id 跨帧稳定 (F11): 本帧 region 与存活 tracker 按质心最近邻匹配
              (距离 <= REGION_MATCH_DIST 继承 id); 质心超距时, 若本帧 COP 落在上一帧
              region 足迹内 (滑动中 COP 尚未移出原区域) 则视为同一手指, 继承 id
              (不重置 origin); 都不满足才分配新 id
            - 首次出现的 region 在该帧的 CoP 视为其 origin, delta = (0, 0);
              coords (足迹) 仅在状态新建时记录 (F7), 供 reset_region_origin
              判定"已放手"——每帧刷新会让比较恒真 (coords 永远来自当前 mask)
            - 后续该 region 再次出现时, delta = 当前 CoP - 该 region 的 origin
            - 二次精修 (refine): 与全帧 _compute_delta_cop 同算法, per-region 独立
            - 本帧不再出现的 region 保留 3 帧 stale 缓冲 (frames_since_seen>3 才删),
              短暂丢失 id 不丢; 调用 reset_region_origin(region_id) 可单独清掉"已放手"

            :param frame2d: (rows, cols) 2D 压力帧
            :return: list[dict]，按 area 降序，每个 dict 包含 _compute_region_cop 输出
                    + 'delta': (delta_x, delta_y) — 该域 CoP 与其 origin 的差
                    + 'id':    稳定 region 标识 (跨帧不随面积排名变化)

            用法:
                for region in p._compute_region_delta_cop(frame2d):
                    print(region['id'], region['cop'], region['delta'], region['area'])
            """
            regions_info = self._compute_region_cop(frame2d)
            with self._lock:
                current_region_ids = set()
                taken_ids = set()   # 本帧已匹配的 id, 防两个 region 继承同一 id

                for region in regions_info:
                    cop_x, cop_y = region['cop']

                    # F11: 质心最近邻 → COP 落点判据, 继承稳定 id
                    # 1) 质心匹配（静止/慢速）: 候选池 = 所有存活 tracker 的 last_cop
                    region_id = None
                    best_d = 1e9
                    for rid, s in self._region_states.items():
                        if rid in taken_ids or s['last_cop_x'] is None:
                            continue
                        d = float(np.hypot(cop_x - s['last_cop_x'], cop_y - s['last_cop_y']))
                        if d < best_d:
                            best_d, region_id = d, rid
                    # 2) 质心超距 → 本帧 COP 落在上一帧 region 足迹内 → 同一手指滑动, 继承 id
                    if region_id is None or best_d > self._region_match_dist:
                        region_id = None
                        hit_cell = (int(round(cop_y)), int(round(cop_x)))
                        for rid, s in self._region_states.items():
                            if rid in taken_ids or s.get('last_coords') is None:
                                continue
                            if hit_cell in s['last_coords']:
                                region_id = rid
                                break
                    # 3) 兜底: 新 region 分配新 id
                    if region_id is None:
                        region_id = 1
                        while region_id in self._region_states or region_id in taken_ids:
                            region_id += 1
                    taken_ids.add(region_id)
                    current_region_ids.add(region_id)

                    s = self._get_region_state(region_id)
                    # F7: coords 仅在状态新建时记录 (足迹), 不再每帧刷新
                    if s.get('coords') is None:
                        s['coords'] = region['coords']
                    # 帧间追踪状态: 记录本帧 COP 与足迹（COP 落点判据用）
                    s['last_cop_x'] = cop_x
                    s['last_cop_y'] = cop_y
                    s['last_coords'] = region['coords']
                    s['frames_since_seen'] = 0

                    if not s['contact_init']:
                        # 首次接触: 锁 origin (origin 锁定时同步把候选点设到这一帧的 CoP)
                        if self._total_thresh is not None and region['total_pressure'] > self._total_thresh:
                            s['origin_x'] = cop_x
                            s['origin_y'] = cop_y
                            s['contact_init'] = True
                            if self._refine_enabled:
                                s['refine_cand_x'] = cop_x
                                s['refine_cand_y'] = cop_y
                                s['refine_curr'] = 1
                            delta_x, delta_y = 0.0, 0.0
                        else:
                            # 阈值未确定 / 总压不够高, 不锁 origin
                            delta_x, delta_y = 0.0, 0.0
                    else:
                        # 已接触: delta = cop - origin
                        delta_x = cop_x - s['origin_x']
                        delta_y = cop_y - s['origin_y']

                        # 二次精修: 寻找连续 N 帧 CoP 几乎不动的"稳定点"
                        if self._refine_enabled and not s['refined']:
                            if s['refine_cand_x'] is None:
                                s['refine_cand_x'] = cop_x
                                s['refine_cand_y'] = cop_y
                                s['refine_curr'] = 1
                            else:
                                dist = float(np.hypot(cop_x - s['refine_cand_x'],
                                                    cop_y - s['refine_cand_y']))
                                if dist <= self._refine_distance:
                                    s['refine_curr'] += 1
                                else:
                                    s['refine_cand_x'] = cop_x
                                    s['refine_cand_y'] = cop_y
                                    s['refine_curr'] = 1

                            if s['refine_curr'] >= self._refine_cnt:
                                # 候选点作为新 origin
                                s['origin_x'] = s['refine_cand_x']
                                s['origin_y'] = s['refine_cand_y']
                                s['refined'] = True

                    region['delta'] = (delta_x, delta_y)
                    region['id'] = region_id
                    region['contact_init'] = s['contact_init']

                # stale 缓冲（参考项目 frames_since_seen>3）: 未匹配 tracker 保留 3 帧, 超过才删
                for rid, s in self._region_states.items():
                    if rid not in current_region_ids:
                        s['frames_since_seen'] += 1
                stale_ids = [rid for rid, s in self._region_states.items() if s['frames_since_seen'] > 3]
                for stale_id in stale_ids:
                    self._region_states.pop(stale_id, None)

            return regions_info


    