"""F6/F7/F11 覆盖: region 状态机 (轻压不误 pop / 足迹 reset / 稳定 id) + 全帧状态机"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tang_7_12_InitCOP_realtime_package_note import PZTSensorAngle

ROWS, COLS = 12, 7
BG = 20.0   # 背景压力: 使 _total_thresh 非零 (84*20 = 1680)


def _frame(*blobs) -> np.ndarray:
    """blobs: list of (rows, cols, value); 未覆盖处为背景 BG"""
    f = np.full((ROWS, COLS), BG, dtype=np.float32)
    for rs, cs, v in blobs:
        f[rs, cs] = v
    return f


def _blob(center_r, center_c, radius, value):
    rs, cs = [], []
    for r in range(ROWS):
        for c in range(COLS):
            if (r - center_r) ** 2 + (c - center_c) ** 2 <= radius ** 2:
                rs.append(r)
                cs.append(c)
    return rs, cs, value


def _sensor() -> PZTSensorAngle:
    return PZTSensorAngle(rows=ROWS, cols=COLS,
                          threshold_factor=1.0, collect_frames=3,
                          stability_frames=2, reset_at_frame=0,
                          refine_cnt=5, refine_distance=0.5)


def _learn_thresh(s: PZTSensorAngle):
    for _ in range(3):
        s.dynamic_threshold(_frame())   # _total_thresh = 1680, _pixel_thresh = 20


def _expected_cop(blob):
    """带背景偏置的全帧 cop 解析值 (背景 BG 参与加权)"""
    f = _frame(blob)
    xx, yy = np.meshgrid(np.arange(COLS), np.arange(ROWS))
    total = f.sum()
    return float((f * xx).sum() / total), float((f * yy).sum() / total)


# ---------- 全帧状态机 ----------

def test_whole_frame_state_machine():
    s = _sensor()
    _learn_thresh(s)
    _, dx, dy, _, _ = s.get_all(_frame(_blob(4, 2, 2, 200)).flatten())   # 接触: 锁 origin
    assert s.get_state() == 1
    assert (dx, dy) == (0.0, 0.0)
    _, dx2, _, _, _ = s.get_all(_frame(_blob(4, 3, 2, 200)).flatten())   # 右移 1 格
    assert dx2 > 0.5
    for _ in range(2):                                                   # 低压 2 帧 → reset
        s.get_all(np.zeros(ROWS * COLS))
    assert s.get_state() == 0


def test_whole_frame_refine():
    s = _sensor()
    _learn_thresh(s)
    s.get_all(_frame(_blob(4, 2, 2, 200)).flatten())                     # 锁 origin
    for _ in range(5):                                                   # 稳定 5 帧 → 精修
        s.get_all(_frame(_blob(4, 2, 2, 200)).flatten())
    assert s.get_state() == 2
    ex, ey = _expected_cop(_blob(4, 2, 2, 200))
    ox, oy = s.get_origin()
    assert abs(ox - ex) < 1e-6 and abs(oy - ey) < 1e-6


# ---------- F6: region 轻压 (总压 < 全帧阈值) 不再误 pop ----------

def test_f6_region_survives_lightening():
    s = _sensor()
    _learn_thresh(s)
    r1 = s._compute_region_delta_cop(_frame(_blob(4, 2, 2, 200)))   # 大指: 总压 2200 > 1680, 锁 origin
    assert len(r1) == 1 and r1[0]['delta'] == (0.0, 0.0)
    assert s._region_states[r1[0]['id']]['contact_init'] is True
    # 同一指轻压: 仍在 mask 内 (150 > 20), 但 region 总压 750 < 全帧阈值 1680
    r2 = s._compute_region_delta_cop(_frame(_blob(4, 2, 1, 150)))
    assert len(r2) == 1
    assert r2[0]['total_pressure'] < s._total_thresh   # 复现旧 bug 的前提
    assert s._region_states[r2[0]['id']]['contact_init'] is True   # F6: 状态未被 pop
    # 再滑 1 格 → delta 继续非零
    r3 = s._compute_region_delta_cop(_frame(_blob(4, 3, 1, 150)))
    assert r3[0]['delta'][0] > 0.5


def test_f6_two_fingers_both_tracked():
    s = _sensor()
    _learn_thresh(s)
    r1 = s._compute_region_delta_cop(_frame(_blob(4, 2, 2, 200), _blob(8, 5, 2, 200)))
    assert len(r1) == 2
    for region in r1:
        assert s._region_states[region['id']]['contact_init'] is True
    r2 = s._compute_region_delta_cop(_frame(_blob(4, 3, 2, 200), _blob(8, 5, 2, 200)))   # 第一指右移
    deltas = [region['delta'][0] for region in r2]
    assert max(deltas) > 0.5 and min(deltas) < 0.3


# ---------- F7: 足迹低于 pixel_thresh → 在该 region 范围内 reset ----------

def test_f7_reset_region_origin_on_footprint_release():
    s = _sensor()
    _learn_thresh(s)
    f1 = _frame(_blob(4, 2, 2, 200))                  # 总压 2600 > 1680 → 锁 origin
    r1 = s._compute_region_delta_cop(f1)
    rid = r1[0]['id']
    assert s._region_states[rid]['contact_init'] is True
    # 手指滑动 1 格: region 仍被追踪 (F11 匹配), 原足迹左侧 cell 压力回落
    f2 = _frame(_blob(4, 3, 2, 200))
    r2 = s._compute_region_delta_cop(f2)
    assert len(r2) == 1 and r2[0]['id'] == rid        # 稳定 id 继承
    assert r2[0]['delta'][0] > 0.5                    # 滑动被量测
    # F7: 足迹内任一 cell <= _pixel_thresh → pop 该 region 状态
    s.reset_region_origin(rid, f2)
    assert rid not in s._region_states
    # 下一帧同位置 → 重新锁 origin, delta 归零
    r3 = s._compute_region_delta_cop(_frame(_blob(4, 3, 2, 200)))
    assert len(r3) == 1 and r3[0]['delta'] == (0.0, 0.0)


# ---------- F11: 面积排名互换时 region id 稳定 ----------

def test_f11_region_id_stable_across_area_swap():
    s = _sensor()
    _learn_thresh(s)
    r1 = s._compute_region_delta_cop(_frame(_blob(4, 2, 2, 200), _blob(8, 5, 1, 150)))
    assert len(r1) == 2
    by_cop = {round(region['cop'][1], 1): region['id'] for region in r1}
    left_id, right_id = by_cop[4.0], by_cop[8.0]
    assert left_id != right_id
    # 下一帧: 左指变小、右指变大 → 面积排名互换 (旧代码按 idx+1 会交换 id)
    r2 = s._compute_region_delta_cop(_frame(_blob(4, 2, 1, 150), _blob(8, 5, 2, 200)))
    by_cop2 = {round(region['cop'][1], 1): region['id'] for region in r2}
    assert by_cop2[4.0] == left_id      # 左指仍继承原 id
    assert by_cop2[8.0] == right_id
