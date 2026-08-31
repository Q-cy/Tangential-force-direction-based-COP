"""审计 0827_4 fraction-only score2 与连续窗参数。

本脚本只读取已经生成的 ``target_band_window_metrics.csv``，不重跑硬件采集、
不修改实时分类器，也不把单次记录的阈值当作生产规则。score2 与实时实现
保持同一公式：

``score2 = target_fraction``

``target_fraction >= 0.30`` 仅作为进入判定；退出判定使用同一阈值的严格小于关系。

目标带局部峰突出度仍保存为观察值，但不参与 score2、阈值或状态模拟。

运行方式（项目唯一验收环境）：

    PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/pzt-mplconfig \
    /home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python \
    analysis/offline_score2_audit_0827_4.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from offline_friction_spectrum_0827_4 import (
    OUTPUT_DIR,
    PHASE_INTERVALS_S,
)


RESULT_DIR = OUTPUT_DIR / "feature_evaluation"
INPUT_METRICS_CSV = RESULT_DIR / "target_band_window_metrics.csv"
TARGET_BAND_HZ = (24.0, 28.0)
FRACTION_THRESHOLD = 0.30
ENTER_WINDOWS = 9
EXIT_WINDOWS = 8
SLIDING_ONSET_S = float(PHASE_INTERVALS_S["sliding"][0])


def _load_metrics(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """读取已有目标带窗口表中的时间、阶段、比例和突出度。"""
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "time_s",
        "phase",
        "power_fraction_24_28_hz",
        "peak_prominence_24_28_hz_db",
    }
    actual = set(rows[0]) if rows else set()
    missing = sorted(required - actual)
    if not rows:
        raise ValueError(f"目标带窗口表没有数据行: {path}")
    if missing:
        raise ValueError(f"目标带窗口表缺少列 {missing}: {path}")
    time_s = np.asarray([float(row["time_s"]) for row in rows], dtype=np.float64)
    phase = np.asarray([row["phase"] for row in rows], dtype=object)
    fraction = np.asarray(
        [float(row["power_fraction_24_28_hz"]) for row in rows], dtype=np.float64
    )
    prominence = np.asarray(
        [float(row["peak_prominence_24_28_hz_db"]) for row in rows], dtype=np.float64
    )
    if not (
        time_s.ndim == phase.ndim == fraction.ndim == prominence.ndim == 1
        and time_s.size == phase.size == fraction.size == prominence.size
        and time_s.size > 0
    ):
        raise ValueError("目标带窗口表的列长度不一致")
    if np.any(np.diff(time_s) <= 0.0):
        raise ValueError("目标带窗口表 time_s 必须严格递增")
    if not np.all(np.isfinite(fraction)) or not np.all(np.isfinite(prominence)):
        raise ValueError("目标带窗口表的 score2 输入不能包含 NaN/无穷值")
    return time_s, phase, fraction, prominence


def calculate_score2(fraction: np.ndarray) -> np.ndarray:
    """按实时 fraction-only 公式返回未经归一化的 score2。"""
    return np.asarray(fraction, dtype=np.float64).copy()


def _max_true_run(mask: np.ndarray) -> int:
    """返回布尔序列最长连续 True 窗数。"""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return int(max((stop - start for start, stop in zip(starts, stops)), default=0))


def _phase_run_lengths(mask: np.ndarray, phase: np.ndarray) -> dict[str, int]:
    """只在各候选阶段内部计算最长连续阳性窗。"""
    return {
        name: _max_true_run(mask & (phase == name))
        for name in (*PHASE_INTERVALS_S, "transition_or_unlabelled")
    }


def _oriented_auc(negative: np.ndarray, positive: np.ndarray) -> tuple[float, str]:
    """返回滑动较高方向的两类逐窗 AUC。"""
    comparisons = positive[:, None] - negative[None, :]
    raw = float(
        np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0)
    )
    return (raw, "higher_is_sliding") if raw >= 0.5 else (1.0 - raw, "lower_is_sliding")


def _balanced_accuracy(
    negative: np.ndarray,
    positive: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """在固定 score2 进入门限下计算灵敏度、特异度和平衡准确率。"""
    sensitivity = float(np.mean(positive >= threshold))
    specificity = float(np.mean(negative < threshold))
    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
    }


def simulate_hysteresis(
    score2: np.ndarray,
    threshold: float,
    enter_windows: int,
    exit_windows: int,
) -> np.ndarray:
    """按同一 fraction 阈值和连续窗数模拟 STICK/SLIP 状态。"""
    slipping = False
    enter_count = 0
    exit_count = 0
    states: list[bool] = []
    for value in np.asarray(score2, dtype=np.float64):
        if not slipping:
            enter_count = enter_count + 1 if value >= threshold else 0
            if enter_count >= enter_windows:
                slipping = True
                enter_count = 0
                exit_count = 0
        else:
            exit_count = exit_count + 1 if value < threshold else 0
            if exit_count >= exit_windows:
                slipping = False
                enter_count = 0
                exit_count = 0
        states.append(slipping)
    return np.asarray(states, dtype=bool)


def _first_true_time(states: np.ndarray, time_s: np.ndarray) -> float | None:
    """返回第一次由 STICK 进入 SLIP 的窗口结束时间。"""
    starts = np.flatnonzero(states & ~np.r_[False, states[:-1]])
    return float(time_s[starts[0]]) if starts.size else None


def _phase_statistics(
    phase: np.ndarray,
    fraction: np.ndarray,
    prominence: np.ndarray,
    score2: np.ndarray,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """输出各候选阶段三项 score2 数值的中位数和 IQR。"""
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    for name in PHASE_INTERVALS_S:
        selected = phase == name
        result[name] = {}
        for metric_name, values in (
            ("target_fraction", fraction),
            ("prominence_db", prominence),
            ("score2", score2),
        ):
            selected_values = values[selected]
            q25, median, q75 = np.quantile(selected_values, (0.25, 0.5, 0.75))
            result[name][metric_name] = {
                "n": int(selected_values.size),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "iqr": float(q75 - q25),
            }
    return result


def write_window_csv(
    path: Path,
    time_s: np.ndarray,
    phase: np.ndarray,
    fraction: np.ndarray,
    prominence: np.ndarray,
    score2: np.ndarray,
    states: np.ndarray,
) -> None:
    """保存每个窗口的 fraction、观察突出度和滞回状态。"""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "time_s", "phase", "target_fraction", "prominence_db",
                "score2", "fraction_positive", "slip_state",
            ),
        )
        writer.writeheader()
        for index in range(time_s.size):
            writer.writerow(
                {
                    "time_s": float(time_s[index]),
                    "phase": str(phase[index]),
                    "target_fraction": float(fraction[index]),
                    "prominence_db": float(prominence[index]),
                    "score2": float(score2[index]),
                    "fraction_positive": int(score2[index] >= FRACTION_THRESHOLD),
                    "slip_state": int(states[index]),
                }
            )


def write_hysteresis_grid(
    path: Path,
    time_s: np.ndarray,
    phase: np.ndarray,
    score2: np.ndarray,
) -> list[dict[str, object]]:
    """保存 1--12 连续窗参数网格，供选择过程审计。"""
    records: list[dict[str, object]] = []
    with path.open("w", newline="", encoding="utf-8") as stream:
        fields = (
            "enter_windows", "exit_windows", "static_contact_max_slip_windows",
            "static_friction_max_slip_windows", "sliding_max_slip_windows",
            "first_slip_time_s", "detection_delay_s", "slip_state_windows",
        )
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for enter_windows in range(1, 13):
            for exit_windows in range(1, 13):
                states = simulate_hysteresis(
                    score2,
                    FRACTION_THRESHOLD,
                    enter_windows,
                    exit_windows,
                )
                first_time = _first_true_time(states, time_s)
                record: dict[str, object] = {
                    "enter_windows": enter_windows,
                    "exit_windows": exit_windows,
                    "static_contact_max_slip_windows": _max_true_run(
                        states & (phase == "static_contact_hold")
                    ),
                    "static_friction_max_slip_windows": _max_true_run(
                        states & (phase == "static_friction")
                    ),
                    "sliding_max_slip_windows": _max_true_run(
                        states & (phase == "sliding")
                    ),
                    "first_slip_time_s": first_time,
                    "detection_delay_s": (
                        first_time - SLIDING_ONSET_S if first_time is not None else None
                    ),
                    "slip_state_windows": int(np.sum(states)),
                }
                records.append(record)
                writer.writerow(record)
    return records


def main() -> None:
    """读取窗口表并写出 score2 审计 JSON/CSV。"""
    time_s, phase, fraction, prominence = _load_metrics(INPUT_METRICS_CSV)
    score2 = calculate_score2(fraction)
    states = simulate_hysteresis(
        score2,
        FRACTION_THRESHOLD,
        ENTER_WINDOWS,
        EXIT_WINDOWS,
    )
    static = score2[phase == "static_friction"]
    sliding = score2[phase == "sliding"]
    auc, direction = _oriented_auc(static, sliding)
    fixed_positive = score2 >= FRACTION_THRESHOLD
    first_time = _first_true_time(states, time_s)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_window_csv(
        RESULT_DIR / "score2_audit_window_metrics.csv",
        time_s,
        phase,
        fraction,
        prominence,
        score2,
        states,
    )
    grid = write_hysteresis_grid(
        RESULT_DIR / "score2_audit_hysteresis_grid.csv",
        time_s,
        phase,
        score2,
    )
    selected_static_false_positive = max(
        _max_true_run(states & (phase == name))
        for name in ("static_contact_hold", "static_friction")
    )
    summary = {
        "input_metrics_csv": str(INPUT_METRICS_CSV),
        "formula": {
            "target_band_hz": list(TARGET_BAND_HZ),
            "score2": "target_band_power_fraction (no normalization)",
            "fraction_threshold": FRACTION_THRESHOLD,
            "enter_rule": "target_band_power_fraction >= fraction_threshold",
            "exit_rule": "target_band_power_fraction < fraction_threshold",
            "prominence_observation": (
                "target_band_peak_prominence_db is saved for observation only and "
                "does not participate in score2 or state"
            ),
        },
        "fixed_enter_threshold_metrics": {
            "threshold": FRACTION_THRESHOLD,
            "positive_window_count": int(np.sum(fixed_positive)),
            "positive_window_runs_by_phase": _phase_run_lengths(fixed_positive, phase),
            **_balanced_accuracy(static, sliding, FRACTION_THRESHOLD),
            "static_friction_vs_sliding_oriented_window_auc": auc,
            "direction": direction,
        },
        "selected_hysteresis": {
            "fraction_threshold": FRACTION_THRESHOLD,
            "enter_windows": ENTER_WINDOWS,
            "exit_windows": EXIT_WINDOWS,
            "slip_state_runs_by_phase": _phase_run_lengths(states, phase),
            "static_phase_max_slip_state_run_windows": selected_static_false_positive,
            "first_slip_state_time_s": first_time,
            "detection_delay_from_10.70_s": (
                first_time - SLIDING_ONSET_S if first_time is not None else None
            ),
            "slip_state_window_count": int(np.sum(states)),
            "triggered_in_0827_4": first_time is not None,
        },
        "phase_statistics_median_iqr": _phase_statistics(
            phase, fraction, prominence, score2
        ),
        "continuous_window_grid": {
            "enter_windows_range": [1, 12],
            "exit_windows_range": [1, 12],
            "csv": str(RESULT_DIR / "score2_audit_hysteresis_grid.csv"),
            "selection_note": (
                "enter=9 is the smallest tested enter window with no SLIP state in "
                "the two labeled static phases. This recording never reaches nine "
                "consecutive fraction-positive windows, so exit=8 is retained as the "
                "configured time hysteresis but is not validated by a state transition."
            ),
        },
        "candidate_grid_records": len(grid),
        "phase_boundaries_are_ground_truth": False,
        "caveat": (
            "All values come from one recording and candidate phase boundaries; they "
            "are an audit candidate, not a production validation set or final rule."
        ),
    }
    with (RESULT_DIR / "score2_audit_summary.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(summary["selected_hysteresis"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
