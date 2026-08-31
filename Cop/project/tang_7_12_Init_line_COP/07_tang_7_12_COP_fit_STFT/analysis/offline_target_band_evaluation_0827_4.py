"""评估 0827_4 中频目标带对静摩擦和滑动的区分能力。

本脚本只做维护者离线分析，不导入实时会话入口，也不修改
``CopSpectrumAnalyzer`` 的分类规则。CoP、160 Hz 重采样、0.5 秒速度 STFT
和时间轴全部复用已有离线回放脚本；阶段边界固定为本次实验的候选区间：

* 静止接触保持：4.10--6.20 s
* 完整静摩擦（包含开始段）：6.20--10.30 s
* 滑动：10.70--13.20 s

每个目标带的局部峰突出度定义为：

``10 * log10((目标带峰值功率 + floor) /
             (相邻背景带平均功率 + floor))``。

相邻背景带取目标带两侧、各自宽度等于目标带宽度的频率区间，目标频点
不进入背景均值。所有比例的分母都是完整显示 2--70 Hz 频带的功率总和。
报告中的AUC、阈值和连续阳性结果在同一条记录上计算，只是探索性结果，
不能直接写入实时滑移检测器。

运行方式（项目唯一验收环境）：

    PYTHONPATH=src:analysis MPLCONFIGDIR=/tmp/pzt-mplconfig \
    /home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python \
    analysis/offline_target_band_evaluation_0827_4.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from offline_friction_feature_evaluation_0827_4 import (
    best_threshold,
    max_true_run,
    oriented_auc,
)
from offline_friction_spectrum_0827 import (
    SPECTRUM_CONFIG,
    build_spectrum,
    replay_csv,
)
from offline_friction_spectrum_0827_4 import (
    INPUT_CSV,
    OUTPUT_DIR,
    PHASE_INTERVALS_S,
    phase_mask,
)


RESULT_DIR = OUTPUT_DIR / "feature_evaluation"

# 这三组频率按照 0.5 秒窗口的 2 Hz 频率分辨率实际对应 20--30、22--30
# 和 24--28 Hz 的离散频点。上下限均为闭区间。
TARGET_BANDS_HZ: dict[str, tuple[float, float]] = {
    "20_30": (20.0, 30.0),
    "22_30": (22.0, 30.0),
    "24_28": (24.0, 28.0),
}
SLIDING_ONSET_S = 10.70


def _safe_fraction(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """计算功率比例；无有效分母时返回 NaN。"""
    result = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator)
    valid &= denominator > np.finfo(float).eps
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _band_mask(
    frequency_hz: np.ndarray,
    band_hz: tuple[float, float],
) -> np.ndarray:
    """返回闭区间频带的离散 FFT 频点 mask。"""
    low, high = band_hz
    return (frequency_hz >= low) & (frequency_hz <= high)


def _adjacent_background_mask(
    frequency_hz: np.ndarray,
    band_hz: tuple[float, float],
) -> tuple[np.ndarray, list[list[float]]]:
    """返回目标带两侧等宽背景频点及其连续区间说明。"""
    low, high = band_hz
    width = high - low
    left_low = max(float(frequency_hz[0]), low - width)
    right_high = min(float(frequency_hz[-1]), high + width)
    left = (frequency_hz >= left_low) & (frequency_hz < low)
    right = (frequency_hz > high) & (frequency_hz <= right_high)
    mask = left | right
    if not np.any(mask):
        raise ValueError(f"目标带 {band_hz} 没有可用的相邻背景频点")
    return mask, [[left_low, low], [high, right_high]]


def calculate_target_band_metrics(
    frequency_hz: np.ndarray,
    power: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, object]]]:
    """计算三种目标带的比例、绝对功率/幅值和局部峰突出度。

    Args:
        frequency_hz: ``(F,)`` 的完整显示频率轴，默认覆盖 2--70 Hz。
        power: ``(T, F)`` 的 CoP X/Y 合成速度功率，定义为
            ``amplitude_x**2 + amplitude_y**2``。

    Returns:
        ``(metrics, metadata)``。``metrics`` 的每个数组都是 ``(T,)``；
        ``metadata`` 记录目标带、实际背景频点和公式，便于审计。
    """
    frequency_hz = np.asarray(frequency_hz, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    if frequency_hz.ndim != 1 or power.ndim != 2:
        raise ValueError("频率轴必须是一维，功率必须是二维")
    if power.shape[1] != frequency_hz.size:
        raise ValueError("功率列数必须与频率轴长度一致")

    total_power = np.sum(power, axis=1)
    floor = float(SPECTRUM_CONFIG.baseline_power_floor)
    metrics: dict[str, np.ndarray] = {}
    metadata: dict[str, dict[str, object]] = {}
    for label, band_hz in TARGET_BANDS_HZ.items():
        target_mask = _band_mask(frequency_hz, band_hz)
        if not np.any(target_mask):
            raise ValueError(f"目标带 {band_hz} 没有对应的 FFT 频点")
        background_mask, background_intervals = _adjacent_background_mask(
            frequency_hz, band_hz
        )
        target_power = np.sum(power[:, target_mask], axis=1)
        peak_index = np.argmax(
            np.where(target_mask[None, :], power, -np.inf), axis=1
        )
        peak_power = np.take_along_axis(
            power, peak_index[:, None], axis=1
        ).reshape(-1)
        background_mean_power = np.mean(power[:, background_mask], axis=1)
        target_label = f"{label}_hz"
        metrics[f"power_fraction_{target_label}"] = _safe_fraction(
            target_power, total_power
        )
        metrics[f"power_{target_label}"] = target_power
        metrics[f"amplitude_{target_label}"] = np.sqrt(
            np.maximum(target_power, 0.0)
        )
        metrics[f"peak_power_{target_label}"] = peak_power
        metrics[f"peak_amplitude_{target_label}"] = np.sqrt(
            np.maximum(peak_power, 0.0)
        )
        metrics[f"peak_prominence_{target_label}_db"] = 10.0 * np.log10(
            (peak_power + floor) / (background_mean_power + floor)
        )
        metrics[f"peak_frequency_{target_label}"] = frequency_hz[peak_index]
        metadata[label] = {
            "target_band_hz": list(band_hz),
            "target_frequency_bins_hz": frequency_hz[target_mask].tolist(),
            "background_intervals_hz": background_intervals,
            "background_frequency_bins_hz": frequency_hz[background_mask].tolist(),
            "peak_prominence_formula": (
                "10*log10((max(target_band_power)+floor)/"
                "(mean(adjacent_background_power)+floor))"
            ),
        }
    return metrics, metadata


def median_iqr(values: np.ndarray) -> dict[str, float | int]:
    """返回一个阶段的样本数、中位数、IQR和四分位数。"""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "q25": float("nan"), "median": float("nan"),
                "q75": float("nan"), "iqr": float("nan")}
    q25, median, q75 = np.quantile(finite, (0.25, 0.50, 0.75))
    return {
        "n": int(finite.size),
        "q25": float(q25),
        "median": float(median),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def _positive_mask(values: np.ndarray, threshold_result: dict[str, object]) -> np.ndarray:
    """按同记录最优阈值生成滑动阳性窗 mask。"""
    threshold = float(threshold_result["threshold"])
    direction = str(threshold_result["direction"])
    if direction == "higher_is_sliding":
        return np.asarray(values) >= threshold
    return np.asarray(values) <= threshold


def evaluate_metric(
    metric_name: str,
    values: np.ndarray,
    spectrum_time_s: np.ndarray,
    phase_masks: dict[str, np.ndarray],
) -> dict[str, object]:
    """评估一个指标的阶段统计、AUC、阈值、连续阳性和检测延迟。"""
    static_values = values[phase_masks["static_friction"]]
    sliding_values = values[phase_masks["sliding"]]
    if not np.all(np.isfinite(static_values)) or not np.all(np.isfinite(sliding_values)):
        raise ValueError(f"指标 {metric_name} 含有无法评估的 NaN/无穷值")
    auc, direction = oriented_auc(static_values, sliding_values)
    threshold = best_threshold(static_values, sliding_values, direction)
    positive = _positive_mask(values, threshold)
    runs = {
        phase: max_true_run(positive[selected])
        for phase, selected in phase_masks.items()
    }
    sliding_indices = np.flatnonzero(phase_masks["sliding"])
    positive_sliding = sliding_indices[positive[sliding_indices]]
    if positive_sliding.size:
        first_index = int(positive_sliding[0])
        first_time = float(spectrum_time_s[first_index])
        delay = first_time - SLIDING_ONSET_S
    else:
        first_time = None
        delay = None
    static_median = float(np.median(static_values))
    sliding_median = float(np.median(sliding_values))
    return {
        "metric": metric_name,
        "static_friction_vs_sliding_oriented_window_auc": float(auc),
        "sliding_direction": direction,
        "static_friction_median": static_median,
        "sliding_median": sliding_median,
        "sliding_minus_static_friction_median": sliding_median - static_median,
        "same_recording_best_threshold": threshold,
        "threshold_direction": str(threshold["direction"]),
        "threshold_sensitivity": float(threshold["sensitivity"]),
        "threshold_specificity": float(threshold["specificity"]),
        "threshold_balanced_accuracy": float(threshold["balanced_accuracy"]),
        "max_consecutive_positive_windows_by_phase": runs,
        "sliding_onset_s": SLIDING_ONSET_S,
        "first_positive_sliding_window_end_s": first_time,
        "detection_delay_s": delay,
        "positive_sliding_window_count": int(positive_sliding.size),
        "positive_window_definition": (
            "single STFT window crossing the same-recording best threshold; "
            "no production hysteresis is applied"
        ),
    }


def _phase_for_index(index: int, phase_masks: dict[str, np.ndarray]) -> str:
    """返回窗口所属候选阶段；阶段边界外标记为 transition_or_unlabelled。"""
    for phase, selected in phase_masks.items():
        if bool(selected[index]):
            return phase
    return "transition_or_unlabelled"


def save_window_metrics(
    spectrum_time_s: np.ndarray,
    metrics: dict[str, np.ndarray],
    phase_masks: dict[str, np.ndarray],
    evaluations: dict[str, dict[str, object]],
) -> Path:
    """保存所有频谱窗、阶段标签和阈值阳性结果。"""
    target = RESULT_DIR / "target_band_window_metrics.csv"
    evaluated_names = tuple(evaluations)
    positive_columns = {
        name: _positive_mask(values, evaluations[name]["same_recording_best_threshold"])
        for name, values in metrics.items()
        if name in evaluations
    }
    fields = [
        "time_s",
        "phase",
        *metrics,
        *[f"positive_{name}" for name in evaluated_names],
    ]
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, timestamp in enumerate(spectrum_time_s):
            row: dict[str, object] = {
                "time_s": float(timestamp),
                "phase": _phase_for_index(index, phase_masks),
            }
            row.update({name: float(values[index]) for name, values in metrics.items()})
            row.update(
                {
                    f"positive_{name}": int(positive_columns[name][index])
                    for name in evaluated_names
                }
            )
            writer.writerow(row)
    return target


def _add_phase_spans(axes) -> None:
    """在图中标记三个候选阶段。"""
    colors = {
        "static_contact_hold": "#4c78a8",
        "static_friction": "#f2a541",
        "sliding": "#d64b4b",
    }
    for axis in axes:
        for phase, (start, stop) in PHASE_INTERVALS_S.items():
            axis.axvspan(start, stop, color=colors[phase], alpha=0.10)
        axis.grid(alpha=0.25)


def plot_timeline(
    spectrum_time_s: np.ndarray,
    metrics: dict[str, np.ndarray],
) -> Path:
    """绘制三种比例、峰突出度、绝对功率和绝对幅值的时间线。"""
    labels = tuple(TARGET_BANDS_HZ)
    families = (
        (
            "power fraction",
            [f"power_fraction_{label}_hz" for label in labels],
            "fraction of full 2-70 Hz power",
        ),
        (
            "local peak prominence",
            [f"peak_prominence_{label}_hz_db" for label in labels],
            "peak prominence (dB)",
        ),
        (
            "absolute band power",
            [f"power_{label}_hz" for label in labels],
            "band power ((cell/s)^2)",
        ),
        (
            "absolute band amplitude",
            [f"amplitude_{label}_hz" for label in labels],
            "band amplitude (cell/s)",
        ),
    )
    figure, axes = plt.subplots(
        len(families), 1, figsize=(16, 14), sharex=True, constrained_layout=True
    )
    for axis, (_, names, ylabel) in zip(axes, families):
        for label, name in zip(labels, names):
            axis.plot(spectrum_time_s, metrics[name], label=label.replace("_", "-") + " Hz")
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right", ncol=3)
    axes[-1].set_xlabel("time from recording start (s); window timestamp is STFT end")
    _add_phase_spans(axes)
    target = RESULT_DIR / "target_band_metrics_timeline.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def plot_phase_distributions(
    metrics: dict[str, np.ndarray],
    phase_masks: dict[str, np.ndarray],
) -> Path:
    """绘制三个目标带的比例、局部突出度和绝对功率阶段分布。"""
    labels = tuple(TARGET_BANDS_HZ)
    families = (
        (
            "power fraction",
            [f"power_fraction_{label}_hz" for label in labels],
        ),
        (
            "peak prominence (dB)",
            [f"peak_prominence_{label}_hz_db" for label in labels],
        ),
        (
            "absolute band power",
            [f"power_{label}_hz" for label in labels],
        ),
    )
    phases = ("static_contact_hold", "static_friction", "sliding")
    figure, axes = plt.subplots(
        len(families), len(labels), figsize=(15, 11), constrained_layout=True
    )
    for row, (family_label, names) in enumerate(families):
        for column, (label, name) in enumerate(zip(labels, names)):
            axis = axes[row, column]
            axis.boxplot(
                [metrics[name][phase_masks[phase]] for phase in phases],
                tick_labels=["contact", "static", "sliding"],
                showfliers=False,
            )
            axis.set_title(f"{label.replace('_', '-') } Hz")
            if column == 0:
                axis.set_ylabel(family_label)
            axis.grid(axis="y", alpha=0.25)
    target = RESULT_DIR / "target_band_phase_distributions.png"
    figure.savefig(target, dpi=180)
    plt.close(figure)
    return target


def build_report(
    spectrum,
    metrics: dict[str, np.ndarray],
    metadata: dict[str, dict[str, object]],
    phase_masks: dict[str, np.ndarray],
    evaluations: dict[str, dict[str, object]],
    outputs: dict[str, Path],
) -> dict[str, object]:
    """构造可审计 JSON 报告。"""
    phase_statistics = {
        phase: {
            name: median_iqr(values[selected])
            for name, values in metrics.items()
        }
        for phase, selected in phase_masks.items()
    }
    fraction_support = {}
    for label in TARGET_BANDS_HZ:
        name = f"power_fraction_{label}_hz"
        evaluation = evaluations[name]
        fraction_support[label] = {
            "static_friction_median": evaluation["static_friction_median"],
            "sliding_median": evaluation["sliding_median"],
            "sliding_minus_static_friction_median": evaluation[
                "sliding_minus_static_friction_median"
            ],
            "oriented_window_auc": evaluation[
                "static_friction_vs_sliding_oriented_window_auc"
            ],
            "higher_in_sliding": evaluation["sliding_direction"] == "higher_is_sliding",
        }
    return {
        "input_csv": str(INPUT_CSV),
        "phase_boundaries_are_ground_truth": False,
        "phase_boundary_note": (
            "The three intervals are fixed candidate intervals from one recording; "
            "they are not synchronized event labels or ground truth."
        ),
        "phase_intervals_s": {
            name: list(interval) for name, interval in PHASE_INTERVALS_S.items()
        },
        "analysis": {
            "sample_rate_hz": float(SPECTRUM_CONFIG.sample_rate_hz),
            "window_duration_s": float(SPECTRUM_CONFIG.detection_window_duration_s),
            "update_interval_s": float(SPECTRUM_CONFIG.detection_update_interval_s),
            "frequency_resolution_hz": 1.0 / SPECTRUM_CONFIG.detection_window_duration_s,
            "frequency_range_hz": [
                float(spectrum.frequency_hz[0]),
                float(spectrum.frequency_hz[-1]),
            ],
            "ignored_frequency_bands_hz": [
                list(band) for band in SPECTRUM_CONFIG.ignored_frequency_bands_hz
            ],
            "max_gap_s": float(SPECTRUM_CONFIG.max_gap_s),
            "power_definition": "amplitude_x**2 + amplitude_y**2",
            "fraction_denominator": "sum of all displayed 2-70 Hz power bins",
            "local_peak_prominence_definition": (
                "10*log10((target-band peak power + floor) / "
                "(mean of adjacent equal-width background-band power + floor))"
            ),
            "window_timestamp_definition": (
                "spectrum_time_s is the end timestamp of the 0.5 s STFT window"
            ),
        },
        "target_band_metadata": metadata,
        "window_counts": {
            phase: int(np.sum(selected)) for phase, selected in phase_masks.items()
        },
        "phase_statistics_median_iqr": phase_statistics,
        "static_friction_vs_sliding": evaluations,
        "target_fraction_observation": fraction_support,
        "is_production_rule": False,
        "warning": (
            "AUC, threshold, continuous-positive runs and delay are exploratory "
            "same-recording measurements. They must not be promoted to the real-time "
            "classifier without independent recordings and synchronized labels."
        ),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }


def main() -> None:
    """执行单次记录的目标带离线评估并保存全部结果。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    replay = replay_csv("full_process", INPUT_CSV)
    spectrum = build_spectrum(replay)
    metrics, metadata = calculate_target_band_metrics(
        spectrum.frequency_hz, spectrum.power
    )
    phase_masks = {
        phase: phase_mask(spectrum.time_s, interval)
        for phase, interval in PHASE_INTERVALS_S.items()
    }
    evaluations = {
        name: evaluate_metric(name, values, spectrum.time_s, phase_masks)
        for name, values in metrics.items()
        if not name.startswith("peak_frequency_")
    }
    window_csv = save_window_metrics(
        spectrum.time_s, metrics, phase_masks, evaluations
    )
    timeline_png = plot_timeline(spectrum.time_s, metrics)
    distributions_png = plot_phase_distributions(metrics, phase_masks)
    report_path = RESULT_DIR / "target_band_summary.json"
    report = build_report(
        spectrum,
        metrics,
        metadata,
        phase_masks,
        evaluations,
        {
            "window_metrics_csv": window_csv,
            "timeline_png": timeline_png,
            "phase_distributions_png": distributions_png,
            "summary_json": report_path,
        },
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
