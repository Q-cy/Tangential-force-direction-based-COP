"""评估CV、平坦度、高频占比和频谱质心区分静摩擦/滑动的能力。

数据和阶段边界复用 ``offline_friction_spectrum_0827_4.py``。所有指标按同一
0.5秒、160 Hz CoP速度窗计算；静摩擦负样本包含开始施加切向力和保持阶段。
报告中的阈值和组合分数都在同一份记录上选择，只能用于探索，不能视为独立
测试集性能或直接写入实时规则。
"""

from __future__ import annotations

import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from offline_friction_spectrum_0827 import build_spectrum, replay_csv
from offline_friction_spectrum_0827_4 import (
    INPUT_CSV,
    OUTPUT_DIR,
    PHASE_INTERVALS_S,
    phase_mask,
)


RESULT_DIR = OUTPUT_DIR / "feature_evaluation"
MAIN_HIGH_FREQUENCY_BAND_HZ = (40.0, 70.0)
SENSITIVITY_HIGH_FREQUENCY_BANDS_HZ = (
    (40.0, 70.0),
    (50.0, 70.0),
    (56.0, 70.0),
)


def power_fraction(
    power: np.ndarray,
    frequency_hz: np.ndarray,
    band_hz: tuple[float, float],
) -> np.ndarray:
    """返回指定闭区间频带占2--70 Hz总功率的比例。"""
    low, high = band_hz
    selected = (frequency_hz >= low) & (frequency_hz <= high)
    return np.sum(power[:, selected], axis=1) / (
        np.sum(power, axis=1) + np.finfo(float).tiny
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    """返回P10、P25、P50、P75和P90。"""
    probabilities = (0.10, 0.25, 0.50, 0.75, 0.90)
    result = np.quantile(values, probabilities)
    return {
        name: float(value)
        for name, value in zip(("p10", "p25", "p50", "p75", "p90"), result)
    }


def oriented_auc(negative: np.ndarray, positive: np.ndarray) -> tuple[float, str]:
    """计算两类逐窗两两比较AUC，并返回滑动对应的数值方向。"""
    comparisons = positive[:, None] - negative[None, :]
    raw_auc = float(
        np.mean(comparisons > 0.0) + 0.5 * np.mean(comparisons == 0.0)
    )
    if raw_auc >= 0.5:
        return raw_auc, "higher_is_sliding"
    return 1.0 - raw_auc, "lower_is_sliding"


def best_threshold(
    negative: np.ndarray,
    positive: np.ndarray,
    direction: str,
) -> dict[str, float | str]:
    """在当前记录上穷举单阈值，报告最高平衡准确率。"""
    values = np.unique(np.concatenate((negative, positive)))
    if values.size == 1:
        candidates = values
    else:
        candidates = np.concatenate(
            (
                [values[0] - np.finfo(float).eps],
                (values[:-1] + values[1:]) / 2.0,
                [values[-1] + np.finfo(float).eps],
            )
        )
    best = {
        "threshold": float(candidates[0]),
        "direction": direction,
        "sensitivity": 0.0,
        "specificity": 0.0,
        "balanced_accuracy": -1.0,
    }
    for threshold in candidates:
        if direction == "higher_is_sliding":
            true_positive = positive >= threshold
            true_negative = negative < threshold
        else:
            true_positive = positive <= threshold
            true_negative = negative > threshold
        sensitivity = float(np.mean(true_positive))
        specificity = float(np.mean(true_negative))
        balanced = (sensitivity + specificity) / 2.0
        if balanced > float(best["balanced_accuracy"]):
            best = {
                "threshold": float(threshold),
                "direction": direction,
                "sensitivity": sensitivity,
                "specificity": specificity,
                "balanced_accuracy": balanced,
            }
    return best


def evaluate_metric(
    negative: np.ndarray,
    positive: np.ndarray,
) -> dict[str, object]:
    """汇总一个指标的分布、AUC和同数据最优阈值。"""
    auc, direction = oriented_auc(negative, positive)
    return {
        "static_friction": quantiles(negative),
        "sliding": quantiles(positive),
        "oriented_window_auc": auc,
        "sliding_direction": direction,
        "same_recording_best_threshold": best_threshold(
            negative, positive, direction
        ),
    }


def combined_robust_score(
    static_metrics: dict[str, np.ndarray],
    sliding_metrics: dict[str, np.ndarray],
    metric_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """按静摩擦中位数/IQR标准化四指标，并沿滑动中位数方向等权组合。"""
    static_columns = []
    sliding_columns = []
    metadata: dict[str, object] = {}
    for name in metric_names:
        negative = static_metrics[name]
        positive = sliding_metrics[name]
        median = float(np.median(negative))
        iqr = float(np.quantile(negative, 0.75) - np.quantile(negative, 0.25))
        scale = max(iqr, np.finfo(float).eps)
        direction = 1.0 if np.median(positive) >= median else -1.0
        static_columns.append(direction * (negative - median) / scale)
        sliding_columns.append(direction * (positive - median) / scale)
        metadata[name] = {
            "static_median": median,
            "static_iqr": iqr,
            "direction_multiplier": direction,
        }
    static_score = np.mean(np.stack(static_columns, axis=1), axis=1)
    sliding_score = np.mean(np.stack(sliding_columns, axis=1), axis=1)
    return static_score, sliding_score, metadata


def apply_combined_score(
    metrics: dict[str, np.ndarray],
    metric_names: tuple[str, ...],
    metadata: dict[str, object],
) -> np.ndarray:
    """把已由完整静摩擦确定的方向和尺度应用到任意阶段窗口。"""
    columns = []
    for name in metric_names:
        parameters = metadata[name]
        scale = max(float(parameters["static_iqr"]), np.finfo(float).eps)
        columns.append(
            float(parameters["direction_multiplier"])
            * (metrics[name] - float(parameters["static_median"]))
            / scale
        )
    return np.mean(np.stack(columns, axis=1), axis=1)


def max_true_run(mask: np.ndarray) -> int:
    """返回布尔序列最长连续True窗数。"""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return max((int(stop - start) for start, stop in zip(starts, stops)), default=0)


def save_windows(
    spectrum,
    all_metrics: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
) -> None:
    """保存完整静摩擦和滑动逐窗指标。"""
    fields = ["phase", "time_s", *all_metrics]
    with (RESULT_DIR / "window_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for phase, selected in masks.items():
            for index in np.flatnonzero(selected):
                row = {
                    "phase": phase,
                    "time_s": float(spectrum.time_s[index]),
                }
                row.update(
                    {name: float(values[index]) for name, values in all_metrics.items()}
                )
                writer.writerow(row)


def plot_distributions(
    static_metrics: dict[str, np.ndarray],
    sliding_metrics: dict[str, np.ndarray],
    metric_names: tuple[str, ...],
) -> None:
    """绘制四个主指标的静摩擦/滑动箱线图。"""
    labels = {
        "speed_cv": "CoP speed CV",
        "spectral_flatness": "spectral flatness",
        "high_40_70_fraction": "40-70 Hz power fraction",
        "spectral_centroid_hz": "spectral centroid (Hz)",
    }
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, name in zip(axes.flat, metric_names):
        axis.boxplot(
            [static_metrics[name], sliding_metrics[name]],
            tick_labels=["static friction", "sliding"],
            showfliers=False,
        )
        axis.set_title(labels[name])
        axis.grid(axis="y", alpha=0.25)
    figure.savefig(RESULT_DIR / "metric_distributions.png", dpi=180)
    plt.close(figure)


def plot_timeline(
    spectrum,
    all_metrics: dict[str, np.ndarray],
    metric_names: tuple[str, ...],
) -> None:
    """绘制四指标全接触时间线，检查阶段开始瞬态和持续性。"""
    figure, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
    for axis, name in zip(axes, metric_names):
        axis.plot(spectrum.time_s, all_metrics[name])
        axis.set_ylabel(name)
        axis.grid(alpha=0.25)
        for phase, (start, stop) in PHASE_INTERVALS_S.items():
            color = {
                "static_contact_hold": "#4c78a8",
                "static_friction": "#f2a541",
                "sliding": "#d64b4b",
            }[phase]
            axis.axvspan(start, stop, color=color, alpha=0.10)
    axes[-1].set_xlabel("time from recording start (s)")
    figure.savefig(RESULT_DIR / "metric_timeline.png", dpi=180)
    plt.close(figure)


def main() -> None:
    """执行指标计算、区分能力评估并永久保存结果。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    replay = replay_csv("full_process", INPUT_CSV)
    spectrum = build_spectrum(replay)
    all_metrics = {
        "speed_cv": spectrum.speed_cv,
        "spectral_flatness": spectrum.flatness,
        "high_40_70_fraction": power_fraction(
            spectrum.power, spectrum.frequency_hz, (40.0, 70.0)
        ),
        "high_50_70_fraction": power_fraction(
            spectrum.power, spectrum.frequency_hz, (50.0, 70.0)
        ),
        "high_56_70_fraction": power_fraction(
            spectrum.power, spectrum.frequency_hz, (56.0, 70.0)
        ),
        "spectral_centroid_hz": spectrum.centroid_hz,
    }
    masks = {
        "static_contact_hold": phase_mask(
            spectrum.time_s, PHASE_INTERVALS_S["static_contact_hold"]
        ),
        "static_friction": phase_mask(
            spectrum.time_s, PHASE_INTERVALS_S["static_friction"]
        ),
        "sliding": phase_mask(spectrum.time_s, PHASE_INTERVALS_S["sliding"]),
    }
    static_metrics = {
        name: values[masks["static_friction"]] for name, values in all_metrics.items()
    }
    sliding_metrics = {
        name: values[masks["sliding"]] for name, values in all_metrics.items()
    }
    static_contact_metrics = {
        name: values[masks["static_contact_hold"]]
        for name, values in all_metrics.items()
    }
    evaluations = {
        name: evaluate_metric(static_metrics[name], sliding_metrics[name])
        for name in all_metrics
    }
    main_metric_names = (
        "speed_cv",
        "spectral_flatness",
        "high_40_70_fraction",
        "spectral_centroid_hz",
    )
    static_score, sliding_score, combination_metadata = combined_robust_score(
        static_metrics, sliding_metrics, main_metric_names
    )
    combined_evaluation = evaluate_metric(static_score, sliding_score)
    combined_threshold = float(
        combined_evaluation["same_recording_best_threshold"]["threshold"]
    )
    all_combined_score = apply_combined_score(
        all_metrics, main_metric_names, combination_metadata
    )
    combined_runs = {
        phase: max_true_run(all_combined_score[selected] >= combined_threshold)
        for phase, selected in masks.items()
    }
    report = {
        "input_csv": str(INPUT_CSV),
        "phase_boundaries_are_ground_truth": False,
        "static_friction_interval_s": list(PHASE_INTERVALS_S["static_friction"]),
        "sliding_interval_s": list(PHASE_INTERVALS_S["sliding"]),
        "window_duration_s": 0.5,
        "update_interval_s": 0.05,
        "static_friction_windows": int(np.sum(masks["static_friction"])),
        "sliding_windows": int(np.sum(masks["sliding"])),
        "cv_definition": "std(COP speed magnitude) / mean(COP speed magnitude)",
        "high_frequency_sensitivity_bands_hz": [
            list(item) for item in SENSITIVITY_HIGH_FREQUENCY_BANDS_HZ
        ],
        "metrics": evaluations,
        "static_contact_safety_check": {
            name: quantiles(values) for name, values in static_contact_metrics.items()
        },
        "equal_weight_robust_combination": {
            "metric_names": list(main_metric_names),
            "normalization": combination_metadata,
            "evaluation": combined_evaluation,
            "static_contact_score": quantiles(
                all_combined_score[masks["static_contact_hold"]]
            ),
            "max_consecutive_positive_windows_by_phase": combined_runs,
            "warning": (
                "Directions, scaling and threshold were selected on this same recording; "
                "not an independent validation result. Static-contact false runs show "
                "that this combination is not a complete production detector."
            ),
        },
    }
    save_windows(spectrum, all_metrics, masks)
    plot_distributions(static_metrics, sliding_metrics, main_metric_names)
    plot_timeline(spectrum, all_metrics, main_metric_names)
    (RESULT_DIR / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
