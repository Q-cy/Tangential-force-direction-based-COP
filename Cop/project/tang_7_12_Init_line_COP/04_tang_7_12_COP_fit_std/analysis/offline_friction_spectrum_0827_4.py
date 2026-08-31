"""分析 COP_test_0827_4.csv 的单次完整摩擦过程。

实验顺序为无接触、静止接触保持、静摩擦保持、滑动、释放。CSV没有人工
事件标记列，因此 ``PHASE_INTERVALS_S`` 是依据 CoP/ADC 时间线选取的候选稳定
区间；每个STFT窗口必须完整落在区间内。需要精确验收实时判定前，应使用同步
按钮、视频或数字事件通道替换这些候选边界。

运行方式：

    PYTHONPATH=src MPLCONFIGDIR=/tmp/pzt-mplconfig \
    /home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python \
    analysis/offline_friction_spectrum_0827_4.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from offline_friction_spectrum_0827 import (
    PROJECT_ROOT,
    SPECTRUM_CONFIG,
    _quantiles,
    _true_runs,
    build_spectrum,
    replay_csv,
)


INPUT_CSV = PROJECT_ROOT / "data/COP_test_0827_4.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/offline_spectrum_0827_4"

# 单位为相对CSV第一帧的秒。避开阶段切换，并只分析候选稳定保持区间。
PHASE_INTERVALS_S = {
    "static_contact_hold": (4.10, 6.20),
    # 包含开始施加切向力、压力斑块快速形变以及随后保持的完整静摩擦阶段。
    "static_friction": (6.20, 10.30),
    "sliding": (10.70, 13.20),
}

# 本批数据的探索性规则，不写入实时配置。0.5秒窗每0.05秒更新，连续9窗
# 从首窗到末窗跨约0.4秒，并且每窗自身覆盖0.5秒数据。
SLIP_BAND_HZ = (24.0, 28.0)
SLIP_BAND_FRACTION_THRESHOLD = 0.095
SLIP_PEAK_FREQUENCY_HZ = 26.0
SLIP_PEAK_NEIGHBOR_BAND_HZ = (20.0, 32.0)
SLIP_PEAK_PROMINENCE_DB = 1.09
SLIP_BAND_CONSECUTIVE_WINDOWS = 9


def phase_mask(time_s: np.ndarray, interval: tuple[float, float]) -> np.ndarray:
    """选择完整落在候选阶段内的0.5秒STFT窗口。"""
    start, stop = interval
    return (
        (time_s >= start + SPECTRUM_CONFIG.detection_window_duration_s)
        & (time_s <= stop)
    )


def band_fraction(power: np.ndarray, frequency_hz: np.ndarray) -> np.ndarray:
    """返回每个窗口中24--28 Hz占2--70 Hz总功率的比例。"""
    low, high = SLIP_BAND_HZ
    selected = (frequency_hz >= low) & (frequency_hz <= high)
    return np.sum(power[:, selected], axis=1) / (
        np.sum(power, axis=1) + np.finfo(float).tiny
    )


def peak_prominence_db(power: np.ndarray, frequency_hz: np.ndarray) -> np.ndarray:
    """返回26 Hz功率相对20--32 Hz其余频点平均功率的dB差。"""
    low, high = SLIP_PEAK_NEIGHBOR_BAND_HZ
    peak = np.isclose(frequency_hz, SLIP_PEAK_FREQUENCY_HZ)
    neighbors = (
        (frequency_hz >= low)
        & (frequency_hz <= high)
        & ~peak
    )
    floor = SPECTRUM_CONFIG.baseline_power_floor
    return 10.0 * np.log10(
        (power[:, peak].reshape(-1) + floor)
        / (np.mean(power[:, neighbors], axis=1) + floor)
    )


def save_window_features(
    spectrum, fractions: np.ndarray, prominence_db: np.ndarray
) -> None:
    """保存每个窗口的阶段、频谱形状和24--28 Hz功率占比。"""
    target = OUTPUT_DIR / "window_features.csv"
    fields = [
        "time_s", "phase", "centroid_hz", "entropy", "flatness",
        "upper_56_70_fraction", "band_24_28_fraction",
        "peak_26_prominence_db",
    ]
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, timestamp in enumerate(spectrum.time_s):
            phase = "transition_or_unlabelled"
            for name, interval in PHASE_INTERVALS_S.items():
                if phase_mask(spectrum.time_s, interval)[index]:
                    phase = name
                    break
            writer.writerow(
                {
                    "time_s": float(timestamp),
                    "phase": phase,
                    "centroid_hz": float(spectrum.centroid_hz[index]),
                    "entropy": float(spectrum.entropy[index]),
                    "flatness": float(spectrum.flatness[index]),
                    "upper_56_70_fraction": float(
                        spectrum.upper_band_fraction[index]
                    ),
                    "band_24_28_fraction": float(fractions[index]),
                    "peak_26_prominence_db": float(prominence_db[index]),
                }
            )


def plot_timeline(
    replay, spectrum, fractions: np.ndarray, prominence_db: np.ndarray
) -> None:
    """绘制CoP、ADC、状态以及24--28 Hz比例的完整实验时间线。"""
    frame_time = replay.time_s - replay.time_s[0]
    figure, axes = plt.subplots(
        5, 1, figsize=(16, 15), sharex=True, constrained_layout=True
    )
    axes[0].plot(frame_time, replay.cop_x, label="CoP X")
    axes[0].plot(frame_time, replay.cop_y, label="CoP Y")
    axes[0].set_ylabel("absolute CoP (cell)")
    axes[0].legend()
    axes[1].plot(frame_time, replay.adc_sum, color="tab:purple")
    axes[1].set_ylabel("ADC sum")
    axes[2].step(frame_time, replay.state, where="post", color="black")
    axes[2].set_ylabel("CoP state")
    axes[3].plot(
        spectrum.time_s,
        fractions,
        color="tab:red",
        label="24-28 Hz power fraction",
    )
    axes[3].axhline(
        SLIP_BAND_FRACTION_THRESHOLD,
        color="black",
        linestyle=":",
        label="exploratory threshold",
    )
    axes[3].set_ylabel("power fraction")
    axes[3].legend()
    axes[4].plot(
        spectrum.time_s,
        prominence_db,
        color="tab:orange",
        label="26 Hz prominence",
    )
    axes[4].axhline(
        SLIP_PEAK_PROMINENCE_DB,
        color="black",
        linestyle=":",
        label="exploratory threshold",
    )
    axes[4].set_ylabel("prominence (dB)")
    axes[4].set_xlabel("time from recording start (s)")
    axes[4].legend()
    colors = {
        "static_contact_hold": "#4c78a8",
        "static_friction": "#f2a541",
        "sliding": "#d64b4b",
    }
    for axis in axes:
        for name, (start, stop) in PHASE_INTERVALS_S.items():
            axis.axvspan(start, stop, color=colors[name], alpha=0.10)
        axis.grid(alpha=0.22)
    figure.savefig(OUTPUT_DIR / "full_process_timeline.png", dpi=180)
    plt.close(figure)


def plot_spectrogram(spectrum) -> None:
    """绘制完整过程的CoP速度时频图并标出候选稳定阶段。"""
    power_reference = np.median(
        spectrum.power[
            phase_mask(spectrum.time_s, PHASE_INTERVALS_S["static_contact_hold"])
        ],
        axis=0,
    )
    floor = SPECTRUM_CONFIG.baseline_power_floor
    relative_db = 10.0 * np.log10(
        (spectrum.power + floor) / (power_reference[None, :] + floor)
    )
    figure, axis = plt.subplots(figsize=(16, 7), constrained_layout=True)
    mesh = axis.pcolormesh(
        spectrum.time_s,
        spectrum.frequency_hz,
        relative_db.T,
        shading="auto",
        cmap="magma",
        vmin=-10.0,
        vmax=35.0,
    )
    colors = {
        "static_contact_hold": "cyan",
        "static_friction": "lime",
        "sliding": "white",
    }
    for name, (start, stop) in PHASE_INTERVALS_S.items():
        axis.axvline(start, color=colors[name], linestyle="--", linewidth=1.5)
        axis.axvline(stop, color=colors[name], linestyle=":", linewidth=1.2)
        axis.text(
            (start + stop) / 2.0,
            69.0,
            name,
            color=colors[name],
            horizontalalignment="center",
            verticalalignment="top",
        )
    axis.set_title("COP_test_0827_4: CoP velocity STFT")
    axis.set_xlabel("time from recording start (s)")
    axis.set_ylabel("frequency (Hz)")
    figure.colorbar(mesh, ax=axis, label="power vs static-contact median (dB)")
    figure.savefig(OUTPUT_DIR / "full_process_spectrogram.png", dpi=180)
    plt.close(figure)


def plot_phase_spectra(spectrum) -> None:
    """比较同一次接触三个候选稳定阶段的速度谱。"""
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    phase_power: dict[str, np.ndarray] = {}
    for name, interval in PHASE_INTERVALS_S.items():
        selected = spectrum.power[phase_mask(spectrum.time_s, interval)]
        phase_power[name] = selected
        amplitude = np.sqrt(selected)
        median = np.median(amplitude, axis=0)
        lower = np.quantile(amplitude, 0.10, axis=0)
        upper = np.quantile(amplitude, 0.90, axis=0)
        line = axes[0].plot(spectrum.frequency_hz, median, label=name)[0]
        axes[0].fill_between(
            spectrum.frequency_hz,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.12,
        )
    axes[0].axvspan(*SLIP_BAND_HZ, color="red", alpha=0.08)
    axes[0].set_title("Median CoP velocity spectrum and P10-P90")
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("single-sided amplitude (cell/s)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    floor = SPECTRUM_CONFIG.baseline_power_floor
    static_friction = np.median(phase_power["static_friction"], axis=0)
    sliding = np.median(phase_power["sliding"], axis=0)
    ratio_db = 10.0 * np.log10(
        (sliding + floor) / (static_friction + floor)
    )
    axes[1].plot(spectrum.frequency_hz, ratio_db, color="tab:red")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].axvspan(*SLIP_BAND_HZ, color="red", alpha=0.08)
    axes[1].set_title("Sliding median power / static-friction median power")
    axes[1].set_xlabel("frequency (Hz)")
    axes[1].set_ylabel("dB")
    axes[1].grid(alpha=0.25)
    figure.savefig(OUTPUT_DIR / "phase_spectrum_comparison.png", dpi=180)
    plt.close(figure)


def build_summary(
    replay,
    spectrum,
    fractions: np.ndarray,
    prominence_db: np.ndarray,
) -> dict[str, object]:
    """构造阶段统计、频率差异和探索性连续窗规则报告。"""
    phases: dict[str, object] = {}
    phase_power: dict[str, np.ndarray] = {}
    for name, interval in PHASE_INTERVALS_S.items():
        selected_mask = phase_mask(spectrum.time_s, interval)
        selected_power = spectrum.power[selected_mask]
        phase_power[name] = selected_power
        phases[name] = {
            "candidate_interval_s": list(interval),
            "complete_stft_windows": int(np.sum(selected_mask)),
            "centroid_hz": _quantiles(spectrum.centroid_hz[selected_mask]),
            "entropy": _quantiles(spectrum.entropy[selected_mask]),
            "flatness": _quantiles(spectrum.flatness[selected_mask]),
            "upper_56_70_fraction": _quantiles(
                spectrum.upper_band_fraction[selected_mask]
            ),
            "band_24_28_fraction": _quantiles(fractions[selected_mask]),
            "peak_26_prominence_db": _quantiles(
                prominence_db[selected_mask]
            ),
        }

    floor = SPECTRUM_CONFIG.baseline_power_floor
    stick_power = np.median(phase_power["static_friction"], axis=0)
    slip_power = np.median(phase_power["sliding"], axis=0)
    difference_db = 10.0 * np.log10(
        (slip_power + floor) / (stick_power + floor)
    )
    strongest = np.argsort(difference_db)[-12:][::-1]

    threshold_mask = (
        (fractions >= SLIP_BAND_FRACTION_THRESHOLD)
        & (prominence_db >= SLIP_PEAK_PROMINENCE_DB)
    )
    runs = []
    for start, stop in _true_runs(threshold_mask):
        if stop - start < SLIP_BAND_CONSECUTIVE_WINDOWS:
            continue
        confirmation_index = start + SLIP_BAND_CONSECUTIVE_WINDOWS - 1
        runs.append(
            {
                "start_s": float(spectrum.time_s[start]),
                "confirmation_s": float(spectrum.time_s[confirmation_index]),
                "stop_s": float(spectrum.time_s[stop - 1]),
                "windows": int(stop - start),
            }
        )
    max_runs_by_phase: dict[str, int] = {}
    for name, interval in PHASE_INTERVALS_S.items():
        selected = threshold_mask[phase_mask(spectrum.time_s, interval)]
        max_runs_by_phase[name] = max(
            (stop - start for start, stop in _true_runs(selected)),
            default=0,
        )
    return {
        "input_csv": str(INPUT_CSV),
        "phase_boundaries_are_ground_truth": False,
        "phase_boundary_note": (
            "Candidate stable intervals inferred from CoP/ADC timeline; replace with "
            "synchronized event labels for validation."
        ),
        "analysis": {
            "sample_rate_hz": SPECTRUM_CONFIG.sample_rate_hz,
            "window_duration_s": SPECTRUM_CONFIG.detection_window_duration_s,
            "update_interval_s": SPECTRUM_CONFIG.detection_update_interval_s,
            "frequency_resolution_hz": 2.0,
            "frequency_range_hz": [2.0, 70.0],
            "ignored_frequency_bands_hz": [],
            "max_gap_s": SPECTRUM_CONFIG.max_gap_s,
        },
        "input_quality": {
            "rows": int(replay.time_s.size),
            "duration_s": float(replay.time_s[-1] - replay.time_s[0]),
            "state_replay_agreement": float(np.mean(replay.state == replay.csv_state)),
            "max_frame_gap_s": replay.max_gap_s,
            "gaps_over_30_ms": replay.gaps_over_30_ms,
            "gaps_over_75_ms": replay.gaps_over_75_ms,
            "gaps_over_160_ms": replay.gaps_over_160_ms,
        },
        "phases": phases,
        "sliding_over_static_friction_top_frequencies": [
            {
                "frequency_hz": float(spectrum.frequency_hz[index]),
                "power_difference_db": float(difference_db[index]),
            }
            for index in strongest
        ],
        "exploratory_rule": {
            "rule": (
                f"24-28 Hz power fraction >= {SLIP_BAND_FRACTION_THRESHOLD:.3f} "
                f"and 26 Hz prominence >= {SLIP_PEAK_PROMINENCE_DB:.2f} dB "
                f"for {SLIP_BAND_CONSECUTIVE_WINDOWS} consecutive windows"
            ),
            "is_production_rule": False,
            "max_consecutive_matching_windows_by_phase": max_runs_by_phase,
            "matching_runs": runs,
        },
    }


def main() -> None:
    """运行完整过程分析并永久保存脚本可复现的图表和表格。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    replay = replay_csv("full_process", INPUT_CSV)
    spectrum = build_spectrum(replay)
    fractions = band_fraction(spectrum.power, spectrum.frequency_hz)
    prominence_db = peak_prominence_db(spectrum.power, spectrum.frequency_hz)
    save_window_features(spectrum, fractions, prominence_db)
    plot_timeline(replay, spectrum, fractions, prominence_db)
    plot_spectrogram(spectrum)
    plot_phase_spectra(spectrum)
    summary = build_summary(replay, spectrum, fractions, prominence_db)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
