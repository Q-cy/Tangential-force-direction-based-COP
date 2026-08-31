"""复现 2026-08-27 三次摩擦实验的离线 CoP 频谱分析。

本脚本不被 ``tangential`` 运行时导入。它从 CSV 的84通道数据重新执行现有
CoP处理链，再调用运行时 ``CopSpectrumAnalyzer`` 生成完全相同的160 Hz、
0.5秒速度STFT。第三次实验没有人工滑动时刻列，因此输出只把频谱变化点标为
候选滑动开始时刻，不能替代同步视频或人工事件标记。

运行方式（项目验收环境）：

    PYTHONPATH=src MPLCONFIGDIR=/tmp/pzt-mplconfig \
    /home/qcy/miniconda3/envs/TimeDrift_GRU/bin/python \
    analysis/offline_friction_spectrum_0827.py
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tangential.config import (
    ConsistenceCalibrationConfig,
    ProcessingConfig,
    SpectrumConfig,
)
from tangential.processing.spectrum import CopSpectrumAnalyzer
from tangential.runtime.sensor import TangentialSampleProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILES = {
    "static_contact": PROJECT_ROOT / "data/COP_test_0827_1.csv",
    "static_friction": PROJECT_ROOT / "data/COP_test_0827_2.csv",
    "stick_to_slip": PROJECT_ROOT / "data/COP_test_0827_3.csv",
}
OUTPUT_DIR = PROJECT_ROOT / "data/offline_spectrum_0827"

# 固定为本次可复现实验参数，不修改实时应用的全局对象。
SPECTRUM_CONFIG = SpectrumConfig(
    ignored_frequency_bands_hz=(),
    max_gap_s=0.160,
)
UPPER_BAND_MIN_HZ = 56.0
SUSTAINED_WINDOWS = 8
REFERENCE_QUANTILE = 0.90


@dataclass
class ReplayResult:
    """保存一份 CSV 的逐帧回放结果和协议质量信息。"""

    name: str
    path: Path
    time_s: np.ndarray
    cop_x: np.ndarray
    cop_y: np.ndarray
    state: np.ndarray
    csv_state: np.ndarray
    adc_sum: np.ndarray
    max_gap_s: float
    gaps_over_30_ms: int
    gaps_over_75_ms: int
    gaps_over_160_ms: int


@dataclass
class SpectrumResult:
    """保存同实时分析器生成的快照及派生的无基线频谱形状特征。"""

    name: str
    frequency_hz: np.ndarray
    time_s: np.ndarray
    power: np.ndarray
    amplitude: np.ndarray
    centroid_hz: np.ndarray
    entropy: np.ndarray
    flatness: np.ndarray
    upper_band_fraction: np.ndarray
    speed_cv: np.ndarray
    motion_db: np.ndarray | None = None


def _read_rows(path: Path) -> list[dict[str, str]]:
    """读取一份108列采集CSV并返回按文件顺序排列的行。"""
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"press_t", "CoP_state", *(f"ch{i}" for i in range(1, 85))}
    missing = sorted(required.difference(rows[0] if rows else {}))
    if not rows:
        raise ValueError(f"CSV没有数据行: {path}")
    if missing:
        raise ValueError(f"CSV缺少分析列 {missing}: {path}")
    return rows


def replay_csv(name: str, path: Path) -> ReplayResult:
    """从ADC重新执行现有CoP状态机，避免依赖CSV中的滤波后CoP偏移。"""
    rows = _read_rows(path)
    processor = TangentialSampleProcessor(
        calibration=None,
        processing_config=ProcessingConfig(
            consistence=ConsistenceCalibrationConfig(enabled=False),
        ),
    )
    times: list[float] = []
    cop_x: list[float] = []
    cop_y: list[float] = []
    states: list[int] = []
    csv_states: list[int] = []
    adc_sums: list[float] = []
    for sequence, row in enumerate(rows):
        raw_data = np.asarray(
            [float(row[f"ch{channel}"]) for channel in range(1, 85)],
            dtype=np.float64,
        )
        timestamp = float(row["press_t"])
        sample = processor._process_sample(
            raw_data,
            {
                "request_seq": sequence,
                "tx_t": timestamp,
                "rx_t": timestamp,
                "latency_s": 0.0,
            },
        )
        times.append(timestamp)
        cop_x.append(sample.cop_x)
        cop_y.append(sample.cop_y)
        states.append(sample.state)
        csv_states.append(int(float(row["CoP_state"])))
        adc_sums.append(sample.adc_sum)

    time_array = np.asarray(times, dtype=np.float64)
    gaps = np.diff(time_array)
    return ReplayResult(
        name=name,
        path=path,
        time_s=time_array,
        cop_x=np.asarray(cop_x, dtype=np.float64),
        cop_y=np.asarray(cop_y, dtype=np.float64),
        state=np.asarray(states, dtype=np.int16),
        csv_state=np.asarray(csv_states, dtype=np.int16),
        adc_sum=np.asarray(adc_sums, dtype=np.float64),
        max_gap_s=float(np.max(gaps)),
        gaps_over_30_ms=int(np.sum(gaps > 0.030)),
        gaps_over_75_ms=int(np.sum(gaps > 0.075)),
        gaps_over_160_ms=int(np.sum(gaps > 0.160)),
    )


def build_spectrum(replay: ReplayResult) -> SpectrumResult:
    """调用实时频谱分析器，并从同一快照计算绝对谱形状特征。"""
    analyzer = CopSpectrumAnalyzer(config=SPECTRUM_CONFIG)
    speed_cv_values: list[float] = []
    for timestamp, cop_x, cop_y, state in zip(
        replay.time_s, replay.cop_x, replay.cop_y, replay.state
    ):
        snapshot = analyzer.process(timestamp, cop_x, cop_y, int(state))
        if snapshot is not None:
            # 离线指标必须与该快照使用完全相同的81个重采样位置点。分析脚本
            # 读取分析器短窗但不修改它；运行时快照/API不增加实验字段。
            position_x = np.asarray(analyzer._resampled_x, dtype=np.float64)[
                -analyzer.required_samples:
            ]
            position_y = np.asarray(analyzer._resampled_y, dtype=np.float64)[
                -analyzer.required_samples:
            ]
            velocity_magnitude = np.hypot(
                np.diff(position_x) * analyzer.sample_rate_hz,
                np.diff(position_y) * analyzer.sample_rate_hz,
            )
            mean_speed = float(np.mean(velocity_magnitude))
            speed_cv_values.append(
                float(np.std(velocity_magnitude) / mean_speed)
                if mean_speed > np.finfo(float).eps
                else 0.0
            )
    snapshots = analyzer.snapshots
    if not snapshots:
        raise ValueError(f"没有形成频谱快照: {replay.path}")

    frequency = np.asarray(snapshots[0].frequency_hz, dtype=np.float64)
    amplitude_x = np.stack(
        [snapshot.velocity_amplitude_x for snapshot in snapshots]
    ).astype(np.float64)
    amplitude_y = np.stack(
        [snapshot.velocity_amplitude_y for snapshot in snapshots]
    ).astype(np.float64)
    power = amplitude_x * amplitude_x + amplitude_y * amplitude_y
    amplitude = np.sqrt(power)
    total_power = np.sum(power, axis=1) + np.finfo(float).tiny
    probability = power / total_power[:, None]
    centroid = np.sum(probability * frequency[None, :], axis=1)
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, np.finfo(float).tiny)), axis=1
    ) / np.log(float(frequency.size))
    positive_power = power + np.finfo(float).tiny
    flatness = np.exp(np.mean(np.log(positive_power), axis=1)) / np.mean(
        positive_power, axis=1
    )
    upper = frequency >= UPPER_BAND_MIN_HZ
    upper_fraction = np.sum(power[:, upper], axis=1) / total_power
    return SpectrumResult(
        name=replay.name,
        frequency_hz=frequency,
        time_s=np.asarray(
            [snapshot.spectrum_time_s for snapshot in snapshots], dtype=np.float64
        ),
        power=power,
        amplitude=amplitude,
        centroid_hz=centroid,
        entropy=entropy,
        flatness=flatness,
        upper_band_fraction=upper_fraction,
        speed_cv=np.asarray(speed_cv_values, dtype=np.float64),
    )


def add_static_reference(spectra: dict[str, SpectrumResult]) -> np.ndarray:
    """以静止接触逐频点中位功率为共同参考，计算所有窗口的运动dB。"""
    baseline = np.median(spectra["static_contact"].power, axis=0)
    floor = SPECTRUM_CONFIG.baseline_power_floor
    baseline_mean = float(np.mean(baseline))
    for spectrum in spectra.values():
        spectrum.motion_db = 10.0 * np.log10(
            (np.mean(spectrum.power, axis=1) + floor) / (baseline_mean + floor)
        )
    return baseline


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """返回布尔序列中所有半开连续True区间。"""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    differences = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(differences == 1)
    stops = np.flatnonzero(differences == -1)
    return list(zip(starts.tolist(), stops.tolist()))


def estimate_transition(
    static_friction: SpectrumResult,
    mixed: SpectrumResult,
) -> tuple[int | None, dict[str, float], np.ndarray]:
    """以独立静摩擦记录P90建立候选转换规则。

    质心、平坦度和56--70 Hz功率占比中至少两项超过静摩擦P90，并连续满足
    ``SUSTAINED_WINDOWS`` 个窗口时，把该连续段首窗标为候选滑动开始。该规则
    只用于本次探索，不写回实时配置。
    """
    thresholds = {
        "centroid_hz": float(
            np.quantile(static_friction.centroid_hz, REFERENCE_QUANTILE)
        ),
        "flatness": float(
            np.quantile(static_friction.flatness, REFERENCE_QUANTILE)
        ),
        "upper_band_fraction": float(
            np.quantile(static_friction.upper_band_fraction, REFERENCE_QUANTILE)
        ),
    }
    evidence_count = (
        (mixed.centroid_hz > thresholds["centroid_hz"]).astype(np.int8)
        + (mixed.flatness > thresholds["flatness"]).astype(np.int8)
        + (
            mixed.upper_band_fraction > thresholds["upper_band_fraction"]
        ).astype(np.int8)
    )
    evidence = evidence_count >= 2
    for start, stop in _true_runs(evidence):
        if stop - start >= SUSTAINED_WINDOWS:
            return start, thresholds, evidence_count
    return None, thresholds, evidence_count


def _quantiles(values: np.ndarray) -> dict[str, float]:
    """返回用于报告的P10/P50/P90。"""
    return {
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
    }


def write_window_csv(
    spectra: dict[str, SpectrumResult],
    transition_index: int | None,
    evidence_count: np.ndarray,
) -> None:
    """保存每个STFT窗口的可审计特征和第三次实验候选阶段。"""
    target = OUTPUT_DIR / "window_features.csv"
    fields = [
        "recording", "time_s", "phase", "motion_db", "centroid_hz",
        "entropy", "flatness", "upper_56_70_fraction", "evidence_count",
    ]
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for name, spectrum in spectra.items():
            for index in range(spectrum.time_s.size):
                if name == "stick_to_slip":
                    phase = (
                        "candidate_slip"
                        if transition_index is not None and index >= transition_index
                        else "pre_transition_stick"
                    )
                    evidence = int(evidence_count[index])
                else:
                    phase = name
                    evidence = ""
                writer.writerow(
                    {
                        "recording": name,
                        "time_s": float(spectrum.time_s[index]),
                        "phase": phase,
                        "motion_db": float(spectrum.motion_db[index]),
                        "centroid_hz": float(spectrum.centroid_hz[index]),
                        "entropy": float(spectrum.entropy[index]),
                        "flatness": float(spectrum.flatness[index]),
                        "upper_56_70_fraction": float(
                            spectrum.upper_band_fraction[index]
                        ),
                        "evidence_count": evidence,
                    }
                )


def plot_cop(replays: dict[str, ReplayResult], transition_time_s: float | None) -> None:
    """绘制三次实验的CoP、ADC总和和状态时间线。"""
    figure, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    for axis, (name, replay) in zip(axes, replays.items()):
        relative_time = replay.time_s - replay.time_s[0]
        axis.plot(relative_time, replay.cop_x, label="CoP X", linewidth=1.1)
        axis.plot(relative_time, replay.cop_y, label="CoP Y", linewidth=1.1)
        state_axis = axis.twinx()
        state_axis.step(
            relative_time, replay.state, where="post", color="black", alpha=0.22,
            label="CoP state",
        )
        state_axis.set_ylim(-0.1, 2.3)
        state_axis.set_ylabel("state")
        if name == "stick_to_slip" and transition_time_s is not None:
            axis.axvline(
                transition_time_s,
                color="red",
                linestyle="--",
                label="candidate slip onset",
            )
        axis.set_title(name)
        axis.set_xlabel("time from recording start (s)")
        axis.set_ylabel("absolute CoP (cell)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
    figure.savefig(OUTPUT_DIR / "cop_and_state_timeline.png", dpi=180)
    plt.close(figure)


def plot_spectrograms(
    spectra: dict[str, SpectrumResult],
    baseline: np.ndarray,
    transition_time_s: float | None,
) -> None:
    """绘制所有频点相对共同静止基线的时频图。"""
    floor = SPECTRUM_CONFIG.baseline_power_floor
    figure, axes = plt.subplots(
        3, 1, figsize=(14, 12), sharex=False, constrained_layout=True
    )
    for axis, (name, spectrum) in zip(axes, spectra.items()):
        relative_db = 10.0 * np.log10(
            (spectrum.power + floor) / (baseline[None, :] + floor)
        )
        mesh = axis.pcolormesh(
            spectrum.time_s,
            spectrum.frequency_hz,
            relative_db.T,
            shading="auto",
            cmap="magma",
            vmin=-10.0,
            vmax=35.0,
        )
        if name == "stick_to_slip" and transition_time_s is not None:
            axis.axvline(transition_time_s, color="cyan", linestyle="--", linewidth=2)
        axis.set_title(name)
        axis.set_xlabel("time from recording start (s)")
        axis.set_ylabel("frequency (Hz)")
        figure.colorbar(mesh, ax=axis, label="power vs static baseline (dB)")
    figure.savefig(OUTPUT_DIR / "spectrograms.png", dpi=180)
    plt.close(figure)


def plot_feature_timeline(
    static_friction: SpectrumResult,
    mixed: SpectrumResult,
    thresholds: dict[str, float],
    transition_time_s: float | None,
) -> None:
    """绘制静摩擦参考与混合记录的三个候选滑动特征。"""
    features = (
        ("centroid_hz", "spectral centroid (Hz)"),
        ("flatness", "spectral flatness"),
        ("upper_band_fraction", "56-70 Hz power fraction"),
    )
    figure, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False, constrained_layout=True)
    for axis, (field_name, label) in zip(axes, features):
        axis.plot(
            static_friction.time_s,
            getattr(static_friction, field_name),
            label="static friction reference",
            alpha=0.8,
        )
        axis.plot(
            mixed.time_s,
            getattr(mixed, field_name),
            label="stick-to-slip recording",
            alpha=0.9,
        )
        axis.axhline(
            thresholds[field_name], color="black", linestyle=":", label="stick P90"
        )
        if transition_time_s is not None:
            axis.axvline(
                transition_time_s,
                color="red",
                linestyle="--",
                label="candidate slip onset",
            )
        axis.set_ylabel(label)
        axis.set_xlabel("time from recording start (s)")
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    figure.savefig(OUTPUT_DIR / "feature_timeline.png", dpi=180)
    plt.close(figure)


def plot_phase_spectra(
    spectra: dict[str, SpectrumResult], transition_index: int | None
) -> None:
    """比较静止、静摩擦以及第三次实验转换前后的中位速度谱。"""
    phases: list[tuple[str, np.ndarray, np.ndarray]] = [
        (
            "static contact",
            spectra["static_contact"].frequency_hz,
            spectra["static_contact"].amplitude,
        ),
        (
            "static friction",
            spectra["static_friction"].frequency_hz,
            spectra["static_friction"].amplitude,
        ),
    ]
    mixed = spectra["stick_to_slip"]
    if transition_index is not None and 0 < transition_index < mixed.time_s.size:
        phases.extend(
            [
                ("mixed: pre-transition", mixed.frequency_hz, mixed.amplitude[:transition_index]),
                ("mixed: candidate slip", mixed.frequency_hz, mixed.amplitude[transition_index:]),
            ]
        )
    else:
        phases.append(("mixed: unsplit", mixed.frequency_hz, mixed.amplitude))

    figure, axis = plt.subplots(figsize=(14, 7), constrained_layout=True)
    for label, frequency, amplitude in phases:
        median = np.median(amplitude, axis=0)
        lower = np.quantile(amplitude, 0.10, axis=0)
        upper = np.quantile(amplitude, 0.90, axis=0)
        line = axis.plot(frequency, median, label=label)[0]
        axis.fill_between(frequency, lower, upper, color=line.get_color(), alpha=0.10)
    axis.set_title("CoP velocity spectrum by experiment phase")
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("single-sided amplitude (cell/s)")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "phase_spectrum_comparison.png", dpi=180)
    plt.close(figure)


def build_summary(
    replays: dict[str, ReplayResult],
    spectra: dict[str, SpectrumResult],
    transition_index: int | None,
    thresholds: dict[str, float],
    evidence_count: np.ndarray,
) -> dict[str, object]:
    """构造包含输入质量、特征分布和候选转换的JSON报告。"""
    recordings: dict[str, object] = {}
    for name, replay in replays.items():
        spectrum = spectra[name]
        state_two = np.flatnonzero(replay.state == SPECTRUM_CONFIG.required_cop_state)
        recordings[name] = {
            "input_csv": str(replay.path),
            "rows": int(replay.time_s.size),
            "duration_s": float(replay.time_s[-1] - replay.time_s[0]),
            "state_replay_agreement": float(np.mean(replay.state == replay.csv_state)),
            "state_2_rows": int(state_two.size),
            "state_2_start_s": float(replay.time_s[state_two[0]] - replay.time_s[0]),
            "state_2_stop_s": float(replay.time_s[state_two[-1]] - replay.time_s[0]),
            "max_frame_gap_s": replay.max_gap_s,
            "gaps_over_30_ms": replay.gaps_over_30_ms,
            "gaps_over_75_ms": replay.gaps_over_75_ms,
            "gaps_over_160_ms": replay.gaps_over_160_ms,
            "spectrum_windows": int(spectrum.time_s.size),
            "features": {
                "motion_db": _quantiles(spectrum.motion_db),
                "centroid_hz": _quantiles(spectrum.centroid_hz),
                "entropy": _quantiles(spectrum.entropy),
                "flatness": _quantiles(spectrum.flatness),
                "upper_56_70_fraction": _quantiles(spectrum.upper_band_fraction),
            },
        }

    transition: dict[str, object] = {
        "is_ground_truth": False,
        "method": (
            "At least two of centroid, flatness, and 56-70 Hz fraction exceed "
            f"the independent static-friction P{int(REFERENCE_QUANTILE * 100)} "
            f"for {SUSTAINED_WINDOWS} consecutive windows."
        ),
        "thresholds": thresholds,
        "candidate_found": transition_index is not None,
    }
    if transition_index is not None:
        mixed = spectra["stick_to_slip"]
        transition.update(
            {
                "candidate_window_index": int(transition_index),
                "candidate_time_from_recording_start_s": float(
                    mixed.time_s[transition_index]
                ),
                "evidence_count_at_candidate": int(evidence_count[transition_index]),
                "pre_transition_windows": int(transition_index),
                "candidate_slip_windows": int(mixed.time_s.size - transition_index),
            }
        )
    return {
        "analysis": {
            "sample_rate_hz": SPECTRUM_CONFIG.sample_rate_hz,
            "window_duration_s": SPECTRUM_CONFIG.detection_window_duration_s,
            "update_interval_s": SPECTRUM_CONFIG.detection_update_interval_s,
            "frequency_resolution_hz": 1.0 / SPECTRUM_CONFIG.detection_window_duration_s,
            "frequency_range_hz": [
                SPECTRUM_CONFIG.detection_min_frequency_hz,
                SPECTRUM_CONFIG.detection_max_frequency_hz,
            ],
            "ignored_frequency_bands_hz": [],
            "max_gap_s": SPECTRUM_CONFIG.max_gap_s,
            "note": (
                "Gaps no longer than max_gap_s are linearly interpolated; generated "
                "grid points are not hardware measurements."
            ),
        },
        "recordings": recordings,
        "candidate_transition": transition,
    }


def main() -> None:
    """运行回放、频谱分析、候选转换估计并保存全部可复现结果。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    replays = {
        name: replay_csv(name, path) for name, path in INPUT_FILES.items()
    }
    spectra = {
        name: build_spectrum(replay) for name, replay in replays.items()
    }
    baseline = add_static_reference(spectra)
    transition_index, thresholds, evidence_count = estimate_transition(
        spectra["static_friction"], spectra["stick_to_slip"]
    )
    transition_time_s = (
        None
        if transition_index is None
        else float(spectra["stick_to_slip"].time_s[transition_index])
    )

    write_window_csv(spectra, transition_index, evidence_count)
    plot_cop(replays, transition_time_s)
    plot_spectrograms(spectra, baseline, transition_time_s)
    plot_feature_timeline(
        spectra["static_friction"],
        spectra["stick_to_slip"],
        thresholds,
        transition_time_s,
    )
    plot_phase_spectra(spectra, transition_index)
    summary = build_summary(
        replays, spectra, transition_index, thresholds, evidence_count
    )
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
