"""分析 App 导出的 raw 单边幅度谱，并保存频谱图片。

用法：
    python analysis/analyze_ecg_raw_spectrum.py data.csv
    python analysis/analyze_ecg_raw_spectrum.py data.csv --min-amplitude 0.0001
    python analysis/analyze_ecg_raw_spectrum.py data.csv --exclude-band 49.5 50.5
    python analysis/analyze_ecg_raw_spectrum.py data.csv --output spectrum.png

依赖：pip install numpy matplotlib
fmax_99 是排除 49.5–50.5 Hz 后，累计频谱能量达到 99% 时的频率。
排除范围可用 --exclude-band 修改。最高非零频点仅表示
数值非零的最高 FFT 频点，不等同于去除噪声后的 ECG 真实频率上限。
可选 --min-amplitude 的单位为 V，由使用者自行确定噪声阈值。
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_app_spectrum(path):
    """读取 App CSV；谱列在前半部分，时域列共有 N 行。"""
    required = {"Time_s", "Raw_V", "SamplingRate_Hz", "Frequency_Hz", "Raw_Spectrum_V"}
    frequencies, amplitudes = [], []
    sample_count, fs = 0, None
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("CSV 缺少必要列：" + ", ".join(sorted(missing)))
        for line, row in enumerate(reader, 2):
            try:
                time_s = float(row["Time_s"])
                raw = float(row["Raw_V"])
                row_fs = float(row["SamplingRate_Hz"])
                freq = float(row["Frequency_Hz"] or "nan")
                amplitude = float(row["Raw_Spectrum_V"] or "nan")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {line} 行包含无效数值。") from exc
            if not np.isfinite(time_s) or not np.isfinite(raw) or not np.isfinite(row_fs) or row_fs <= 0:
                raise ValueError(f"第 {line} 行的时间、raw 电压或采样率无效。")
            if fs is None:
                fs = row_fs
            elif not np.isclose(row_fs, fs, rtol=1e-9, atol=0):
                raise ValueError("CSV 中的采样率不一致。")
            sample_count += 1
            freq_finite, amp_finite = np.isfinite(freq), np.isfinite(amplitude)
            if freq_finite != amp_finite:
                raise ValueError(f"第 {line} 行的频率与幅度必须同时有值或同时为空。")
            if freq_finite:
                if freq < 0 or amplitude < 0:
                    raise ValueError(f"第 {line} 行的频率和幅度不能为负数。")
                frequencies.append(freq)
                amplitudes.append(amplitude)

    if sample_count < 2:
        raise ValueError("至少需要两个采样点。")
    frequency = np.asarray(frequencies, dtype=float)
    amplitude = np.asarray(amplitudes, dtype=float)
    expected_count = sample_count // 2 + 1
    if len(frequency) != expected_count:
        raise ValueError(f"频谱有 {len(frequency)} 个点，按 {sample_count} 个时域采样点应有 {expected_count} 个。")
    expected_frequency = np.arange(expected_count) * fs / sample_count
    if not np.allclose(frequency, expected_frequency, rtol=1e-7, atol=max(1e-9, fs * 1e-10)):
        raise ValueError("频谱频率与采样率、采样点数不一致，请检查 CSV 是否完整。")
    return frequency, amplitude, fs, sample_count


def analyze_spectrum(frequency, amplitude, sample_count, min_amplitude=None,
                     exclude_band=(49.5, 50.5)):
    """计算频谱能量分位点和超过给定幅度阈值的最高频点。

    App 使用单边峰值幅度：正频率内点翻倍，DC 和偶数长度的 Nyquist
    频点未翻倍。因此其均方贡献分别为 A²/2 和 A²。
    """
    energy = amplitude ** 2 / 2.0
    energy[0] = amplitude[0] ** 2
    if sample_count % 2 == 0:
        energy[-1] = amplitude[-1] ** 2
    excluded = (frequency >= exclude_band[0]) & (frequency <= exclude_band[1])
    excluded_energy = float(np.sum(energy[excluded]))
    original_energy = float(np.sum(energy))
    energy[excluded] = 0.0
    total_energy = float(np.sum(energy))
    if not np.isfinite(total_energy) or total_energy <= 0:
        raise ValueError("频谱能量为零，无法计算 99% 能量频率。")
    cumulative = np.cumsum(energy) / total_energy
    index_99 = min(int(np.searchsorted(cumulative, 0.99, side="left")), len(frequency) - 1)
    nonzero = np.flatnonzero(amplitude > 0)
    above_threshold = None
    if min_amplitude is not None:
        above = np.flatnonzero(amplitude > min_amplitude)
        if len(above):
            above_threshold = float(frequency[above[-1]])
    return {
        "fmax_99_hz": float(frequency[index_99]),
        "highest_nonzero_hz": float(frequency[nonzero[-1]]),
        "highest_above_threshold_hz": above_threshold,
        "cumulative": cumulative,
        "excluded_energy_fraction": excluded_energy / original_energy if original_energy > 0 else 0.0,
        "excluded_bin_count": int(np.count_nonzero(excluded)),
    }


def save_figure(path, frequency, amplitude, result, min_amplitude, exclude_band):
    """在一张图片中显示完整频谱和低频细节。"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    zoom_end = min(frequency[-1], max(20.0, result["fmax_99_hz"] * 1.3))
    for ax in axes:
        ax.plot(frequency, amplitude, color="#1c6a9a", linewidth=0.9, label="Raw amplitude")
        ax.axvspan(*exclude_band, color="#8c8c8c", alpha=0.25,
                   label=f"Excluded from energy: {exclude_band[0]:g}–{exclude_band[1]:g} Hz")
        ax.axvline(result["fmax_99_hz"], color="#cb4b36", linestyle="--",
                   label=f"99% energy: {result['fmax_99_hz']:.3g} Hz")
        if min_amplitude is not None:
            ax.axhline(min_amplitude, color="#916b19", linestyle=":",
                       label=f"Amplitude threshold: {min_amplitude:.3g} V")
            if result["highest_above_threshold_hz"] is not None:
                ax.axvline(result["highest_above_threshold_hz"], color="#7951a6", linestyle="-.",
                           label=f"Highest above threshold: {result['highest_above_threshold_hz']:.3g} Hz")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Amplitude (V)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_title("Raw ECG single-sided amplitude spectrum — full range")
    axes[0].set_xlim(0, frequency[-1])
    axes[1].set_title("Low-frequency detail")
    axes[1].set_xlim(0, zoom_end)
    visible = amplitude[frequency <= zoom_end]
    if len(visible) and np.max(visible) > 0:
        axes[1].set_ylim(0, np.max(visible) * 1.1)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_file", type=Path, help="App 导出的 CSV 文件")
    parser.add_argument("--output", type=Path, help="频谱图片路径，默认保存至 results/new_runs/spectrum")
    parser.add_argument("--min-amplitude", type=float, help="可选的幅度阈值（V），用于报告超过阈值的最高频点")
    parser.add_argument("--exclude-band", nargs=2, type=float, default=(49.5, 50.5),
                        metavar=("LOW_HZ", "HIGH_HZ"), help="能量计算排除的频段，默认 49.5–50.5 Hz（含两端）")
    args = parser.parse_args()
    if args.min_amplitude is not None and (not np.isfinite(args.min_amplitude) or args.min_amplitude < 0):
        parser.error("--min-amplitude 必须是非负有限数值，单位 V。")
    if not np.all(np.isfinite(args.exclude_band)) or not 0 <= args.exclude_band[0] < args.exclude_band[1]:
        parser.error("--exclude-band 必须满足 0 ≤ 下限 < 上限，单位 Hz。")
    try:
        frequency, amplitude, fs, count = read_app_spectrum(args.csv_file)
        result = analyze_spectrum(frequency, amplitude, count, args.min_amplitude, args.exclude_band)
        output = args.output or Path(__file__).resolve().parent.parent / "results" / "new_runs" / "spectrum" / (args.csv_file.stem + "_raw_spectrum.png")
        if output.resolve() == args.csv_file.resolve():
            raise ValueError("图片路径不能与输入 CSV 相同。")
        output.parent.mkdir(parents=True, exist_ok=True)
        save_figure(output, frequency, amplitude, result, args.min_amplitude, args.exclude_band)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.exit(2, f"错误：{exc}\n")

    print(f"采样率: {fs:g} Hz；采样点数: {count}；频率间隔: {fs / count:g} Hz")
    print(f"能量积分排除: {args.exclude_band[0]:g}–{args.exclude_band[1]:g} Hz（含边界），"
          f"{result['excluded_bin_count']} 个频点，占原总能量 {100 * result['excluded_energy_fraction']:.3f}%")
    print(f"99% 累计能量 fmax: {result['fmax_99_hz']:g} Hz")
    print(f"最高非零 FFT 频点: {result['highest_nonzero_hz']:g} Hz")
    print("提示：最高非零频点可能包含噪声和数值误差，不等于 ECG 的真实有效频率上限。")
    if args.min_amplitude is not None:
        value = result["highest_above_threshold_hz"]
        print(f"超过 {args.min_amplitude:g} V 的最高频点: " + (f"{value:g} Hz" if value is not None else "无"))
    print(f"频谱图片: {output.resolve()}")


if __name__ == "__main__":
    main()
