"""对先后采集的三个 ECG 导联做第一个 QRS 的探索性比较。

示例：
    python analysis/analyze_einthoven.py
    python analysis/analyze_einthoven.py --lead-i data/einthoven/1.csv --lead-ii data/einthoven/2.csv --lead-iii data/einthoven/3.csv

默认读取各 CSV 的 Filtered_V 和 R_Peak。以每份记录中标记的第一个
R 峰为中心，截取前 0.08 秒到后 0.12 秒的 QRS 比较窗口，并用 R 峰
前 0.2–0.1 秒的平均值做基线校正。比较导联 II 与导联 I + III 的
单个波形，不做幅度拟合、
极性翻转或时间拉伸。三个导联是先后采集时，这只是探索性比较，
不能证明 Einthoven 定律成立或不成立。

依赖：numpy、pandas、matplotlib。
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_record(path, column):
    frame = pd.read_csv(path)
    required = {"Time_s", "SamplingRate_Hz", column, "R_Peak"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"{path} 缺少列：{', '.join(sorted(missing))}")
    if len(frame) < 2:
        raise ValueError(f"{path} 数据点不足。")
    t = pd.to_numeric(frame["Time_s"], errors="coerce").to_numpy(float)
    signal = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
    rates = pd.to_numeric(frame["SamplingRate_Hz"], errors="coerce").to_numpy(float)
    flags = pd.to_numeric(frame["R_Peak"], errors="coerce").to_numpy(float)
    if not all(np.all(np.isfinite(x)) for x in (t, signal, rates, flags)):
        raise ValueError(f"{path} 的时间、信号、采样率或 R 峰标记含缺失或无效值。")
    fs = float(rates[0])
    if fs <= 0 or not np.allclose(rates, fs, rtol=1e-9, atol=0):
        raise ValueError(f"{path} 的采样率无效或不一致。")
    if not np.allclose(np.diff(t), 1 / fs, rtol=0.01, atol=1e-9):
        raise ValueError(f"{path} 的时间间隔与采样率不一致。")
    if not np.all(np.isin(flags, [0, 1])):
        raise ValueError(f"{path} 的 R_Peak 列应为 0 或 1。")
    return {"path": str(Path(path).resolve()), "time": t, "signal": signal,
            "fs": fs, "peaks": np.flatnonzero(flags == 1)}


def first_qrs(record):
    fs = record["fs"]
    before, after = round(0.08 * fs), round(0.12 * fs)
    baseline_start, baseline_end = round(-0.2 * fs), round(-0.1 * fs)
    if before < 1 or after < 1 or baseline_start >= baseline_end:
        raise ValueError("采样率过低，无法截取 QRS 和基线窗口。")
    if len(record["peaks"]) == 0:
        raise ValueError(f"{record['path']} 没有标记 R 峰。")
    peak = int(record["peaks"][0])
    if peak + baseline_start < 0 or peak + after >= len(record["signal"]):
        raise ValueError(f"{record['path']} 的第一个 R 峰缺少完整基线或 QRS 窗口。")
    relative = np.arange(-before, after + 1)
    baseline = float(np.mean(record["signal"][peak + baseline_start:peak + baseline_end]))
    segment = record["signal"][peak + relative] - baseline
    return {"time": relative / fs, "waveform": segment, "peak": peak, "baseline": baseline}


def compare_qrs(lead_i, lead_ii, lead_iii):
    measured = lead_ii["waveform"]
    predicted = lead_i["waveform"] + lead_iii["waveform"]
    residual = measured - predicted
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    ref_rms = float(np.sqrt(np.mean(measured ** 2)))
    corr = float(np.corrcoef(measured, predicted)[0, 1]) if np.std(measured) > 0 and np.std(predicted) > 0 else None
    return {"predicted": predicted, "residual": residual, "rmse_v": rmse,
            "mae_v": mae, "reference_rms_v": ref_rms,
            "nrmse_percent": 100 * rmse / ref_rms if ref_rms > 0 else None,
            "correlation": corr}


def save_plot(path, templates, comparison):
    t = templates["II"]["time"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for name, color in [("I", "#3568a8"), ("II", "#d66829"), ("III", "#31915c")]:
        axes[0].plot(t, templates[name]["waveform"], label=f"Lead {name}", color=color)
    axes[0].set_title("First R-aligned QRS from each separate recording")
    axes[0].legend(ncol=3, fontsize=9)
    axes[1].plot(t, templates["II"]["waveform"], label="Lead II", color="#d66829")
    axes[1].plot(t, comparison["predicted"], label="Lead I + Lead III", color="#3568a8", linestyle="--")
    axes[1].set_title("Exploratory comparison: Lead II and Lead I + III")
    axes[1].legend(fontsize=9)
    axes[2].plot(t, comparison["residual"], color="#8a4290", label="II - (I + III)")
    axes[2].axhline(0, color="#777777", linewidth=0.8)
    axes[2].set_title("Difference between first QRS waveforms")
    axes[2].legend(fontsize=9)
    for ax in axes:
        ax.axvline(0, color="#888888", linewidth=0.8, linestyle=":")
        ax.set_ylabel("Voltage (V)")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time relative to detected R peak (s)")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lead-i", type=Path, default=PROJECT_ROOT / "data" / "einthoven" / "1.csv")
    parser.add_argument("--lead-ii", type=Path, default=PROJECT_ROOT / "data" / "einthoven" / "2.csv")
    parser.add_argument("--lead-iii", type=Path, default=PROJECT_ROOT / "data" / "einthoven" / "3.csv")
    parser.add_argument("--column", default="Filtered_V", help="信号列名，默认为 Filtered_V")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "new_runs" / "einthoven_first_qrs")
    args = parser.parse_args()
    try:
        records = {name: load_record(path, args.column) for name, path in
                   [("I", args.lead_i), ("II", args.lead_ii), ("III", args.lead_iii)]}
        rates = [records[name]["fs"] for name in ("I", "II", "III")]
        if not np.allclose(rates, rates[0], rtol=1e-9, atol=0):
            raise ValueError("三个导联采样率不同；请先确认采集方案。脚本不会自动重采样。")
        templates = {name: first_qrs(record) for name, record in records.items()}
        comparison = compare_qrs(templates["I"], templates["II"], templates["III"])
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Time_relative_R_s": templates["II"]["time"],
                      "Lead_I_V": templates["I"]["waveform"], "Lead_II_V": templates["II"]["waveform"],
                      "Lead_III_V": templates["III"]["waveform"], "Lead_I_plus_III_V": comparison["predicted"],
                      "Difference_II_minus_sum_V": comparison["residual"]}).to_csv(
                          args.output_dir / "einthoven_templates.csv", index=False)
        summary = {"recording_mode": "separate recordings; exploratory comparison, not simultaneous verification",
                   "column": args.column, "sampling_rate_hz": rates[0],
                   "r_window_s": [-0.08, 0.12], "baseline_window_s": [-0.2, -0.1],
                   "leads": {name: {"path": rec["path"], "samples": len(rec["time"]),
                                    "r_peak_count": len(rec["peaks"]),
                                    "first_r_sample_index_1_based": templates[name]["peak"] + 1,
                                    "first_r_time_s": float(rec["time"][templates[name]["peak"]]),
                                    "baseline_v": templates[name]["baseline"]}
                             for name, rec in records.items()},
                   "metrics": {k: v for k, v in comparison.items() if k not in ("predicted", "residual")}}
        (args.output_dir / "einthoven_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        save_plot(args.output_dir / "einthoven_comparison.png", templates, comparison)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.exit(2, f"错误：{exc}\n")

    print("先后采集的第一个 R 峰对齐 QRS 波形比较，不能作为同步验证。")
    print(f"采样率: {rates[0]:g} Hz；信号列: {args.column}")
    print("第一个 R 峰时间: " + ", ".join(f"{k}={records[k]['time'][templates[k]['peak']]:.3f} s" for k in ("I", "II", "III")))
    print(f"II 与 I+III: RMSE={comparison['rmse_v']:.6g} V；MAE={comparison['mae_v']:.6g} V")
    print(f"归一化 RMSE={comparison['nrmse_percent']:.3g}%（除以 II 模板的 RMS）" if comparison["nrmse_percent"] is not None else "归一化 RMSE 不可计算")
    print(f"Pearson r={comparison['correlation']:.4f}" if comparison["correlation"] is not None else "Pearson r 不可计算")
    print(f"结果目录: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
