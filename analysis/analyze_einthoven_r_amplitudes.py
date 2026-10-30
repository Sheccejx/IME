"""比较先后采集的 I、II、III 导联的 R 峰幅值。

默认读取 1.csv、2.csv、3.csv 的前 11 个已标记 R 峰，分别对应
I、II、III 导联。幅值 = 指定信号在 R 峰处的值，减去该峰前
0.2–0.1 秒窗口的均值。按出现顺序组成探索性配对，比较 II 与
I + III，计算双侧配对 t 检验和平均差值的 95% 置信区间。

示例：
    python analysis/analyze_einthoven_r_amplitudes.py
    python analysis/analyze_einthoven_r_amplitudes.py --signal-source raw-notch50
依赖：numpy、pandas、scipy、matplotlib。

注意：三次记录并不同步；序号配对不对应同一次心跳，多个心搏也
不能视为独立受试者。因此 p 值只作探索性描述，不能验证定律、
推断总体，也不能用 p > 0.05 证明两组等效。
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from analyze_einthoven import load_record

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def notch_raw(record):
    """只做 50 Hz 陷波：Q=30、前后向滤波、最多 1 秒奇对称延拓。"""
    if record["fs"] <= 100:
        raise ValueError(f"{record['path']} 的采样率必须大于 100 Hz 才能进行 50 Hz 陷波。")
    b, a = signal.iirnotch(50.0, 30.0, fs=record["fs"])
    sos = signal.tf2sos(b, a)
    padlen = min(len(record["signal"]) - 2, max(1, round(record["fs"])))
    result = dict(record)
    result["signal"] = signal.sosfiltfilt(sos, record["signal"], padtype="odd", padlen=padlen)
    return result


def extract_peaks(record, count):
    """取时间顺序上的前 count 个标记峰，不跳过或替换任何峰。"""
    if len(record["peaks"]) < count:
        raise ValueError(f"{record['path']} 只有 {len(record['peaks'])} 个 R 峰，少于 {count} 个。")
    peaks = record["peaks"][:count]
    start, end = round(-0.2 * record["fs"]), round(-0.1 * record["fs"])
    if start >= end or any(int(peak) + start < 0 for peak in peaks):
        raise ValueError(f"{record['path']} 的某个 R 峰缺少完整基线窗口。")
    baselines = np.array([record["signal"][p + start:p + end].mean() for p in peaks])
    amplitudes = record["signal"][peaks] - baselines
    return {"indices": peaks, "times": record["time"][peaks],
            "baselines": baselines, "amplitudes": amplitudes}


def analyze(lead_i, lead_ii, lead_iii):
    lhs = lead_ii["amplitudes"]
    rhs = lead_i["amplitudes"] + lead_iii["amplitudes"]
    differences = lhs - rhs
    count = len(differences)
    if count < 2 or not np.all(np.isfinite(differences)):
        raise ValueError("至少需要两组有效数值。")
    diff_sd = float(np.std(differences, ddof=1))
    if diff_sd == 0:
        raise ValueError("配对差值没有变异，常规 t 检验不可用。")
    result = stats.ttest_rel(lhs, rhs, alternative="two-sided")
    ci = stats.t.interval(0.95, df=count - 1, loc=float(np.mean(differences)),
                          scale=diff_sd / np.sqrt(count))
    return {"rhs": rhs, "differences": differences, "n": count,
            "ii_mean_v": float(np.mean(lhs)), "ii_sd_v": float(np.std(lhs, ddof=1)),
            "i_plus_iii_mean_v": float(np.mean(rhs)), "i_plus_iii_sd_v": float(np.std(rhs, ddof=1)),
            "difference_mean_v": float(np.mean(differences)), "difference_sd_v": diff_sd,
            "difference_ci95_v": [float(ci[0]), float(ci[1])],
            "t_statistic": float(result.statistic), "df": count - 1, "p_value_two_sided": float(result.pvalue)}


def save_plot(path, ii, result, signal_label):
    rank = np.arange(1, result["n"] + 1)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    axes[0].plot(rank, ii["amplitudes"], "o-", color="#d66829", label="Lead II")
    axes[0].plot(rank, result["rhs"], "s--", color="#3568a8", label="Lead I + Lead III")
    axes[0].set_title(f"R-peak amplitudes by ordinal beat ({signal_label}, separate recordings)")
    axes[0].set_ylabel("Baseline-corrected amplitude (V)")
    axes[0].legend()
    axes[1].bar(rank, result["differences"], color="#80549c")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Ordinal difference: II − (I + III)")
    axes[1].set_xlabel("R-peak order within each recording")
    axes[1].set_ylabel("Difference (V)")
    axes[1].set_xticks(rank)
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lead-i", type=Path, default=PROJECT_ROOT / "data" / "einthoven" / "1.csv")
    parser.add_argument("--lead-ii", type=Path, default=PROJECT_ROOT / "data" / "einthoven" / "2.csv")
    parser.add_argument("--lead-iii", type=Path, default=PROJECT_ROOT / "data" / "einthoven" / "3.csv")
    parser.add_argument("--column", default="Filtered_V")
    parser.add_argument("--signal-source", choices=("filtered", "raw-notch50"), default="filtered",
                        help="filtered: 使用 --column；raw-notch50: Raw_V 仅进行 50 Hz 陷波")
    parser.add_argument("--num-beats", type=int, default=11)
    parser.add_argument("--output-dir", type=Path, help="输出目录；默认按信号类型分别保存")
    args = parser.parse_args()
    if args.num_beats < 2:
        parser.error("--num-beats 必须至少为 2。")
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "results" / "new_runs" / ("einthoven_r_amplitudes_notch50" if args.signal_source == "raw-notch50"
                                                                  else "einthoven_r_amplitudes_filtered")
    source_column = "Raw_V" if args.signal_source == "raw-notch50" else args.column
    try:
        records = {name: load_record(path, source_column) for name, path in
                   [("I", args.lead_i), ("II", args.lead_ii), ("III", args.lead_iii)]}
        rates = [records[k]["fs"] for k in ("I", "II", "III")]
        if not np.allclose(rates, rates[0], rtol=1e-9, atol=0):
            raise ValueError("三个导联的采样率不一致。")
        if args.signal_source == "raw-notch50":
            records = {name: notch_raw(record) for name, record in records.items()}
        peaks = {name: extract_peaks(record, args.num_beats) for name, record in records.items()}
        result = analyze(peaks["I"], peaks["II"], peaks["III"])
        table = {"Beat_order": np.arange(1, args.num_beats + 1)}
        for name in ("I", "II", "III"):
            table.update({f"Lead_{name}_sample_index_1based": peaks[name]["indices"] + 1,
                          f"Lead_{name}_time_s": peaks[name]["times"],
                          f"Lead_{name}_baseline_V": peaks[name]["baselines"],
                          f"Lead_{name}_R_amplitude_V": peaks[name]["amplitudes"]})
        table["Lead_I_plus_III_V"] = result["rhs"]
        table["Difference_II_minus_sum_V"] = result["differences"]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table).to_csv(args.output_dir / "r_peak_amplitudes.csv", index=False)
        summary = {"recording_mode": "separate recordings; ordinal pairing is exploratory, not simultaneous",
                   "paths": {name: rec["path"] for name, rec in records.items()},
                   "column": source_column, "signal_source": args.signal_source,
                   "notch": {"frequency_hz": 50.0, "q": 30.0, "filtering": "forward-backward",
                             "padding": "odd; at most one second"} if args.signal_source == "raw-notch50" else None,
                   "peak_source": "original CSV R_Peak markers", "baseline_window_s": [-0.2, -0.1],
                   "test": "two-sided paired t-test on ordinal beat differences; exploratory",
                   "metrics": {key: value for key, value in result.items() if key not in ("rhs", "differences")}}
        (args.output_dir / "r_peak_statistics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        save_plot(args.output_dir / "r_peak_comparison.png", peaks["II"], result,
                  "raw + 50 Hz notch" if args.signal_source == "raw-notch50" else args.column)
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        parser.exit(2, f"错误：{exc}\n")
    print(f"信号: {'Raw_V + 50 Hz notch, Q=30' if args.signal_source == 'raw-notch50' else args.column}；R 峰沿用原 CSV 标记")
    print(f"II: {result['ii_mean_v']:.6f} ± {result['ii_sd_v']:.6f} V")
    print(f"I+III: {result['i_plus_iii_mean_v']:.6f} ± {result['i_plus_iii_sd_v']:.6f} V")
    print(f"平均差 II-(I+III): {result['difference_mean_v']:.6f} V；95% CI "
          f"[{result['difference_ci95_v'][0]:.6f}, {result['difference_ci95_v'][1]:.6f}] V")
    print(f"双侧配对 t 检验（仅序号配对、探索性）: t({result['df']})={result['t_statistic']:.4f}, p={result['p_value_two_sided']:.6g}")
    print(f"结果目录: {args.output_dir.resolve()}")
    print("警告：导联先后采集，序号配对不是同一次心跳；结果不能作为同步验证或总体推断。")


if __name__ == "__main__":
    main()
