#!/usr/bin/env python3
"""
RQ3 Data Preparation: Merge entropy buckets with per-task results.
Then call rq3_analysis.R for statistical tests + LaTeX table.

Usage:
  python rq3_prepare.py --entropy_csv /scratch/oldhome/safrin/projects/Empirical-Quantization-Study/RQ3/output_dir/entropy_buckets_per_task.csv \
                        --config rq3_config.json \
                        --output_dir ./rq3_results
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def load_results_csv(path):
    """Load TaskID, Pass CSV."""
    df = pd.read_csv(path)
    col_map = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("taskid", "task_id", "id", "_id", "question_id", "name"):
            col_map[c] = "task_id"
        elif cl in ("pass", "pass@1", "correct", "is_pass", "passed"):
            col_map[c] = "passed"
    df = df.rename(columns=col_map)
    df["task_id"] = df["task_id"].astype(str)
    df["passed"] = df["passed"].astype(int)
    return df[["task_id", "passed"]]


def main():
    parser = argparse.ArgumentParser(description="RQ3: Prepare merged data and run R analysis")
    parser.add_argument("--entropy_csv", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default="./rq3_results")
    parser.add_argument("--r_script", default=None,
                        help="Path to rq3_analysis.R (default: same dir as this script)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_entropy = pd.read_csv(args.entropy_csv)
    df_entropy["task_id"] = df_entropy["task_id"].astype(str)
    print(f"Entropy data: {len(df_entropy)} tasks")

    with open(args.config) as f:
        config = json.load(f)

    # ── Merge ──
    all_rows = []
    for model_name, model_cfg in config["models"].items():
        print(f"\nModel: {model_name}")
        for bench_name, bench_cfg in model_cfg["benchmarks"].items():
            print(f"  Benchmark: {bench_name}")
            df_fp = load_results_csv(bench_cfg["fp"])
            print(f"    FP: {df_fp['passed'].sum()}/{len(df_fp)} ({df_fp['passed'].mean()*100:.1f}%)")

            for tech_name, tech_path in bench_cfg["techniques"].items():
                df_q = load_results_csv(tech_path)
                print(f"    {tech_name}: {df_q['passed'].sum()}/{len(df_q)} ({df_q['passed'].mean()*100:.1f}%)")

                merged = df_entropy[df_entropy["benchmark"] == bench_name].copy()
                merged = merged.merge(df_fp.rename(columns={"passed": "pass_fp"}), on="task_id", how="inner")
                merged = merged.merge(df_q.rename(columns={"passed": "pass_q"}), on="task_id", how="inner")
                merged["model"] = model_name
                merged["technique"] = tech_name
                merged["degraded"] = ((merged["pass_fp"] == 1) & (merged["pass_q"] == 0)).astype(int)
                merged["helped"] = ((merged["pass_fp"] == 0) & (merged["pass_q"] == 1)).astype(int)
                all_rows.append(merged)

    df_all = pd.concat(all_rows, ignore_index=True)
    merged_path = output_dir / "rq3_merged.csv"
    df_all.to_csv(merged_path, index=False)
    print(f"\nMerged CSV: {merged_path} ({len(df_all)} rows)")

    # ── Run R ──
    if args.r_script:
        r_path = args.r_script
    else:
        r_path = str(Path(__file__).parent / "rq3_analysis.R")

    print(f"\nRunning: Rscript {r_path} {merged_path} {output_dir}")
    result = subprocess.run(
        ["Rscript", r_path, str(merged_path), str(output_dir)],
        capture_output=False
    )
    if result.returncode != 0:
        print(f"\nR exited with code {result.returncode}")
        print(f"Run manually: Rscript {r_path} {merged_path} {output_dir}")


if __name__ == "__main__":
    main()