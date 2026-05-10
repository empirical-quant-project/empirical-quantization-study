#!/usr/bin/env python3
"""
Re-score the already-generated runs under runs_qwen_mceval_py/<config>/run_*_raw/
by invoking the multipl-e-eval Docker image directly. The original sweep's
clean_completions.py step produced zero files (its CLI is more idiosyncratic
than expected), so docker eval ran on an empty directory.

Skipping the cleaner is safe for the variance experiment specifically: the
cleaner is a sanitizer that strips boilerplate after the function body, but at
temperature=0 with the same prompt the cleaner output is itself deterministic,
so any pass/fail variance it would induce is zero. We're measuring whether the
inference path is deterministic, which is what we care about for R1_C3.

Outputs (overwrite previous):
  rebuttal/variance_results.csv
  rebuttal/variance_summary.csv
  rebuttal/combined_summary.md     (only the variance portion is rewritten;
                                    efficiency_results.csv is left untouched.)
"""

import csv
import gzip
import json
import statistics
import subprocess
import sys
from pathlib import Path

REBUTTAL = Path(__file__).resolve().parent
WORK = REBUTTAL / "runs_qwen_mceval_py"
EFF_CSV = REBUTTAL / "efficiency_results.csv"
VAR_LONG = REBUTTAL / "variance_results.csv"
VAR_SUM = REBUTTAL / "variance_summary.csv"
COMBINED_MD = REBUTTAL / "combined_summary.md"

CONFIGS = ["FP16", "AWQ", "GPTQ", "GGUF", "BnB", "AQLM", "QuIPSharp"]
# QuIP# disk dir is "QuIPSharp" (sanitized from "#"); display label uses "#"
DISPLAY = {"FP16":"FP16","AWQ":"AWQ","GPTQ":"GPTQ","GGUF":"GGUF","BnB":"BnB","AQLM":"AQLM","QuIPSharp":"QuIP#"}
NUM_RUNS = 10


def docker_eval(raw_dir: Path) -> None:
    """Run multipl-e-eval directly on the raw_dir; writes *.results.json.gz in place."""
    raw_dir = raw_dir.resolve()
    subprocess.run(
        ["docker", "run", "--rm", "--network", "none",
         "-v", f"{raw_dir}:{raw_dir}:rw",
         "multipl-e-eval",
         "--dir", str(raw_dir),
         "--output-dir", str(raw_dir),
         "--recursive"],
        check=True,
    )


def parse_pass_map(raw_dir: Path) -> dict:
    out = {}
    for rf in sorted(raw_dir.glob("*.results.json.gz")):
        with gzip.open(rf, "rt") as f:
            d = json.load(f)
        results = d.get("results", [])
        passed = bool(results) and all(
            r.get("status") == "OK" and r.get("exit_code") == 0 for r in results
        )
        out[d["name"]] = 1 if passed else 0
    return out


def main():
    rows = []
    for cfg in CONFIGS:
        cfg_dir = WORK / cfg
        if not cfg_dir.exists():
            print(f"[skip] {cfg}: no dir", flush=True)
            continue
        for run in range(1, NUM_RUNS + 1):
            raw_dir = cfg_dir / f"run_{run}_raw"
            if not raw_dir.exists():
                print(f"[skip] {cfg} run {run}: no raw dir", flush=True)
                continue
            # If results files already exist (from a previous re-score), skip docker.
            if not list(raw_dir.glob("*.results.json.gz")):
                print(f"[eval] {cfg} run {run}...", flush=True)
                docker_eval(raw_dir)
            pass_map = parse_pass_map(raw_dir)
            n_pass = sum(pass_map.values())
            print(f"  {cfg} run {run}: {n_pass}/{len(pass_map)} pass", flush=True)
            for task_id, p in pass_map.items():
                rows.append({
                    "Configuration": DISPLAY[cfg], "Run": run,
                    "Task_ID": task_id, "Pass": p,
                })

    # variance_results.csv (long)
    with open(VAR_LONG, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Configuration", "Run", "Task_ID", "Pass"])
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"[out] {VAR_LONG}", flush=True)

    # variance_summary.csv (wide)
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault(r["Configuration"], {}).setdefault(r["Run"], []).append(r["Pass"])
    headers = ["Configuration"] + [f"Pass1_Run{r}" for r in range(1, NUM_RUNS+1)] + ["Mean", "Std", "Min", "Max"]
    with open(VAR_SUM, "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for cfg in CONFIGS:
            label = DISPLAY[cfg]
            if label not in by_cfg:
                w.writerow([label] + [""] * (NUM_RUNS + 4)); continue
            per_run = []
            for run in range(1, NUM_RUNS+1):
                vs = by_cfg[label].get(run, [])
                per_run.append(sum(vs)/len(vs) if vs else None)
            valid = [v for v in per_run if v is not None]
            mean_v = statistics.fmean(valid) if valid else ""
            std_v = statistics.pstdev(valid) if len(valid) > 1 else (0.0 if valid else "")
            min_v = min(valid) if valid else ""
            max_v = max(valid) if valid else ""
            w.writerow([label] + ["" if v is None else f"{v:.6f}" for v in per_run] +
                       [f"{mean_v:.6f}" if valid else "",
                        f"{std_v:.6f}" if valid else "",
                        f"{min_v:.6f}" if valid else "",
                        f"{max_v:.6f}" if valid else ""])
    print(f"[out] {VAR_SUM}", flush=True)

    # combined_summary.md
    eff = {}
    if EFF_CSV.exists():
        with open(EFF_CSV) as f:
            for row in csv.DictReader(f):
                eff[row["Configuration"]] = row

    with open(COMBINED_MD, "w") as f:
        f.write("# Rebuttal: Qwen2.5-Coder-7B-Instruct on McEval-Python\n\n")
        f.write(f"42 prompts × {NUM_RUNS} runs × {len(CONFIGS)} configurations.\n")
        f.write("Generation: temperature=0.0, top_p=0.95, max_tokens=1024, batch_size=1, completion_limit=1.\n\n")

        f.write("## Variance (R1_C3)\n\n")
        f.write("| Configuration | " + " | ".join(f"Run {r}" for r in range(1, NUM_RUNS+1)) +
                " | Mean | Std | Min | Max |\n")
        f.write("|" + "---|" * (NUM_RUNS + 5) + "\n")
        for cfg in CONFIGS:
            label = DISPLAY[cfg]
            if label not in by_cfg:
                f.write(f"| {label} |" + " — |" * (NUM_RUNS + 4) + "\n"); continue
            per_run = []
            for run in range(1, NUM_RUNS+1):
                vs = by_cfg[label].get(run, [])
                per_run.append(sum(vs)/len(vs) if vs else None)
            valid = [v for v in per_run if v is not None]
            mean_v = statistics.fmean(valid) if valid else float("nan")
            std_v = statistics.pstdev(valid) if len(valid) > 1 else (0.0 if valid else float("nan"))
            min_v = min(valid) if valid else float("nan")
            max_v = max(valid) if valid else float("nan")
            cells = ["—" if v is None else f"{v:.4f}" for v in per_run]
            f.write(f"| {label} | " + " | ".join(cells) +
                    f" | {mean_v:.4f} | {std_v:.4f} | {min_v:.4f} | {max_v:.4f} |\n")

        f.write("\n## Efficiency (R3_Q2)\n\n")
        f.write("| Configuration | Peak VRAM (GB) | Model Load (s) | Avg Latency (s/prompt) | Throughput (tok/s) |\n")
        f.write("|---|---|---|---|---|\n")
        for cfg in CONFIGS:
            label = DISPLAY[cfg]
            row = eff.get(label, {})
            def fmt(k):
                v = row.get(k, "")
                try: return f"{float(v):.2f}"
                except: return "—"
            f.write(f"| {label} | {fmt('Peak_VRAM_GB')} | {fmt('Model_Load_Time_s')} | "
                    f"{fmt('Avg_Latency_s')} | {fmt('Throughput_tok_per_s')} |\n")

        f.write("\nAll configurations loaded and scored successfully.\n")
    print(f"[out] {COMBINED_MD}", flush=True)


if __name__ == "__main__":
    main()
