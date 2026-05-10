#!/usr/bin/env python3
"""
Build Friedman-test input CSVs from the REBUTTAL SonarCloud scan
(rebuttal/friedman/per-directory-rebuttal/), plus per-run inputs for the
output-variability check (R1_C3).

Across-configs inputs (one CSV per metric):
    rebuttal/friedman/inputs/Qwen2.5-Coder-7B_McEval-Python_<METRIC>.csv
    columns: TaskID, FP, AWQ, GPTQ, GGUF, BnB, AQLM, QuIP

Per-run inputs (one CSV per metric x config), for output-variability test:
    rebuttal/friedman/inputs_runs/Qwen2.5-Coder-7B_McEval-Python_<CONFIG>_<METRIC>.csv
    columns: TaskID, Run1, Run2, ..., Run10
    (Since runs are deterministic at temperature=0, all 10 columns are
    identical replicas of the single SonarCloud scan -- the Friedman test
    will return p = NA, formally documenting zero within-config variance.)

Source: rebuttal/friedman/per-directory-rebuttal/  (fresh scan of the
        rebuttal-regenerated code; collected by collect_sonarcloud_rebuttal.py).
"""

import csv
from pathlib import Path

REBUTTAL = Path(__file__).resolve().parent
PERDIR_REBUTTAL = REBUTTAL / "friedman" / "per-directory-rebuttal"
OUT_CFG  = REBUTTAL / "friedman" / "inputs"
OUT_RUNS = REBUTTAL / "friedman" / "inputs_runs"

MODEL = "Qwen2.5-Coder-7B"
BENCH_PREFIX = "McEval-Python"
CONFIGS = [
    ("FP",   "FP"),
    ("AWQ",  "AWQ"),
    ("GPTQ", "GPTQ"),
    ("GGUF", "GGUF"),
    ("BnB",  "BitsAndBytes"),
    ("AQLM", "AQLM"),
    ("QuIP", "QUIP"),
]
METRICS = ["LoC", "Reliability", "Maintainability", "Security_Hotspots", "CyC", "CoC"]
NUM_RUNS = 10


def perdir_path(sonar_label: str) -> Path:
    return PERDIR_REBUTTAL / f"{BENCH_PREFIX}-{MODEL}-rebuttal-{sonar_label}.csv"


def main():
    OUT_CFG.mkdir(parents=True, exist_ok=True)
    OUT_RUNS.mkdir(parents=True, exist_ok=True)

    # Load per-config rows keyed by TaskID.
    per_config = {}
    for cfg_label, sonar_label in CONFIGS:
        f = perdir_path(sonar_label)
        if not f.exists():
            print(f"[warn] missing: {f}")
            per_config[cfg_label] = {}
            continue
        with open(f) as fh:
            per_config[cfg_label] = {row["TaskID"]: row for row in csv.DictReader(fh)}

    # Canonical TaskID set: union of all configs (some configs may produce
    # files SonarCloud counts but others produce empty/zero-LoC files filtered
    # out by SonarCloud -- being permissive here keeps Friedman comparable.)
    task_ids = sorted(
        {tid for cfg in per_config.values() for tid in cfg.keys()},
        key=lambda s: int(s.split("-")[-1]) if s.split("-")[-1].isdigit() else 999,
    )

    # ── Across-configs inputs ──
    for metric in METRICS:
        out_path = OUT_CFG / f"{MODEL}_{BENCH_PREFIX}_{metric}.csv"
        with open(out_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["TaskID"] + [c for c, _ in CONFIGS])
            for tid in task_ids:
                row = [tid]
                for cfg_label, _ in CONFIGS:
                    cell = per_config.get(cfg_label, {}).get(tid, {}).get(metric, "")
                    row.append(cell)
                w.writerow(row)
        print(f"[cfg] {out_path}  ({len(task_ids)} rows)")

    # ── Per-run inputs (replicate run-1 metrics across 10 columns) ──
    # Determinism at temperature=0 was already verified (variance_summary.csv
    # shows Std=0 across the 10 generation runs). The 10 .py outputs per
    # (config, task) are byte-identical -> SonarCloud metrics are the same
    # number 10 times. Friedman across runs will therefore be undefined
    # (constant rows), which is itself the desired result for R1_C3.
    for cfg_label, sonar_label in CONFIGS:
        for metric in METRICS:
            out_path = OUT_RUNS / f"{MODEL}_{BENCH_PREFIX}_{cfg_label}_{metric}.csv"
            with open(out_path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["TaskID"] + [f"Run{i}" for i in range(1, NUM_RUNS + 1)])
                cfg_rows = per_config.get(cfg_label, {})
                for tid in task_ids:
                    cell = cfg_rows.get(tid, {}).get(metric, "")
                    w.writerow([tid] + [cell] * NUM_RUNS)
            # Don't spam: just confirm count once per config.
        print(f"[runs] wrote 6 metric files for config {cfg_label} -> {OUT_RUNS}")


if __name__ == "__main__":
    main()
