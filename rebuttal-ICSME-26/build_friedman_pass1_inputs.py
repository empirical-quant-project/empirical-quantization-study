#!/usr/bin/env python3
"""
Build Friedman-test input CSVs for pass@1 (functional correctness).

Source: rebuttal/variance_results.csv -- long format with columns
    Configuration, Run, Task_ID, Pass

Outputs:
  rebuttal/friedman/inputs_pass1/
    Qwen2.5-Coder-7B_McEval-Python_pass1_configs.csv
        wide: TaskID, FP, AWQ, GPTQ, GGUF, BnB, AQLM, QuIP   (run-1 values)
    Qwen2.5-Coder-7B_McEval-Python_pass1_runs_<CONFIG>.csv
        wide: TaskID, Run1..Run10  (per-run pass/fail per task)

R is invoked elsewhere (friedman-test-pass1.r) -- this script just builds
inputs.
"""

import csv
from pathlib import Path

REBUTTAL = Path(__file__).resolve().parent
LONG_CSV = REBUTTAL / "variance_results.csv"
OUT_DIR = REBUTTAL / "friedman" / "inputs_pass1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = ["FP", "AWQ", "GPTQ", "GGUF", "BnB", "AQLM", "QuIP"]
# variance_results.csv uses display labels FP16 and QuIP#; map to the cleaner
# labels used everywhere else in this folder.
CFG_FROM_VAR = {"FP16": "FP", "AWQ": "AWQ", "GPTQ": "GPTQ", "GGUF": "GGUF",
                "BnB": "BnB", "AQLM": "AQLM", "QuIP#": "QuIP"}


def main():
    if not LONG_CSV.exists():
        raise SystemExit(f"missing {LONG_CSV}")

    # records[(cfg, run, task_id)] = pass
    by_cfg_run = {c: {} for c in CONFIGS}
    task_ids = set()
    with open(LONG_CSV) as f:
        for row in csv.DictReader(f):
            cfg_disp = row["Configuration"]
            cfg = CFG_FROM_VAR.get(cfg_disp)
            if cfg is None: continue
            run = int(row["Run"])
            tid = row["Task_ID"]
            task_ids.add(tid)
            by_cfg_run[cfg].setdefault(run, {})[tid] = int(row["Pass"])

    task_ids = sorted(task_ids,
                      key=lambda s: int(s.split("-")[-1]) if s.split("-")[-1].isdigit() else 999)

    # ── Across-configs (use run 1 since deterministic) ──
    cfg_csv = OUT_DIR / "Qwen2.5-Coder-7B_McEval-Python_pass1_configs.csv"
    with open(cfg_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["TaskID"] + CONFIGS)
        for tid in task_ids:
            row = [tid]
            for c in CONFIGS:
                row.append(by_cfg_run[c].get(1, {}).get(tid, ""))
            w.writerow(row)
    print(f"[cfg] {cfg_csv}  ({len(task_ids)} rows, {len(CONFIGS)} configs)")

    # ── Per-run (one CSV per config, columns = Run1..Run10) ──
    for c in CONFIGS:
        path = OUT_DIR / f"Qwen2.5-Coder-7B_McEval-Python_pass1_runs_{c}.csv"
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["TaskID"] + [f"Run{i}" for i in range(1, 11)])
            for tid in task_ids:
                row = [tid] + [by_cfg_run[c].get(r, {}).get(tid, "") for r in range(1, 11)]
                w.writerow(row)
        print(f"[runs:{c}] {path}")


if __name__ == "__main__":
    main()
