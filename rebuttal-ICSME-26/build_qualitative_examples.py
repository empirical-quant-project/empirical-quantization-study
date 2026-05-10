#!/usr/bin/env python3
"""
Build qualitative_examples.csv: for each (example task x configuration),
emit prompt, generated code, pass/fail, and per-task SonarCloud metrics.

Layout (long format, one row per example x config):
  example_id, benchmark, language, model, task_id, config,
  prompt, generated_code, passed,
  LoC, Reliability, Maintainability, Security_Hotspots, CyC, CoC

Examples (curated for the rebuttal qualitative figure):
  Java-21  -- 6/7 pass, only QuIP# fails  (QuIP-only failure)
  Java-2   -- FP+AWQ+GPTQ pass, all 4-bit fail  (broad low-bit degradation)
  Java-24  -- FP fails; GPTQ+AQLM pass  (fits the AQLM/GPTQ-help narrative)
  Java-12  -- only AQLM passes  (sole-AQLM win)
  Java-43  -- FP fails; AWQ+BnB+AQLM pass  (multi-method help)
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METHODS = REPO / "Methods"
PERDIR = REPO / "Analysis-and-Reports" / "Java" / "per-directory"
OUT_CSV = Path(__file__).resolve().parent / "qualitative_examples.csv"

EXAMPLES = [
    # (example_id, benchmark, language, model, task_id, narrative)
    ("E1", "McEval", "Java", "Qwen2.5-Coder-7B", "Java-21", "QuIP-only failure (6/7 pass, only QuIP# fails)"),
    ("E2", "McEval", "Java", "Qwen2.5-Coder-7B", "Java-2",  "Broad low-bit degradation (4-bit techniques all fail)"),
    ("E3", "McEval", "Java", "Qwen2.5-Coder-7B", "Java-24", "AQLM and GPTQ pass tasks FP fails"),
    ("E4", "McEval", "Java", "Qwen2.5-Coder-7B", "Java-12", "Sole AQLM win on a date-arithmetic task"),
    ("E5", "McEval", "Java", "Qwen2.5-Coder-7B", "Java-43", "Multi-method help on a classic algorithms task"),
]

# Config label -> Methods/<dir> label  +  per-directory CSV technique label
CONFIGS = [
    ("FP",    "FP",            "FP"),
    ("AWQ",   "AWQ",           "AWQ"),
    ("GPTQ",  "GPTQ",          "GPTQ"),
    ("GGUF",  "GGUF",          "GGUF"),
    ("BnB",   "BitsAndBytes",  "BitsAndBytes"),
    ("AQLM",  "AQLM",          "AQLM"),
    ("QuIP#", "QUIP",          "QUIP"),
]


def find_results_json(method_dir: str, model: str, task_id: str) -> Path | None:
    base = METHODS / method_dir / "MC-java"
    candidates = [
        base / "completion" / model / f"{task_id}.results.json",
        base / "completion" / f"{task_id}.results.json",  # GPTQ has no model subdir
    ]
    candidates += list(base.rglob(f"{task_id}.results.json"))
    for c in candidates:
        if c.exists():
            return c
    return None


def load_perdir_metrics(model: str, sonar_label: str) -> dict:
    csv_path = PERDIR / f"McEval-Java-{model}-{sonar_label}.csv"
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["TaskID"]] = row
    return out


def extract_passed(rec: dict) -> int:
    results = rec.get("results", [])
    if not results:
        return 0
    r = results[0]
    return 1 if (r.get("status") == "OK" and r.get("exit_code") == 0) else 0


def main():
    rows = []

    # Pre-load SonarCloud per-task metrics for each (model, technique).
    sonar_cache = {}

    for ex_id, bench, lang, model, task_id, narrative in EXAMPLES:
        for cfg_label, method_dir, sonar_label in CONFIGS:
            key = (model, sonar_label)
            if key not in sonar_cache:
                sonar_cache[key] = load_perdir_metrics(model, sonar_label)
            metrics = sonar_cache[key].get(task_id, {})

            results_path = find_results_json(method_dir, model, task_id)
            if results_path is None:
                rows.append({
                    "example_id": ex_id, "narrative": narrative,
                    "benchmark": bench, "language": lang, "model": model,
                    "task_id": task_id, "config": cfg_label,
                    "prompt": "", "generated_code": "",
                    "passed": "", "load_error": f"missing results file for {method_dir}/{task_id}",
                    "LoC": metrics.get("LoC", ""),
                    "Reliability": metrics.get("Reliability", ""),
                    "Maintainability": metrics.get("Maintainability", ""),
                    "Security_Hotspots": metrics.get("Security_Hotspots", ""),
                    "CyC": metrics.get("CyC", ""),
                    "CoC": metrics.get("CoC", ""),
                })
                continue

            with open(results_path) as f:
                rec = json.load(f)
            prompt = rec.get("prompt", "")
            program = rec.get("results", [{}])[0].get("program", "") if rec.get("results") else ""
            passed = extract_passed(rec)

            rows.append({
                "example_id": ex_id, "narrative": narrative,
                "benchmark": bench, "language": lang, "model": model,
                "task_id": task_id, "config": cfg_label,
                "prompt": prompt, "generated_code": program,
                "passed": passed, "load_error": "",
                "LoC": metrics.get("LoC", ""),
                "Reliability": metrics.get("Reliability", ""),
                "Maintainability": metrics.get("Maintainability", ""),
                "Security_Hotspots": metrics.get("Security_Hotspots", ""),
                "CyC": metrics.get("CyC", ""),
                "CoC": metrics.get("CoC", ""),
            })

    fieldnames = [
        "example_id", "narrative", "benchmark", "language", "model",
        "task_id", "config",
        "prompt", "generated_code", "passed",
        "LoC", "Reliability", "Maintainability", "Security_Hotspots", "CyC", "CoC",
        "load_error",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {len(rows)} rows -> {OUT_CSV}")
    # Summary
    print("\nPass matrix:")
    print(f"  {'example':<6} {'task':<8} {'narrative':<55}", end="")
    for c, _, _ in CONFIGS:
        print(f" {c:>6}", end="")
    print()
    for ex_id, _, _, model, task_id, narrative in EXAMPLES:
        print(f"  {ex_id:<6} {task_id:<8} {narrative[:55]:<55}", end="")
        for c, _, _ in CONFIGS:
            r = next((r for r in rows if r["example_id"] == ex_id and r["config"] == c), None)
            cell = ("PASS" if r and r["passed"] == 1 else "FAIL") if r else "—"
            print(f" {cell:>6}", end="")
        print()


if __name__ == "__main__":
    main()
