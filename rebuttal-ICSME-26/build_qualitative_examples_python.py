#!/usr/bin/env python3
"""
Python counterpart to build_qualitative_examples.py.

5 McEval-Python tasks (Qwen2.5-Coder-7B) curated for qualitative narrative.
Same long-form schema, written to qualitative_examples_python.csv.
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METHODS = REPO / "Methods"
PERDIR = REPO / "Analysis-and-Reports" / "Python" / "per-directory"
OUT_CSV = Path(__file__).resolve().parent / "qualitative_examples_python.csv"

EXAMPLES = [
    ("E1", "McEval", "Python", "Qwen2.5-Coder-7B", "Python-10", "QuIP-only failure (6/7 pass, only QuIP# fails)"),
    ("E2", "McEval", "Python", "Qwen2.5-Coder-7B", "Python-11", "FP-only pass (all 6 quantized fail)"),
    ("E3", "McEval", "Python", "Qwen2.5-Coder-7B", "Python-43", "Multi-method help (5/6 quantized pass while FP fails)"),
    ("E4", "McEval", "Python", "Qwen2.5-Coder-7B", "Python-36", "Sole AQLM win on a road-clearing problem"),
    ("E5", "McEval", "Python", "Qwen2.5-Coder-7B", "Python-17", "QuIP# + AQLM + GGUF help (counter-narrative: QuIP# can also help)"),
]

CONFIGS = [
    ("FP",    "FP",            "FP"),
    ("AWQ",   "AWQ",           "AWQ"),
    ("GPTQ",  "GPTQ",          "GPTQ"),
    ("GGUF",  "GGUF",          "GGUF"),
    ("BnB",   "BitsAndBytes",  "BitsAndBytes"),
    ("AQLM",  "AQLM",          "AQLM"),
    ("QuIP#", "QUIP",          "QUIP"),
]


def find_results_json(method_dir: str, model: str, task_id: str):
    base = METHODS / method_dir / "MC-py"
    candidates = [
        base / "completion" / model / f"{task_id}.results.json",
        base / "completion" / f"{task_id}.results.json",
    ]
    candidates += list(base.rglob(f"{task_id}.results.json"))
    for c in candidates:
        if c.exists():
            return c
    return None


def load_perdir_metrics(model: str, sonar_label: str) -> dict:
    csv_path = PERDIR / f"McEval-Python-{model}-{sonar_label}.csv"
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
    print("\nPass matrix:")
    print(f"  {'example':<6} {'task':<11} {'narrative':<60}", end="")
    for c, _, _ in CONFIGS:
        print(f" {c:>6}", end="")
    print()
    for ex_id, _, _, model, task_id, narrative in EXAMPLES:
        print(f"  {ex_id:<6} {task_id:<11} {narrative[:60]:<60}", end="")
        for c, _, _ in CONFIGS:
            r = next((r for r in rows if r["example_id"] == ex_id and r["config"] == c), None)
            cell = ("PASS" if r and r["passed"] == 1 else "FAIL") if r else "—"
            print(f" {cell:>6}", end="")
        print()

    # Uniqueness summary
    import hashlib
    print("\nGenerated-code uniqueness per example:")
    for ex_id, _, _, _, task_id, _ in EXAMPLES:
        sub = [r for r in rows if r["example_id"] == ex_id]
        hashes = {}
        for r in sub:
            h = hashlib.sha1((r["generated_code"] or "").encode("utf-8", errors="replace")).hexdigest()[:8]
            hashes.setdefault(h, []).append(r["config"])
        n_unique = len(hashes)
        groups = " | ".join("[" + ",".join(g) + "]" for g in hashes.values())
        print(f"  {ex_id} ({task_id}): {n_unique} unique  {groups}")


if __name__ == "__main__":
    main()
