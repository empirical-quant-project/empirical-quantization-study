#!/usr/bin/env python3
"""
Prepares per-instance pass@1 CSV files for McNemar's test from Python benchmark results.

For McEval-Python:
  Source: Methods/{METHOD}/MC-py/completion/{MODEL}/*.results.json
  Pass condition: results[0]["status"] == "OK"

For CoderEval-Python:
  Source: Methods/{METHOD}/CE-py/completion/{MODEL}/{SUBFOLDER}/generations.jsonl_out.jsonl
  Pass condition: generate_results[0]["is_pass"] == true

For BigCodeBench-Python:
  Source: Methods/{METHOD}/BCB-py/completion/{MODEL}/*.jsonl
  Pass condition: determined from separate eval results
  NOTE: BCB uses a different eval pipeline. The JSONL files contain solutions,
        but pass/fail comes from the BigCodeBench evaluation output.
        We look for *_eval_results.json or *_results.jsonl files.

Output: One CSV per directory with columns: TaskID, Pass (0 or 1)

Usage:
    python prepare_pass1_python.py [--dry-run]
"""

import os
import sys
import json
import glob
import csv
import re

# ──────────────────────────── CONFIG ────────────────────────────
METHODS_BASE = "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Methods"
OUTPUT_BASE = "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Analysis-and-Reports/Python/pass@1-values"

DRY_RUN = "--dry-run" in sys.argv

METHODS = ["FP", "AWQ", "GPTQ", "BitsAndBytes", "AQLM", "GGUF", "QUIP"]
MODELS = ["CodeLlama-7B", "Qwen2.5-Coder-7B"]

# ──────────────────────── McEval EXTRACTION ─────────────────────

def extract_mceval_pass1(src_dir):
    """Extract pass/fail from McEval .results.json files."""
    results = []
    json_files = sorted(glob.glob(os.path.join(src_dir, "*.results.json")))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        task_id = data.get("name", os.path.basename(json_path).replace(".results.json", ""))

        res_list = data.get("results", [])
        if res_list and res_list[0].get("status") == "OK":
            passed = 1
        else:
            passed = 0

        results.append((task_id, passed))

    def sort_key(item):
        match = re.search(r"(\d+)$", item[0])
        return int(match.group(1)) if match else 0

    results.sort(key=sort_key)
    return results


# ──────────────────── CoderEval EXTRACTION ──────────────────────

def find_out_jsonl(base_dir):
    """Find generations.jsonl_out.jsonl recursively."""
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith("_out.jsonl"):
                return os.path.join(root, f)
    return None


def extract_codereval_pass1(out_jsonl_path):
    """Extract pass/fail from CoderEval generations.jsonl_out.jsonl."""
    results = []

    with open(out_jsonl_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            task_id = data.get("_id", f"task-{i}")

            gen_results = data.get("generate_results", [])
            if gen_results and gen_results[0].get("is_pass", False):
                passed = 1
            else:
                passed = 0

            results.append((task_id, passed))

    results.sort(key=lambda x: x[0])
    return results


# ──────────────────── BigCodeBench EXTRACTION ───────────────────

def find_bcb_eval_results(base_dir):
    """Find BigCodeBench evaluation results file.
    
    Looks for:
      - *_eval_results.json (standard BCB output)
      - eval_results.json
      - *_results.jsonl
      - The JSONL file itself may contain pass info in 'result' or 'pass' field
    """
    # Check for eval_results.json files
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if "eval_results" in f and f.endswith(".json"):
                return os.path.join(root, f), "eval_json"

    # Check for *_results.jsonl
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith("_results.jsonl"):
                return os.path.join(root, f), "results_jsonl"

    # Fall back to the sanitized_calibrated.jsonl — check if it has pass info
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.endswith(".jsonl") and "sanitized_calibrated" in f:
                return os.path.join(root, f), "solution_jsonl"

    return None, None


def extract_bcb_pass1_from_eval_json(eval_path):
    """Extract pass/fail from BigCodeBench eval_results.json.
    Format: {"date": "...", "eval": {"BigCodeBench/0": [{"status": "pass"/"fail", ...}], ...}}
    """
    with open(eval_path, "r") as f:
        data = json.load(f)

    # Handle nested "eval" key
    eval_data = data.get("eval", data)

    results = []
    for task_id, info in eval_data.items():
        if task_id in ("date",):  # skip non-task keys
            continue
        if isinstance(info, list) and len(info) > 0:
            status = info[0].get("status", "").lower()
            passed = 1 if status == "pass" else 0
        elif isinstance(info, dict):
            status = info.get("status", "").lower()
            passed = 1 if status == "pass" else 0
        else:
            passed = 0
        results.append((task_id, passed))

    results.sort(key=lambda x: (x[0].split("/")[0] if "/" in x[0] else x[0],
                                 int(x[0].split("/")[1]) if "/" in x[0] else 0))
    return results


def extract_bcb_pass1_from_results_jsonl(results_path):
    """Extract pass/fail from BigCodeBench results JSONL."""
    results = []
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_id = data.get("task_id", data.get("id", ""))
            # Check various pass indicators
            passed = 0
            if data.get("passed", False) or data.get("pass", False) or data.get("is_pass", False):
                passed = 1
            elif data.get("status", "").lower() == "pass":
                passed = 1
            elif data.get("result", "") == "passed":
                passed = 1
            results.append((task_id, passed))

    results.sort(key=lambda x: (x[0].split("/")[0] if "/" in x[0] else x[0],
                                 int(x[0].split("/")[1]) if "/" in x[0] else 0))
    return results


def extract_bcb_pass1(src_dir):
    """Extract BigCodeBench pass/fail from eval results."""
    eval_path, eval_type = find_bcb_eval_results(src_dir)

    if eval_path is None:
        return None, "no eval file found"

    if eval_type == "eval_json":
        return extract_bcb_pass1_from_eval_json(eval_path), eval_path
    elif eval_type == "results_jsonl":
        return extract_bcb_pass1_from_results_jsonl(eval_path), eval_path
    else:
        return None, f"unsupported format: {eval_type}"


# ──────────────────────── SAVE CSV ──────────────────────────────

def save_pass1_csv(results, output_path):
    """Save list of (TaskID, Pass) to CSV."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["TaskID", "Pass"])
        for task_id, passed in results:
            writer.writerow([task_id, passed])


# ──────────────────────── MAIN ──────────────────────────────────

def main():
    print()
    print("=" * 60)
    print("  Prepare Pass@1 CSVs for McNemar's Test (Python)")
    print("=" * 60)
    if DRY_RUN:
        print("  *** DRY RUN MODE ***")
    print(f"  Output: {OUTPUT_BASE}/")
    print()

    total_files = 0
    total_skipped = 0

    # ──── McEval-Python ────
    mceval_out = os.path.join(OUTPUT_BASE, "McEval")
    if not DRY_RUN:
        os.makedirs(mceval_out, exist_ok=True)

    print("━━━ McEval-Python ━━━")
    for method in METHODS:
        for model in MODELS:
            src_dir = os.path.join(METHODS_BASE, method, "MC-py", "completion", model)

            if not os.path.isdir(src_dir):
                print(f"  ⏭  No dir: {method}/MC-py/completion/{model}/")
                total_skipped += 1
                continue

            json_files = glob.glob(os.path.join(src_dir, "*.results.json"))
            if not json_files:
                print(f"  ⏭  No .results.json: {method}/MC-py/completion/{model}/")
                total_skipped += 1
                continue

            dir_name = f"McEval-Python-{model}-{method}"
            output_path = os.path.join(mceval_out, f"{dir_name}.csv")

            if DRY_RUN:
                print(f"  [DRY RUN] {dir_name}: {len(json_files)} tasks")
                total_files += 1
                continue

            results = extract_mceval_pass1(src_dir)
            save_pass1_csv(results, output_path)

            pass_count = sum(1 for _, p in results if p == 1)
            print(f"  ✓ {dir_name}: {len(results)} tasks, {pass_count} passed ({pass_count/len(results)*100:.1f}%)")
            total_files += 1

    # ──── McEval-Python QUIP run2 ────
    run2_src = os.path.join(METHODS_BASE, "QUIP", "MC-py", "completion", "CodeLlama-7B-run2")
    if os.path.isdir(run2_src) and glob.glob(os.path.join(run2_src, "*.results.json")):
        dir_name = "McEval-Python-CodeLlama-7B-run2-QUIP"
        output_path = os.path.join(mceval_out, f"{dir_name}.csv")

        if DRY_RUN:
            print(f"  [DRY RUN] {dir_name}")
        else:
            results = extract_mceval_pass1(run2_src)
            save_pass1_csv(results, output_path)
            pass_count = sum(1 for _, p in results if p == 1)
            print(f"  ✓ {dir_name}: {len(results)} tasks, {pass_count} passed ({pass_count/len(results)*100:.1f}%)")
        total_files += 1

    # ──── CoderEval-Python ────
    print()
    print("━━━ CoderEval-Python ━━━")
    codereval_out = os.path.join(OUTPUT_BASE, "CoderEval")
    if not DRY_RUN:
        os.makedirs(codereval_out, exist_ok=True)

    for method in METHODS:
        for model in MODELS:
            src_dir = os.path.join(METHODS_BASE, method, "CE-py", "completion", model)

            if not os.path.isdir(src_dir):
                print(f"  ⏭  No dir: {method}/CE-py/completion/{model}/")
                total_skipped += 1
                continue

            out_jsonl = find_out_jsonl(src_dir)
            if not out_jsonl:
                print(f"  ⏭  No *_out.jsonl: {method}/CE-py/completion/{model}/")
                total_skipped += 1
                continue

            dir_name = f"CoderEval-Python-{model}-{method}"
            output_path = os.path.join(codereval_out, f"{dir_name}.csv")

            if DRY_RUN:
                with open(out_jsonl, "r") as f:
                    n = sum(1 for line in f if line.strip())
                print(f"  [DRY RUN] {dir_name}: {n} tasks")
                total_files += 1
                continue

            results = extract_codereval_pass1(out_jsonl)
            save_pass1_csv(results, output_path)

            pass_count = sum(1 for _, p in results if p == 1)
            print(f"  ✓ {dir_name}: {len(results)} tasks, {pass_count} passed ({pass_count/len(results)*100:.1f}%)")
            total_files += 1

    # ──── BigCodeBench-Python ────
    print()
    print("━━━ BigCodeBench-Python ━━━")
    bcb_out = os.path.join(OUTPUT_BASE, "BCB")
    if not DRY_RUN:
        os.makedirs(bcb_out, exist_ok=True)

    for method in METHODS:
        for model in MODELS:
            src_dir = os.path.join(METHODS_BASE, method, "BCB-py", "completion", model)

            if not os.path.isdir(src_dir):
                print(f"  ⏭  No dir: {method}/BCB-py/completion/{model}/")
                total_skipped += 1
                continue

            bcb_results, eval_info = extract_bcb_pass1(src_dir)

            if bcb_results is None:
                print(f"  ⏭  No eval results: {method}/BCB-py/completion/{model}/ ({eval_info})")
                total_skipped += 1
                continue

            dir_name = f"BCB-Python-{model}-{method}"
            output_path = os.path.join(bcb_out, f"{dir_name}.csv")

            if DRY_RUN:
                print(f"  [DRY RUN] {dir_name}: {len(bcb_results)} tasks (from {os.path.basename(eval_info)})")
                total_files += 1
                continue

            save_pass1_csv(bcb_results, output_path)

            pass_count = sum(1 for _, p in bcb_results if p == 1)
            print(f"  ✓ {dir_name}: {len(bcb_results)} tasks, {pass_count} passed ({pass_count/len(bcb_results)*100:.1f}%)")
            total_files += 1

    # Summary
    print()
    print("=" * 60)
    print(f"  Files created: {total_files}")
    print(f"  Skipped: {total_skipped}")
    print("=" * 60)
    if DRY_RUN:
        print("  This was a dry run. Re-run without --dry-run to execute.")
    print()


if __name__ == "__main__":
    main()