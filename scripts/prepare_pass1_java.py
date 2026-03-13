#!/usr/bin/env python3
"""
Prepares per-instance pass@1 CSV files for McNemar's test from Java benchmark results.

For McEval-Java:
  Source: Methods/{METHOD}/MC-java/completion/{MODEL}/*.results.json
  Pass condition: results[0]["status"] == "OK"

For CoderEval-Java:
  Source: Methods/{METHOD}/CE-java/completion/{MODEL}/{SUBFOLDER}/generations.jsonl_out.jsonl
  Pass condition: generate_results[0]["is_pass"] == true

Output: One CSV per directory with columns: TaskID, Pass (0 or 1)
  e.g. McEval-Java-CodeLlama-7B-AWQ.csv

Usage:
    python prepare_pass1_java.py [--dry-run]
"""

import os
import sys
import json
import glob
import csv

# ──────────────────────────── CONFIG ────────────────────────────
METHODS_BASE = "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Methods"
OUTPUT_DIR = "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Analysis-and-Reports/Java/pass@1-values"

DRY_RUN = "--dry-run" in sys.argv

METHODS = ["FP", "AWQ", "GPTQ", "BitsAndBytes", "AQLM", "GGUF", "QUIP"]
MODELS = ["CodeLlama-7B", "Qwen2.5-Coder-7B"]

# ──────────────────────── McEval EXTRACTION ─────────────────────

def extract_mceval_pass1(src_dir):
    """Extract pass/fail from McEval .results.json files.
    Returns sorted list of (TaskID, Pass) tuples.
    """
    results = []
    json_files = sorted(glob.glob(os.path.join(src_dir, "*.results.json")))
    
    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)
        
        task_id = data.get("name", os.path.basename(json_path).replace(".results.json", ""))
        
        # Check if first result passed
        res_list = data.get("results", [])
        if res_list and res_list[0].get("status") == "OK":
            passed = 1
        else:
            passed = 0
        
        results.append((task_id, passed))
    
    # Sort by task ID numerically
    def sort_key(item):
        import re
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
    """Extract pass/fail from CoderEval generations.jsonl_out.jsonl.
    Returns sorted list of (TaskID, Pass) tuples.
    """
    results = []
    
    with open(out_jsonl_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            
            # Task ID from _id field
            task_id = data.get("_id", f"task-{i}")
            
            # Check pass/fail from generate_results
            gen_results = data.get("generate_results", [])
            if gen_results and gen_results[0].get("is_pass", False):
                passed = 1
            else:
                passed = 0
            
            results.append((task_id, passed))
    
    # Sort by task ID (hex strings sort lexicographically)
    results.sort(key=lambda x: x[0])
    return results


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
    print("  Prepare Pass@1 CSVs for McNemar's Test (Java)")
    print("=" * 60)
    if DRY_RUN:
        print("  *** DRY RUN MODE ***")
    print(f"  Output: {OUTPUT_DIR}/")
    print()

    if not DRY_RUN:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    total_files = 0
    total_skipped = 0

    # ──── McEval-Java ────
    print("━━━ McEval-Java ━━━")
    for method in METHODS:
        for model in MODELS:
            src_dir = os.path.join(METHODS_BASE, method, "MC-java", "completion", model)
            
            if not os.path.isdir(src_dir):
                print(f"  ⏭  No dir: {method}/MC-java/completion/{model}/")
                total_skipped += 1
                continue
            
            json_files = glob.glob(os.path.join(src_dir, "*.results.json"))
            if not json_files:
                print(f"  ⏭  No .results.json: {method}/MC-java/completion/{model}/")
                total_skipped += 1
                continue
            
            dir_name = f"McEval-Java-{model}-{method}"
            output_path = os.path.join(OUTPUT_DIR, f"{dir_name}.csv")
            
            if DRY_RUN:
                print(f"  [DRY RUN] {dir_name}: {len(json_files)} tasks")
                total_files += 1
                continue
            
            results = extract_mceval_pass1(src_dir)
            save_pass1_csv(results, output_path)
            
            pass_count = sum(1 for _, p in results if p == 1)
            print(f"  ✓ {dir_name}: {len(results)} tasks, {pass_count} passed ({pass_count/len(results)*100:.1f}%)")
            total_files += 1

    # ──── CoderEval-Java ────
    print()
    print("━━━ CoderEval-Java ━━━")
    for method in METHODS:
        for model in MODELS:
            src_dir = os.path.join(METHODS_BASE, method, "CE-java", "completion", model)
            
            if not os.path.isdir(src_dir):
                print(f"  ⏭  No dir: {method}/CE-java/completion/{model}/")
                total_skipped += 1
                continue
            
            out_jsonl = find_out_jsonl(src_dir)
            if not out_jsonl:
                print(f"  ⏭  No *_out.jsonl: {method}/CE-java/completion/{model}/")
                total_skipped += 1
                continue
            
            dir_name = f"CoderEval-Java-{model}-{method}"
            output_path = os.path.join(OUTPUT_DIR, f"{dir_name}.csv")
            
            if DRY_RUN:
                # Count lines
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

    # ──── McEval-Java QUIP run2 ────
    run2_src = os.path.join(METHODS_BASE, "QUIP", "MC-java", "completion", "CodeLlama-7B-run2")
    if os.path.isdir(run2_src) and glob.glob(os.path.join(run2_src, "*.results.json")):
        dir_name = "McEval-Java-CodeLlama-7B-run2-QUIP"
        output_path = os.path.join(OUTPUT_DIR, f"{dir_name}.csv")
        
        if DRY_RUN:
            print(f"\n  [DRY RUN] {dir_name}")
        else:
            results = extract_mceval_pass1(run2_src)
            save_pass1_csv(results, output_path)
            pass_count = sum(1 for _, p in results if p == 1)
            print(f"\n  ✓ {dir_name}: {len(results)} tasks, {pass_count} passed ({pass_count/len(results)*100:.1f}%)")
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