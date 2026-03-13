#!/usr/bin/env python3
"""
Extracts generated code from BigCodeBench JSONL files into individual .py files
for SonarQube analysis.

Source:  Methods/{METHOD}/BCB-py/completion/{MODEL}/*.jsonl
Dest:    sonarQube-analysis/emp-quant/BCB-Python-{MODEL}-{METHOD}/

Each JSONL line has: task_id (e.g. "BigCodeBench/0"), solution (the code)
Output filename: BigCodeBench_0.py (slash replaced with underscore)

Usage:
    python extract_bcb_for_sonarqube.py [--dry-run]
"""

import os
import sys
import json
import glob

# ──────────────────────────── CONFIG ────────────────────────────
METHODS_BASE = "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Methods"
PY_DEST_BASE = "/scratch/oldhome/safrin/projects/sonarQube-analysis/emp-quant"

DRY_RUN = "--dry-run" in sys.argv

METHODS = ["AWQ", "GPTQ", "BitsAndBytes", "FP", "AQLM", "GGUF", "QUIP"]
MODELS = ["CodeLlama-7B", "Qwen2.5-Coder-7B"]

# ──────────────────────────── MAIN ──────────────────────────────

def find_jsonl(base_dir):
    """Find a .jsonl file in a directory (there should be exactly one)."""
    jsonl_files = glob.glob(os.path.join(base_dir, "*.jsonl"))
    if jsonl_files:
        return jsonl_files[0]
    # Check one level deeper
    jsonl_files = glob.glob(os.path.join(base_dir, "*", "*.jsonl"))
    if jsonl_files:
        return jsonl_files[0]
    return None


def extract_py_files(jsonl_path, output_dir):
    """Extract solution field from each line into a .py file."""
    count = 0
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_id = data.get("task_id", f"task-{i}")
            solution = data.get("solution", "")
            if not solution.strip():
                continue

            # Convert task_id to filename: "BigCodeBench/0" -> "BigCodeBench_0.py"
            safe_name = task_id.replace("/", "_")
            output_filename = f"{safe_name}.py"
            output_path = os.path.join(output_dir, output_filename)

            if not DRY_RUN:
                with open(output_path, "w") as out:
                    out.write(solution)
            count += 1
    return count


def main():
    print()
    print("=" * 55)
    print("  Extract BCB Python files for SonarQube Analysis")
    print("=" * 55)
    if DRY_RUN:
        print("  *** DRY RUN MODE — nothing will be written ***")
    print()
    print(f"  Source : {METHODS_BASE}/{{METHOD}}/BCB-py/completion/{{MODEL}}/")
    print(f"  Dest   : {PY_DEST_BASE}/BCB-Python-{{MODEL}}-{{METHOD}}/")
    print()

    total_extracted = 0
    total_skipped = 0
    total_errors = 0

    for method in METHODS:
        for model in MODELS:
            src_dir = os.path.join(METHODS_BASE, method, "BCB-py", "completion", model)

            if not os.path.isdir(src_dir):
                print(f"  ⏭  No source dir: {method}/BCB-py/completion/{model}/")
                total_skipped += 1
                continue

            # Find the JSONL file
            jsonl_path = find_jsonl(src_dir)
            if not jsonl_path:
                print(f"  ⏭  No .jsonl file in: {method}/BCB-py/completion/{model}/")
                total_skipped += 1
                continue

            # Count lines
            with open(jsonl_path, "r") as f:
                num_lines = sum(1 for line in f if line.strip())

            # Destination folder
            dest_dir = os.path.join(PY_DEST_BASE, f"BCB-Python-{model}-{method}")

            print(f"  ─────────────────────────────────────────")
            print(f"  Method : {method}")
            print(f"  Model  : {model}")
            print(f"  Source : {os.path.basename(jsonl_path)}")
            print(f"  Tasks  : {num_lines}")
            print(f"  Dest   : BCB-Python-{model}-{method}/")

            if DRY_RUN:
                print(f"  [DRY RUN] Would extract {num_lines} .py files")
                total_extracted += num_lines
                continue

            # Create destination directory
            os.makedirs(dest_dir, exist_ok=True)

            try:
                count = extract_py_files(jsonl_path, dest_dir)
                print(f"  ✓ Extracted {count} .py files")
                total_extracted += count
            except Exception as e:
                print(f"  ✗ Error: {e}")
                total_errors += 1

    # Summary
    print()
    print("=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"  Extracted : {total_extracted} .py files")
    print(f"  Skipped   : {total_skipped} method/model combos")
    print(f"  Errors    : {total_errors}")
    print()
    if DRY_RUN:
        print("  This was a dry run. Re-run without --dry-run to execute.")
        print()


if __name__ == "__main__":
    main()