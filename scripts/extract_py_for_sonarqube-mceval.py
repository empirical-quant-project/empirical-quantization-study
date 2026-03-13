#!/usr/bin/env python3
"""
Extracts generated code from McEval Python .results.json files into individual .py files
for SonarQube analysis.

Source:  Methods/{METHOD}/MC-py/completion/{MODEL}/*.results.json
Dest:    emp-quant/McEval-Python-{MODEL}-{METHOD}/

Usage:
    python extract_py_for_sonarqube.py [--dry-run]
"""

import os
import sys
import json
import glob

# ──────────────────────────── CONFIG ────────────────────────────
METHODS_BASE = "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Methods"
EMPQUANT_BASE = "/scratch/oldhome/safrin/projects/sonarQube-analysis/emp-quant"

DRY_RUN = "--dry-run" in sys.argv

# All quantization methods to process
METHODS = ["AWQ", "GPTQ", "BitsAndBytes", "FP", "AQLM", "GGUF", "QUIP"]

# All model subdirectories (under completion/)
MODELS = ["CodeLlama-7B", "Qwen2.5-Coder-7B"]

# ──────────────────────────── MAIN ──────────────────────────────

def extract_py_files(json_path, output_dir):
    """Extract the program field from a .results.json file into a .py file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        return 0

    # Use the task name from the JSON (e.g., "Python-7")
    task_name = data.get("name", "")
    if not task_name:
        # Fallback: derive from filename (Python-7.results.json → Python-7)
        task_name = os.path.basename(json_path).replace(".results.json", "")

    # Extract the first (and only) result's program
    program_code = results[0].get("program", "")
    if not program_code.strip():
        return 0

    output_filename = f"{task_name}.py"
    output_path = os.path.join(output_dir, output_filename)

    if not DRY_RUN:
        with open(output_path, "w") as f:
            f.write(program_code)

    return 1


def main():
    print()
    print("=" * 50)
    print("  Extract .py files for SonarQube Analysis")
    print("=" * 50)
    if DRY_RUN:
        print("  *** DRY RUN MODE — nothing will be written ***")
    print()
    print(f"  Source : {METHODS_BASE}/{{METHOD}}/MC-py/completion/{{MODEL}}/")
    print(f"  Dest   : {EMPQUANT_BASE}/McEval-Python-{{MODEL}}-{{METHOD}}/")
    print()

    total_extracted = 0
    total_skipped = 0
    total_errors = 0

    for method in METHODS:
        for model in MODELS:
            src_dir = os.path.join(METHODS_BASE, method, "MC-py", "completion", model)

            if not os.path.isdir(src_dir):
                print(f"  ⏭  No source dir: {method}/MC-py/completion/{model}/")
                total_skipped += 1
                continue

            # Find all .results.json files
            json_files = sorted(glob.glob(os.path.join(src_dir, "*.results.json")))
            if not json_files:
                print(f"  ⏭  No .results.json files in: {method}/MC-py/completion/{model}/")
                total_skipped += 1
                continue

            # Destination folder
            dest_dir = os.path.join(EMPQUANT_BASE, f"McEval-Python-{model}-{method}")

            print(f"  ─────────────────────────────────────────")
            print(f"  Method : {method}")
            print(f"  Model  : {model}")
            print(f"  Files  : {len(json_files)} .results.json")
            print(f"  Dest   : McEval-Python-{model}-{method}/")

            if DRY_RUN:
                print(f"  [DRY RUN] Would extract {len(json_files)} .py files")
                total_extracted += len(json_files)
                continue

            # Create destination directory
            os.makedirs(dest_dir, exist_ok=True)

            count = 0
            errors = 0
            for json_path in json_files:
                try:
                    extracted = extract_py_files(json_path, dest_dir)
                    count += extracted
                except Exception as e:
                    print(f"  ✗ Error processing {os.path.basename(json_path)}: {e}")
                    errors += 1

            print(f"  ✓ Extracted {count} .py files" + (f" ({errors} errors)" if errors else ""))
            total_extracted += count
            total_errors += errors

    # Also handle QuIP CodeLlama-7B-run2 if it exists
    run2_src = os.path.join(METHODS_BASE, "QUIP", "MC-py", "completion", "CodeLlama-7B-run2")
    if os.path.isdir(run2_src):
        json_files = sorted(glob.glob(os.path.join(run2_src, "*.results.json")))
        if json_files:
            dest_dir = os.path.join(EMPQUANT_BASE, "McEval-Python-CodeLlama-7B-run2-QUIP")
            print(f"  ─────────────────────────────────────────")
            print(f"  Method : QUIP (run2)")
            print(f"  Model  : CodeLlama-7B-run2")
            print(f"  Files  : {len(json_files)} .results.json")
            print(f"  Dest   : McEval-Python-CodeLlama-7B-run2-QUIP/")

            if DRY_RUN:
                print(f"  [DRY RUN] Would extract {len(json_files)} .py files")
                total_extracted += len(json_files)
            else:
                os.makedirs(dest_dir, exist_ok=True)
                count = 0
                for json_path in json_files:
                    try:
                        count += extract_py_files(json_path, dest_dir)
                    except Exception as e:
                        print(f"  ✗ Error: {e}")
                        total_errors += 1
                print(f"  ✓ Extracted {count} .py files")
                total_extracted += count

    # Summary
    print()
    print("=" * 50)
    print("  Summary")
    print("=" * 50)
    print(f"  Extracted : {total_extracted} .py files")
    print(f"  Skipped   : {total_skipped} method/model combos")
    print(f"  Errors    : {total_errors}")
    print()
    if DRY_RUN:
        print("  This was a dry run. Re-run without --dry-run to execute.")
        print()


if __name__ == "__main__":
    main()