#!/usr/bin/env python3
"""
Extracts generated code from CoderEval generations.jsonl files into individual
.py / .java files for SonarQube analysis.

Source:  Methods/{METHOD}/CE-{lang}/completion/{MODEL}/{SUBFOLDER}/generations.jsonl
Dest Python: sonarQube-analysis/emp-quant/CoderEval-Python-{MODEL}-{METHOD}/
Dest Java:   SonarQube-Analysis-java/emp-quant/CoderEval-Java-{MODEL}-{METHOD}/

Usage:
    python extract_codereval_for_sonarqube.py [--dry-run]
"""

import os
import sys
import json
import glob

# ──────────────────────────── CONFIG ────────────────────────────
METHODS_BASE = "/scratch/oldhome/user/projects/Empirical-Quantization-Study/Methods"
PY_DEST_BASE = "/scratch/oldhome/user/projects/sonarQube-analysis/emp-quant"
JAVA_DEST_BASE = "/scratch/oldhome/user/projects/SonarQube-Analysis-java/emp-quant"

DRY_RUN = "--dry-run" in sys.argv

METHODS = ["AWQ", "GPTQ", "BitsAndBytes", "FP", "AQLM", "GGUF", "QUIP"]
MODELS = ["CodeLlama-7B", "Qwen2.5-Coder-7B"]

# Language config: CE folder name → (file extension, dest base, dest prefix)
LANG_CONFIG = {
    "CE-py": (".py", PY_DEST_BASE, "CoderEval-Python"),
    "CE-java": (".java", JAVA_DEST_BASE, "CoderEval-Java"),
}

# ──────────────────────────── MAIN ──────────────────────────────

def find_generations_jsonl(base_dir):
    """Find generations.jsonl recursively under a directory."""
    for root, dirs, files in os.walk(base_dir):
        if "generations.jsonl" in files:
            return os.path.join(root, "generations.jsonl")
    return None


def extract_files(jsonl_path, output_dir, ext):
    """Extract completion field from each line of generations.jsonl into individual files."""
    count = 0
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            task_id = data.get("task_id", f"task-{i}")
            completion = data.get("completion", "")
            if not completion.strip():
                continue

            output_filename = f"{task_id}{ext}"
            output_path = os.path.join(output_dir, output_filename)

            if not DRY_RUN:
                with open(output_path, "w") as out:
                    out.write(completion)
            count += 1
    return count


def main():
    print()
    print("=" * 55)
    print("  Extract CoderEval files for SonarQube Analysis")
    print("=" * 55)
    if DRY_RUN:
        print("  *** DRY RUN MODE — nothing will be written ***")
    print()

    total_extracted = 0
    total_skipped = 0
    total_errors = 0

    for method in METHODS:
        for lang_folder, (ext, dest_base, dest_prefix) in LANG_CONFIG.items():
            for model in MODELS:
                # Source: Methods/{METHOD}/{CE-lang}/completion/{MODEL}/
                src_dir = os.path.join(METHODS_BASE, method, lang_folder, "completion", model)

                if not os.path.isdir(src_dir):
                    print(f"  ⏭  No source dir: {method}/{lang_folder}/completion/{model}/")
                    total_skipped += 1
                    continue

                # Find generations.jsonl (could be in a subfolder like AWQ/, GPTQ/, BNB/, etc.)
                jsonl_path = find_generations_jsonl(src_dir)
                if not jsonl_path:
                    print(f"  ⏭  No generations.jsonl in: {method}/{lang_folder}/completion/{model}/")
                    total_skipped += 1
                    continue

                # Count lines
                with open(jsonl_path, "r") as f:
                    num_lines = sum(1 for line in f if line.strip())

                # Destination folder
                dest_dir = os.path.join(dest_base, f"{dest_prefix}-{model}-{method}")

                print(f"  ─────────────────────────────────────────")
                print(f"  Method : {method}")
                print(f"  Lang   : {lang_folder}")
                print(f"  Model  : {model}")
                print(f"  Tasks  : {num_lines}")
                print(f"  Dest   : {dest_prefix}-{model}-{method}/")

                if DRY_RUN:
                    print(f"  [DRY RUN] Would extract {num_lines} {ext} files")
                    total_extracted += num_lines
                    continue

                # Create destination directory
                os.makedirs(dest_dir, exist_ok=True)

                try:
                    count = extract_files(jsonl_path, dest_dir, ext)
                    print(f"  ✓ Extracted {count} {ext} files")
                    total_extracted += count
                except Exception as e:
                    print(f"  ✗ Error: {e}")
                    total_errors += 1

    # Summary
    print()
    print("=" * 55)
    print("  Summary")
    print("=" * 55)
    print(f"  Extracted : {total_extracted} files")
    print(f"  Skipped   : {total_skipped} method/model/lang combos")
    print(f"  Errors    : {total_errors}")
    print()
    if DRY_RUN:
        print("  This was a dry run. Re-run without --dry-run to execute.")
        print()


if __name__ == "__main__":
    main()