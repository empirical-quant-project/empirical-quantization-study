#!/usr/bin/env python3
"""
Scrape specific SonarCloud directories, create per-instance files,
and append to existing combined CSVs.

Usage:
    python collect_sonarcloud_specific_dirs_python.py <dir1> <dir2> ...

Example:
    python collect_sonarcloud_specific_dirs_python.py \
        BCB-Python-CodeLlama-7B-BitsAndBytes \
        BCB-Python-Qwen2.5-Coder-7B-BitsAndBytes \
        CoderEval-Python-CodeLlama-7B-GPTQ
"""

import requests
import pandas as pd
import json
import os
import sys
import re
import time

# ──────────────────────────── CONFIG ────────────────────────────
SONAR_TOKEN = "ba3c05737b78f966e3ff084cf69fb0f003930efa"  # Replace with your SonarCloud token
PROJECT_KEY = "saimaafrin_sonarQube-analysis"
SONAR_URL = "https://sonarcloud.io"

EMP_QUANT_PREFIX = "emp-quant"
METRICS = ["ncloc", "complexity", "cognitive_complexity"]
API_DELAY = 0.1

# Output paths
PER_DIR_OUT = "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Analysis-and-Reports/Python/per-directory"
COMBINED_OUT = "/scratch/oldhome/safrin/projects/Empirical-Quantization-Study/Analysis-and-Reports/Python/combined"

# ──────────────────────── API FUNCTIONS ─────────────────────────

def fetch_issues_for_file(file_key):
    issue_counts = {"BUG": 0, "CODE_SMELL": 0, "SECURITY_HOTSPOT": 0}

    url = f"{SONAR_URL}/api/issues/search"
    params = {"componentKeys": file_key, "ps": 500, "facets": "types"}
    try:
        response = requests.get(url, auth=(SONAR_TOKEN, ""), params=params)
        if response.status_code == 200:
            data = response.json()
            for facet in data.get("facets", []):
                if facet["property"] == "types":
                    for value in facet["values"]:
                        if value["val"] in issue_counts:
                            issue_counts[value["val"]] = value["count"]
    except Exception as e:
        print(f"    Warning: {e}")

    time.sleep(API_DELAY)

    sec_url = f"{SONAR_URL}/api/hotspots/search"
    sec_params = {"projectKey": PROJECT_KEY, "componentKeys": file_key, "ps": 500}
    try:
        sec_response = requests.get(sec_url, auth=(SONAR_TOKEN, ""), params=sec_params)
        if sec_response.status_code == 200:
            issue_counts["SECURITY_HOTSPOT"] = sec_response.json().get("total", 0)
    except Exception as e:
        print(f"    Warning: {e}")

    time.sleep(API_DELAY)
    return issue_counts


def fetch_files_for_directory(target_dir):
    """Fetch all files with metrics for a specific directory."""
    all_components = []
    page = 1
    page_size = 500
    component_key = f"{PROJECT_KEY}:{EMP_QUANT_PREFIX}/{target_dir}"

    while True:
        url = f"{SONAR_URL}/api/measures/component_tree"
        params = {
            "component": component_key,
            "metricKeys": ",".join(METRICS),
            "ps": page_size,
            "p": page,
            "qualifiers": "FIL",
        }
        response = requests.get(url, auth=(SONAR_TOKEN, ""), params=params)
        if response.status_code != 200:
            print(f"    Error fetching (page {page}): {response.status_code} - {response.text}")
            break

        data = response.json()
        components = data.get("components", [])
        all_components.extend(components)

        if len(components) < page_size:
            break
        page += 1
        time.sleep(API_DELAY)

    return all_components


# ──────────────────── TASK ID EXTRACTION ────────────────────────

def extract_taskid(file_path):
    filename = os.path.basename(file_path)
    name = os.path.splitext(filename)[0]
    if name.startswith("BigCodeBench_"):
        return name.replace("_", "/", 1)
    return name


def sort_taskids(taskid_list):
    def sort_key(tid):
        match = re.search(r"(\d+)$", tid)
        if match:
            return (tid[:match.start()], int(match.group(1)))
        return (tid, 0)
    return sorted(taskid_list, key=sort_key)


# ──────────────────── PROCESSING ────────────────────────────────

def process_directory(target_dir):
    """Fetch and process all files in a specific directory."""
    print(f"\n  Fetching files for: {target_dir}")
    components = fetch_files_for_directory(target_dir)
    print(f"  Found {len(components)} files")

    if not components:
        return [], {}

    task_data = {}
    for comp in components:
        file_path = comp.get("path", "")
        file_key = comp.get("key", "")
        task_id = extract_taskid(file_path)

        measures = {m: 0.0 for m in METRICS}
        for m in comp.get("measures", []):
            measures[m["metric"]] = float(m.get("value", 0))

        issue_counts = fetch_issues_for_file(file_key)

        if task_id not in task_data:
            task_data[task_id] = {
                "TaskID": task_id,
                "LoC": 0.0,
                "Reliability": 0,
                "Maintainability": 0,
                "Security_Hotspots": 0,
                "CyC": 0.0,
                "CoC": 0.0,
            }

        row = task_data[task_id]
        row["LoC"] += measures.get("ncloc", 0)
        row["CyC"] += measures.get("complexity", 0)
        row["CoC"] += measures.get("cognitive_complexity", 0)
        row["Reliability"] += issue_counts.get("BUG", 0)
        row["Maintainability"] += issue_counts.get("CODE_SMELL", 0)
        row["Security_Hotspots"] += issue_counts.get("SECURITY_HOTSPOT", 0)

    sorted_ids = sort_taskids(list(task_data.keys()))
    sorted_data = [task_data[tid] for tid in sorted_ids]

    totals = {"LoC": 0, "Reliability": 0, "Maintainability": 0, "Security_Hotspots": 0, "CyC": 0, "CoC": 0}
    for row in sorted_data:
        for key in totals:
            totals[key] += row.get(key, 0)

    return sorted_data, totals


def get_benchmark_and_model_method(dir_name):
    """Parse directory name into benchmark, model, method."""
    method_candidates = ["AWQ", "GPTQ", "BitsAndBytes", "FP", "AQLM", "GGUF", "QUIP"]
    
    # Find benchmark prefix
    for i, part in enumerate(dir_name.split("-")):
        if part in ["CodeLlama", "Qwen2.5"]:
            benchmark = "-".join(dir_name.split("-")[:i])
            suffix = dir_name[len(benchmark) + 1:]
            break
    else:
        return dir_name, "", ""

    method = ""
    model = suffix
    for mc in method_candidates:
        if suffix.endswith(f"-{mc}"):
            method = mc
            model = suffix[:-(len(mc) + 1)]
            break

    return benchmark, model, method


# ──────────────────────── MAIN ──────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python collect_sonarcloud_specific_dirs_python.py <dir1> <dir2> ...")
        print("Example: python collect_sonarcloud_specific_dirs_python.py BCB-Python-CodeLlama-7B-BitsAndBytes")
        sys.exit(1)

    target_dirs = [d for d in sys.argv[1:] if not d.startswith("--")]

    print()
    print("=" * 60)
    print("  Scrape Specific SonarCloud Directories")
    print("=" * 60)
    print(f"  Directories to process: {len(target_dirs)}")
    print()

    os.makedirs(PER_DIR_OUT, exist_ok=True)
    os.makedirs(COMBINED_OUT, exist_ok=True)

    # Group by benchmark for combined CSV appending
    benchmark_new_data = {}

    for dir_name in target_dirs:
        print(f"─────────────────────────────────────────")
        print(f"  Processing: {dir_name}")

        data_rows, totals = process_directory(dir_name)

        if not data_rows:
            print(f"  ⏭  No data found, skipping")
            continue

        print(f"  Tasks: {len(data_rows)}")
        print(f"  Totals: LoC={totals['LoC']}, Reliability={totals['Reliability']}, "
              f"Maintainability={totals['Maintainability']}, CyC={totals['CyC']}, CoC={totals['CoC']}")

        # Save per-directory CSV
        csv_path = os.path.join(PER_DIR_OUT, f"{dir_name}.csv")
        df = pd.DataFrame(data_rows)
        df.to_csv(csv_path, index=False)

        # Save per-directory JSONL
        jsonl_path = os.path.join(PER_DIR_OUT, f"{dir_name}.jsonl")
        with open(jsonl_path, "w") as f:
            for row in data_rows:
                json.dump(row, f)
                f.write("\n")
            json.dump({"TOTALS": {k: str(v) for k, v in totals.items()}}, f)
            f.write("\n")

        print(f"  ✓ Saved {csv_path}")

        # Prepare for combined CSV
        benchmark, model, method = get_benchmark_and_model_method(dir_name)
        if benchmark not in benchmark_new_data:
            benchmark_new_data[benchmark] = []

        for row in data_rows:
            combined_row = {"Directory": dir_name, "Model": model, "Method": method}
            combined_row.update(row)
            benchmark_new_data[benchmark].append(combined_row)

    # Append to combined CSVs
    print()
    print("=" * 60)
    print("  Appending to combined CSVs")
    print("=" * 60)

    for benchmark, new_rows in benchmark_new_data.items():
        combined_path = os.path.join(COMBINED_OUT, f"{benchmark}-combined.csv")

        if os.path.exists(combined_path):
            existing_df = pd.read_csv(combined_path)
            new_df = pd.DataFrame(new_rows)

            # Remove any existing rows for these directories (in case of re-run)
            new_dirs = new_df["Directory"].unique()
            existing_df = existing_df[~existing_df["Directory"].isin(new_dirs)]

            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"  ✓ {benchmark}: appended {len(new_rows)} rows (total: {len(combined_df)})")
        else:
            combined_df = pd.DataFrame(new_rows)
            print(f"  ✓ {benchmark}: created with {len(new_rows)} rows")

        combined_df.to_csv(combined_path, index=False)

    print()
    print("Done!")


if __name__ == "__main__":
    main()