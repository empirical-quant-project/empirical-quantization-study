#!/usr/bin/env python3
"""
Collects SonarCloud metrics for all directories in the Java SonarQube-Analysis-java repo.

For each directory (e.g. McEval-Java-CodeLlama-7B-AWQ):
  - Fetches per-file metrics (LoC, CyC, CoC) and issues (Reliability, Maintainability, Security_Hotspots)
  - Saves per-instance CSV and JSONL
  - Groups by benchmark and creates combined CSVs

Output structure:
  reports/
    per-directory/
      McEval-Java-CodeLlama-7B-AWQ.csv
      McEval-Java-CodeLlama-7B-AWQ.jsonl
      ...
    combined/
      McEval-Java-combined.csv
      CoderEval-Java-combined.csv

Usage:
    python collect_sonarcloud_java.py [--output-dir reports]
"""

import requests
import pandas as pd
import json
import os
import sys
import re
import time

# ──────────────────────────── CONFIG ────────────────────────────
SONAR_TOKEN = "748648749bb73630fcf96d51cd4e8764e2150987"  # Replace with your SonarCloud token
PROJECT_KEY = "user_SonarQube-Analysis-java"
SONAR_URL = "https://sonarcloud.io"

# Target directories under emp-quant/
EMP_QUANT_PREFIX = "emp-quant"

# Metrics to fetch from SonarCloud
METRICS = ["ncloc", "complexity", "cognitive_complexity"]

# Issue types
ISSUE_TYPES = ["BUG", "VULNERABILITY", "CODE_SMELL"]

# Rate limiting
API_DELAY = 0.1  # seconds between API calls

# Output directory
OUTPUT_DIR = "reports"
for arg in sys.argv[1:]:
    if arg.startswith("--output-dir="):
        OUTPUT_DIR = arg.split("=", 1)[1]
    elif sys.argv[sys.argv.index(arg) - 1] == "--output-dir":
        OUTPUT_DIR = arg

# ──────────────────────── API FUNCTIONS ─────────────────────────

def fetch_issues_for_file(file_key):
    """Fetch issue counts (BUG, CODE_SMELL, SECURITY_HOTSPOT) for a file."""
    issue_counts = {"BUG": 0, "CODE_SMELL": 0, "SECURITY_HOTSPOT": 0}

    url = f"{SONAR_URL}/api/issues/search"
    params = {
        "componentKeys": file_key,
        "ps": 500,
        "facets": "types",
    }
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
        print(f"    Warning: Error fetching issues for {file_key}: {e}")

    time.sleep(API_DELAY)

    # Fetch security hotspots separately
    sec_url = f"{SONAR_URL}/api/hotspots/search"
    sec_params = {
        "projectKey": PROJECT_KEY,
        "componentKeys": file_key,
        "ps": 500,
    }
    try:
        sec_response = requests.get(sec_url, auth=(SONAR_TOKEN, ""), params=sec_params)
        if sec_response.status_code == 200:
            sec_data = sec_response.json()
            issue_counts["SECURITY_HOTSPOT"] = sec_data.get("total", 0)
    except Exception as e:
        print(f"    Warning: Error fetching hotspots for {file_key}: {e}")

    time.sleep(API_DELAY)
    return issue_counts


def fetch_all_files_with_metrics():
    """Fetch all files with their metrics from SonarCloud, handling pagination."""
    all_components = []
    page = 1
    page_size = 500

    while True:
        url = f"{SONAR_URL}/api/measures/component_tree"
        params = {
            "component": PROJECT_KEY,
            "metricKeys": ",".join(METRICS),
            "ps": page_size,
            "p": page,
            "qualifiers": "FIL",
        }
        response = requests.get(url, auth=(SONAR_TOKEN, ""), params=params)
        if response.status_code != 200:
            print(f"Error fetching data (page {page}): {response.status_code} - {response.text}")
            break

        data = response.json()
        components = data.get("components", [])
        all_components.extend(components)
        print(f"  Fetched page {page}: {len(components)} files (total: {len(all_components)})")

        if len(components) < page_size:
            break
        page += 1
        time.sleep(API_DELAY)

    return all_components


# ──────────────────── TASK ID EXTRACTION ────────────────────────

def extract_taskid_java(file_path):
    """Extract task ID from Java file path.
    
    Handles:
      - Java-1.java -> Java-1
      - 636766a81a6d9265ec01758e.java -> 636766a81a6d9265ec01758e
    """
    filename = os.path.basename(file_path)
    name = os.path.splitext(filename)[0]
    return name


def sort_taskids(taskid_list):
    """Sort task IDs numerically where possible."""
    def sort_key(tid):
        match = re.search(r"(\d+)$", tid)
        if match:
            prefix = tid[: match.start()]
            return (prefix, int(match.group(1)))
        return (tid, 0)

    return sorted(taskid_list, key=sort_key)


# ──────────────────── PROCESSING LOGIC ──────────────────────────

def get_directory_from_path(file_path):
    """Extract the emp-quant subdirectory from a file path."""
    parts = file_path.split("/")
    if len(parts) >= 2 and parts[0] == EMP_QUANT_PREFIX:
        return parts[1]
    return None


def process_directory(components, target_dir):
    """Process all files in a target directory, returning per-task data."""
    task_data = {}
    target_prefix = f"{EMP_QUANT_PREFIX}/{target_dir}/"

    for comp in components:
        file_path = comp.get("path", "")
        if not file_path.startswith(target_prefix):
            continue

        file_key = comp.get("key", "")
        task_id = extract_taskid_java(file_path)

        # Get metrics
        measures = {m: 0.0 for m in METRICS}
        for m in comp.get("measures", []):
            measures[m["metric"]] = float(m.get("value", 0))

        # Get issues
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


# ──────────────────────── MAIN ──────────────────────────────────

def main():
    print()
    print("=" * 60)
    print("  SonarCloud Report Collection - Java Repo")
    print("=" * 60)
    print(f"  Project: {PROJECT_KEY}")
    print(f"  Output:  {OUTPUT_DIR}/")
    print()

    per_dir_out = os.path.join(OUTPUT_DIR, "per-directory")
    combined_out = os.path.join(OUTPUT_DIR, "combined")
    os.makedirs(per_dir_out, exist_ok=True)
    os.makedirs(combined_out, exist_ok=True)

    print("Fetching all files from SonarCloud...")
    components = fetch_all_files_with_metrics()
    if not components:
        print("No data fetched. Check your SONAR_TOKEN and PROJECT_KEY.")
        return

    print(f"Total files fetched: {len(components)}")
    print()

    # Discover directories
    directories = set()
    for comp in components:
        dir_name = get_directory_from_path(comp.get("path", ""))
        if dir_name:
            directories.add(dir_name)

    directories = sorted(directories)
    print(f"Found {len(directories)} directories to process")
    print()

    # Group by benchmark
    benchmark_groups = {}
    for dir_name in directories:
        parts = dir_name.split("-")
        if len(parts) >= 2:
            for i, part in enumerate(parts):
                if part in ["CodeLlama", "Qwen2.5"]:
                    benchmark = "-".join(parts[:i])
                    break
            else:
                benchmark = parts[0]
        else:
            benchmark = dir_name

        if benchmark not in benchmark_groups:
            benchmark_groups[benchmark] = []
        benchmark_groups[benchmark].append(dir_name)

    # Process each directory
    all_results = {}
    for dir_name in directories:
        print(f"─────────────────────────────────────────")
        print(f"  Processing: {dir_name}")

        data_rows, totals = process_directory(components, dir_name)

        if not data_rows:
            print(f"  ⏭  No files found, skipping")
            continue

        print(f"  Tasks: {len(data_rows)}")
        print(f"  Totals: LoC={totals['LoC']}, Reliability={totals['Reliability']}, "
              f"Maintainability={totals['Maintainability']}, CyC={totals['CyC']}, CoC={totals['CoC']}")

        # Save per-directory CSV
        csv_path = os.path.join(per_dir_out, f"{dir_name}.csv")
        df = pd.DataFrame(data_rows)
        df.to_csv(csv_path, index=False)

        # Save per-directory JSONL
        jsonl_path = os.path.join(per_dir_out, f"{dir_name}.jsonl")
        with open(jsonl_path, "w") as f:
            for row in data_rows:
                json.dump(row, f)
                f.write("\n")
            json.dump({"TOTALS": {k: str(v) for k, v in totals.items()}}, f)
            f.write("\n")

        print(f"  ✓ Saved {csv_path}")
        all_results[dir_name] = data_rows

    # Combined CSVs per benchmark
    print()
    print("=" * 60)
    print("  Creating combined reports per benchmark")
    print("=" * 60)

    for benchmark, dir_list in sorted(benchmark_groups.items()):
        combined_rows = []
        for dir_name in sorted(dir_list):
            if dir_name not in all_results:
                continue

            suffix = dir_name[len(benchmark) + 1:]
            method_candidates = ["AWQ", "GPTQ", "BitsAndBytes", "FP", "AQLM", "GGUF", "QUIP"]
            method = ""
            model = suffix
            for mc in method_candidates:
                if suffix.endswith(f"-{mc}"):
                    method = mc
                    model = suffix[: -(len(mc) + 1)]
                    break

            for row in all_results[dir_name]:
                combined_row = {
                    "Directory": dir_name,
                    "Model": model,
                    "Method": method,
                }
                combined_row.update(row)
                combined_rows.append(combined_row)

        if combined_rows:
            combined_path = os.path.join(combined_out, f"{benchmark}-combined.csv")
            df = pd.DataFrame(combined_rows)
            df.to_csv(combined_path, index=False)
            print(f"  ✓ {benchmark}: {len(combined_rows)} rows -> {combined_path}")

    print()
    print("Done!")


if __name__ == "__main__":
    main()