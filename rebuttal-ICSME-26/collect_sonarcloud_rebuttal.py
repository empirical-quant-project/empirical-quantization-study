#!/usr/bin/env python3
"""
Collect SonarCloud metrics for the 7 rebuttal directories only.

Outputs (under rebuttal/friedman/per-directory-rebuttal/):
  McEval-Python-Qwen2.5-Coder-7B-rebuttal-FP.csv
  ... (one per technique)

Each CSV has columns: TaskID, LoC, Reliability, Maintainability,
Security_Hotspots, CyC, CoC.

Adapted from scripts/collect_sonarcloud_specific_dirs_python.py but writes
under rebuttal/ to keep the rebuttal artifacts self-contained.
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

import requests

SONAR_TOKEN = "ba3c05737b78f966e3ff084cf69fb0f003930efa"
PROJECT_KEY = "userafrin_sonarQube-analysis"
SONAR_URL = "https://sonarcloud.io"
EMP_QUANT_PREFIX = "emp-quant"
METRICS = ["ncloc", "complexity", "cognitive_complexity"]
API_DELAY = 0.1

REBUTTAL = Path(__file__).resolve().parent
OUT_DIR = REBUTTAL / "friedman" / "per-directory-rebuttal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIRS = [
    "McEval-Python-Qwen2.5-Coder-7B-rebuttal-FP",
    "McEval-Python-Qwen2.5-Coder-7B-rebuttal-AWQ",
    "McEval-Python-Qwen2.5-Coder-7B-rebuttal-GPTQ",
    "McEval-Python-Qwen2.5-Coder-7B-rebuttal-GGUF",
    "McEval-Python-Qwen2.5-Coder-7B-rebuttal-BitsAndBytes",
    "McEval-Python-Qwen2.5-Coder-7B-rebuttal-AQLM",
    "McEval-Python-Qwen2.5-Coder-7B-rebuttal-QUIP",
]


def fetch_issues(file_key: str) -> dict:
    counts = {"BUG": 0, "CODE_SMELL": 0, "SECURITY_HOTSPOT": 0}
    try:
        r = requests.get(
            f"{SONAR_URL}/api/issues/search",
            auth=(SONAR_TOKEN, ""),
            params={"componentKeys": file_key, "ps": 500, "facets": "types"},
            timeout=15,
        )
        if r.status_code == 200:
            for facet in r.json().get("facets", []):
                if facet["property"] == "types":
                    for v in facet["values"]:
                        if v["val"] in counts:
                            counts[v["val"]] = v["count"]
    except Exception as e:
        print(f"    warn issues {file_key}: {e}")
    time.sleep(API_DELAY)
    try:
        r = requests.get(
            f"{SONAR_URL}/api/hotspots/search",
            auth=(SONAR_TOKEN, ""),
            params={"projectKey": PROJECT_KEY, "componentKeys": file_key, "ps": 500},
            timeout=15,
        )
        if r.status_code == 200:
            counts["SECURITY_HOTSPOT"] = r.json().get("total", 0)
    except Exception as e:
        print(f"    warn hotspots {file_key}: {e}")
    time.sleep(API_DELAY)
    return counts


def fetch_files(target_dir: str) -> list:
    """Return list of (filepath, ncloc, complexity, cognitive_complexity, key)."""
    out = []
    page = 1
    component_key = f"{PROJECT_KEY}:{EMP_QUANT_PREFIX}/{target_dir}"
    while True:
        r = requests.get(
            f"{SONAR_URL}/api/measures/component_tree",
            auth=(SONAR_TOKEN, ""),
            params={
                "component": component_key,
                "metricKeys": ",".join(METRICS),
                "ps": 500,
                "p": page,
                "qualifiers": "FIL",
            },
            timeout=15,
        )
        if r.status_code != 200:
            print(f"    fetch_files HTTP {r.status_code}: {r.text[:200]}")
            break
        data = r.json()
        for c in data.get("components", []):
            measures = {m["metric"]: m.get("value", "0") for m in c.get("measures", [])}
            out.append({
                "key": c["key"],
                "path": c["path"],
                "ncloc": float(measures.get("ncloc", 0) or 0),
                "complexity": float(measures.get("complexity", 0) or 0),
                "cognitive_complexity": float(measures.get("cognitive_complexity", 0) or 0),
            })
        paging = data.get("paging", {})
        if page * paging.get("pageSize", 500) >= paging.get("total", 0):
            break
        page += 1
        time.sleep(API_DELAY)
    return out


def task_id_from_path(path: str) -> str:
    """emp-quant/<dir>/Python-7.py -> Python-7"""
    base = os.path.basename(path)
    return re.sub(r"\.py$", "", base)


def collect_for_dir(target_dir: str) -> int:
    print(f"\n=== {target_dir} ===")
    files = fetch_files(target_dir)
    if not files:
        print(f"  (no files returned -- analysis may not be complete yet)")
        return 0

    rows = []
    for f in files:
        issues = fetch_issues(f["key"])
        rows.append({
            "TaskID": task_id_from_path(f["path"]),
            "LoC": f["ncloc"],
            "Reliability": issues["BUG"],
            "Maintainability": issues["CODE_SMELL"],
            "Security_Hotspots": issues["SECURITY_HOTSPOT"],
            "CyC": f["complexity"],
            "CoC": f["cognitive_complexity"],
        })

    rows.sort(key=lambda r: (r["TaskID"].split("-")[0],
                             int(r["TaskID"].split("-")[-1]) if r["TaskID"].split("-")[-1].isdigit() else 999))

    out_path = OUT_DIR / f"{target_dir}.csv"
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["TaskID", "LoC", "Reliability", "Maintainability", "Security_Hotspots", "CyC", "CoC"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {len(rows)} rows -> {out_path}")
    return len(rows)


def main():
    total = 0
    for d in DIRS:
        total += collect_for_dir(d)
    print(f"\nTotal rows collected: {total}")
    print(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
