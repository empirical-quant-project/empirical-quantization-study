#!/usr/bin/env python3
"""
Aggregates per-instance combined CSVs into summary CSVs.
Each row = one directory (model+method combo) with summed metrics.

Input:  McEval-Java-combined.csv (per-task rows)
Output: McEval-Java-summary.csv (one row per directory)

Usage:
    python aggregate_combined.py <input_csv> [<input_csv2> ...]
    
    Or process all combined CSVs in a folder:
    python aggregate_combined.py reports/combined/*.csv
"""

import pandas as pd
import sys
import os

METRIC_COLS = ["LoC", "Reliability", "Maintainability", "Security_Hotspots", "CyC", "CoC"]

for input_path in sys.argv[1:]:
    if not input_path.endswith(".csv"):
        continue

    df = pd.read_csv(input_path)

    # Group by Directory, Model, Method and sum metrics
    summary = df.groupby(["Directory", "Model", "Method"])[METRIC_COLS].sum().reset_index()

    # Sort by Model then Method
    method_order = ["FP", "AWQ", "GPTQ", "BitsAndBytes", "AQLM", "GGUF", "QUIP"]
    summary["_method_rank"] = summary["Method"].map({m: i for i, m in enumerate(method_order)}).fillna(99)
    summary = summary.sort_values(["Model", "_method_rank"]).drop(columns=["_method_rank"])

    # Output path: replace -combined with -summary
    output_path = input_path.replace("-combined.csv", "-summary.csv")
    summary.to_csv(output_path, index=False)

    print(f"✓ {os.path.basename(input_path)} -> {os.path.basename(output_path)} ({len(summary)} rows)")
    print(summary.to_string(index=False))
    print()