#!/usr/bin/env python3
"""
Extracts rebuttal-generated McEval-Python code (one config x one run, since
runs are deterministic) into SonarQube staging directories.

Source:  rebuttal/runs_qwen_mceval_py/<CONFIG>/run_1_raw/*.results.json.gz
Dest:    sonarQube-analysis/emp-quant/Rebuttal-McEval-Python-Qwen2.5-Coder-7B-<TECH>/<task>.py

This only stages the files locally. Pushing to GitHub (where SonarCloud picks
them up via CI) is a separate, user-gated step:

    cd /scratch/oldhome/safrin/projects/sonarQube-analysis
    git add emp-quant/Rebuttal-McEval-Python-Qwen2.5-Coder-7B-*
    git commit -m "Rebuttal: McEval-Python Qwen2.5-Coder-7B per-config snapshots"
    git push

After the SonarCloud scan completes, re-collect metrics and rebuild Friedman
inputs (overriding the existing per-directory CSVs) using:

    python scripts/collect_sonarcloud_python.py
    python rebuttal/build_friedman_inputs.py     # rebuilds inputs from per-directory
    Rscript rebuttal/friedman-test-sonarqube.r
"""

import gzip
import json
import shutil
from pathlib import Path

REBUTTAL = Path(__file__).resolve().parent
WORK = REBUTTAL / "runs_qwen_mceval_py"
DEST_BASE = Path("/scratch/oldhome/safrin/projects/sonarQube-analysis/emp-quant")

# (run-dir name, SonarQube tech label)
CONFIG_MAP = [
    ("FP16",     "FP"),
    ("AWQ",      "AWQ"),
    ("GPTQ",     "GPTQ"),
    ("GGUF",     "GGUF"),
    ("BnB",      "BitsAndBytes"),
    ("AQLM",     "AQLM"),
    ("QuIPSharp","QUIP"),
]


def extract_one(results_gz: Path, dest_dir: Path) -> int:
    with gzip.open(results_gz, "rt") as f:
        d = json.load(f)
    results = d.get("results", [])
    if not results:
        return 0
    program = results[0].get("program", "")
    if not program.strip():
        return 0
    name = d.get("name", results_gz.name.replace(".results.json.gz", ""))
    out = dest_dir / f"{name}.py"
    out.write_text(program)
    return 1


def main():
    print("Staging rebuttal-generated McEval-Python code into SonarQube layout")
    print("-" * 70)
    for run_label, tech_label in CONFIG_MAP:
        src = WORK / run_label / "run_1_raw"
        dest = DEST_BASE / f"McEval-Python-Qwen2.5-Coder-7B-rebuttal-{tech_label}"
        if not src.exists():
            print(f"  [skip] {tech_label}: no source dir {src}")
            continue
        # Wipe existing dest (idempotent re-runs).
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for rf in sorted(src.glob("*.results.json.gz")):
            n += extract_one(rf, dest)
        print(f"  {tech_label:12s}: {n} files -> {dest.relative_to(DEST_BASE.parent)}")


if __name__ == "__main__":
    main()
