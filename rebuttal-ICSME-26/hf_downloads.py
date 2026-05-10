#!/usr/bin/env python3
"""
hf_downloads.py
---------------
Collects all-time and last-30-days download counts from the Hugging Face Hub
for every model used in the empirical study:

    "Quantize with Confidence? An Empirical Study of Quantization
     for Code Generation."

Coverage:
  - 2 full-precision (FP16) baselines: Qwen2.5-Coder-7B-Instruct, CodeLlama-7B-Instruct
  - 6 quantized variants per family: GPTQ, AWQ, QuIP#, AQLM, BitsAndBytes, GGUF

Notes on provenance:
  * For AWQ, GPTQ, GGUF, BitsAndBytes the paper used pre-quantized public
    checkpoints from Hugging Face (see Sec. IV.B), so the listed repos are the
    canonical releases used in the study.
  * For AQLM and QuIP# the paper performed local quantization with WikiText-2
    calibration; the IDs below point to the closest community-released
    checkpoints, since the authors' own quantized weights are not on the Hub.
    Adjust REPOS as needed before running.

Output:
  CSV to stdout with columns:
    base_model, technique, repo_id, downloads_all_time, downloads_last_30d,
    likes, created_at, last_modified

Usage:
  python hf_downloads.py                      # writes CSV to stdout
  python hf_downloads.py > downloads.csv      # save to file
  HF_TOKEN=hf_xxx python hf_downloads.py      # higher API rate limits

Dependencies:
  Standard library only (urllib + json). No extra installs required.
  An optional HF_TOKEN env var raises rate limits and is recommended.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

HF_API = "https://huggingface.co/api/models"
TIMEOUT = 30
SLEEP_BETWEEN_CALLS = 0.5  # be polite to the public API

# ---------------------------------------------------------------------------
# Models used in the study.
# Tuple format: (base_model_label, technique_label, hf_repo_id_or_None)
# ---------------------------------------------------------------------------
REPOS: list[tuple[str, str, str | None]] = [
    # ---- Qwen2.5-Coder-7B family ------------------------------------------
    ("Qwen2.5-Coder-7B-Instruct", "FP16",         "Qwen/Qwen2.5-Coder-7B-Instruct"),
    ("Qwen2.5-Coder-7B-Instruct", "AWQ",          "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ"),
    ("Qwen2.5-Coder-7B-Instruct", "GPTQ",         "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4"),
    ("Qwen2.5-Coder-7B-Instruct", "GGUF",         "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF"),
    ("Qwen2.5-Coder-7B-Instruct", "BitsAndBytes", "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"),
    ("Qwen2.5-Coder-7B-Instruct", "AQLM",         None),  # self-quantized; no public checkpoint
    ("Qwen2.5-Coder-7B-Instruct", "QuIP#",        None),  # self-quantized via QuIP-for-all

    # ---- CodeLlama-7B-Instruct family -------------------------------------
    ("CodeLlama-7B-Instruct",     "FP16",         "codellama/CodeLlama-7b-Instruct-hf"),
    ("CodeLlama-7B-Instruct",     "AWQ",          "TheBloke/CodeLlama-7B-Instruct-AWQ"),
    ("CodeLlama-7B-Instruct",     "GPTQ",         "TheBloke/CodeLlama-7B-Instruct-GPTQ"),
    ("CodeLlama-7B-Instruct",     "GGUF",         "TheBloke/CodeLlama-7B-Instruct-GGUF"),
    # BnB on CodeLlama was applied at runtime to the FP16 base; no separate HF repo.
    ("CodeLlama-7B-Instruct",     "BitsAndBytes", None),
    # AQLM 4-bit mixed-2x15 -- this is the exact repo whose variant dir name
    # (CodeLlama-7b-hf-AQLM-4bit-mixed-2x15) drives copy_results_mceval.sh.
    ("CodeLlama-7B-Instruct",     "AQLM",         "Devy1/CodeLlama-7b-hf-AQLM-4bit-mixed-2x15"),
    ("CodeLlama-7B-Instruct",     "QuIP#",        None),  # self-quantized via QuIP-for-all
]


def fetch_model_info(repo_id: str, token: str | None = None) -> dict | None:
    """
    Query the Hugging Face Hub for a single model's metadata.
    Returns the parsed JSON dict, or None on failure.
    """
    # ?expand[]=downloadsAllTime is required to get cumulative downloads;
    # the default endpoint only returns the last-30-days "downloads" field.
    url = (f"{HF_API}/{repo_id}"
           "?expand[]=downloadsAllTime"
           "&expand[]=downloads"
           "&expand[]=likes"
           "&expand[]=createdAt"
           "&expand[]=lastModified")
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        print(f"  [HTTP {e.code}] {repo_id}", file=sys.stderr)
    except URLError as e:
        print(f"  [URL error] {repo_id}: {e.reason}", file=sys.stderr)
    except json.JSONDecodeError as e:
        print(f"  [JSON error] {repo_id}: {e}", file=sys.stderr)
    return None


def extract_row(base: str, tech: str, repo: str | None,
                meta: dict | None) -> list[str]:
    """Build one CSV row from the API response."""
    if repo is None:
        # tech-specific marker for what "no repo" means
        marker = ("(runtime BnB applied to FP16 base; no separate HF repo)"
                  if tech == "BitsAndBytes"
                  else "(self-quantized; not released on HF)")
        return [base, tech, marker, "", "", "", "", ""]
    if meta is None:
        return [base, tech, repo, "ERROR", "ERROR", "", "", ""]

    # The Hub exposes:
    #   downloads          -> last-30-days download count
    #   downloadsAllTime   -> cumulative since model creation (= "time 0")
    # Some legacy responses omit downloadsAllTime; fall back to downloads.
    downloads_30d = meta.get("downloads", "")
    downloads_all = meta.get("downloadsAllTime", "")
    likes         = meta.get("likes", "")
    created_at    = meta.get("createdAt", "")
    last_modified = meta.get("lastModified", "")

    return [base, tech, repo,
            str(downloads_all), str(downloads_30d), str(likes),
            created_at, last_modified]


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("# (no HF_TOKEN set; using anonymous API access)", file=sys.stderr)

    writer = csv.writer(sys.stdout)
    writer.writerow([
        "base_model", "technique", "repo_id",
        "downloads_all_time", "downloads_last_30d", "likes",
        "created_at", "last_modified",
    ])

    print(f"# Collecting download stats for {len(REPOS)} models", file=sys.stderr)
    for base, tech, repo in REPOS:
        print(f"  -> {base:32s} {tech:14s} {repo or '(self-quantized)'}",
              file=sys.stderr)
        meta = fetch_model_info(repo, token=token) if repo else None
        writer.writerow(extract_row(base, tech, repo, meta))
        sys.stdout.flush()
        if repo:
            time.sleep(SLEEP_BETWEEN_CALLS)

    print("# done.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
