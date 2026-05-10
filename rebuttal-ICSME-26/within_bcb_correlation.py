#!/usr/bin/env python3
"""
Within-BigCodeBench point-biserial correlations between prompt complexity
(entropy, length) and a binary degradation outcome.

Addresses a reviewer concern that the pooled (McEval + CoderEval + BCB)
correlations in Table V may capture benchmark identity rather than complexity,
since the High-entropy bucket is ~95% BCB and CoderEval is ~95% in Low.
BCB is the only benchmark with substantial sample size in both High and Low
buckets, so restricting to BCB controls for benchmark identity.

Inputs (reused from the RQ3 pipeline):
  RQ3/rq3_results/rq3_merged.csv  -- per (model, technique, task) rows with
                                      entropy, length_words, pass_fp, pass_q,
                                      degraded, benchmark, model, technique.

Output:
  rebuttal/bcb_within_correlations.csv  -- one row per
                                            (model, technique, complexity-metric).
  Stdout: a Table V-style summary with entropy and length side-by-side.

Definitions:
  - degraded = 1 iff pass_fp == 1 and pass_q == 0; else 0.
  - Sample is restricted to FP-pass tasks (pass_fp == 1) -- tasks where FP failed
    cannot have a degradation outcome.
  - Point-biserial r is Pearson r with one binary variable; we use
    scipy.stats.pointbiserialr.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import pointbiserialr

REPO_ROOT = Path(__file__).resolve().parent.parent
MERGED_CSV = REPO_ROOT / "RQ3" / "rq3_results" / "rq3_merged.csv"
OUT_CSV = Path(__file__).resolve().parent / "bcb_within_correlations.csv"

MODELS = ["CodeLlama-7B-Instruct", "Qwen2.5-Coder-7B-Instruct"]
TECHNIQUES = ["AWQ", "GPTQ", "GGUF", "BnB", "AQLM", "QuIP#"]
METRICS = [("entropy", "entropy"), ("length", "length_words")]


def sig_flag(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def main() -> None:
    df = pd.read_csv(MERGED_CSV)
    bcb = df[df["benchmark"] == "BigCodeBench"].copy()

    rows = []
    for model in MODELS:
        for tech in TECHNIQUES:
            sub = bcb[(bcb["model"] == model) & (bcb["technique"] == tech)]
            fp_pass = sub[sub["pass_fp"] == 1]
            n = len(fp_pass)
            n_deg = int(fp_pass["degraded"].sum())

            for metric_label, col in METRICS:
                # Edge case: degraded all 0 or all 1 -> r undefined.
                if n_deg == 0 or n_deg == n:
                    r, p = float("nan"), float("nan")
                    flag = ""
                else:
                    res = pointbiserialr(fp_pass[col].values, fp_pass["degraded"].values)
                    r, p = float(res.statistic), float(res.pvalue)
                    flag = sig_flag(p)

                rows.append({
                    "Model": model,
                    "Technique": tech,
                    "Metric": metric_label,
                    "n": n,
                    "n_degraded": n_deg,
                    "r": r,
                    "p_value": p,
                    "sig": flag,
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    # --- Sanity check: print n per (model, technique) ---
    print("Sample size sanity check (BCB FP-pass tasks per model x technique):")
    print("-" * 64)
    sanity = (
        out[out["Metric"] == "entropy"][["Model", "Technique", "n", "n_degraded"]]
        .reset_index(drop=True)
    )
    print(sanity.to_string(index=False))
    print()

    # --- Table V-style summary ---
    print("Within-BigCodeBench point-biserial correlations (degraded ~ complexity)")
    print("=" * 88)
    header = f"{'Model':<28} {'Technique':<8}  {'Entropy':>22}     {'Length':>22}"
    print(header)
    print(f"{'':<28} {'':<8}  {'r':>10} {'p':>10}     {'r':>10} {'p':>10}")
    print("-" * 88)

    pivot = out.pivot_table(
        index=["Model", "Technique"],
        columns="Metric",
        values=["r", "p_value", "sig"],
        aggfunc="first",
    )
    # Preserve our model/technique ordering.
    pivot = pivot.reindex(
        pd.MultiIndex.from_product([MODELS, TECHNIQUES], names=["Model", "Technique"])
    )

    last_model = None
    for (model, tech), row in pivot.iterrows():
        r_e = row[("r", "entropy")]
        p_e = row[("p_value", "entropy")]
        s_e = row[("sig", "entropy")] or ""
        r_l = row[("r", "length")]
        p_l = row[("p_value", "length")]
        s_l = row[("sig", "length")] or ""

        model_disp = model if model != last_model else ""
        last_model = model
        print(
            f"{model_disp:<28} {tech:<8}  "
            f"{r_e:>+10.3f} {p_e:>10.4f}{s_e:<3}  "
            f"{r_l:>+10.3f} {p_l:>10.4f}{s_l:<3}"
        )

    print("-" * 88)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    print(f"\nWrote: {OUT_CSV}")


if __name__ == "__main__":
    main()
