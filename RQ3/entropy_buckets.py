#!/usr/bin/env python3
"""
Categorize benchmark tasks into High/Low entropy buckets.
Computes entropy per task, splits at global median, shows distribution.
"""

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Entropy ──

def shannon_entropy(text):
    if not text:
        return 0.0
    symbols = text.split()
    if not symbols:
        return 0.0
    counts = Counter(symbols)
    total = len(symbols)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def extract_docstring(prompt):
    for pattern in [r'"""(.*?)"""', r"'''(.*?)'''", r'/\*\*(.*?)\*/', r'/\*(.*?)\*/']:
        match = re.search(pattern, prompt, re.DOTALL)
        if match:
            text = match.group(1)
            lines = [re.sub(r'^\s*\*\s?', '', l) for l in text.split('\n')]
            return '\n'.join(lines).strip()
    return prompt.strip()


# ── Load datasets ──

def load_all(mceval_path, codereval_path, bigcodebench_path):
    rows = []

    # McEval
    with open(mceval_path) as f:
        for line in f:
            if not line.strip(): continue
            t = json.loads(line)
            text = extract_docstring(t.get("prompt", ""))
            rows.append({"task_id": t.get("name",""), "benchmark": "McEval",
                         "text": text, "entropy": shannon_entropy(text),
                         "length_words": len(text.split())})

    # CoderEval
    with open(codereval_path) as f:
        for line in f:
            if not line.strip(): continue
            t = json.loads(line)
            text = t.get("docstring", "")
            rows.append({"task_id": t.get("question_id",""), "benchmark": "CoderEval",
                         "text": text, "entropy": shannon_entropy(text),
                         "length_words": len(text.split())})

    # BigCodeBench
    with open(bigcodebench_path) as f:
        for line in f:
            if not line.strip(): continue
            t = json.loads(line)
            text = t.get("instruct_prompt", "")
            rows.append({"task_id": t.get("task_id",""), "benchmark": "BigCodeBench",
                         "text": text, "entropy": shannon_entropy(text),
                         "length_words": len(text.split())})

    return pd.DataFrame(rows)


# ── Main ──

mceval_path = sys.argv[1]
codereval_path = sys.argv[2]
bigcodebench_path = sys.argv[3]
output_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("./entropy_buckets")
output_dir.mkdir(parents=True, exist_ok=True)

df = load_all(mceval_path, codereval_path, bigcodebench_path)

# Global median split
median_entropy = df["entropy"].median()
df["bucket"] = np.where(df["entropy"] >= median_entropy, "High", "Low")

print(f"Total tasks: {len(df)}")
print(f"Global median entropy: {median_entropy:.2f}")
print()

# Summary table
summary = df.groupby(["bucket", "benchmark"]).agg(
    count=("entropy", "size"),
    mean_entropy=("entropy", "mean"),
    std_entropy=("entropy", "std"),
    mean_length=("length_words", "mean"),
).reset_index()

# Reorder
summary["bucket"] = pd.Categorical(summary["bucket"], categories=["High", "Low"], ordered=True)
summary["benchmark"] = pd.Categorical(summary["benchmark"],
    categories=["McEval", "CoderEval", "BigCodeBench"], ordered=True)
summary = summary.sort_values(["bucket", "benchmark"]).reset_index(drop=True)

print(summary.to_string(index=False))

# Save CSV (per-task)
csv_path = output_dir / "entropy_buckets_per_task.csv"
df[["task_id", "benchmark", "entropy", "length_words", "bucket"]].to_csv(csv_path, index=False)
print(f"\nPer-task CSV: {csv_path}")

# Save summary CSV
summary_path = output_dir / "entropy_buckets_summary.csv"
summary.to_csv(summary_path, index=False)
print(f"Summary CSV:  {summary_path}")


# ══════════════════════════════════════════════
#  VISUALIZATION
# ══════════════════════════════════════════════

benchmarks = ["McEval", "CoderEval", "BigCodeBench"]
bench_colors = {"McEval": "#2ecc71", "CoderEval": "#f39c12", "BigCodeBench": "#e74c3c"}

fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

# ── Plot 1: Stacked bar showing count per bucket × benchmark ──
ax1 = fig.add_subplot(gs[0, 0])

buckets = ["High", "Low"]
x = np.arange(len(buckets))
width = 0.22
for i, bench in enumerate(benchmarks):
    counts = []
    for bucket in buckets:
        row = summary[(summary["bucket"] == bucket) & (summary["benchmark"] == bench)]
        counts.append(row["count"].values[0] if len(row) > 0 else 0)
    bars = ax1.bar(x + (i - 1) * width, counts, width,
                   label=bench, color=bench_colors[bench], alpha=0.85,
                   edgecolor="black", linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., h + 3,
                     str(int(h)), ha="center", va="bottom", fontsize=9, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(["High Entropy", "Low Entropy"], fontsize=11)
ax1.set_ylabel("Number of Tasks", fontsize=11)
ax1.set_title("Task Count per Bucket", fontsize=12, fontweight="bold")
ax1.legend(fontsize=9)
ax1.grid(axis="y", alpha=0.2)

# ── Plot 2: Box plot of entropy, colored by benchmark, split by bucket ──
ax2 = fig.add_subplot(gs[0, 1])

positions = []
box_data = []
box_colors_list = []
tick_labels = []
pos = 0
for bucket in buckets:
    for bench in benchmarks:
        sub = df[(df["bucket"] == bucket) & (df["benchmark"] == bench)]
        box_data.append(sub["entropy"].values)
        box_colors_list.append(bench_colors[bench])
        positions.append(pos)
        tick_labels.append(f"{bench}\n({bucket})")
        pos += 1
    pos += 0.5  # gap between buckets

bp = ax2.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6,
                 showfliers=True, flierprops=dict(markersize=3, alpha=0.5))
for patch, color in zip(bp["boxes"], box_colors_list):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Median line
ax2.axhline(y=median_entropy, color="black", linestyle="--", linewidth=1, alpha=0.6,
            label=f"Global Median ({median_entropy:.2f})")

ax2.set_xticks(positions)
ax2.set_xticklabels(tick_labels, fontsize=8)
ax2.set_ylabel("Shannon Entropy (bits)", fontsize=11)
ax2.set_title("Entropy Distribution per Bucket", fontsize=12, fontweight="bold")
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.2)

# ── Plot 3: Histogram overlay ──
ax3 = fig.add_subplot(gs[1, 0])

for bench in benchmarks:
    sub = df[df["benchmark"] == bench]
    ax3.hist(sub["entropy"], bins=20, alpha=0.5, color=bench_colors[bench],
             label=f"{bench} (n={len(sub)})", edgecolor="white", linewidth=0.5)

ax3.axvline(x=median_entropy, color="black", linestyle="--", linewidth=2,
            label=f"Median = {median_entropy:.2f}")
ax3.set_xlabel("Shannon Entropy (bits)", fontsize=11)
ax3.set_ylabel("Frequency", fontsize=11)
ax3.set_title("Entropy Distribution by Benchmark", fontsize=12, fontweight="bold")
ax3.legend(fontsize=9)
ax3.grid(axis="y", alpha=0.2)

# ── Plot 4: Proportion pie/donut per benchmark ──
ax4 = fig.add_subplot(gs[1, 1])

# Grouped horizontal bar: % High vs % Low per benchmark
y = np.arange(len(benchmarks))
high_pcts = []
low_pcts = []
for bench in benchmarks:
    total = len(df[df["benchmark"] == bench])
    n_high = len(df[(df["benchmark"] == bench) & (df["bucket"] == "High")])
    high_pcts.append(n_high / total * 100)
    low_pcts.append((total - n_high) / total * 100)

bars_h = ax4.barh(y + 0.15, high_pcts, 0.3, label="High Entropy",
                   color="#e74c3c", alpha=0.75, edgecolor="black", linewidth=0.5)
bars_l = ax4.barh(y - 0.15, low_pcts, 0.3, label="Low Entropy",
                   color="#3498db", alpha=0.75, edgecolor="black", linewidth=0.5)

for bar, pct in zip(bars_h, high_pcts):
    ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
             f"{pct:.0f}%", va="center", fontsize=10, fontweight="bold")
for bar, pct in zip(bars_l, low_pcts):
    ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2.,
             f"{pct:.0f}%", va="center", fontsize=10, fontweight="bold")

ax4.set_yticks(y)
ax4.set_yticklabels(benchmarks, fontsize=11)
ax4.set_xlabel("Percentage of Tasks (%)", fontsize=11)
ax4.set_title("High vs Low Split per Benchmark", fontsize=12, fontweight="bold")
ax4.legend(fontsize=9)
ax4.set_xlim(0, 115)
ax4.grid(axis="x", alpha=0.2)

fig.suptitle("Entropy-Based Task Categorization Across Benchmarks\n"
             f"(Global Median Split at {median_entropy:.2f} bits)",
             fontsize=14, fontweight="bold", y=1.02)

plot_path = output_dir / "entropy_buckets_visualization.png"
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close()
print(f"Figure:       {plot_path}")