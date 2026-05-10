# Rebuttal: Within-BigCodeBench Complexity Correlations

This folder contains a rebuttal analysis addressing a reviewer's concern that the
pooled Table V correlations between prompt complexity and quantization-induced
degradation may be confounded by benchmark identity. The pooled analysis mixes
McEval, CoderEval, and BigCodeBench tasks, but the High-entropy bucket is ~95%
BigCodeBench while CoderEval is ~95% Low, so the pooled point-biserial r could
reflect "this task is from BCB" rather than complexity itself. To control for
benchmark identity, we re-compute the same point-biserial correlations using
only BigCodeBench-Python tasks — BCB is the one benchmark with substantial
sample size in both High (650) and Low (490) entropy buckets. For each of the
12 (model x technique) combinations and each complexity metric (Shannon entropy,
prompt token length), we restrict to FP-pass tasks (since tasks where FP failed
have no degradation outcome) and report n, n_degraded, Pearson r, p-value, and
a significance flag. Results are written to `bcb_within_correlations.csv` and a
Table V-style summary is printed to stdout.

## Reproduce

The script reuses the per-task merged file produced by the existing RQ3
pipeline (`RQ3/rq3_results/rq3_merged.csv`), so no upstream re-run is needed.
From the repository root:

```bash
python3 rebuttal/within_bcb_correlation.py
```

Requires `pandas` and `scipy`. Outputs `rebuttal/bcb_within_correlations.csv`.

---

# Rebuttal: Variance + Efficiency for Qwen2.5-Coder-7B-Instruct on McEval-Python

Combined experiment addressing two reviewer concerns at once:
**(R1_C3)** is our pass@1 sensitive to run-to-run noise? and
**(R3_Q2)** what is the per-technique efficiency profile?

We run all 42 McEval-Python tasks against Qwen2.5-Coder-7B-Instruct under 7
configurations (FP16 + AWQ, GPTQ, GGUF, BitsAndBytes, AQLM, QuIP#), repeating
generation 10 times per configuration. Generation parameters are identical to
the main experiments (`temperature=0`, `top_p=0.95`, `max_tokens=1024`,
`batch_size=1`, `do_sample=True` — at temperature 0 transformers falls back to
greedy decoding, so results are deterministic; we still run 10× to confirm).
Model-loading dispatch and the scoring pipeline (clean_completions.py +
multipl-e-eval Docker image) are reused unchanged from
`lowbit-quantization-D070/`. On run 1 of each configuration we additionally
log peak VRAM (`torch.cuda.max_memory_allocated`), wall-clock model load time,
average per-prompt latency (excluding 3 warm-up prompts), and throughput
(tokens/sec).

**Outputs**

- `variance_results.csv` — long: `Configuration, Run, Task_ID, Pass`
- `variance_summary.csv` — wide: `Configuration, Pass1_Run1..Pass1_Run10, Mean, Std, Min, Max`
- `efficiency_results.csv` — `Configuration, Peak_VRAM_GB, Model_Load_Time_s, Avg_Latency_s, Throughput_tok_per_s`
- `combined_summary.md` — markdown tables for both, plus a section listing any configurations that failed to load.

**Reproduce**

Use the same conda env as the main experiments
(`/scratch/oldhome/user/anaconda3/envs/aqlm-fse` — torch 2.5.1+cu124,
transformers 4.47.0, autoawq 0.2.7, auto_gptq 0.7.1, bitsandbytes 0.48.2,
aqlm 1.1.6, gguf 0.17.1, quiptools_cuda 0.0.0). The MultiPL-E Docker image must
be tagged `multipl-e-eval`. Pin a single GPU (the production scripts use GPU 2):

```bash
cd /scratch/oldhome/user/projects/Empirical-Quantization-Study
conda activate /scratch/oldhome/user/anaconda3/envs/aqlm-fse
CUDA_VISIBLE_DEVICES=2 python3 rebuttal/combined_variance_efficiency.py
```

Useful flags while iterating:

- `--configs FP16,AWQ` — restrict to a subset.
- `--num-runs 1` — quick smoke-test before launching the full sweep.
- `--skip-eval` — generate completions without running Docker (e.g., to debug loaders).
- `--skip-gen` — re-score existing completions without re-generating.

Per-config completions and cleaned/eval outputs are written under
`rebuttal/runs_qwen_mceval_py/<config>/run_<n>_{raw,cleaned}/`. The full
sweep takes several GPU-hours; partial results are usable since CSV/Markdown
outputs are written once at the end.

**Model paths used**

| Configuration | Source |
|---|---|
| FP16  | `Qwen/Qwen2.5-Coder-7B-Instruct` (HF) |
| AWQ   | `Qwen/Qwen2.5-Coder-7B-Instruct-AWQ` (HF, official) |
| GPTQ  | `Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4` (HF, official) |
| GGUF  | `Methods/GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf` (local) |
| BnB   | `Qwen/Qwen2.5-Coder-7B-Instruct` loaded with NF4 4-bit BnB config (the loader strips a `_bnb` suffix from the name to find the base model — same pattern as `automodel_quantize-2.py`) |
| AQLM  | `lowbit-quantization-D070/2-quantize-models/output/Qwen2.5-Coder-7B-Instruct` (local; quant_method=aqlm in config.json) |
| QuIP# | `Methods/QUIP/QuIP-for-all/qwen-coder-7B_4bit_quip` (local; loaded via the in-tree QuIP-for-all `quantizer.load_quantized_model`) |

