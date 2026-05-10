#!/usr/bin/env python3
"""
Combined rebuttal experiment for Qwen2.5-Coder-7B-Instruct on McEval-Python (42 tasks):

  (1) 10-run pass@1 variance analysis  (R1_C3)
  (2) Per-technique efficiency profile (R3_Q2):
        peak VRAM, model load time, avg per-prompt latency, throughput.

Reuses the existing model-loading and scoring pipeline:
  - Model dispatch mirrors lowbit-quantization-D070/utils/MultiPL-E/automodel_quantize-2.py
    (QuIP / GGUF / BnB / plain HF) plus AQLM via plain HF (auto-loads via aqlm package).
  - Generation parameters: temperature=0.0, max_tokens=1024, batch_size=1,
    completion_limit=1, do_sample=True (matches the existing scripts; transformers
    falls back to greedy at temperature=0, so output is deterministic).
  - Scoring: writes completions in MultiPL-E's .json.gz format, runs
    clean_completions.py, then `docker run multipl-e-eval`, then reads the
    per-task .results.json.gz files (status == "OK" and exit_code == 0 -> pass).

Outputs (all under rebuttal/):
  - variance_results.csv   long form: Configuration,Run,Task_ID,Pass
  - variance_summary.csv   one row per config: Configuration,Pass1_Run1..10,Mean,Std,Min,Max
  - efficiency_results.csv one row per config: Configuration,Peak_VRAM_GB,
                            Model_Load_Time_s,Avg_Latency_s,Throughput_tok_per_s
  - combined_summary.md    markdown tables, plus a section listing any
                            configurations that failed to load.

Usage:
  CUDA_VISIBLE_DEVICES=2 python3 rebuttal/combined_variance_efficiency.py \
        [--num-runs 10] [--configs FP16,AWQ,...] [--skip-eval]

Notes:
  - Set CUDA_VISIBLE_DEVICES before launching (the existing scripts pin GPU 2).
  - The MultiPL-E docker image must be available (multipl-e-eval tag).
  - The QuIP weights live next to hadamard.safetensors; the loader chdirs into
    the QuIP-for-all directory exactly as the production loader does.
"""

import argparse
import gc
import gzip
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------------------

THIS = Path(__file__).resolve()
REBUTTAL_DIR = THIS.parent
REPO_ROOT = REBUTTAL_DIR.parent  # Empirical-Quantization-Study/
LOWBIT_ROOT = Path("/scratch/oldhome/user/projects/lowbit-quantization-D070")
UTILS_DIR = LOWBIT_ROOT / "utils"
MULTIPLE_DIR = UTILS_DIR / "MultiPL-E"
CLEAN_SCRIPT = UTILS_DIR / "clean_completions.py"
MCEVAL_PY_JSONL = LOWBIT_ROOT / "4-run-benchmarks" / "mceval" / "mceval_py.jsonl"

WORK_DIR = REBUTTAL_DIR / "runs_qwen_mceval_py"

CONFIGS = [
    # (label, model_name_or_path, dispatch)
    ("FP16",  "Qwen/Qwen2.5-Coder-7B-Instruct",                                                       "fp"),
    ("AWQ",   "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",                                                   "fp"),  # AWQ weights load via plain HF
    ("GPTQ",  "Qwen/Qwen2.5-Coder-7B-Instruct-GPTQ-Int4",                                             "fp"),  # GPTQ weights load via plain HF
    ("GGUF",  str(REPO_ROOT / "Methods/GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf"),                  "gguf"),
    ("BnB",   "Qwen/Qwen2.5-Coder-7B-Instruct_bnb",                                                   "bnb"),
    ("AQLM",  str(LOWBIT_ROOT / "2-quantize-models/output/Qwen2.5-Coder-7B-Instruct"),                "fp"),
    ("QuIP#", str(REPO_ROOT / "Methods/QUIP/QuIP-for-all/qwen-coder-7B_4bit_quip"),                   "quip"),
]

TEMPERATURE = 0.0
TOP_P = 0.95
MAX_TOKENS = 1024
WARMUP_PROMPTS = 3  # excluded from latency average


# ----------------------------------------------------------------------
# Model loaders (mirror existing pipeline dispatch)
# ----------------------------------------------------------------------

def load_model(name: str, dispatch: str):
    """Return (model, tokenizer). Raises on failure."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if dispatch == "quip":
        # Replicates automodel_quantize-2.py's QuIP branch.
        quip_dir = str(REPO_ROOT / "Methods/QUIP/QuIP-for-all")
        sys.path.insert(0, quip_dir)
        original_dir = os.getcwd()
        os.chdir(quip_dir)  # QuIP needs hadamard.safetensors from its directory
        try:
            from quantizer import load_quantized_model  # type: ignore
            model = load_quantized_model(name).cuda()
        finally:
            os.chdir(original_dir)
        tok_path = "Qwen/Qwen2.5-Coder-7B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(
            tok_path, padding_side="left", trust_remote_code=True
        )

    elif dispatch == "gguf":
        model = AutoModelForCausalLM.from_pretrained(
            os.path.dirname(name),
            gguf_file=os.path.basename(name),
            torch_dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-Coder-7B-Instruct",
            padding_side="left",
            trust_remote_code=True,
        )

    elif dispatch == "bnb":
        from transformers import BitsAndBytesConfig
        actual_model = re.sub(r"[_-]bnb.*", "", name)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            actual_model,
            quantization_config=quantization_config,
            device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            actual_model, padding_side="left", trust_remote_code=True
        )

    else:  # "fp" -- plain HF (also handles AWQ, GPTQ, AQLM whose configs auto-dispatch)
        model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype="auto",
            device_map="cuda",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            name, padding_side="left", trust_remote_code=True
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------

def stop_at_stop_token(decoded: str, stop_tokens) -> str:
    min_idx = len(decoded)
    for tok in stop_tokens:
        i = decoded.find(tok)
        if i != -1 and i < min_idx:
            min_idx = i
    return decoded[:min_idx]


def generate_one(model, tokenizer, prompt: str, max_tokens: int, temperature: float, top_p: float):
    """At temperature 0 we use greedy (do_sample=False); otherwise sample.
    The legacy automodel scripts pass do_sample=True even at temp=0 -- that
    was a no-op warning on older transformers but is now a hard error
    (transformers >=4.47), so we explicitly switch to greedy here. Output is
    identical to the legacy fall-back behavior."""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        return_token_type_ids=False,
        truncation=True,
        max_length=max_tokens - 1,
    ).to("cuda")
    input_len = inputs["input_ids"].shape[1]
    max_length = input_len + max_tokens

    gen_kwargs = dict(
        use_cache=True,
        max_length=max_length,
        pad_token_id=tokenizer.pad_token_id,
    )
    if temperature > 0.0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    gen_ids = out[0, input_len:]
    n_gen = int(gen_ids.shape[0])
    decoded = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return decoded, n_gen


# ----------------------------------------------------------------------
# Per-task completion files (MultiPL-E .json.gz format)
# ----------------------------------------------------------------------

def write_completion(out_dir: Path, problem: dict, completion: str) -> None:
    payload = {
        "name": problem["name"],
        "language": problem["language"],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "prompt": problem["prompt"],
        "tests": problem["tests"],
        "completions": [completion],
        "stop_tokens": problem["stop_tokens"],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_dir / f"{problem['name']}.json.gz", "wt") as f:
        f.write(json.dumps(payload))


# ----------------------------------------------------------------------
# Scoring (clean_completions + docker eval + parse .results.json.gz)
# ----------------------------------------------------------------------

def score_run(raw_dir: Path, cleaned_dir: Path) -> dict:
    """Run clean_completions.py + docker eval; return {task_id: 0/1}."""
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            sys.executable, "-u", str(CLEAN_SCRIPT),
            "--input_dir", str(raw_dir),
            "--language", "py",
            "--output_dir", str(cleaned_dir),
        ],
        check=True,
        cwd=str(UTILS_DIR),
    )

    # Docker eval -- mount cleaned_dir at the same absolute path inside the container.
    subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none",
            "-v", f"{cleaned_dir}:{cleaned_dir}:rw",
            "multipl-e-eval",
            "--dir", str(cleaned_dir),
            "--output-dir", str(cleaned_dir),
            "--recursive",
        ],
        check=True,
    )

    pass_map = {}
    for results_file in sorted(cleaned_dir.glob("*.results.json.gz")):
        with gzip.open(results_file, "rt") as f:
            d = json.load(f)
        results = d.get("results", [])
        passed = bool(results) and all(
            r.get("status") == "OK" and r.get("exit_code") == 0 for r in results
        )
        pass_map[d["name"]] = 1 if passed else 0
    return pass_map


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def load_prompts():
    prompts = []
    with open(MCEVAL_PY_JSONL) as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def free_model(model):
    try:
        model.to("cpu")
    except Exception:
        pass
    del model
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--configs", default=None,
                        help="Comma-separated subset of config labels to run (default: all).")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip generation; only re-score existing completions.")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Skip docker eval (useful when iterating on the script).")
    args = parser.parse_args()

    selected = set(args.configs.split(",")) if args.configs else None
    configs = [c for c in CONFIGS if (selected is None or c[0] in selected)]
    print(f"[info] running {len(configs)} configs x {args.num_runs} runs on "
          f"{MCEVAL_PY_JSONL.name}", flush=True)

    prompts = load_prompts()
    print(f"[info] loaded {len(prompts)} prompts", flush=True)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    variance_rows = []           # Configuration, Run, Task_ID, Pass
    efficiency_rows = []         # Configuration, Peak_VRAM_GB, Model_Load_Time_s, Avg_Latency_s, Throughput_tok_per_s
    failures = []                # (Configuration, error_message)

    for label, name, dispatch in configs:
        print(f"\n========== {label} ({name}) ==========", flush=True)
        config_dir = WORK_DIR / label.replace("#", "Sharp")
        config_dir.mkdir(parents=True, exist_ok=True)

        peak_vram_gb = float("nan")
        load_time_s = float("nan")
        avg_latency_s = float("nan")
        throughput = float("nan")

        # ---- Load + generation ----
        model = tokenizer = None
        if not args.skip_gen:
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
                model, tokenizer = load_model(name, dispatch)
                # Force a full GPU sync so timing is honest.
                torch.cuda.synchronize()
                load_time_s = time.time() - t0
                peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"[load] {label}: {load_time_s:.2f}s, peak VRAM {peak_vram_gb:.2f} GB",
                      flush=True)
            except Exception as e:
                err = f"load failure: {type(e).__name__}: {e}"
                print(f"[FAIL] {label}: {err}\n{traceback.format_exc()}", flush=True)
                failures.append((label, err))
                continue

            # Run N sweeps. Time only run 1. Catch generation errors per config
            # so one failure does not abort the sweep.
            try:
                for run in range(1, args.num_runs + 1):
                    run_dir = config_dir / f"run_{run}_raw"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    latencies = []
                    tokens_generated = 0

                    for idx, problem in enumerate(prompts):
                        t0 = time.time()
                        decoded, n_tok = generate_one(
                            model, tokenizer, problem["prompt"],
                            MAX_TOKENS, TEMPERATURE, TOP_P,
                        )
                        torch.cuda.synchronize()
                        dt = time.time() - t0
                        completion = stop_at_stop_token(decoded, problem["stop_tokens"])
                        write_completion(run_dir, problem, completion)
                        if run == 1 and idx >= WARMUP_PROMPTS:
                            latencies.append(dt)
                            tokens_generated += n_tok

                    if run == 1 and latencies:
                        avg_latency_s = statistics.fmean(latencies)
                        total_active_time = sum(latencies)
                        throughput = tokens_generated / total_active_time if total_active_time > 0 else float("nan")
                        print(f"[run1] {label}: avg latency {avg_latency_s:.2f}s, "
                              f"throughput {throughput:.1f} tok/s "
                              f"(over {len(latencies)} prompts)", flush=True)

                    print(f"  [{label}] run {run}/{args.num_runs} generated.", flush=True)
            except Exception as e:
                err = f"generation failure: {type(e).__name__}: {e}"
                print(f"[FAIL] {label}: {err}\n{traceback.format_exc()}", flush=True)
                failures.append((label, err))
            finally:
                if model is not None:
                    free_model(model)

        # ---- Scoring ----
        if not args.skip_eval:
            try:
                for run in range(1, args.num_runs + 1):
                    raw_dir = config_dir / f"run_{run}_raw"
                    cleaned_dir = config_dir / f"run_{run}_cleaned"
                    if not raw_dir.exists():
                        continue
                    print(f"  [{label}] scoring run {run}...", flush=True)
                    pass_map = score_run(raw_dir, cleaned_dir)
                    for task_id, p in pass_map.items():
                        variance_rows.append({
                            "Configuration": label, "Run": run,
                            "Task_ID": task_id, "Pass": p,
                        })
            except Exception as e:
                err = f"scoring failure: {type(e).__name__}: {e}"
                print(f"[FAIL] {label}: {err}\n{traceback.format_exc()}", flush=True)
                failures.append((label, err))

        efficiency_rows.append({
            "Configuration": label,
            "Peak_VRAM_GB": peak_vram_gb,
            "Model_Load_Time_s": load_time_s,
            "Avg_Latency_s": avg_latency_s,
            "Throughput_tok_per_s": throughput,
        })

    # ----------------------------------------------------------------------
    # Write outputs
    # ----------------------------------------------------------------------
    import csv

    var_csv = REBUTTAL_DIR / "variance_results.csv"
    with open(var_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Configuration", "Run", "Task_ID", "Pass"])
        w.writeheader()
        for row in variance_rows:
            w.writerow(row)
    print(f"[out] {var_csv}", flush=True)

    # variance_summary.csv: per-config Pass1_Run1..PassN, Mean, Std, Min, Max
    sum_csv = REBUTTAL_DIR / "variance_summary.csv"
    headers = ["Configuration"] + [f"Pass1_Run{r}" for r in range(1, args.num_runs + 1)] + \
              ["Mean", "Std", "Min", "Max"]
    by_cfg = {}
    for row in variance_rows:
        by_cfg.setdefault(row["Configuration"], {}).setdefault(row["Run"], []).append(row["Pass"])
    with open(sum_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for label, _, _ in configs:
            if label not in by_cfg:
                w.writerow([label] + [""] * (args.num_runs + 4))
                continue
            per_run_pass1 = []
            for run in range(1, args.num_runs + 1):
                vals = by_cfg[label].get(run, [])
                if vals:
                    per_run_pass1.append(sum(vals) / len(vals))
                else:
                    per_run_pass1.append(None)
            valid = [v for v in per_run_pass1 if v is not None]
            mean_v = statistics.fmean(valid) if valid else ""
            std_v = statistics.pstdev(valid) if len(valid) > 1 else (0.0 if valid else "")
            min_v = min(valid) if valid else ""
            max_v = max(valid) if valid else ""
            w.writerow([label] + ["" if v is None else f"{v:.6f}" for v in per_run_pass1] +
                       [f"{mean_v:.6f}" if valid else "",
                        f"{std_v:.6f}" if valid else "",
                        f"{min_v:.6f}" if valid else "",
                        f"{max_v:.6f}" if valid else ""])
    print(f"[out] {sum_csv}", flush=True)

    eff_csv = REBUTTAL_DIR / "efficiency_results.csv"
    with open(eff_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["Configuration", "Peak_VRAM_GB", "Model_Load_Time_s",
                           "Avg_Latency_s", "Throughput_tok_per_s"],
        )
        w.writeheader()
        for row in efficiency_rows:
            w.writerow({k: ("" if (isinstance(v, float) and v != v) else v) for k, v in row.items()})
    print(f"[out] {eff_csv}", flush=True)

    # combined_summary.md
    md = REBUTTAL_DIR / "combined_summary.md"
    with open(md, "w") as f:
        f.write("# Rebuttal: Qwen2.5-Coder-7B-Instruct on McEval-Python\n\n")
        f.write(f"42 prompts × {args.num_runs} runs × {len(configs)} configurations.\n")
        f.write(f"Generation: temperature={TEMPERATURE}, top_p={TOP_P}, max_tokens={MAX_TOKENS}, batch_size=1, completion_limit=1.\n\n")

        f.write("## Variance (R1_C3)\n\n")
        f.write("| Configuration | " +
                " | ".join(f"Run {r}" for r in range(1, args.num_runs + 1)) +
                " | Mean | Std | Min | Max |\n")
        f.write("|" + "---|" * (args.num_runs + 5) + "\n")
        for label, _, _ in configs:
            if label not in by_cfg:
                f.write(f"| {label} |" + " — |" * (args.num_runs + 4) + "\n")
                continue
            per_run_pass1 = []
            for run in range(1, args.num_runs + 1):
                vals = by_cfg[label].get(run, [])
                per_run_pass1.append(sum(vals) / len(vals) if vals else None)
            valid = [v for v in per_run_pass1 if v is not None]
            mean_v = statistics.fmean(valid) if valid else float("nan")
            std_v = statistics.pstdev(valid) if len(valid) > 1 else (0.0 if valid else float("nan"))
            min_v = min(valid) if valid else float("nan")
            max_v = max(valid) if valid else float("nan")
            cells = ["—" if v is None else f"{v:.4f}" for v in per_run_pass1]
            f.write(f"| {label} | " + " | ".join(cells) +
                    f" | {mean_v:.4f} | {std_v:.4f} | {min_v:.4f} | {max_v:.4f} |\n")

        f.write("\n## Efficiency (R3_Q2)\n\n")
        f.write("| Configuration | Peak VRAM (GB) | Model Load (s) | Avg Latency (s/prompt) | Throughput (tok/s) |\n")
        f.write("|---|---|---|---|---|\n")
        for row in efficiency_rows:
            def fmt(x):
                return "—" if (isinstance(x, float) and x != x) else f"{x:.2f}"
            f.write(f"| {row['Configuration']} | {fmt(row['Peak_VRAM_GB'])} | "
                    f"{fmt(row['Model_Load_Time_s'])} | {fmt(row['Avg_Latency_s'])} | "
                    f"{fmt(row['Throughput_tok_per_s'])} |\n")

        if failures:
            f.write("\n## Configurations that failed to load\n\n")
            for label, msg in failures:
                f.write(f"- **{label}**: {msg}\n")
        else:
            f.write("\nAll configurations loaded successfully.\n")

    print(f"[out] {md}", flush=True)
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
