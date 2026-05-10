#!/usr/bin/env python3
"""
Final qualitative-examples CSV: 5 tasks where ALL 7 configurations PASS,
selected for the largest SonarCloud metric spread (LoC + CyC + CoC).

Selections (Qwen2.5-Coder-7B):
  E1  CoderEval-Python  62b8b4c1eb7e40a82d2d1139 (verifyClass)
  E2  McEval-Java       Java-1
  E3  McEval-Python     Python-28
  E4  McEval-Java       Java-53
  E5  CoderEval-Python  62ece4992e6aefcf4aabbd84 (is_ipv4)

Layout: long format, one row per (example x config). Same schema as the
earlier qualitative_examples CSVs.
"""

import csv
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METHODS = REPO / "Methods"
PERDIR_J = REPO / "Analysis-and-Reports" / "Java" / "per-directory"
PERDIR_P = REPO / "Analysis-and-Reports" / "Python" / "per-directory"
OUT_CSV = Path(__file__).resolve().parent / "qualitative_examples_final.csv"

# (example_id, benchmark, language, model, task_id, narrative, source_kind)
# source_kind: "mc-java" | "mc-py" | "ce-py" | "bcb-py"
# Group A: all-7-pass (quality-under-correctness story)
# Group B: QuIP-only-fail (matches paper's headline pass@1 finding)
EXAMPLES = [
    # --- A: all 7 configs PASS ---
    ("E1", "CoderEval", "Python", "Qwen2.5-Coder-7B", "62b8b4c1eb7e40a82d2d1139",
     "ALL PASS: verifyClass — largest metric spread (LoC 2-18, CyC 9, CoC 12)", "ce-py"),
    ("E2", "McEval", "Java", "Qwen2.5-Coder-7B", "Java-1",
     "ALL PASS: strongest McEval-Java all-pass spread (LoC 13-20, CyC 3, CoC 3)", "mc-java"),
    ("E3", "McEval", "Python", "Qwen2.5-Coder-7B", "Python-28",
     "ALL PASS: strongest McEval-Python all-pass spread (LoC 38-42, CoC 6)", "mc-py"),
    ("E4", "McEval", "Java", "Qwen2.5-Coder-7B", "Java-53",
     "ALL PASS: largest McEval-Java all-pass LoC range (LoC 15-26)", "mc-java"),
    ("E5", "CoderEval", "Python", "Qwen2.5-Coder-7B", "62ece4992e6aefcf4aabbd84",
     "ALL PASS: is_ipv4 — balanced LoC + CoC spread (LoC 6-12, CoC 6)", "ce-py"),
    # --- B: QuIP# fails while all 6 others PASS (matches BCB-Python QuIP# pass@1 effect, p_adj=9.6e-12) ---
    ("E6", "BCB", "Python", "Qwen2.5-Coder-7B", "BigCodeBench/1053",
     "QUIP-ONLY FAIL: largest QuIP#-fail metric spread (FP=32 LoC, QuIP#=0 — empty body)", "bcb-py"),
    ("E7", "McEval", "Java", "Qwen2.5-Coder-7B", "Java-28",
     "QUIP-ONLY FAIL: strongest McEval-Java QuIP-fail spread (LoC 12-32, CyC 4, CoC 6)", "mc-java"),
    ("E8", "BCB", "Python", "Qwen2.5-Coder-7B", "BigCodeBench/0",
     "QUIP-ONLY FAIL: BCB-Python balanced spread (LoC 11-22, CyC 6, CoC 5)", "bcb-py"),
]

BCB_PROMPTS = Path("/scratch/oldhome/user/projects/BigCodeBench/scripts/bcb_generations/inputs_instruct_full.jsonl")

CONFIGS = [
    ("FP",    "FP",            "FP"),
    ("AWQ",   "AWQ",           "AWQ"),
    ("GPTQ",  "GPTQ",          "GPTQ"),
    ("GGUF",  "GGUF",          "GGUF"),
    ("BnB",   "BitsAndBytes",  "BitsAndBytes"),
    ("AQLM",  "AQLM",          "AQLM"),
    ("QuIP#", "QUIP",          "QUIP"),
]


def find_mc_results_json(method_dir: str, model: str, task_id: str, lang: str):
    sub = "MC-java" if lang == "Java" else "MC-py"
    base = METHODS / method_dir / sub
    candidates = [
        base / "completion" / model / f"{task_id}.results.json",
        base / "completion" / f"{task_id}.results.json",
    ]
    candidates += list(base.rglob(f"{task_id}.results.json"))
    for c in candidates:
        if c.exists():
            return c
    return None


_CE_CACHE: dict = {}

def load_ce_row(method_dir: str, model: str, sonar_label: str, task_id: str):
    """Return (prompt, generated_code, passed). Caches per (technique) jsonl."""
    cache_key = (method_dir, sonar_label, model)
    if cache_key not in _CE_CACHE:
        # Find the jsonl regardless of inner subdir naming (e.g. BitsAndBytes -> BnB).
        base = METHODS / method_dir / "CE-py" / "completion" / model
        idx = {}
        path = None
        if base.exists():
            matches = list(base.rglob("generations.jsonl_out.jsonl"))
            if matches:
                path = matches[0]
                with open(path) as f:
                    for line in f:
                        if not line.strip(): continue
                        d = json.loads(line)
                        idx[d.get("_id")] = d
        _CE_CACHE[cache_key] = (idx, path)
    idx, path = _CE_CACHE[cache_key]
    if task_id not in idx:
        return None, None, None
    d = idx[task_id]
    # Prompt: signature + docstring.  We assemble: "<name>(...)\n    \"\"\"<docstring>\"\"\""
    prompt = f"# CoderEval task: {d.get('name','')}\n# project: {d.get('project','')}\n# file: {d.get('file_path','')}\n\n\"\"\"{d.get('docstring','')}\"\"\""
    gr = d.get("generate_results") or []
    if isinstance(gr, list) and gr:
        first = gr[0]
        generated = first.get("generate_code", "") or ""
        passed = 1 if first.get("is_pass") else 0
    else:
        # fall back to code list
        code = d.get("code") or []
        generated = code[0] if isinstance(code, list) and code else ""
        passed = 0  # unknown -> conservative
    return prompt, generated, passed


_BCB_PROMPT_CACHE: dict = {}
_BCB_RESULT_CACHE: dict = {}

def _load_bcb_prompts() -> dict:
    if _BCB_PROMPT_CACHE: return _BCB_PROMPT_CACHE
    if BCB_PROMPTS.exists():
        with open(BCB_PROMPTS) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                d = json.loads(line)
                _BCB_PROMPT_CACHE[d["task_id"]] = d.get("instruct_prompt", "")
    return _BCB_PROMPT_CACHE


def load_bcb_row(method_dir: str, model: str, task_id: str):
    """Return (prompt, generated_code, passed) for a BCB-Python task."""
    prompts = _load_bcb_prompts()
    cache_key = (method_dir, model)
    if cache_key not in _BCB_RESULT_CACHE:
        base = METHODS / method_dir / "BCB-py" / "completion" / model
        idx_sol, idx_eval = {}, {}
        if base.exists():
            jsonls = list(base.glob("*sanitized_calibrated.jsonl"))
            if jsonls:
                with open(jsonls[0]) as f:
                    for line in f:
                        if not line.strip(): continue
                        d = json.loads(line)
                        idx_sol[d.get("task_id")] = d.get("solution") or d.get("raw_solution") or ""
            evals = list(base.glob("*sanitized_calibrated_eval_results.json"))
            if evals:
                with open(evals[0]) as f:
                    ev = json.load(f)
                for tid, entries in ev.get("eval", {}).items():
                    if entries and isinstance(entries, list):
                        idx_eval[tid] = entries[0]
        _BCB_RESULT_CACHE[cache_key] = (idx_sol, idx_eval)
    idx_sol, idx_eval = _BCB_RESULT_CACHE[cache_key]
    prompt = prompts.get(task_id, "")
    generated = idx_sol.get(task_id, "")
    eval_entry = idx_eval.get(task_id) or {}
    status = (eval_entry.get("status") or "").lower()
    passed = 1 if status == "pass" else 0
    return prompt, generated, passed


def load_perdir_metrics(perdir: Path, label_prefix: str, model: str, sonar_label: str) -> dict:
    csv_path = perdir / f"{label_prefix}-{model}-{sonar_label}.csv"
    out = {}
    if not csv_path.exists():
        return out
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["TaskID"]] = row
    return out


def extract_passed_mc(rec: dict) -> int:
    results = rec.get("results", [])
    if not results: return 0
    r = results[0]
    return 1 if (r.get("status") == "OK" and r.get("exit_code") == 0) else 0


def main():
    rows = []
    sonar_cache = {}

    for ex_id, bench, lang, model, task_id, narrative, kind in EXAMPLES:
        # Pick the right per-directory metrics root + label prefix.
        if kind == "mc-java":
            perdir, label_prefix = PERDIR_J, "McEval-Java"
        elif kind == "mc-py":
            perdir, label_prefix = PERDIR_P, "McEval-Python"
        elif kind == "ce-py":
            perdir, label_prefix = PERDIR_P, "CoderEval-Python"
        elif kind == "bcb-py":
            perdir, label_prefix = PERDIR_P, "BCB-Python"
        else:
            raise ValueError(kind)

        for cfg_label, method_dir, sonar_label in CONFIGS:
            # SonarCloud metrics
            ck = (kind, model, sonar_label)
            if ck not in sonar_cache:
                sonar_cache[ck] = load_perdir_metrics(perdir, label_prefix, model, sonar_label)
            metrics = sonar_cache[ck].get(task_id, {})

            err = ""
            if kind in ("mc-java", "mc-py"):
                rp = find_mc_results_json(method_dir, model, task_id, lang)
                if rp is None:
                    prompt = ""; generated_code = ""; passed = ""
                    err = f"missing results file for {method_dir}/{task_id}"
                else:
                    with open(rp) as f:
                        rec = json.load(f)
                    prompt = rec.get("prompt", "")
                    generated_code = rec.get("results", [{}])[0].get("program", "") if rec.get("results") else ""
                    passed = extract_passed_mc(rec)
            elif kind == "ce-py":
                p, g, pa = load_ce_row(method_dir, model, sonar_label, task_id)
                if p is None:
                    prompt = ""; generated_code = ""; passed = ""
                    err = f"missing CE-py row for {method_dir}/{task_id}"
                else:
                    prompt, generated_code, passed = p, g, pa
            else:  # bcb-py
                prompt, generated_code, passed = load_bcb_row(method_dir, model, task_id)
                if not generated_code:
                    err = f"missing BCB-py row for {method_dir}/{task_id}"

            rows.append({
                "example_id": ex_id, "narrative": narrative,
                "benchmark": bench, "language": lang, "model": model,
                "task_id": task_id, "config": cfg_label,
                "prompt": prompt, "generated_code": generated_code,
                "passed": passed,
                "LoC": metrics.get("LoC", ""),
                "Reliability": metrics.get("Reliability", ""),
                "Maintainability": metrics.get("Maintainability", ""),
                "Security_Hotspots": metrics.get("Security_Hotspots", ""),
                "CyC": metrics.get("CyC", ""),
                "CoC": metrics.get("CoC", ""),
                "load_error": err,
            })

    fieldnames = [
        "example_id", "narrative", "benchmark", "language", "model",
        "task_id", "config",
        "prompt", "generated_code", "passed",
        "LoC", "Reliability", "Maintainability", "Security_Hotspots", "CyC", "CoC",
        "load_error",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"wrote {len(rows)} rows -> {OUT_CSV}")

    # Summary
    print("\nPass matrix (every cell expected to be PASS):")
    print(f"  {'ex':<3} {'task':<26}", end="")
    for c, _, _ in CONFIGS: print(f" {c:>6}", end="")
    print()
    for ex in [r["example_id"] for r in rows][::7]:
        sub = [r for r in rows if r["example_id"] == ex]
        t = sub[0]["task_id"]
        print(f"  {ex:<3} {t:<26}", end="")
        for c, _, _ in CONFIGS:
            r = next(r for r in sub if r["config"] == c)
            cell = "PASS" if r["passed"] == 1 else ("FAIL" if r["passed"] == 0 else "—")
            print(f" {cell:>6}", end="")
        print()

    print("\nMetric spread per example (LoC / CyC / CoC ranges):")
    print(f"  {'ex':<3} {'task':<26}  {'LoC':<14} {'CyC':<10} {'CoC':<10}")
    for ex in [r["example_id"] for r in rows][::7]:
        sub = [r for r in rows if r["example_id"] == ex]
        t = sub[0]["task_id"]
        loc = [float(r["LoC"]) for r in sub if r["LoC"] not in ("", None)]
        cyc = [float(r["CyC"]) for r in sub if r["CyC"] not in ("", None)]
        coc = [float(r["CoC"]) for r in sub if r["CoC"] not in ("", None)]
        print(f"  {ex:<3} {t:<26}  {f'{min(loc):.0f}-{max(loc):.0f} (Δ{max(loc)-min(loc):.0f})':<14} "
              f"{f'Δ{max(cyc)-min(cyc):.0f}':<10} {f'Δ{max(coc)-min(coc):.0f}':<10}")

    # Code uniqueness
    import hashlib
    print("\nCode uniqueness per example:")
    for ex in [r["example_id"] for r in rows][::7]:
        sub = [r for r in rows if r["example_id"] == ex]
        hashes = {}
        for r in sub:
            h = hashlib.sha1((r["generated_code"] or "").encode("utf-8", errors="replace")).hexdigest()[:8]
            hashes.setdefault(h, []).append(r["config"])
        groups = " | ".join("[" + ",".join(g) + "]" for g in hashes.values())
        print(f"  {ex} ({sub[0]['task_id']}): {len(hashes)}/7 unique  {groups}")


if __name__ == "__main__":
    main()
