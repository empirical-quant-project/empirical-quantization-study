#!/bin/bash
#
# Automates copying .results.json.gz files from results_cleaned into
# the Empirical-Quantization-Study/Methods folder structure, then gunzips them.
#
# Handles two source patterns:
#   1. Top-level model variant dirs (AWQ, GPTQ, FP, BnB, CodeLlama-AQLM)
#   2. scratch/ dir (GGUF, AQLM-Qwen, QuIP — long path-encoded dataset names)
#
# Destination: Methods/{METHOD}/{benchmark}-{lang}/completion/{model}/
#   where {model} is "CodeLlama-7B" or "Qwen2.5-Coder-7B"
#
# Usage:   bash copy_results.sh [--dry-run]
#          --dry-run : show what would be copied without actually doing it

set -euo pipefail

# ──────────────────────────── CONFIG ────────────────────────────
SRC_BASE="/scratch/oldhome/user/projects/lowbit-quantization-D070/4-run-benchmarks/results_cleaned"
DST_BASE="/scratch/oldhome/user/projects/Empirical-Quantization-Study/Methods"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

# Counters
COPIED=0
SKIPPED=0
ERRORS=0

# ──────────────────────── HELPER FUNCTIONS ──────────────────────

# Map dataset folder name prefix → destination benchmark folder
get_benchmark_folder() {
    local dataset_name="$1"
    if [[ "$dataset_name" == humaneval-java-* ]]; then
        echo "multipLE-java"
    elif [[ "$dataset_name" == humaneval-py-* ]]; then
        echo "multipLE-py"
    elif [[ "$dataset_name" == mceval_java-* ]]; then
        echo "MC-java"
    elif [[ "$dataset_name" == mceval_py-* ]]; then
        echo "MC-py"
    else
        echo "UNKNOWN"
    fi
}

# Process a single dataset directory: copy .results.json.gz → dest/model/, then gunzip
# Args: $1=dataset_dir  $2=method  $3=benchmark_folder  $4=model_name  $5=label (for logging)
process_dataset_dir() {
    local dataset_dir="$1"
    local method="$2"
    local benchmark_folder="$3"
    local model_name="$4"
    local label="$5"

    local dest_path="$DST_BASE/$method/$benchmark_folder/completion/$model_name"

    # Count .results.json.gz files
    local file_count
    file_count=$(find "$dataset_dir" -maxdepth 1 -name "*.results.json.gz" 2>/dev/null | wc -l)
    if [[ "$file_count" -eq 0 ]]; then
        return
    fi

    echo -e "${CYAN}─────────────────────────────────────────${NC}"
    echo -e "  Source  : $label"
    echo -e "  Method  : ${GREEN}$method${NC}"
    echo -e "  Bench   : $benchmark_folder"
    echo -e "  Model   : $model_name"
    echo -e "  Files   : $file_count .results.json.gz"
    echo -e "  Dest    : $dest_path/"

    if $DRY_RUN; then
        echo -e "  ${CYAN}[DRY RUN] Would copy $file_count files${NC}"
        ((COPIED += file_count))
        return
    fi

    # Create destination directory
    mkdir -p "$dest_path"

    # Copy files
    if cp "$dataset_dir"/*.results.json.gz "$dest_path/" 2>/dev/null; then
        # Gunzip (force overwrite existing)
        if gunzip -f "$dest_path"/*.results.json.gz 2>/dev/null; then
            local result_count
            result_count=$(find "$dest_path" -maxdepth 1 -name "*.results.json" | wc -l)
            echo -e "  ${GREEN}✓ Copied & gunzipped → $result_count .results.json files${NC}"
            ((COPIED += file_count))
        else
            echo -e "  ${RED}✗ Copy succeeded but gunzip failed${NC}"
            ((ERRORS++))
        fi
    else
        echo -e "  ${RED}✗ Copy failed${NC}"
        ((ERRORS++))
    fi
}

# ──────────────── VARIANT → METHOD + MODEL MAPS ────────────────
# Explicit mapping: variant dir name → quantization method
declare -A VARIANT_METHOD_MAP=(
    # FP (full precision baselines)
    ["CodeLlama-7b-Instruct-hf"]="FP"
    ["Qwen2.5-Coder-7B-Instruct"]="FP"
    # AWQ
    ["CodeLlama-7B-Instruct-AWQ"]="AWQ"
    ["Qwen2.5-Coder-7B-Instruct-AWQ"]="AWQ"
    # GPTQ
    ["CodeLlama-7B-Instruct-GPTQ"]="GPTQ"
    ["Qwen2.5-Coder-7B-Instruct-GPTQ-Int4"]="GPTQ"
    # BitsAndBytes
    ["CodeLlama-7b-Instruct-hf_bnb"]="BitsAndBytes"
    ["Qwen2.5-Coder-7B-bnb-4bit"]="BitsAndBytes"
    # AQLM (CodeLlama — top-level dir)
    ["CodeLlama-7b-hf-AQLM-4bit-mixed-2x15"]="AQLM"
)

# Explicit mapping: variant dir name → model subdirectory name
declare -A VARIANT_MODEL_MAP=(
    ["CodeLlama-7b-Instruct-hf"]="CodeLlama-7B"
    ["Qwen2.5-Coder-7B-Instruct"]="Qwen2.5-Coder-7B"
    ["CodeLlama-7B-Instruct-AWQ"]="CodeLlama-7B"
    ["Qwen2.5-Coder-7B-Instruct-AWQ"]="Qwen2.5-Coder-7B"
    ["CodeLlama-7B-Instruct-GPTQ"]="CodeLlama-7B"
    ["Qwen2.5-Coder-7B-Instruct-GPTQ-Int4"]="Qwen2.5-Coder-7B"
    ["CodeLlama-7b-Instruct-hf_bnb"]="CodeLlama-7B"
    ["Qwen2.5-Coder-7B-bnb-4bit"]="Qwen2.5-Coder-7B"
    ["CodeLlama-7b-hf-AQLM-4bit-mixed-2x15"]="CodeLlama-7B"
)

# ──────────────────── SCRATCH DIR HELPERS ───────────────────────

# Detect method from the benchmark subdir name inside scratch/
get_scratch_method() {
    local bench_dir_name="$1"
    if [[ "$bench_dir_name" == *"_AQLM" ]]; then
        echo "AQLM"
    elif [[ "$bench_dir_name" == *"_GGUF" ]]; then
        echo "GGUF"
    elif [[ "$bench_dir_name" == *"_quip" ]]; then
        echo "QUIP"
    else
        echo "UNKNOWN"
    fi
}

# Filter: only process 7B models from scratch/ dataset dirs
is_7b_model() {
    local dataset_name="$1"
    if [[ "$dataset_name" == *"7b"* ]] || [[ "$dataset_name" == *"7B"* ]]; then
        return 0
    fi
    return 1
}

# Detect model name from scratch/ dataset folder names.
# QuIP has a duplicate CodeLlama run (trailing underscore) → CodeLlama-7B-run2
get_scratch_model() {
    local dataset_name="$1"

    # QuIP: trailing underscore variant = duplicate CodeLlama run
    if [[ "$dataset_name" == *"CodeLlama_7b_4bit_quip_-"* ]]; then
        echo "CodeLlama-7B-run2"
        return
    fi

    # Case-insensitive model detection
    local lower
    lower=$(echo "$dataset_name" | tr '[:upper:]' '[:lower:]')

    if [[ "$lower" == *"codellama"* ]]; then
        echo "CodeLlama-7B"
    elif [[ "$lower" == *"qwen"* ]]; then
        echo "Qwen2.5-Coder-7B"
    else
        echo "UNKNOWN"
    fi
}

# ──────────────────────── MAIN LOGIC ────────────────────────────

echo ""
echo "=========================================="
echo "  Results Copy & Organize Automation"
echo "=========================================="
$DRY_RUN && echo -e "${CYAN}  *** DRY RUN MODE — nothing will be copied ***${NC}"
echo ""
echo "  Source : $SRC_BASE"
echo "  Dest   : $DST_BASE"
echo ""

# ────────── PART 1: Top-level model variant directories ──────────
echo -e "${GREEN}━━━ PART 1: Standard model variant directories ━━━${NC}"

for variant_dir in "$SRC_BASE"/*/; do
    variant=$(basename "$variant_dir")

    # Skip Archive, scratch, and CodeLlama-7b-Instruct-bnb (empty — use hf_bnb instead)
    [[ "$variant" == "Archive" ]] && continue
    [[ "$variant" == "scratch" ]] && continue
    [[ "$variant" == "CodeLlama-7b-Instruct-bnb" ]] && continue

    # Look up method and model from explicit maps
    method="${VARIANT_METHOD_MAP[$variant]:-}"
    model_name="${VARIANT_MODEL_MAP[$variant]:-}"
    if [[ -z "$method" ]] || [[ -z "$model_name" ]]; then
        echo -e "${YELLOW}⚠  Skipping unmapped variant: $variant${NC}"
        continue
    fi

    # Iterate over benchmark subdirs (e.g. java_benchmark_temperature_0.0_AWQ)
    for bench_dir in "$variant_dir"/*/; do
        [ -d "$bench_dir" ] || continue

        # Iterate over dataset folders inside the benchmark dir
        for dataset_dir in "$bench_dir"/*/; do
            [ -d "$dataset_dir" ] || continue

            dataset_name=$(basename "$dataset_dir")
            benchmark_folder=$(get_benchmark_folder "$dataset_name")

            if [[ "$benchmark_folder" == "UNKNOWN" ]]; then
                echo -e "${YELLOW}⚠  Skipping unknown dataset: $dataset_name${NC}"
                continue
            fi

            process_dataset_dir "$dataset_dir" "$method" "$benchmark_folder" "$model_name" "$variant / $dataset_name"
        done
    done
done

# ────────── PART 2: scratch/ directory (GGUF, AQLM-Qwen, QuIP) ──────────
echo ""
echo -e "${GREEN}━━━ PART 2: scratch/ directory (GGUF, AQLM, QuIP) ━━━${NC}"

SCRATCH_DIR="$SRC_BASE/scratch"
if [ -d "$SCRATCH_DIR" ]; then
    for bench_dir in "$SCRATCH_DIR"/*/; do
        [ -d "$bench_dir" ] || continue
        bench_dir_name=$(basename "$bench_dir")

        method=$(get_scratch_method "$bench_dir_name")
        if [[ "$method" == "UNKNOWN" ]]; then
            echo -e "${YELLOW}⚠  Skipping unknown scratch bench dir: $bench_dir_name${NC}"
            continue
        fi

        for dataset_dir in "$bench_dir"/*/; do
            [ -d "$dataset_dir" ] || continue

            dataset_name=$(basename "$dataset_dir")

            # Only process 7B models
            if ! is_7b_model "$dataset_name"; then
                echo -e "${YELLOW}  ⏭  Skipping non-7B: $dataset_name${NC}"
                continue
            fi

            benchmark_folder=$(get_benchmark_folder "$dataset_name")
            if [[ "$benchmark_folder" == "UNKNOWN" ]]; then
                echo -e "${YELLOW}⚠  Skipping unknown dataset: $dataset_name${NC}"
                continue
            fi

            model_name=$(get_scratch_model "$dataset_name")
            if [[ "$model_name" == "UNKNOWN" ]]; then
                echo -e "${YELLOW}⚠  Cannot detect model from: $dataset_name${NC}"
                continue
            fi

            process_dataset_dir "$dataset_dir" "$method" "$benchmark_folder" "$model_name" "scratch / $dataset_name"
        done
    done
else
    echo -e "${YELLOW}⚠  scratch/ directory not found${NC}"
fi

# ──────────────────────── SUMMARY ───────────────────────────────
echo ""
echo "=========================================="
echo "  Summary"
echo "=========================================="
echo -e "  ${GREEN}Copied  : $COPIED files${NC}"
echo -e "  ${YELLOW}Skipped : $SKIPPED files${NC}"
echo -e "  ${RED}Errors  : $ERRORS${NC}"
echo ""
$DRY_RUN && echo -e "${CYAN}  This was a dry run. Re-run without --dry-run to execute.${NC}"
echo ""