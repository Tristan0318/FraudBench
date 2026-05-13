#!/usr/bin/env bash
# run_ablation.sh — Unified ablation runner.
#
# Covers both ablation sub-experiments and auto-generates any missing indices:
#
#   Prompt Ablation  — 5 prompt variants × 27 categories (single-image, no review)
#                      Index: PromptAblation/sample_index.json
#                      Output:   PromptAblation/{variant}/{category}/summary.json
#
#   Mismatch Review  — V1 Base prompt + cross-category review × 27 categories
#                      Index: MismatchReview/mismatch_index.json
#                      Output:   MismatchReview/{category}/summary.json
#
# Index generation is automatic: if the required JSON does not exist it is
# created before the detection loop starts.  Pass --regen-* flags to force
# regeneration even when an index already exists.
#
# Usage
# -----
#   bash run_ablation.sh                   # run both sub-experiments
#   bash run_ablation.sh --prompt          # prompt ablation only
#   bash run_ablation.sh --mismatch        # mismatch review only
#   bash run_ablation.sh --prompt --mismatch --concurrency 6
#   bash run_ablation.sh --regen-sample    # regenerate sample index, then run both
#   bash run_ablation.sh --regen-mismatch  # regenerate mismatch index, then run both
#
# Optional overrides
#   --concurrency N     API calls per category process (default 4)
#   --sample-ratio R    Sampling ratio for sample index (default 0.1)
#   --sample-seed  S    RNG seed for sample index (default 42)
#   --mismatch-seed S   RNG seed for mismatch index (default 42)
#   --python PATH       Python interpreter (default python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$(dirname "$EVAL_DIR")"
ROOT="$(dirname "$TOOLS_DIR")"
PYTHON="${PYTHON:-python3}"

# ── Output roots ──────────────────────────────────────────────────────────────
ABLATION_ROOT="$ROOT/PromptAblation"
MISMATCH_ROOT="$ROOT/MismatchReview"

# ── Manifest paths ────────────────────────────────────────────────────────────
SAMPLE_INDEX="$ABLATION_ROOT/sample_index.json"
MISMATCH_INDEX="$MISMATCH_ROOT/mismatch_index.json"

# ── Detection scripts ─────────────────────────────────────────────────────────
ABLATION_DETECT="$EVAL_DIR/tools/detect_prompt_ablation.py"
MISMATCH_DETECT="$EVAL_DIR/tools/detect_mismatch.py"
GEN_SAMPLE_INDEX="$EVAL_DIR/tools/generate_sample_index.py"
GEN_MISMATCH_INDEX="$EVAL_DIR/tools/generate_mismatch_index.py"

# ── Defaults ──────────────────────────────────────────────────────────────────
RUN_PROMPT=false
RUN_MISMATCH=false
REGEN_SAMPLE_INDEX=false
REGEN_MISMATCH_INDEX=false
CONCURRENCY=4
SAMPLE_RATIO=0.1
SAMPLE_SEED=42
MISMATCH_SEED=42

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt)          RUN_PROMPT=true;    shift ;;
        --mismatch)        RUN_MISMATCH=true;  shift ;;
        --regen-sample)    REGEN_SAMPLE_INDEX=true;  shift ;;
        --regen-mismatch)  REGEN_MISMATCH_INDEX=true; shift ;;
        --concurrency)     CONCURRENCY="$2";   shift 2 ;;
        --sample-ratio)    SAMPLE_RATIO="$2";  shift 2 ;;
        --sample-seed)     SAMPLE_SEED="$2";   shift 2 ;;
        --mismatch-seed)   MISMATCH_SEED="$2"; shift 2 ;;
        --python)          PYTHON="$2";        shift 2 ;;
        *) echo "[error] unknown arg: $1"; exit 1 ;;
    esac
done

# Default: run both if neither flag is set
if ! $RUN_PROMPT && ! $RUN_MISMATCH; then
    RUN_PROMPT=true
    RUN_MISMATCH=true
fi

# ── Prompt variants ───────────────────────────────────────────────────────────
VARIANTS=(
    "v1_baseline"
    "v2_merged_role"
    "v3_no_artifacts"
    "v4_generic_role"
    "v5_minimal"
)

# ── Categories (27 Amazon categories; excludes Delivery and Hotels) ───────────
CATEGORIES=(
    "All Beauty"
    "Amazon Fashion"
    "Appliances"
    "Arts, Crafts & Sewing"
    "Automotive"
    "Baby Products"
    "Beauty & Personal Care"
    "Books"
    "CDs & Vinyl"
    "Cell Phones & Accessories"
    "Clothing, Shoes & Jewelry"
    "Electronics"
    "Grocery & Gourmet Food"
    "Handmade Products"
    "Health & Household"
    "Health & Personal Care"
    "Home & Kitchen"
    "Industrial & Scientific"
    "Magazine Subscriptions"
    "Musical Instruments"
    "Office Products"
    "Patio, Lawn & Garden"
    "Pet Supplies"
    "Sports & Outdoors"
    "Tools & Home Improvement"
    "Toys & Games"
    "Video Games"
)

# ── Header ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "  Ablation Runner"
echo "  Prompt ablation : $RUN_PROMPT"
echo "  Mismatch review : $RUN_MISMATCH"
echo "  Concurrency     : $CONCURRENCY"
echo "  Sample ratio    : $SAMPLE_RATIO  (seed $SAMPLE_SEED)"
echo "  Mismatch seed   : $MISMATCH_SEED"
echo "  Root            : $ROOT"
echo "================================================================"
echo

# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Generate sample index (shared by both sub-experiments)
# ════════════════════════════════════════════════════════════════════════════
mkdir -p "$ABLATION_ROOT" "$MISMATCH_ROOT"

if [[ -f "$SAMPLE_INDEX" && "$REGEN_SAMPLE_INDEX" == false ]]; then
    echo "[index] sample index exists — reusing"
    echo "  $SAMPLE_INDEX"
else
    echo "[index] generating sample index ..."
    "$PYTHON" "$GEN_SAMPLE_INDEX" \
        --root         "$ROOT" \
        --output       "$SAMPLE_INDEX" \
        --sample-ratio "$SAMPLE_RATIO" \
        --sample-seed  "$SAMPLE_SEED" \
        || { echo "[ERROR] sample index generation failed — aborting"; exit 1; }
    echo "[index] saved: $SAMPLE_INDEX"
fi
echo

# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Generate mismatch index (only if --mismatch or --all)
# ════════════════════════════════════════════════════════════════════════════
if $RUN_MISMATCH; then
    if [[ -f "$MISMATCH_INDEX" && "$REGEN_MISMATCH_INDEX" == false ]]; then
        echo "[index] mismatch index exists — reusing"
        echo "  $MISMATCH_INDEX"
    else
        echo "[index] generating mismatch index ..."
        "$PYTHON" "$GEN_MISMATCH_INDEX" \
            --root   "$ROOT" \
            --output "$MISMATCH_INDEX" \
            --seed   "$MISMATCH_SEED" \
            || { echo "[ERROR] mismatch index generation failed — aborting"; exit 1; }
        echo "[index] saved: $MISMATCH_INDEX"
    fi
    echo
fi

# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Prompt Ablation (5 variants × 27 categories)
# ════════════════════════════════════════════════════════════════════════════
if $RUN_PROMPT; then
    n_variants=${#VARIANTS[@]}
    n_cats=${#CATEGORIES[@]}
    total_runs=$(( n_variants * n_cats ))

    echo "════════════════════════════════════════════════════════════"
    echo "  PROMPT ABLATION"
    echo "  $n_variants variants × $n_cats categories = $total_runs runs"
    echo "════════════════════════════════════════════════════════════"
    echo

    run_idx=0
    for variant in "${VARIANTS[@]}"; do
        echo "── Variant: $variant ────────────────────────────────────────"
        echo

        cat_idx=0
        for cat in "${CATEGORIES[@]}"; do
            cat_idx=$(( cat_idx + 1 ))
            run_idx=$(( run_idx + 1 ))
            cat_dir="$ROOT/$cat"
            out_dir="$ABLATION_ROOT/$variant/$cat"

            if [[ ! -d "$cat_dir/DeepFake" && ! -d "$cat_dir/Negative" ]]; then
                echo "[$(date '+%F %T')] [prompt/$variant] [$cat_idx/$n_cats] $cat  SKIP (no data)"
                continue
            fi

            echo "[$(date '+%F %T')] [prompt/$variant] [$cat_idx/$n_cats] $cat  (run $run_idx/$total_runs)"

            "$PYTHON" "$ABLATION_DETECT" \
                --sample-index "$SAMPLE_INDEX" \
                --category        "$cat" \
                --output          "$out_dir" \
                --prompt-variant  "$variant" \
                --concurrency     "$CONCURRENCY" \
                || echo "[$(date '+%F %T')] [prompt/$variant] [$cat_idx/$n_cats] $cat  FAILED — continuing"
        done

        echo
        echo "[$(date '+%F %T')] [prompt/$variant] done"
        echo
    done

    echo "[$(date '+%F %T')] PROMPT ABLATION complete ($total_runs runs)"
    echo
fi

# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Mismatch Review (27 categories)
# ════════════════════════════════════════════════════════════════════════════
if $RUN_MISMATCH; then
    n_cats=${#CATEGORIES[@]}

    echo "════════════════════════════════════════════════════════════"
    echo "  MISMATCH REVIEW"
    echo "  $n_cats categories"
    echo "════════════════════════════════════════════════════════════"
    echo

    LOG_DIR="$MISMATCH_ROOT/logs"
    mkdir -p "$LOG_DIR"

    cat_idx=0
    for cat in "${CATEGORIES[@]}"; do
        cat_idx=$(( cat_idx + 1 ))
        summary="$MISMATCH_ROOT/$cat/summary.json"
        safe="${cat//[^a-zA-Z0-9]/_}"
        log_file="$LOG_DIR/${safe}.log"

        if [[ -f "$summary" ]]; then
            echo "[$(date '+%F %T')] [mismatch] [$cat_idx/$n_cats] $cat  SKIP (summary exists)"
            continue
        fi

        echo "[$(date '+%F %T')] [mismatch] [$cat_idx/$n_cats] $cat"

        "$PYTHON" "$MISMATCH_DETECT" \
            --category   "$cat" \
            --index "$MISMATCH_INDEX" \
            --output     "$MISMATCH_ROOT" \
            --concurrency "$CONCURRENCY" \
            > "$log_file" 2>&1 \
            && echo "[$(date '+%F %T')] [mismatch] [$cat_idx/$n_cats] $cat  done" \
            || echo "[$(date '+%F %T')] [mismatch] [$cat_idx/$n_cats] $cat  FAILED (see $log_file)"
    done

    echo
    echo "[$(date '+%F %T')] MISMATCH REVIEW complete"
    echo
fi

# ════════════════════════════════════════════════════════════════════════════
# Final summary
# ════════════════════════════════════════════════════════════════════════════
echo "================================================================"
if $RUN_PROMPT; then
    done_prompt=0
    for v in "${VARIANTS[@]}"; do
        for cat in "${CATEGORIES[@]}"; do
            [[ -f "$ABLATION_ROOT/$v/$cat/summary.json" ]] && done_prompt=$(( done_prompt + 1 ))
        done
    done
    total_prompt=$(( ${#VARIANTS[@]} * ${#CATEGORIES[@]} ))
    echo "  Prompt ablation : $done_prompt / $total_prompt categories done"
fi
if $RUN_MISMATCH; then
    done_mm=0
    for cat in "${CATEGORIES[@]}"; do
        [[ -f "$MISMATCH_ROOT/$cat/summary.json" ]] && done_mm=$(( done_mm + 1 ))
    done
    echo "  Mismatch review : $done_mm / ${#CATEGORIES[@]} categories done"
fi
echo "================================================================"
