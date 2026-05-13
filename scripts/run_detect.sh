#!/usr/bin/env bash
# run_detect.sh — Unified runner for all six experiment modes.
#
# Combines run_SingleImage_NoReview.sh, run_SingleImage_withReview.sh,
# run_MultiImage_NoReview.sh, run_MultiImage_SingleTurn_NoReview.sh,
# run_MultiImage_withReview.sh, and run_MultiImage_SingleTurn_withReview.sh
# into a single script.  Flags are forwarded directly to detect.py.
#
# Usage
# -----
#   bash run_detect.sh                                       # SingleImage-NoReview
#   bash run_detect.sh --with-review                        # SingleImage-withReview
#   bash run_detect.sh --review-mode                        # MultiStep-NoReview
#   bash run_detect.sh --review-mode --single-turn          # MultiImage-NoReview
#   bash run_detect.sh --review-mode --with-review          # MultiStep-withReview
#   bash run_detect.sh --review-mode --single-turn --with-review  # MultiImage-withReview
#
# Optional overrides
#   --concurrency N   API calls per model (default 4)
#   --python PATH     Python interpreter (default python3)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$SCRIPT_DIR")"
TOOLS_DIR="$(dirname "$EVAL_DIR")"
ROOT="$(dirname "$TOOLS_DIR")"
DETECT="$EVAL_DIR/tools/detect.py"
PYTHON="${PYTHON:-python3}"
CONCURRENCY=4

# ── Flag parsing ──────────────────────────────────────────────────────────────
REVIEW_MODE=false
SINGLE_TURN=false
WITH_REVIEW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --review-mode)  REVIEW_MODE=true; shift ;;
        --single-turn)  SINGLE_TURN=true; shift ;;
        --with-review)  WITH_REVIEW=true; shift ;;
        --concurrency)  CONCURRENCY="$2"; shift 2 ;;
        --python)       PYTHON="$2"; shift 2 ;;
        *) echo "[error] unknown arg: $1"; exit 1 ;;
    esac
done

# ── Derive MODE name and extra flags ─────────────────────────────────────────
if $REVIEW_MODE; then
    MODE_BASE="MultiStep"
    EXTRA_FLAGS="--review-mode"
    if $SINGLE_TURN; then
        MODE_BASE="MultiImage"
        EXTRA_FLAGS="--review-mode --single-turn"
    fi
else
    MODE_BASE="SingleImage"
    EXTRA_FLAGS=""
fi

if $WITH_REVIEW; then
    MODE="${MODE_BASE}-withReview"
    EXTRA_FLAGS="$EXTRA_FLAGS --with-review"
else
    MODE="${MODE_BASE}-NoReview"
fi

# ── Categories ────────────────────────────────────────────────────────────────
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
    "Delivery, Pickup & Dine-Out"
    "Electronics"
    "Grocery & Gourmet Food"
    "Handmade Products"
    "Health & Household"
    "Health & Personal Care"
    "Home & Kitchen"
    "Hotels & Accommodations"
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

# ── Run ───────────────────────────────────────────────────────────────────────
total=${#CATEGORIES[@]}
echo "============================================================"
echo "  Mode        : $MODE"
echo "  Extra flags : ${EXTRA_FLAGS:-<none>}"
echo "  Concurrency : $CONCURRENCY"
echo "  Categories  : $total"
echo "  Script      : $DETECT"
echo "============================================================"
echo

i=0
for cat in "${CATEGORIES[@]}"; do
    i=$((i+1))
    cat_dir="$ROOT/$cat"
    out_dir="$cat_dir/Results/$MODE"

    if [[ ! -d "$cat_dir/DeepFake" || ! -d "$cat_dir/Negative" ]]; then
        echo "[$(date '+%F %T')] [$MODE] [$i/$total] $cat SKIP (missing DeepFake/ or Negative/)"
        echo
        continue
    fi

    echo "------------------------------------------------------------"
    echo "[$(date '+%F %T')] [$MODE] [$i/$total] $cat"
    echo "  out : $out_dir"
    echo "------------------------------------------------------------"

    # shellcheck disable=SC2086
    "$PYTHON" "$DETECT" \
        --input "Negative=$cat_dir/Negative" \
        --input "DeepFake=$cat_dir/DeepFake" \
        --output "$out_dir" \
        --concurrency "$CONCURRENCY" \
        $EXTRA_FLAGS \
        || echo "[$(date '+%F %T')] [$MODE] [$i/$total] $cat FAILED, continuing"

    echo
done

echo "[$(date '+%F %T')] [$MODE] all $total categories done"
