#!/usr/bin/env python3
"""
generate_mismatch_index.py — Generate mismatched-review index.

Reuses the same sampled images from PromptAblation/sample_index.json,
but pairs EVERY image with a review text from a DIFFERENT category.

Experiment design
-----------------
  Condition A (Matched)    — fake image + its own product review    [existing data]
  Condition B (Mismatched) — fake image + review from wrong category [this script]

Comparing A vs B isolates whether models use text-image consistency as a
detection signal, or rely purely on visual artifacts.

Review text source: Negative/Review_XXX/MetaReview_XXX.json (title + text)
from any category other than the image's own category.  This format is
clean, always available, and unambiguously off-topic for the target image.

Output
------
  MismatchReview/mismatch_index.json

  Each entry adds two fields to the standard index format:
    "review_text"             — the injected mismatched review string
    "review_source_category"  — which category it came from
    "review_source_review_id" — which review folder it came from

Usage
-----
  python generate_mismatch_index.py
  python generate_mismatch_index.py --root /path/to/ANON-NIPS26 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_DATASET = _HERE.parent.parent.parent   # tools/ → Evaluation/ → Tools/ → dataset root

AMAZON_DIRS = [
    "All Beauty", "Amazon Fashion", "Appliances", "Arts, Crafts & Sewing",
    "Automotive", "Baby Products", "Beauty & Personal Care", "Books",
    "CDs & Vinyl", "Cell Phones & Accessories", "Clothing, Shoes & Jewelry",
    "Electronics", "Grocery & Gourmet Food", "Handmade Products",
    "Health & Household", "Health & Personal Care", "Home & Kitchen",
    "Industrial & Scientific", "Magazine Subscriptions", "Musical Instruments",
    "Office Products", "Patio, Lawn & Garden", "Pet Supplies",
    "Sports & Outdoors", "Tools & Home Improvement", "Toys & Games",
    "Video Games",
]

SOURCE_INDEX = _DATASET / "PromptAblation" / "sample_index.json"
OUTPUT_DIR   = _DATASET / "MismatchReview"


# ── Review pool builder ───────────────────────────────────────────────────────
def _load_review_texts(root: Path, category: str) -> list[tuple[str, str]]:
    """Return list of (review_id, text) from all MetaReview files in category."""
    neg_dir = root / category / "Negative"
    if not neg_dir.is_dir():
        return []
    results: list[tuple[str, str]] = []
    for review_dir in sorted(neg_dir.iterdir()):
        if not review_dir.is_dir():
            continue
        num = review_dir.name.split("_")[-1]
        meta = review_dir / f"MetaReview_{num}.json"
        if not meta.exists():
            continue
        try:
            data  = json.loads(meta.read_text(encoding="utf-8"))
            title = (data.get("title") or "").strip()
            body  = (data.get("text") or data.get("content") or "").strip()
            parts = [p for p in (title, body) if p]
            text  = ". ".join(parts)
            if text:
                results.append((review_dir.name, text))
        except (json.JSONDecodeError, OSError):
            continue
    return results


def build_review_pool(root: Path) -> dict[str, list[tuple[str, str]]]:
    """Return {category: [(review_id, text), ...]} for all categories."""
    pool: dict[str, list[tuple[str, str]]] = {}
    for cat in AMAZON_DIRS:
        reviews = _load_review_texts(root, cat)
        if reviews:
            pool[cat] = reviews
    return pool


# ── Index builder ────────────────────────────────────────────────────────────
def build_mismatch_index(root: Path, seed: int) -> dict:
    source_path = root / "PromptAblation" / "sample_index.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Source index not found: {source_path}")

    source = json.loads(source_path.read_text(encoding="utf-8"))
    rng    = random.Random(seed)

    print(f"Loading review pool …")
    pool = build_review_pool(root)
    print(f"  {sum(len(v) for v in pool.values())} reviews across {len(pool)} categories\n")

    out_categories: dict[str, list[dict]] = {}
    total_images = 0

    for cat in AMAZON_DIRS:
        entries = source.get("categories", {}).get(cat)
        if not entries:
            print(f"  SKIP {cat!r} (not in source index)")
            continue

        # Build review pool from all OTHER categories
        other_reviews: list[tuple[str, str, str]] = []  # (source_cat, review_id, text)
        for other_cat, reviews in pool.items():
            if other_cat == cat:
                continue
            for rev_id, text in reviews:
                other_reviews.append((other_cat, rev_id, text))

        if not other_reviews:
            print(f"  WARN {cat!r}: no cross-category reviews available, skipping")
            continue

        out_entries: list[dict] = []
        for entry in entries:
            src_cat, rev_id, text = rng.choice(other_reviews)
            out_entries.append({
                **entry,
                "review_text":            text,
                "review_source_category": src_cat,
                "review_source_review_id": rev_id,
            })
        out_categories[cat] = out_entries
        total_images += len(out_entries)
        print(f"  {cat}: {len(out_entries)} images, reviews from "
              f"{len(set(e['review_source_category'] for e in out_entries))} other categories")

    return {
        "generated_at":    datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root":            str(root),
        "experiment":      "mismatch_review",
        "description":     "Fake/real images paired with reviews from a DIFFERENT category",
        "sample_ratio":    source.get("sample_ratio"),
        "sample_seed":     source.get("sample_seed"),
        "mismatch_seed":   seed,
        "source_index": str(source_path),
        "total_images":    total_images,
        "categories":      out_categories,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root",   default=str(_DATASET))
    p.add_argument("--output", default=str(OUTPUT_DIR / "mismatch_index.json"))
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    root   = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Root   : {root}")
    print(f"Output : {output}")
    print(f"Seed   : {args.seed}")
    print()

    index = build_mismatch_index(root, args.seed)

    output.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone. {index["total_images"]} images → {output}")


if __name__ == "__main__":
    main()
