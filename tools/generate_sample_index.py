#!/usr/bin/env python3
"""
Generate a fixed sample index for the prompt-ablation study.

Scans every eligible category under ROOT, applies stratified sampling, and
writes the sampled image paths to a single JSON file.  Run this ONCE before
starting any ablation runs; every variant then reads the same index so the
compared image sets are byte-for-byte identical.

Usage
-----
  python generate_sample_index.py \
      --root /path/to/dataset \
      --output /path/to/dataset/PromptAblation/sample_index.json \
      --sample-ratio 0.1 \
      --sample-seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse walk_inputs / sample_jobs from the ablation script
sys.path.insert(0, str(Path(__file__).parent))
from detect_prompt_ablation import walk_inputs, sample_jobs

CATEGORIES = [
    "All Beauty",
    "Amazon Fashion",
    "Appliances",
    "Arts, Crafts & Sewing",
    "Automotive",
    "Baby Products",
    "Beauty & Personal Care",
    "Books",
    "CDs & Vinyl",
    "Cell Phones & Accessories",
    "Clothing, Shoes & Jewelry",
    "Electronics",
    "Grocery & Gourmet Food",
    "Handmade Products",
    "Health & Household",
    "Health & Personal Care",
    "Home & Kitchen",
    "Industrial & Scientific",
    "Magazine Subscriptions",
    "Musical Instruments",
    "Office Products",
    "Patio, Lawn & Garden",
    "Pet Supplies",
    "Sports & Outdoors",
    "Tools & Home Improvement",
    "Toys & Games",
    "Video Games",
]


def build_index(root: Path, ratio: float, seed: int) -> dict:
    categories: dict[str, list[dict]] = {}

    for cat in CATEGORIES:
        cat_dir = root / cat
        inputs = []
        if (cat_dir / "Negative").is_dir():
            inputs.append(("Negative", cat_dir / "Negative"))
        if (cat_dir / "DeepFake").is_dir():
            inputs.append(("DeepFake", cat_dir / "DeepFake"))

        if not inputs:
            print(f"  SKIP {cat!r} (no Negative/ or DeepFake/)")
            continue

        all_jobs = walk_inputs(inputs, review_mode=False)
        sampled  = sample_jobs(all_jobs, ratio=ratio, seed=seed)

        entries = [
            {
                "image_path": str(j.image_path),
                "label":      j.label,
                "bucket":     j.bucket,
                "generator":  j.generator,
                "image_rel":  j.image_rel,
            }
            for j in sampled
        ]
        categories[cat] = entries
        print(f"  {cat}: {len(all_jobs)} total → {len(sampled)} sampled")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root":         str(root),
        "sample_ratio": ratio,
        "sample_seed":  seed,
        "categories":   categories,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Generate ablation sample index.")
    p.add_argument("--root",         default=str(Path(__file__).resolve().parent.parent.parent.parent),
                   help="Dataset root directory.")
    p.add_argument("--output",       required=True,
                   help="Output JSON path.")
    p.add_argument("--sample-ratio", type=float, default=0.3)
    p.add_argument("--sample-seed",  type=int,   default=42)
    args = p.parse_args()

    root   = Path(args.root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Root        : {root}")
    print(f"Sample ratio: {args.sample_ratio}  seed: {args.sample_seed}")
    print(f"Output      : {output}")
    print()

    index = build_index(root, args.sample_ratio, args.sample_seed)

    with output.open("w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    total = sum(len(v) for v in index["categories"].values())
    print(f"\nDone. {len(index["categories"])} categories, {total} images → {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
