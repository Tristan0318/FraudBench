#!/usr/bin/env python3
"""
compute_ablation_accuracy.py — Prompt-ablation accuracy tables.

Reads PromptAblation/{variant}/{category}/summary.json for 5 prompt variants
and computes macro-averaged metrics across 27 Amazon categories.

Output: ablation_results.xlsx  (5 sheets, one per variant)
Same metric columns as compute_accuracy.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_TOOLS_DIR   = Path(__file__).resolve().parent
_DATASET     = _TOOLS_DIR.parent.parent.parent   # tools/ → Evaluation/ → Tools/ → dataset root
_ABLATION    = _DATASET / "PromptAblation"

# ── Constants ─────────────────────────────────────────────────────────────────
GENERATORS = [
    "gpt-image-2",
    "grok-imagine-image",
    "nano-banana-2",
    "qwen-image-2.0-pro",
    "qwen-image-edit-max",
    "wan2.7-image-pro",
]

MLLMS = [
    "gemini-3-flash",
    "gpt-5.4-mini",
    "grok-4-1-fast-reasoning",
    "grok-4.20-reasoning-latest",
    "kimi-k2.6",
    "qvq-max-latest",
    "qwen3-vl-flash",
    "qwen3-vl-plus",
    "qwen3.5-omni-plus",
    "qwen3.6-flash",
    "qwen3.6-plus",
]

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

VARIANTS = [
    ("v1_baseline",    "V1 Baseline (forensic expert + artifact checklist)"),
    ("v2_merged_role", "V2 Merged Role (no system/user split)"),
    ("v3_no_artifacts","V3 No Artifacts (role only, no checklist)"),
    ("v4_generic_role","V4 Generic Role (no forensic persona)"),
    ("v5_minimal",     "V5 Minimal (no system, one-sentence user)"),
]

BUCKETS = (
    ["Negative TNR", "Negative Precision", "Negative F1", "Negative Mean Conf"]
    + [col for g in GENERATORS for col in [f"{g} TPR", f"{g} Mean Conf"]]
    + ["Overall F1", "Overall Bal.Acc", "Overall Acc", "Overall Macro-F1"]
)


# ── Metric helpers ────────────────────────────────────────────────────────────
def _f1(tp: int, fp: int, fn: int) -> float | None:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else None


def _precision(tp: int, fp: int) -> float | None:
    d = tp + fp
    return (tp / d) if d > 0 else None


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


def _cell(correct: int, total: int) -> str:
    return f"{correct / total:.3f}" if total > 0 else "—"


def _avg(rates: list[float]) -> str:
    return f"{sum(rates) / len(rates):.3f}" if rates else "—"


# ── Data loading ──────────────────────────────────────────────────────────────
def _load_category(variant: str, category: str) -> list[dict]:
    """Return rows list from PromptAblation/{variant}/{category}/summary.json."""
    path = _ABLATION / variant / category / "summary.json"
    if not path.exists():
        return []
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d.get("rows", [])
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] {path}: {e}", file=sys.stderr)
        return []


def _verdict_counts(rows: list[dict], model: str, is_fake: bool) -> tuple[int, int, float | None]:
    """Returns (correct, total, mean_conf_on_correct) for one model in a row subset."""
    total = correct = 0
    confs: list[float] = []
    for row in rows:
        vd = (row.get("verdicts") or {}).get(model) or {}
        if vd.get("status") != "ok" or vd.get("error") is not None:
            continue
        total += 1
        predicted_fake = bool(vd.get("is_ai_modified"))
        if predicted_fake == is_fake:
            correct += 1
            try:
                confs.append(float(vd["confidence"]))
            except (TypeError, KeyError, ValueError):
                pass
    mean_conf = (sum(confs) / len(confs)) if confs else None
    return correct, total, mean_conf


# ── Table builder ─────────────────────────────────────────────────────────────
def build_variant_table(variant: str) -> pd.DataFrame:
    """Macro-averaged metrics for one ablation variant across all categories."""
    rows_out: list[dict] = []

    for model in MLLMS:
        row: dict = {"Method": model}

        neg_tnr_vals  = []
        neg_prec_vals = []
        neg_f1_vals   = []
        neg_conf_vals = []
        gen_tpr_vals  = {g: [] for g in GENERATORS}
        gen_conf_vals = {g: [] for g in GENERATORS}
        ov_f1_vals    = []
        ov_bal_vals   = []
        ov_acc_vals   = []
        ov_mf1_vals   = []

        for cat in AMAZON_DIRS:
            all_rows = _load_category(variant, cat)
            neg_rows  = [r for r in all_rows if r.get("label") == "Negative"]
            fake_rows = [r for r in all_rows if r.get("label") == "DeepFake"]

            tn, neg_tot, neg_conf = _verdict_counts(neg_rows, model, is_fake=False)
            fp = neg_tot - tn

            gen_counts: dict[str, tuple[int, int, float | None]] = {}
            for g in GENERATORS:
                g_rows = [r for r in fake_rows if r.get("generator") == g]
                gen_counts[g] = _verdict_counts(g_rows, model, is_fake=True)

            all_tp   = sum(tp       for tp, _,   _ in gen_counts.values())
            all_fn   = sum(tot - tp for tp, tot, _ in gen_counts.values())
            fake_tot = sum(tot      for _,  tot, _ in gen_counts.values())
            total    = neg_tot + fake_tot

            if neg_tot > 0:
                neg_tnr_vals.append(tn / neg_tot)
            v = _precision(tn, all_fn)
            if v is not None:
                neg_prec_vals.append(v)
            neg_f1 = _f1(tn, all_fn, fp)
            if neg_f1 is not None:
                neg_f1_vals.append(neg_f1)
            if neg_conf is not None:
                neg_conf_vals.append(neg_conf)

            for g in GENERATORS:
                tp, tot, mean_conf = gen_counts[g]
                if tot > 0:
                    gen_tpr_vals[g].append(tp / tot)
                if mean_conf is not None:
                    gen_conf_vals[g].append(mean_conf)

            fake_f1 = _f1(all_tp, fp, all_fn)
            if fake_f1 is not None:
                ov_f1_vals.append(fake_f1)
            ov_tpr = (all_tp / fake_tot) if fake_tot > 0 else None
            ov_tnr = (tn / neg_tot)      if neg_tot  > 0 else None
            if ov_tpr is not None and ov_tnr is not None:
                ov_bal_vals.append((ov_tpr + ov_tnr) / 2)
            if total > 0:
                ov_acc_vals.append((tn + all_tp) / total)
            if neg_f1 is not None and fake_f1 is not None:
                ov_mf1_vals.append((neg_f1 + fake_f1) / 2)

        row["Negative TNR"]       = _avg(neg_tnr_vals)
        row["Negative Precision"] = _avg(neg_prec_vals)
        row["Negative F1"]        = _avg(neg_f1_vals)
        row["Negative Mean Conf"] = _avg(neg_conf_vals)
        for g in GENERATORS:
            row[f"{g} TPR"]       = _avg(gen_tpr_vals[g])
            row[f"{g} Mean Conf"] = _avg(gen_conf_vals[g])
        row["Overall F1"]       = _avg(ov_f1_vals)
        row["Overall Bal.Acc"]  = _avg(ov_bal_vals)
        row["Overall Acc"]      = _avg(ov_acc_vals)
        row["Overall Macro-F1"] = _avg(ov_mf1_vals)
        rows_out.append(row)

    return pd.DataFrame(rows_out).set_index("Method")


# ── Excel export ──────────────────────────────────────────────────────────────
def save_excel(tables: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[error] openpyxl required: pip install openpyxl", file=sys.stderr)
        sys.exit(1)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name, table in tables:
            sn = sheet_name[:31]
            table[BUCKETS].to_excel(writer, sheet_name=sn)
            ws = writer.sheets[sn]
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(c.value)) if c.value else 0) for c in col_cells
                )
                ws.column_dimensions[col_cells[0].column_letter].width = max_len + 2

    print(f"Saved {len(tables)} sheets → {path}", file=sys.stderr)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    out_path = _TOOLS_DIR / "ablation_results.xlsx"
    all_tables: list[tuple[str, pd.DataFrame]] = []

    for idx, (variant, description) in enumerate(VARIANTS, 1):
        print(f"[{idx}/5] {variant} …", file=sys.stderr)
        table = build_variant_table(variant)
        sheet = f"T{idx}_{variant}"
        all_tables.append((sheet, table))

        print(f"\n{'='*100}")
        print(f"  {description}")
        print(f"  Macro-avg over {len(AMAZON_DIRS)} Amazon categories")
        print(f"{'='*100}")
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 320)
        print(table[BUCKETS].to_string())

    save_excel(all_tables, out_path)


if __name__ == "__main__":
    main()
