#!/usr/bin/env python3
"""
compute_accuracy.py — TPR / Precision / F1 tables for the DeepFake detection benchmark.

Combines 11 MLLMs (4 modes) + 8 Traditional Detection methods (image-level).

Metrics (20 columns total)
--------------------------
  Negative TNR        — correctly classified real / total real
  Negative Precision  — TN / (TN + all_FN)  "when model says Real, how often correct"
  Negative F1         — F1 treating Real as the positive class
  Negative Mean Conf  — mean confidence on correctly-retained real images

  <gen> TPR           — correctly detected / total for that generator
  <gen> Mean Conf     — mean self-reported confidence on correctly-detected items
                        (only computed within that generator's bucket)

  Overall F1          — F1 treating all DeepFake as positive class
  Overall Bal.Acc     — (overall TPR + TNR) / 2
  Overall Acc         — (TN + ΣTP) / total samples (raw accuracy, class-imbalanced)
  Overall Macro-F1    — (Negative F1 + DeepFake F1) / 2

  FP = Negative images wrongly flagged as AI (shared across all generators)

T1-T4  : macro-avg (per-category rate averaged over all 29 categories)
T5-T33 : raw correct/total for each category
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # Tools/ for read_results
from read_results import load_dataframe

# ── Constants ─────────────────────────────────────────────────────────────────
GENERATORS = [
    "gpt-image-2",
    "grok-imagine-image",
    "nano-banana-2",
    "qwen-image-2.0-pro",
    "qwen-image-edit-max",
    "wan2.7-image-pro",
]

BUCKETS = (
    ["Negative TNR", "Negative Precision", "Negative F1", "Negative Mean Conf"]
    + [col for g in GENERATORS for col in [f"{g} TPR", f"{g} Mean Conf"]]
    + ["Overall F1", "Overall Bal.Acc", "Overall Acc", "Overall Macro-F1"]
)

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

MODE_LABELS = {
    "SingleImage-NoReview":  "Single Image w/o Review",
    "SingleImage-withReview": "Single Image w/ Review",
    "MultiStep-NoReview":    "Multi-Step w/o Review",
    "MultiStep-withReview":  "Multi-Step w/ Review",
    "MultiImage-NoReview":   "Multi-Image w/o Review",
    "MultiImage-withReview": "Multi-Image w/ Review",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "Data"

_TRAD_CONFIGS = [
    ("Result-COSPY",     "progan",             "COSPY-ProGAN",       "prediction", {"AI-Generated"}, "score"),
    ("Result-COSPY",     "sd-v1_4",            "COSPY-SDv1.4",       "prediction", {"AI-Generated"}, "score"),
    ("Result-Effort",    "chameleon",          "Effort-Chameleon",   "label",      {"AI"},           "fake_prob"),
    ("Result-Effort",    "sd-v1_4",            "Effort-SDv1.4",      "label",      {"AI"},           "fake_prob"),
    ("Result-ForgeLens", "GenImage",           "ForgeLens-GenImage", "label",      {"AI"},           "score"),
    ("Result-ForgeLens", "training_setting_1", "ForgeLens-Setting1", "label",      {"AI"},           "score"),
    ("Result-IAPL",      "ProGAN",             "IAPL-ProGAN",        "prediction", {"fake"},         "score"),
    ("Result-IAPL",      "SDv1.4",             "IAPL-SDv1.4",        "prediction", {"fake"},         "score"),
]


# ── Cell formatting ───────────────────────────────────────────────────────────
def _cell(correct: int, total: int) -> str:
    if total == 0:
        return "—"
    return f"{correct / total:.3f}"


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "—"
    return f"{numerator / denominator:.3f}"


def _avg(rates: list[float]) -> str:
    if not rates:
        return "—"
    return f"{sum(rates) / len(rates):.3f}"


# ── Metric helpers ────────────────────────────────────────────────────────────
def _f1(tp: int, fp: int, fn: int) -> float | None:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else None


def _precision(tp: int, fp: int) -> float | None:
    d = tp + fp
    return (tp / d) if d > 0 else None


def _fmt(v: float | None) -> str:
    return f"{v:.3f}" if v is not None else "—"


# ── MLLM helpers ──────────────────────────────────────────────────────────────
def _mllm_counts(df_slice: pd.DataFrame, model: str, label: str) -> tuple[int, int, float | None]:
    """Returns (correct, total, mean_confidence_on_correct)."""
    total = len(df_slice)
    if total == 0:
        return 0, 0, None
    ok   = df_slice[f"{model}_ok"].astype(bool)
    verd = df_slice[f"{model}_verdict"]
    correct_mask = ok & (verd == (label != "Negative"))
    correct = int(correct_mask.sum())
    if correct == 0:
        return correct, total, None
    conf = pd.to_numeric(df_slice.loc[correct_mask, f"{model}_confidence"], errors="coerce").dropna()
    mean_conf = float(conf.mean()) if len(conf) > 0 else None
    return correct, total, mean_conf


# ── MLLM row builders ─────────────────────────────────────────────────────────
def _mllm_row_from_counts(model: str,
                          tn: int, neg_tot: int, neg_mean_conf: float | None,
                          gen_counts: dict[str, tuple[int, int, float | None]]) -> dict:
    """Build one table row from pre-computed (tp, total, mean_conf) per generator."""
    fp     = neg_tot - tn
    all_tp = sum(tp        for tp, _, _ in gen_counts.values())
    all_fn = sum(tot - tp  for tp, tot, _ in gen_counts.values())

    fake_tot = sum(tot for _, tot, _ in gen_counts.values())
    total    = neg_tot + fake_tot
    row: dict = {"Method": model}

    # Negative: TNR | Precision | F1 | Mean Conf
    neg_f1 = _f1(tn, all_fn, fp)
    row["Negative TNR"]        = _cell(tn, neg_tot)
    row["Negative Precision"]  = _fmt(_precision(tn, all_fn))
    row["Negative F1"]         = _fmt(neg_f1)
    row["Negative Mean Conf"]  = _fmt(neg_mean_conf)

    # Per-generator: TPR | Mean Conf (on correct)
    for gen in GENERATORS:
        tp, tot, mean_conf = gen_counts[gen]
        row[f"{gen} TPR"]       = _cell(tp, tot)
        row[f"{gen} Mean Conf"] = _fmt(mean_conf)

    # Overall: F1 | Bal.Acc | Acc | Macro-F1
    fake_f1 = _f1(all_tp, fp, all_fn)
    ov_tpr  = (all_tp / fake_tot) if fake_tot > 0 else None
    ov_tnr  = (tn / neg_tot)      if neg_tot  > 0 else None
    bal_acc = ((ov_tpr + ov_tnr) / 2) if (ov_tpr is not None and ov_tnr is not None) else None
    macro_f1 = ((neg_f1 + fake_f1) / 2) if (neg_f1 is not None and fake_f1 is not None) else None
    row["Overall F1"]       = _fmt(fake_f1)
    row["Overall Bal.Acc"]  = _fmt(bal_acc)
    row["Overall Acc"]      = _cell(tn + all_tp, total)
    row["Overall Macro-F1"] = _fmt(macro_f1)
    return row


def _is_multi_mode(mode: str) -> bool:
    return mode.startswith("Multi")


def build_mllm_rows_raw(df: pd.DataFrame, mode: str, category: str) -> list[dict]:
    sub     = df[(df["mode"] == mode) & (df["category"] == category)]
    if _is_multi_mode(mode):
        sub = sub[sub["num_images"] >= 2]
    neg_df  = sub[sub["label"] == "Negative"]
    fake_df = sub[sub["label"] == "DeepFake"]
    rows = []
    for model in MLLMS:
        tn, neg_tot, neg_conf = _mllm_counts(neg_df, model, "Negative")
        gen_counts = {g: _mllm_counts(fake_df[fake_df["generator"] == g], model, "DeepFake")
                      for g in GENERATORS}
        rows.append(_mllm_row_from_counts(model, tn, neg_tot, neg_conf, gen_counts))
    return rows


def build_mllm_rows_avg(df: pd.DataFrame, mode: str, categories: list[str]) -> list[dict]:
    sub = df[df["mode"] == mode]
    if _is_multi_mode(mode):
        sub = sub[sub["num_images"] >= 2]
    rows = []
    for model in MLLMS:
        row: dict = {"Method": model}

        # Collect per-category values for averaging
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

        for cat in categories:
            cat_neg  = sub[(sub["category"] == cat) & (sub["label"] == "Negative")]
            cat_fake = sub[(sub["category"] == cat) & (sub["label"] == "DeepFake")]
            tn, neg_tot, neg_conf = _mllm_counts(cat_neg, model, "Negative")
            fp = neg_tot - tn

            gen_counts = {g: _mllm_counts(cat_fake[cat_fake["generator"] == g], model, "DeepFake")
                          for g in GENERATORS}
            all_tp   = sum(tp       for tp, _,   _ in gen_counts.values())
            all_fn   = sum(tot - tp for tp, tot, _ in gen_counts.values())
            fake_tot = sum(tot      for _,  tot, _ in gen_counts.values())
            total    = neg_tot + fake_tot

            if neg_tot > 0: neg_tnr_vals.append(tn / neg_tot)
            v = _precision(tn, all_fn)
            if v is not None: neg_prec_vals.append(v)
            neg_f1 = _f1(tn, all_fn, fp)
            if neg_f1 is not None: neg_f1_vals.append(neg_f1)
            if neg_conf is not None: neg_conf_vals.append(neg_conf)

            for g in GENERATORS:
                tp, tot, mean_conf = gen_counts[g]
                if tot > 0: gen_tpr_vals[g].append(tp / tot)
                if mean_conf is not None: gen_conf_vals[g].append(mean_conf)

            fake_f1 = _f1(all_tp, fp, all_fn)
            if fake_f1 is not None: ov_f1_vals.append(fake_f1)
            ov_tpr = (all_tp / fake_tot) if fake_tot > 0 else None
            ov_tnr = (tn / neg_tot)      if neg_tot  > 0 else None
            if ov_tpr is not None and ov_tnr is not None:
                ov_bal_vals.append((ov_tpr + ov_tnr) / 2)
            if total > 0: ov_acc_vals.append((tn + all_tp) / total)
            if neg_f1 is not None and fake_f1 is not None:
                ov_mf1_vals.append((neg_f1 + fake_f1) / 2)

        row["Negative TNR"]        = _avg(neg_tnr_vals)
        row["Negative Precision"]  = _avg(neg_prec_vals)
        row["Negative F1"]         = _avg(neg_f1_vals)
        row["Negative Mean Conf"]  = _avg(neg_conf_vals)
        for g in GENERATORS:
            row[f"{g} TPR"]       = _avg(gen_tpr_vals[g])
            row[f"{g} Mean Conf"] = _avg(gen_conf_vals[g])
        row["Overall F1"]       = _avg(ov_f1_vals)
        row["Overall Bal.Acc"]  = _avg(ov_bal_vals)
        row["Overall Acc"]      = _avg(ov_acc_vals)
        row["Overall Macro-F1"] = _avg(ov_mf1_vals)
        rows.append(row)
    return rows


# ── Traditional detection helpers ─────────────────────────────────────────────
def _trad_counts(folder: str, backbone: str, pred_field: str,
                 fake_values: set[str], generator: str | None,
                 category: str, score_field: str = "score") -> tuple[int, int, float | None]:
    """Returns (correct, total, mean_score_on_correct)."""
    root = DATA_DIR / folder
    json_path = (
        root / category / "Negative" / f"Negative_{backbone}.json"
        if generator is None
        else root / category / "Deepfake" / f"{generator}_{backbone}.json"
    )
    if not json_path.exists():
        return 0, 0, None
    try:
        items = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return 0, 0, None
    correct = total = 0
    correct_confs: list[float] = []
    is_neg = generator is None
    for item in items:
        pred = str(item.get(pred_field, "")).strip()
        total += 1
        is_correct = (is_neg and pred not in fake_values) or \
                     (not is_neg and pred in fake_values)
        if is_correct:
            correct += 1
            try:
                s = float(item.get(score_field))
                # score = AI probability; for Negative bucket, use 1-s as "confidence in Real"
                correct_confs.append(1.0 - s if is_neg else s)
            except (TypeError, ValueError):
                pass
    mean_conf = (sum(correct_confs) / len(correct_confs)) if correct_confs else None
    return correct, total, mean_conf


def _trad_row_from_counts(display: str,
                          tn: int, neg_tot: int, neg_mean_conf: float | None,
                          gen_counts: dict[str, tuple[int, int, float | None]]) -> dict:
    fp     = neg_tot - tn
    all_tp = sum(tp        for tp, _,   _ in gen_counts.values())
    all_fn = sum(tot - tp  for tp, tot, _ in gen_counts.values())

    fake_tot = sum(tot for _, tot, _ in gen_counts.values())
    total    = neg_tot + fake_tot
    row: dict = {"Method": display}

    neg_f1 = _f1(tn, all_fn, fp)
    row["Negative TNR"]        = _cell(tn, neg_tot)
    row["Negative Precision"]  = _fmt(_precision(tn, all_fn))
    row["Negative F1"]         = _fmt(neg_f1)
    row["Negative Mean Conf"]  = _fmt(neg_mean_conf)

    for gen in GENERATORS:
        tp, tot, mean_conf = gen_counts[gen]
        row[f"{gen} TPR"]       = _cell(tp, tot)
        row[f"{gen} Mean Conf"] = _fmt(mean_conf)

    fake_f1 = _f1(all_tp, fp, all_fn)
    ov_tpr  = (all_tp / fake_tot) if fake_tot > 0 else None
    ov_tnr  = (tn / neg_tot)      if neg_tot  > 0 else None
    bal_acc = ((ov_tpr + ov_tnr) / 2) if (ov_tpr is not None and ov_tnr is not None) else None
    macro_f1 = ((neg_f1 + fake_f1) / 2) if (neg_f1 is not None and fake_f1 is not None) else None
    row["Overall F1"]       = _fmt(fake_f1)
    row["Overall Bal.Acc"]  = _fmt(bal_acc)
    row["Overall Acc"]      = _cell(tn + all_tp, total)
    row["Overall Macro-F1"] = _fmt(macro_f1)
    return row


def build_trad_rows_raw(category: str) -> list[dict]:
    rows = []
    for folder, backbone, display, pred_field, fake_values, score_field in _TRAD_CONFIGS:
        tn, neg_tot, neg_conf = _trad_counts(folder, backbone, pred_field, fake_values, None, category, score_field)
        gen_counts = {g: _trad_counts(folder, backbone, pred_field, fake_values, g, category, score_field)
                      for g in GENERATORS}
        rows.append(_trad_row_from_counts(display, tn, neg_tot, neg_conf, gen_counts))
    return rows


def build_trad_rows_avg(categories: list[str]) -> list[dict]:
    rows = []
    for folder, backbone, display, pred_field, fake_values, score_field in _TRAD_CONFIGS:
        row: dict = {"Method": display}

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

        for cat in categories:
            tn, neg_tot, neg_conf = _trad_counts(folder, backbone, pred_field, fake_values, None, cat, score_field)
            fp = neg_tot - tn
            gen_counts = {g: _trad_counts(folder, backbone, pred_field, fake_values, g, cat, score_field)
                          for g in GENERATORS}
            all_tp   = sum(tp       for tp, _,   _ in gen_counts.values())
            all_fn   = sum(tot - tp for tp, tot, _ in gen_counts.values())
            fake_tot = sum(tot      for _,  tot, _ in gen_counts.values())
            total    = neg_tot + fake_tot

            if neg_tot > 0: neg_tnr_vals.append(tn / neg_tot)
            v = _precision(tn, all_fn)
            if v is not None: neg_prec_vals.append(v)
            neg_f1 = _f1(tn, all_fn, fp)
            if neg_f1 is not None: neg_f1_vals.append(neg_f1)
            if neg_conf is not None: neg_conf_vals.append(neg_conf)

            for g in GENERATORS:
                tp, tot, mean_conf = gen_counts[g]
                if tot > 0: gen_tpr_vals[g].append(tp / tot)
                if mean_conf is not None: gen_conf_vals[g].append(mean_conf)

            fake_f1 = _f1(all_tp, fp, all_fn)
            if fake_f1 is not None: ov_f1_vals.append(fake_f1)
            ov_tpr = (all_tp / fake_tot) if fake_tot > 0 else None
            ov_tnr = (tn / neg_tot)      if neg_tot  > 0 else None
            if ov_tpr is not None and ov_tnr is not None:
                ov_bal_vals.append((ov_tpr + ov_tnr) / 2)
            if total > 0: ov_acc_vals.append((tn + all_tp) / total)
            if neg_f1 is not None and fake_f1 is not None:
                ov_mf1_vals.append((neg_f1 + fake_f1) / 2)

        row["Negative TNR"]        = _avg(neg_tnr_vals)
        row["Negative Precision"]  = _avg(neg_prec_vals)
        row["Negative F1"]         = _avg(neg_f1_vals)
        row["Negative Mean Conf"]  = _avg(neg_conf_vals)
        for g in GENERATORS:
            row[f"{g} TPR"]       = _avg(gen_tpr_vals[g])
            row[f"{g} Mean Conf"] = _avg(gen_conf_vals[g])
        row["Overall F1"]       = _avg(ov_f1_vals)
        row["Overall Bal.Acc"]  = _avg(ov_bal_vals)
        row["Overall Acc"]      = _avg(ov_acc_vals)
        row["Overall Macro-F1"] = _avg(ov_mf1_vals)
        rows.append(row)
    return rows


# ── Combine into one table ────────────────────────────────────────────────────
def build_table(df: pd.DataFrame, mode: str,
                category: str | None = None,
                categories: list[str] | None = None,
                include_traditional: bool = True) -> pd.DataFrame:
    if category:
        rows = build_mllm_rows_raw(df, mode, category)
        if include_traditional and mode == "SingleImage-NoReview":
            rows += build_trad_rows_raw(category)
    else:
        cats = categories or sorted(df["category"].unique().tolist())
        rows = build_mllm_rows_avg(df, mode, cats)
        if include_traditional and mode == "SingleImage-NoReview":
            rows += build_trad_rows_avg(cats)
    return pd.DataFrame(rows).set_index("Method")


# ── Pretty printer ────────────────────────────────────────────────────────────
def print_table(title: str, table: pd.DataFrame, avg: bool = False) -> None:
    note = "macro-avg over categories" if avg else "raw counts per category"
    print(f"\n{'='*120}")
    print(f"  {title}  [{note}]")
    print(f"  Negative: TNR|Prec|F1|Conf  |  <gen>: TPR|Conf  |  Overall: F1|Bal.Acc|Acc|Macro-F1")
    print(f"{'='*120}")
    pd.set_option("display.max_colwidth", 20)
    pd.set_option("display.width", 320)
    print(table[BUCKETS].to_string())
    print()


# ── Excel export ──────────────────────────────────────────────────────────────
def save_excel(tables: list[tuple[str, pd.DataFrame]], path: Path) -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("[error] openpyxl required: pip install openpyxl", file=sys.stderr)
        return

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


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--excel",    metavar="FILE")
    p.add_argument("--table",    type=int, choices=[1, 2, 3, 4, 5, 6])
    p.add_argument("--category", metavar="NAME")
    p.add_argument("--dataset",  default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("Loading MLLM results…", file=sys.stderr)
    kwargs = {"dataset_root": args.dataset} if args.dataset else {}
    df = load_dataframe(**kwargs)
    print(f"Loaded {len(df):,} rows.", file=sys.stderr)

    categories = sorted(df["category"].unique().tolist())
    mode_order = [
        "SingleImage-NoReview",
        "SingleImage-withReview",
        "MultiImage-NoReview",
        "MultiImage-withReview",
        "MultiStep-NoReview",
        "MultiStep-withReview",
    ]

    all_tables: list[tuple[str, pd.DataFrame]] = []

    for i, mode in enumerate(mode_order, 1):
        title = f"Table {i}: {MODE_LABELS[mode]}  —  Macro-Avg over {len(categories)} Categories"
        table = build_table(df, mode, categories=categories)
        all_tables.append((f"T{i}_{mode[:20]}", table))
        if args.table is None or args.table == i:
            print_table(title, table, avg=True)

    if args.category:
        table = build_table(df, "SingleImage-NoReview", category=args.category)
        print_table(f"Single Image w/o Review  |  {args.category}", table, avg=False)
    elif args.table is None:
        for idx, cat in enumerate(categories, 7):
            table = build_table(df, "SingleImage-NoReview", category=cat)
            title = f"Table {idx}: Single Image w/o Review  |  {cat}"
            all_tables.append((f"T{idx}_{cat[:25]}", table))
            print_table(title, table, avg=False)

    if args.excel:
        if args.table is not None and not args.category:
            for idx, cat in enumerate(categories, 7):
                table = build_table(df, "SingleImage-NoReview", category=cat)
                all_tables.append((f"T{idx}_{cat[:25]}", table))
        save_excel(all_tables, Path(args.excel))


if __name__ == "__main__":
    main()
