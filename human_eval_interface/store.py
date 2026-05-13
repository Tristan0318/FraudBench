from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from threading import Lock
from typing import Optional

from catalog import ImageEntry


_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_name(s: str) -> str:
    s = s.strip()
    s = _NAME_RE.sub("_", s)
    return s or "anonymous"


def _atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


class Store:
    """One Store per (evaluator, scope). All writes are atomic."""

    def __init__(self, root: str, evaluator: str, scope: str, results_root: str):
        self.root = os.path.abspath(root)
        self.results_root = os.path.abspath(results_root)
        self.evaluator = safe_name(evaluator)
        self.scope = scope  # "All" or a category name
        self._lock = Lock()
        self.base_dir = self._compute_base_dir()
        os.makedirs(os.path.join(self.base_dir, "DeepFake"), exist_ok=True)
        self._progress_path = os.path.join(self.base_dir, "_progress.json")
        # Cache: image_id -> True if already submitted
        self._seen: set[str] = set()
        self._history: list[dict] = []  # for undo
        self._load_progress()

    def _compute_base_dir(self) -> str:
        # Layout: <results_root>/<evaluator>/<scope>/...
        # scope is "All" or a category name (may contain spaces) — kept as a literal dir name.
        return os.path.join(self.results_root, self.evaluator, self.scope)

    def _load_progress(self) -> None:
        prog = _read_json(self._progress_path)
        if prog:
            self._seen = set(prog.get("seen", []))
            self._history = prog.get("history", [])

    def _save_progress(self) -> None:
        _atomic_write_json(
            self._progress_path,
            {
                "evaluator": self.evaluator,
                "scope": self.scope,
                "updated_at": _now_iso(),
                "seen": sorted(self._seen),
                "history": self._history[-200:],  # cap to last 200
            },
        )

    @property
    def seen(self) -> set[str]:
        return self._seen

    def _bucket_file(self, entry: ImageEntry) -> str:
        if entry.bucket == "Negative":
            return os.path.join(self.base_dir, "Negative.json")
        return os.path.join(self.base_dir, "DeepFake", f"{entry.gen_model}.json")

    def append_record(
        self,
        entry: ImageEntry,
        human_choice: str,
        elapsed_ms: int,
    ) -> dict:
        with self._lock:
            if entry.image_id in self._seen:
                raise ValueError(f"image already evaluated: {entry.image_id}")
            record = {
                "image_id": entry.image_id,
                "category": entry.category,
                "review_id": entry.review_id,
                "filename": entry.filename,
                "rel_path": entry.rel_path,
                "true_label": entry.true_label,
                "gen_model": entry.gen_model,
                "human_choice": human_choice,
                "correct": human_choice == entry.true_label,
                "elapsed_ms": int(elapsed_ms),
                "decided_at": _now_iso(),
                "evaluator": self.evaluator,
            }

            target = self._bucket_file(entry)
            blob = _read_json(target) or {
                "evaluator": self.evaluator,
                "scope": self.scope,
                "bucket": entry.bucket,
                "gen_model": entry.gen_model,
                "records": [],
            }
            blob["updated_at"] = record["decided_at"]
            blob["records"].append(record)
            _atomic_write_json(target, blob)

            self._seen.add(entry.image_id)
            self._history.append(
                {"image_id": entry.image_id, "target": target}
            )
            self._save_progress()
            return record

    def undo_last(self) -> Optional[dict]:
        with self._lock:
            if not self._history:
                return None
            last = self._history.pop()
            target = last["target"]
            blob = _read_json(target)
            removed = None
            if blob and blob.get("records"):
                # remove the last matching image_id from the end
                for i in range(len(blob["records"]) - 1, -1, -1):
                    if blob["records"][i]["image_id"] == last["image_id"]:
                        removed = blob["records"].pop(i)
                        break
                blob["updated_at"] = _now_iso()
                _atomic_write_json(target, blob)
            self._seen.discard(last["image_id"])
            self._save_progress()
            return removed

    def write_summary(self, catalog_stats: dict) -> dict:
        """Aggregate per-bucket and per-model accuracy across written files."""
        with self._lock:
            buckets: list[tuple[str, str, Optional[str]]] = [
                (os.path.join(self.base_dir, "Negative.json"), "Negative", None),
            ]
            df_dir = os.path.join(self.base_dir, "DeepFake")
            if os.path.isdir(df_dir):
                for fname in sorted(os.listdir(df_dir)):
                    if fname.endswith(".json"):
                        model = fname[:-5]
                        buckets.append(
                            (os.path.join(df_dir, fname), "DeepFake", model)
                        )

            per_bucket = []
            total = 0
            correct = 0
            for path, bucket, model in buckets:
                blob = _read_json(path)
                if not blob:
                    continue
                recs = blob.get("records", [])
                n = len(recs)
                c = sum(1 for r in recs if r.get("correct"))
                total += n
                correct += c
                per_bucket.append(
                    {
                        "bucket": bucket,
                        "gen_model": model,
                        "n": n,
                        "correct": c,
                        "accuracy": (c / n) if n else None,
                    }
                )

            summary = {
                "evaluator": self.evaluator,
                "scope": self.scope,
                "generated_at": _now_iso(),
                "catalog_stats": catalog_stats,
                "evaluated": total,
                "correct": correct,
                "accuracy": (correct / total) if total else None,
                "per_bucket": per_bucket,
            }
            _atomic_write_json(
                os.path.join(self.base_dir, "summary.json"), summary
            )
            return summary


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
