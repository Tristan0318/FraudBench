#!/usr/bin/env python3
"""
detect_mismatch.py
==================================
DeepFake detection with MISMATCHED review text (cross-category).

Reads MismatchReview/mismatch_index.json and injects a review from a
different product category into the v1_baseline prompt.

Experiment design
-----------------
  Condition A (Matched)    — v1_baseline in PromptAblation  [done]
  Condition B (Mismatched) — this script                    [new]

  Δ between A and B shows whether models use text-image consistency
  as a detection cue or rely purely on visual artifacts.

Usage
-----
  python detect_mismatch.py --category "All Beauty"
  python detect_mismatch.py --category "Electronics" \\
      --models qwen3.6-plus kimi-k2.6 --concurrency 4

  # If DASHSCOPE_API_KEY_1 is unavailable, exclude KEY_1 models:
  python detect_mismatch.py --category "All Beauty" \\
      --models qwen3.6-plus kimi-k2.6 qwen3-vl-flash \\
               qwen3.6-flash qwen3.5-omni-plus \\
               grok-4-1-fast-reasoning grok-4.20-reasoning-latest \\
               gpt-5.4-mini gemini-3-flash
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Reuse internals from the ablation script ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
import detect_prompt_ablation as _abl

MODELS       = _abl.MODELS
MODEL_BY_NAME = _abl.MODEL_BY_NAME
ModelCfg     = _abl.ModelCfg

_SYSTEM      = _abl._BASELINE_SYSTEM
_USER        = _abl._BASELINE_USER

_INDEX_DEFAULT = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "MismatchReview" / "mismatch_index.json"
)


# ── Prompt helpers ────────────────────────────────────────────────────────────
def _comment_block(text: str) -> str:
    return (
        f"\n\nThe customer who purchased this product on Amazon left the "
        f"following review:\n\"{text}\"\n"
        f"Consider this customer review when making your assessment.\n\n"
    )


def _build_messages(cfg: ModelCfg, image_path: Path,
                    review_text: str | None,
                    downscale_level: int = 0) -> list[dict]:
    """Single-image messages with optional injected review text."""
    data_uri, _ = _abl.image_to_data_uri(image_path)

    user_body = _USER
    if review_text:
        # Insert review block just before the JSON schema instruction
        split_marker = "\n\nReturn ONLY"
        if split_marker in user_body:
            head, tail = user_body.split(split_marker, 1)
            user_body = head + _comment_block(review_text) + split_marker + tail
        else:
            user_body = user_body + _comment_block(review_text)

    parts: list[dict] = [
        {"type": "image_url", "image_url": {"url": data_uri}},
        {"type": "text",      "text": user_body},
    ]

    merge = cfg.name.startswith(("qwen", "qvq", "kimi"))
    if merge:
        if _SYSTEM:
            parts.insert(0, {"type": "text", "text": _SYSTEM})
        return [{"role": "user", "content": parts}]
    if _SYSTEM:
        return [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": parts},
        ]
    return [{"role": "user", "content": parts}]


# ── Per-job caller ────────────────────────────────────────────────────────────
def _call_once(cfg: ModelCfg, image_path: Path,
               review_text: str | None,
               downscale_level: int = 0) -> dict:
    if cfg.provider == "gemini":
        # Gemini builds its own message format; patch _ACTIVE_USER temporarily
        old_user   = _abl._ACTIVE_USER
        old_system = _abl._ACTIVE_SYSTEM
        patched_user = old_user
        if review_text:
            split_marker = "\n\nReturn ONLY"
            if split_marker in old_user:
                head, tail = old_user.split(split_marker, 1)
                patched_user = head + _comment_block(review_text) + split_marker + tail
        _abl._ACTIVE_USER   = patched_user
        _abl._ACTIVE_SYSTEM = _SYSTEM
        try:
            return _abl.call_gemini(cfg, [image_path])
        finally:
            _abl._ACTIVE_USER   = old_user
            _abl._ACTIVE_SYSTEM = old_system

    msgs = _build_messages(cfg, image_path, review_text, downscale_level)
    # Temporarily override _build_openai_messages via a thin wrapper
    # by calling the HTTP layer directly through call_openai_compat's internals.
    # Simplest: replicate the call_openai_compat logic with our custom messages.
    api_key = _abl._provider_key(cfg)
    url     = f"{cfg.endpoint.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload: dict[str, Any] = {
        "model":   cfg.model_id,
        "messages": msgs,
        "stream":  cfg.stream,
    }
    if cfg.stream:
        payload["stream_options"] = {"include_usage": True}
    if cfg.extra_body:
        payload.update(cfg.extra_body)

    status, result = _abl._http_post_json(url, payload, headers, stream=cfg.stream)

    if cfg.stream:
        content, reasoning = _abl._consume_sse(result)
        return {"content": content, "reasoning_content": reasoning or None,
                "raw_response": content}
    try:
        msg = result["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as e:
        raise _abl.HttpError(status, json.dumps(result)[:500]) from e
    return {"content": msg.get("content") or "",
            "reasoning_content": msg.get("reasoning_content") or None,
            "raw_response": msg.get("content") or ""}


def _call_model(cfg: ModelCfg, image_path: Path,
                review_text: str | None) -> dict:
    """Call with retry / xAI downscale logic."""
    MAX_RETRIES = _abl.MAX_RETRIES
    RETRY_BASE  = _abl.RETRY_BASE
    is_xai      = cfg.provider == "xai"
    downscale   = 0
    max_ds      = len(_abl.XAI_DOWNSCALE_TIERS) if is_xai else 0
    total       = MAX_RETRIES + max_ds
    attempt = backoff_n = 0
    last_err: str | None = None

    while attempt < total:
        attempt += 1
        try:
            res = _call_once(cfg, image_path, review_text,
                             downscale_level=downscale)
            raw   = res.get("content") or res.get("raw_response") or ""
            parsed = _abl.parse_json_reply(raw)
            if parsed is None:
                return {"status": "error", "error": f"unparseable: {raw[:200]}",
                        "is_ai_modified": None, "confidence": None, "reason": None}
            return {
                "status":        "ok",
                "is_ai_modified": parsed.get("is_ai_modified"),
                "confidence":    parsed.get("confidence"),
                "reason":        parsed.get("reason"),
                "error":         None,
            }
        except _abl.HttpError as e:
            last_err = f"HTTP {e.status}: {e.body[:300]}"
            if e.status == 413 and is_xai and downscale < max_ds:
                downscale += 1
                continue
            if 400 <= e.status < 500 and e.status != 429:
                break
            backoff_n += 1
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            backoff_n += 1
        if attempt < total:
            time.sleep(RETRY_BASE * (2 ** (backoff_n - 1)))

    return {"status": "error", "error": last_err,
            "is_ai_modified": None, "confidence": None, "reason": None}


# ── Job dataclass ─────────────────────────────────────────────────────────────
@dataclass
class MismatchJob:
    image_path:              Path
    image_rel:               str
    label:                   str
    bucket:                  str
    generator:               str | None
    review_text:             str | None
    review_source_category:  str | None
    review_source_review_id: str | None

    @property
    def review_id(self) -> str:
        return Path(self.image_rel).parts[0] if self.image_rel else ""

    @property
    def key(self) -> str:
        return str(self.image_path)


def load_jobs(index_path: Path, category: str) -> tuple[list[MismatchJob], dict]:
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    entries  = index_data.get("categories", {}).get(category)
    if entries is None:
        avail = sorted(index_data.get("categories", {}).keys())
        raise KeyError(f"{category!r} not in index. Available: {avail}")
    jobs = []
    for e in entries:
        p = Path(e["image_path"])
        if not p.exists():
            raise FileNotFoundError(f"Image missing: {p}")
        jobs.append(MismatchJob(
            image_path=p,
            image_rel=e.get("image_rel", ""),
            label=e["label"],
            bucket=e["bucket"],
            generator=e.get("generator"),
            review_text=e.get("review_text"),
            review_source_category=e.get("review_source_category"),
            review_source_review_id=e.get("review_source_review_id"),
        ))
    meta = {k: index_data.get(k)
            for k in ("sample_ratio", "sample_seed", "mismatch_seed", "experiment")}
    return jobs, meta


# ── Result store ──────────────────────────────────────────────────────────────
class ResultStore:
    def __init__(self, output_dir: Path, category: str,
                 jobs: list[MismatchJob], models: list[ModelCfg], meta: dict):
        self._dir = output_dir / category
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jobs   = {j.key: j for j in jobs}
        self._models = [m.name for m in models]
        self._meta   = meta
        self._data: dict[str, dict] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        for p in self._dir.glob("*.json"):
            if p.name == "summary.json":
                continue
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if d.get("key"):
                    self._data[d["key"]] = d
            except (json.JSONDecodeError, OSError):
                pass

    def already_done(self, key: str, model: str) -> bool:
        v = self._data.get(key, {}).get("verdicts", {}).get(model, {})
        return v.get("status") == "ok"

    def save(self, key: str, model: str, verdict: dict) -> None:
        if key not in self._data:
            j = self._jobs[key]
            self._data[key] = {
                "key": key, "bucket": j.bucket, "label": j.label,
                "generator": j.generator, "image_rel": j.image_rel,
                "review_id": j.review_id,
                "review_source_category":  j.review_source_category,
                "review_source_review_id": j.review_source_review_id,
                "verdicts": {},
            }
        self._data[key]["verdicts"][model] = verdict
        slug = Path(key).stem[:60]
        (self._dir / f"{slug}.json").write_text(
            json.dumps(self._data[key], ensure_ascii=False, indent=2),
            encoding="utf-8")

    def flush_summary(self, category: str) -> Path:
        summary = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "experiment":   "mismatch_review",
            "category":     category,
            "models":       self._models,
            **self._meta,
            "num_jobs": len(self._data),
            "rows":     list(self._data.values()),
        }
        p = self._dir / "summary.json"
        p.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                     encoding="utf-8")
        return p


# ── Runner ────────────────────────────────────────────────────────────────────
def run(category: str, index_path: Path, output_dir: Path,
        models: list[ModelCfg], concurrency: int, resume: bool) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    jobs, meta = load_jobs(index_path, category)
    store      = ResultStore(output_dir, category, jobs, models, meta)

    tasks = [(job, cfg) for job in jobs for cfg in models
             if not (resume and store.already_done(job.key, cfg.name))]
    total = len(jobs) * len(models)
    done  = total - len(tasks)
    print(f"[{category}] {len(jobs)} images × {len(models)} models = {total} "
          f"({done} done, {len(tasks)} to run)", flush=True)
    if not tasks:
        store.flush_summary(category)
        return

    completed = done
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_call_model, cfg, job.image_path, job.review_text):
                   (job, cfg) for job, cfg in tasks}
        for fut in as_completed(futures):
            job, cfg = futures[fut]
            try:
                verdict = fut.result()
            except Exception as exc:
                verdict = {"status": "error", "error": str(exc),
                           "is_ai_modified": None, "confidence": None}
            store.save(job.key, cfg.name, verdict)
            completed += 1
            print(f"  [{completed}/{total}] {cfg.name} | {job.bucket} | "
                  f"{Path(job.image_rel).name} | {verdict.get('status')} "
                  f"ai={verdict.get('is_ai_modified')}", flush=True)

    path = store.flush_summary(category)
    print(f"[{category}] summary → {path}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--category",    required=True)
    p.add_argument("--index",    default=str(_INDEX_DEFAULT))
    p.add_argument("--output",      default=str(_INDEX_DEFAULT.parent))
    p.add_argument("--models",      nargs="+",
                   choices=[m.name for m in MODELS], default=None,
                   help="Default: all models whose API key env var is set")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--no-resume",   action="store_true")
    args = p.parse_args()

    # Default: all models whose key is available
    if args.models:
        selected = [MODEL_BY_NAME[n] for n in args.models]
    else:
        selected = [m for m in MODELS if os.getenv(m.key_env)]
        skipped  = [m for m in MODELS if not os.getenv(m.key_env)]
        if skipped:
            print(f"[info] skipping {[m.name for m in skipped]} "
                  f"(env var not set)", flush=True)

    if not selected:
        print("[error] no models available — check API key env vars", file=sys.stderr)
        sys.exit(1)

    run(
        category=args.category,
        index_path=Path(args.index).expanduser().resolve(),
        output_dir=Path(args.output).expanduser().resolve(),
        models=selected,
        concurrency=max(1, args.concurrency),
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
