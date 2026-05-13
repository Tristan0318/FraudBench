from __future__ import annotations

import argparse
import os
import sys
from threading import Lock
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_file

# Ensure local imports work whether run as `python app.py` or `python -m`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from catalog import Catalog, list_categories  # noqa: E402
from sampler import Sampler  # noqa: E402
from store import Store, safe_name  # noqa: E402


app = Flask(__name__)


class AppState:
    """Holds active sessions keyed by (evaluator, scope). Single-process."""

    def __init__(self, root: str, results_root: str):
        self.root = os.path.abspath(root)
        self.results_root = os.path.abspath(results_root)
        self._lock = Lock()
        # session key -> dict(catalog, sampler, store)
        self._sessions: dict[tuple[str, str], dict] = {}
        self._catalog_cache: dict[str, Catalog] = {}

    def _get_catalog(self, scope: str) -> Catalog:
        if scope not in self._catalog_cache:
            self._catalog_cache[scope] = Catalog(self.root, scope)
        return self._catalog_cache[scope]

    def get_or_create(self, evaluator: str, scope: str) -> dict:
        key = (safe_name(evaluator), scope)
        with self._lock:
            sess = self._sessions.get(key)
            if sess is None:
                catalog = self._get_catalog(scope)
                store = Store(self.root, evaluator, scope, self.results_root)
                sampler = Sampler(catalog)
                sess = {"catalog": catalog, "sampler": sampler, "store": store}
                self._sessions[key] = sess
            return sess

    def lookup_image(self, scope: str, image_id: str):
        catalog = self._catalog_cache.get(scope) or self._get_catalog(scope)
        return catalog.get(image_id)


STATE: Optional[AppState] = None


@app.route("/")
def index():
    assert STATE is not None
    return render_template(
        "index.html", categories=list_categories(STATE.root)
    )


@app.route("/api/categories")
def api_categories():
    assert STATE is not None
    return jsonify({"categories": list_categories(STATE.root)})


@app.post("/api/session")
def api_session():
    body = request.get_json(force=True) or {}
    evaluator = body.get("evaluator", "").strip()
    scope = body.get("scope", "").strip()
    if not evaluator:
        return jsonify({"error": "evaluator name required"}), 400
    if not scope:
        return jsonify({"error": "scope required"}), 400
    assert STATE is not None
    if scope != "All" and scope not in list_categories(STATE.root):
        return jsonify({"error": f"unknown category: {scope}"}), 400
    sess = STATE.get_or_create(evaluator, scope)
    # Intentionally do NOT return per-class counts (real/fake/per_model)
    # in the session response — that would let the evaluator infer the
    # remaining label distribution mid-session.
    return jsonify(
        {
            "evaluator": safe_name(evaluator),
            "scope": scope,
            "total": sess["catalog"].stats()["total"],
            "progress": sess["sampler"].progress(sess["store"].seen),
        }
    )


@app.post("/api/next")
def api_next():
    body = request.get_json(force=True) or {}
    sess = STATE.get_or_create(body.get("evaluator", ""), body.get("scope", ""))  # type: ignore[union-attr]
    catalog = sess["catalog"]
    sampler = sess["sampler"]
    store = sess["store"]
    entry = sampler.next(store.seen)
    if entry is None:
        summary = store.write_summary(catalog.stats())
        return jsonify(
            {
                "done": True,
                "progress": sampler.progress(store.seen),
                "summary": summary,
            }
        )
    sampler.remember(entry)
    return jsonify(
        {
            "done": False,
            "image_id": entry.image_id,
            "url": f"/img/{entry.image_id}?scope={_url_scope(sess['catalog'].scope)}",
            "progress": sampler.progress(store.seen),
        }
    )


@app.post("/api/submit")
def api_submit():
    body = request.get_json(force=True) or {}
    evaluator = body.get("evaluator", "")
    scope = body.get("scope", "")
    image_id = body.get("image_id", "")
    choice = body.get("choice", "")
    elapsed_ms = int(body.get("elapsed_ms", 0))

    if choice not in ("real", "fake"):
        return jsonify({"error": "choice must be 'real' or 'fake'"}), 400

    sess = STATE.get_or_create(evaluator, scope)  # type: ignore[union-attr]
    entry = sess["catalog"].get(image_id)
    if entry is None:
        return jsonify({"error": "unknown image_id"}), 404
    try:
        record = sess["store"].append_record(entry, choice, elapsed_ms)
    except ValueError as e:
        return jsonify({"error": str(e)}), 409
    return jsonify(
        {
            "ok": True,
            "recorded": {
                "image_id": record["image_id"],
                "human_choice": record["human_choice"],
                "decided_at": record["decided_at"],
            },
            "progress": sess["sampler"].progress(sess["store"].seen),
        }
    )


@app.post("/api/undo")
def api_undo():
    body = request.get_json(force=True) or {}
    sess = STATE.get_or_create(body.get("evaluator", ""), body.get("scope", ""))  # type: ignore[union-attr]
    removed = sess["store"].undo_last()
    if removed is None:
        return jsonify({"ok": False, "error": "nothing to undo"}), 400
    entry = sess["catalog"].get(removed["image_id"])
    if entry is not None:
        sess["sampler"].forget_last(entry)
    return jsonify(
        {
            "ok": True,
            "removed": {"image_id": removed["image_id"]},
            "progress": sess["sampler"].progress(sess["store"].seen),
        }
    )


@app.get("/img/<image_id>")
def serve_image(image_id: str):
    scope = request.args.get("scope", "")
    if not scope:
        abort(400, description="scope query param required")
    assert STATE is not None
    # Decode any URL-encoded characters Flask hasn't already
    entry = STATE.lookup_image(scope, image_id)
    if entry is None:
        abort(404)
    return send_file(entry.abs_path, conditional=True)


def _url_scope(scope: str) -> str:
    # Flask handles encoding when interpolated by the client; pass through here
    # but encode spaces explicitly so the URL is clean in dev tools.
    from urllib.parse import quote

    return quote(scope, safe="")


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_results = os.path.join(here, "Results")

    parser = argparse.ArgumentParser(description="Human eval tool for Real vs DeepFake")
    parser.add_argument(
        "--root",
        default=os.path.normpath(os.path.join(here, "..", "..", "..", "..")),
        help="Dataset root directory (read-only)",
    )
    parser.add_argument(
        "--results-root",
        default=default_results,
        help="Where to write evaluation outputs (default: Tools/HumanEval/Results)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        print(f"[error] root not found: {args.root}", file=sys.stderr)
        return 2

    os.makedirs(args.results_root, exist_ok=True)

    global STATE
    STATE = AppState(args.root, args.results_root)
    cats = list_categories(args.root)
    print(f"[human-eval] root={args.root}")
    print(f"[human-eval] results={args.results_root}")
    print(f"[human-eval] categories={len(cats)} (e.g. {cats[:3]} ...)")
    print(f"[human-eval] open http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
