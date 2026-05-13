from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, Optional

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

KNOWN_GEN_MODELS = {
    "gpt-image-2",
    "grok-imagine-image",
    "nano-banana-2",
    "qwen-image-2.0-pro",
    "qwen-image-edit-max",
    "wan2.7-image-pro",
}


@dataclass(frozen=True)
class ImageEntry:
    image_id: str
    abs_path: str
    rel_path: str
    category: str
    bucket: str  # "Negative" | "DeepFake"
    gen_model: Optional[str]
    review_id: str
    filename: str

    @property
    def true_label(self) -> str:
        return "real" if self.bucket == "Negative" else "fake"


def _iter_images(folder: str) -> Iterable[str]:
    for dirpath, _dirs, files in os.walk(folder):
        for name in files:
            if name.startswith("."):
                continue
            if not name.lower().endswith(IMG_EXTS):
                continue
            yield os.path.join(dirpath, name)


def _hash(rel_path: str) -> str:
    return hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:16]


def _scan_category(root: str, category: str) -> list[ImageEntry]:
    entries: list[ImageEntry] = []
    cat_dir = os.path.join(root, category)

    neg_root = os.path.join(cat_dir, "Negative")
    if os.path.isdir(neg_root):
        for abs_path in _iter_images(neg_root):
            rel = os.path.relpath(abs_path, root)
            review_id = os.path.basename(os.path.dirname(abs_path))
            if not review_id.startswith("Review_"):
                continue
            entries.append(
                ImageEntry(
                    image_id=_hash(rel),
                    abs_path=abs_path,
                    rel_path=rel,
                    category=category,
                    bucket="Negative",
                    gen_model=None,
                    review_id=review_id,
                    filename=os.path.basename(abs_path),
                )
            )

    df_root = os.path.join(cat_dir, "DeepFake")
    if os.path.isdir(df_root):
        for model in sorted(os.listdir(df_root)):
            if model not in KNOWN_GEN_MODELS:
                continue
            model_dir = os.path.join(df_root, model)
            if not os.path.isdir(model_dir):
                continue
            for abs_path in _iter_images(model_dir):
                rel = os.path.relpath(abs_path, root)
                review_id = os.path.basename(os.path.dirname(abs_path))
                if not review_id.startswith("Review_"):
                    continue
                entries.append(
                    ImageEntry(
                        image_id=_hash(rel),
                        abs_path=abs_path,
                        rel_path=rel,
                        category=category,
                        bucket="DeepFake",
                        gen_model=model,
                        review_id=review_id,
                        filename=os.path.basename(abs_path),
                    )
                )
    return entries


def list_categories(root: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if not os.path.isdir(full):
            continue
        if os.path.isdir(os.path.join(full, "Negative")) or os.path.isdir(
            os.path.join(full, "DeepFake")
        ):
            out.append(name)
    return out


class Catalog:
    def __init__(self, root: str, scope: str):
        self.root = os.path.abspath(root)
        self.scope = scope  # "All" or a category name
        self.entries: list[ImageEntry] = []
        self._by_id: dict[str, ImageEntry] = {}
        self._build()

    def _build(self) -> None:
        if self.scope == "All":
            cats = list_categories(self.root)
        else:
            cats = [self.scope]
        for cat in cats:
            self.entries.extend(_scan_category(self.root, cat))
        self._by_id = {e.image_id: e for e in self.entries}

    def get(self, image_id: str) -> Optional[ImageEntry]:
        return self._by_id.get(image_id)

    @property
    def real(self) -> list[ImageEntry]:
        return [e for e in self.entries if e.bucket == "Negative"]

    @property
    def fake(self) -> list[ImageEntry]:
        return [e for e in self.entries if e.bucket == "DeepFake"]

    def stats(self) -> dict:
        per_model: dict[str, int] = {}
        for e in self.fake:
            per_model[e.gen_model or "?"] = per_model.get(e.gen_model or "?", 0) + 1
        return {
            "total": len(self.entries),
            "real": len(self.real),
            "fake": len(self.fake),
            "per_model": per_model,
            "categories": sorted({e.category for e in self.entries}),
        }
