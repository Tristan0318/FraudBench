from __future__ import annotations

import random
from collections import deque
from typing import Optional

from catalog import Catalog, ImageEntry


class Sampler:
    """Real/Fake 50:50 balanced sampler with anti-recency for same Review_xxx."""

    def __init__(self, catalog: Catalog, recent_window: int = 30):
        self.catalog = catalog
        self.recent: deque = deque(maxlen=recent_window)
        self._rng = random.Random()

    def remember(self, entry: ImageEntry) -> None:
        self.recent.append((entry.category, entry.review_id))

    def forget_last(self, entry: ImageEntry) -> None:
        try:
            # remove the most recent occurrence
            tmp = list(self.recent)
            for i in range(len(tmp) - 1, -1, -1):
                if tmp[i] == (entry.category, entry.review_id):
                    del tmp[i]
                    break
            self.recent = deque(tmp, maxlen=self.recent.maxlen)
        except ValueError:
            pass

    def next(self, seen: set[str]) -> Optional[ImageEntry]:
        pool_real = [e for e in self.catalog.real if e.image_id not in seen]
        pool_fake = [e for e in self.catalog.fake if e.image_id not in seen]
        if not pool_real and not pool_fake:
            return None

        if pool_real and pool_fake:
            side = "real" if self._rng.random() < 0.5 else "fake"
        elif pool_real:
            side = "real"
        else:
            side = "fake"

        pool = pool_real if side == "real" else pool_fake
        recent_set = set(self.recent)
        fresh = [e for e in pool if (e.category, e.review_id) not in recent_set]
        candidates = fresh if fresh else pool
        return self._rng.choice(candidates)

    def progress(self, seen: set[str]) -> dict:
        # Public progress intentionally omits per-class (real/fake) counts to
        # avoid leaking the remaining label distribution mid-session — once an
        # evaluator finishes all reals, every later image would otherwise be
        # known to be fake. Per-class numbers stay available in the final
        # summary written at completion.
        n_total = len(self.catalog.entries)
        n_seen = sum(1 for e in self.catalog.entries if e.image_id in seen)
        return {"evaluated": n_seen, "total": n_total}
