# SPDX-License-Identifier: Apache-2.0
"""Numpy-backed distilled concept-tag table (torch-free).

Tables are built once per tokenizer by replaykv/learned_tags.py (embed every
vocab piece with a strong embedder, mean-center, k-means) and converted to .npz
via tools/convert_tags.py. At serving time this is a pure lookup."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class TagTable:
    def __init__(self, topm: np.ndarray, k: int, base_offset: int = 1000):
        if topm.ndim != 2:
            raise ValueError("topm must be [vocab, M]")
        self.topm = topm.astype(np.int64)
        self.k = int(k)
        self.base_offset = base_offset
        self._base_id: int | None = None

    def bind_vocab(self, vocab_size: int) -> None:
        """Synthetic tag ids must not collide with real token ids."""
        self._base_id = vocab_size + self.base_offset

    @property
    def base_id(self) -> int:
        if self._base_id is None:
            raise RuntimeError("call bind_vocab(len(tokenizer)) first")
        return self._base_id

    def _clusters(self, ids: Sequence[int], m: int) -> list[int]:
        n = self.topm.shape[0]
        valid = [int(t) for t in ids if 0 <= int(t) < n]
        if not valid:
            return []
        rows = self.topm[np.asarray(valid), :max(1, min(m, self.topm.shape[1]))]
        return np.unique(rows).tolist()

    def block_tags(self, ids: Sequence[int]) -> list[int]:
        return [self.base_id + c for c in self._clusters(ids, m=1)]

    def query_tags(self, ids: Sequence[int], m: int = 4) -> list[int]:
        return [self.base_id + c for c in self._clusters(ids, m=m)]


def load_tag_table(path: str) -> TagTable:
    data = np.load(path)
    return TagTable(topm=data["topm"], k=int(data["k"]))
