# SPDX-License-Identifier: Apache-2.0
"""Cheap streaming selector for focused replay.

This path intentionally avoids full-transformer prefill over the long context.
It stores compact token/block features, picks a tiny evidence span, and leaves
the expensive LLM work to dense replay over that selected span.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math

import torch

FEATURE_NAMES = [
    "idf_overlap",
    "tfidf_overlap",
    "exact_overlap",
    "bigram_overlap",
    "recency",
]


@dataclass
class StreamingSelection:
    blocks: list[int]
    ranked_blocks: list[int]
    scores: list[float]
    feature_names: list[str]
    feature_bytes: int


def _unique_query_terms(query_ids: list[int]) -> torch.Tensor:
    ids = [int(x) for x in query_ids if int(x) >= 0]
    if not ids:
        return torch.empty(0, dtype=torch.long)
    return torch.tensor(sorted(set(ids)), dtype=torch.long)


def _expand_selection(ranked: list[int], nb: int, radius: int) -> list[int]:
    blocks = set()
    for b in ranked:
        for off in range(-radius, radius + 1):
            bb = b + off
            if 0 <= bb < nb:
                blocks.add(bb)
    return sorted(blocks)


@torch.no_grad()
def block_query_features_from_blocks(blocks: torch.Tensor,
                                     query_ids: list[int]) -> torch.Tensor:
    """Return cheap per-block query features [nb, len(FEATURE_NAMES)] on CPU."""
    if blocks.ndim != 2:
        raise ValueError(f"expected [nb, block_size] blocks, got {tuple(blocks.shape)}")
    blocks = blocks.to(dtype=torch.long, device="cpu")
    nb = blocks.shape[0]
    terms = _unique_query_terms(query_ids)
    if terms.numel() == 0:
        return torch.zeros(nb, len(FEATURE_NAMES), dtype=torch.float32)

    eq = blocks[:, :, None].eq(terms[None, None, :])
    present = eq.any(dim=1).float()
    tf = eq.sum(dim=1).float()
    df = present.sum(dim=0)
    idf = torch.log((float(nb) + 1.0) / (df + 1.0)).clamp_min(0.0)
    idf_sum = idf.sum().clamp_min(1e-6)

    idf_overlap = (present * idf).sum(dim=1) / idf_sum
    tfidf_overlap = (tf.clamp_max(3.0) * idf).sum(dim=1) / (3.0 * idf_sum)
    exact_overlap = present.mean(dim=1)

    q = torch.tensor(query_ids, dtype=torch.long)
    bigram_hits = torch.zeros(nb, dtype=torch.float32)
    bigram_total = 0
    for a, b in zip(q[:-1].tolist(), q[1:].tolist()):
        if a == b:
            continue
        hit = blocks[:, :-1].eq(int(a)) & blocks[:, 1:].eq(int(b))
        bigram_hits += hit.any(dim=1).float()
        bigram_total += 1
    bigram_overlap = bigram_hits / max(1, bigram_total)

    recency = torch.linspace(0.0, 1.0, nb, dtype=torch.float32)
    return torch.stack([
        idf_overlap,
        tfidf_overlap,
        exact_overlap,
        bigram_overlap,
        recency,
    ], dim=-1).contiguous()


@torch.no_grad()
def block_query_features(prompt_ids: list[int], query_ids: list[int],
                         block_size: int = 64) -> torch.Tensor:
    """Return cheap per-block query features [nb, len(FEATURE_NAMES)] on CPU."""
    usable = (len(prompt_ids) // block_size) * block_size
    if usable <= 0:
        raise ValueError("prompt is shorter than one block")
    blocks = torch.tensor(prompt_ids[:usable], dtype=torch.long).view(-1, block_size)
    return block_query_features_from_blocks(blocks, query_ids)


@torch.no_grad()
def score_streaming_features(features: torch.Tensor) -> torch.Tensor:
    """Small fixed linear scorer over normalized cheap features.

    The weights are deliberately simple and interpretable: rare query-token
    overlap dominates, phrase/bigram matches break ties, and recency is weak.
    This can be replaced by a trained tiny MLP without changing the index.
    """
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(
            f"expected [nb, {len(FEATURE_NAMES)}], got {tuple(features.shape)}")
    weights = torch.tensor([5.0, 3.0, 0.5, 2.0, 0.05], dtype=features.dtype)
    return features @ weights


@torch.no_grad()
def select_blocks_from_features(features: torch.Tensor, topk: int = 1,
                                radius: int = 0) -> StreamingSelection:
    scores = score_streaming_features(features)
    kk = max(1, min(int(topk), int(scores.numel())))
    vals, idx = torch.topk(scores, kk)
    ranked = [int(x) for x in idx.tolist()]
    nb = int(scores.numel())
    return StreamingSelection(
        blocks=_expand_selection(ranked, nb, radius),
        ranked_blocks=ranked,
        scores=[round(float(x), 6) for x in vals.tolist()],
        feature_names=list(FEATURE_NAMES),
        feature_bytes=features.numel() * features.element_size(),
    )


@torch.no_grad()
def select_blocks_from_block_tensor(blocks: torch.Tensor, query_ids: list[int],
                                    topk: int = 1, radius: int = 0) -> StreamingSelection:
    features = block_query_features_from_blocks(blocks, query_ids)
    return select_blocks_from_features(features, topk=topk, radius=radius)


def select_blocks_sparse(block_iter: Iterable[Sequence[int]], query_ids: list[int],
                         topk: int = 1, radius: int = 0) -> StreamingSelection:
    """Select blocks from a token stream without dense tensor comparisons.

    This is the product-shaped query scorer for lexical/symbolic signals: it
    stores only compact per-block query-overlap features, not the long prompt
    tokens or full-model KV. It is intentionally CPU-friendly for million-token
    streams with small query term sets.
    """
    q_terms = sorted(set(int(x) for x in query_ids if int(x) >= 0))
    term_to_idx = {t: i for i, t in enumerate(q_terms)}
    q_bigrams = {
        (int(a), int(b))
        for a, b in zip(query_ids[:-1], query_ids[1:])
        if int(a) != int(b)
    }
    m = len(q_terms)
    counts_by_block: list[list[int]] = []
    bigram_hits_by_block: list[int] = []
    df = [0] * m
    for block in block_iter:
        counts = [0] * m
        hit_bigrams = set()
        prev = None
        for tok in block:
            t = int(tok)
            j = term_to_idx.get(t)
            if j is not None:
                counts[j] += 1
            if prev is not None:
                bg = (prev, t)
                if bg in q_bigrams:
                    hit_bigrams.add(bg)
            prev = t
        for j, cnt in enumerate(counts):
            if cnt:
                df[j] += 1
        counts_by_block.append(counts)
        bigram_hits_by_block.append(len(hit_bigrams))

    nb = len(counts_by_block)
    if nb == 0:
        raise ValueError("block iterator produced no full blocks")
    if m == 0:
        ranked = list(range(min(int(topk), nb)))
        return StreamingSelection(
            blocks=_expand_selection(ranked, nb, radius),
            ranked_blocks=ranked,
            scores=[0.0 for _ in ranked],
            feature_names=list(FEATURE_NAMES),
            feature_bytes=nb * len(FEATURE_NAMES) * 4,
        )
    idf = [max(0.0, math.log((float(nb) + 1.0) / (d + 1.0))) for d in df]
    idf_sum = max(sum(idf), 1e-6)
    bigram_total = max(1, len(q_bigrams))
    denom = max(1, nb - 1)
    scored = []
    for b, counts in enumerate(counts_by_block):
        idf_overlap = sum(idf[j] for j, cnt in enumerate(counts) if cnt) / idf_sum
        tfidf_overlap = sum(min(cnt, 3) * idf[j] for j, cnt in enumerate(counts)) / (3.0 * idf_sum)
        exact_overlap = sum(1 for cnt in counts if cnt) / float(m)
        bigram_overlap = bigram_hits_by_block[b] / float(bigram_total)
        recency = b / float(denom)
        score = (
            5.0 * idf_overlap
            + 3.0 * tfidf_overlap
            + 0.5 * exact_overlap
            + 2.0 * bigram_overlap
            + 0.05 * recency
        )
        scored.append((score, b))
    kk = max(1, min(int(topk), nb))
    scored.sort(reverse=True)
    top = scored[:kk]
    ranked = [b for _, b in top]
    return StreamingSelection(
        blocks=_expand_selection(ranked, nb, radius),
        ranked_blocks=ranked,
        scores=[round(float(s), 6) for s, _ in top],
        feature_names=list(FEATURE_NAMES),
        feature_bytes=nb * len(FEATURE_NAMES) * 4,
    )


@torch.no_grad()
def select_blocks(prompt_ids: list[int], query_ids: list[int], block_size: int = 64,
                  topk: int = 1, radius: int = 0) -> StreamingSelection:
    features = block_query_features(prompt_ids, query_ids, block_size=block_size)
    return select_blocks_from_features(features, topk=topk, radius=radius)
