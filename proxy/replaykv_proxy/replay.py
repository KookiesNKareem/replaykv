# SPDX-License-Identifier: Apache-2.0
"""Bounded-replay engine: ingest-time index + gated semantic tags + replay.

Torch-free extraction of the benchmarked selector path
(replaykv/streaming_replay.py + the gating/NMS logic from the product
harnesses). Operates on token ids from any HF tokenizer; semantic tags come
from a static token->cluster table distilled offline (see tags.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from .tags import TagTable


@dataclass
class ReplayConfig:
    block_size: int = 64
    topk: int = 48
    radius: int = 1
    candidate_multiplier: int = 6
    nms_gap: int = -1            # -1 => 2*radius
    gate_score: float = 1.0      # sparse top-1 below this => enable tag expansion
    query_m: int = 4             # query-side soft cluster expansion
    max_replay_tokens: int = 12000
    min_score_frac: float = 0.1  # drop blocks scoring below this fraction of top-1
    min_keep: int = 4            # ...but never fewer than this many blocks


@dataclass
class ReplayResult:
    intervals: list[tuple[int, int]]
    ranked_blocks: list[int]
    scores: list[float]
    replay_tokens: int
    gate_used_tags: bool
    n_blocks: int


def _score_blocks(blocks: list[list[int]], query_ids: list[int],
                  topk: int) -> tuple[list[int], list[float]]:
    """IDF/TF-IDF/bigram lexical scorer over fixed token blocks (pure python)."""
    q_terms = sorted(set(int(x) for x in query_ids if int(x) >= 0))
    term_to_idx = {t: i for i, t in enumerate(q_terms)}
    q_bigrams = {(int(a), int(b)) for a, b in zip(query_ids[:-1], query_ids[1:])
                 if int(a) != int(b)}
    m = len(q_terms)
    nb = len(blocks)
    if nb == 0:
        raise ValueError("no blocks")
    if m == 0:
        return list(range(min(topk, nb))), [0.0] * min(topk, nb)
    counts_by_block: list[list[int]] = []
    bigram_hits: list[int] = []
    df = [0] * m
    for block in blocks:
        counts = [0] * m
        hits = set()
        prev = None
        for tok in block:
            t = int(tok)
            j = term_to_idx.get(t)
            if j is not None:
                counts[j] += 1
            if prev is not None and (prev, t) in q_bigrams:
                hits.add((prev, t))
            prev = t
        for j, c in enumerate(counts):
            if c:
                df[j] += 1
        counts_by_block.append(counts)
        bigram_hits.append(len(hits))
    idf = [max(0.0, math.log((nb + 1.0) / (d + 1.0))) for d in df]
    idf_sum = max(sum(idf), 1e-6)
    bg_total = max(1, len(q_bigrams))
    denom = max(1, nb - 1)
    scored = []
    for b, counts in enumerate(counts_by_block):
        s = (5.0 * sum(idf[j] for j, c in enumerate(counts) if c) / idf_sum
             + 3.0 * sum(min(c, 3) * idf[j] for j, c in enumerate(counts)) / (3.0 * idf_sum)
             + 0.5 * sum(1 for c in counts if c) / float(m)
             + 2.0 * bigram_hits[b] / float(bg_total)
             + 0.05 * b / float(denom))
        scored.append((s, b))
    scored.sort(reverse=True)
    top = scored[:max(1, min(topk, nb))]
    return [b for _, b in top], [round(s, 6) for s, _ in top]


def _diversify(ranked: list[int], scores: list[float], topk: int,
               radius: int, nms_gap: int) -> tuple[list[int], list[float]]:
    gap = nms_gap if nms_gap >= 0 else max(0, 2 * radius)
    chosen: list[int] = []
    chosen_scores: list[float] = []
    for b, s in zip(ranked, scores):
        if all(abs(b - o) > gap for o in chosen):
            chosen.append(b)
            chosen_scores.append(s)
            if len(chosen) >= topk:
                return chosen, chosen_scores
    seen = set(chosen)
    for b, s in zip(ranked, scores):
        if b not in seen:
            chosen.append(b)
            chosen_scores.append(s)
            seen.add(b)
            if len(chosen) >= topk:
                break
    return chosen, chosen_scores


def _merge_intervals(blocks: list[int], radius: int, block_size: int,
                     n_tokens: int) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for b in sorted(set(blocks)):
        lo = max(0, b - radius) * block_size
        hi = min(n_tokens, (b + radius + 1) * block_size)
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


class ReplayEngine:
    """Stateless bounded replay over a tokenized context."""

    def __init__(self, cfg: ReplayConfig, tag_table: TagTable | None = None):
        self.cfg = cfg
        self.tags = tag_table

    def select(self, context_ids: list[int], query_ids: list[int]) -> ReplayResult:
        cfg = self.cfg
        usable = (len(context_ids) // cfg.block_size) * cfg.block_size
        blocks = [context_ids[lo:lo + cfg.block_size]
                  for lo in range(0, usable, cfg.block_size)]
        cand_k = max(cfg.topk, cfg.topk * max(1, cfg.candidate_multiplier))

        ranked, scores = _score_blocks(blocks, query_ids, cand_k)
        used_tags = False
        if self.tags is not None and (not scores or scores[0] < cfg.gate_score):
            # weak lexical anchors -> add the distilled semantic channel
            used_tags = True
            tagged_blocks = [b + self.tags.block_tags(b) for b in blocks]
            tagged_query = list(query_ids) + self.tags.query_tags(
                query_ids, m=cfg.query_m)
            ranked, scores = _score_blocks(tagged_blocks, tagged_query, cand_k)

        ranked, scores = _diversify(ranked, scores, cfg.topk, cfg.radius, cfg.nms_gap)
        # relative-score cutoff: irrelevant blocks score far below the evidence
        # cliff (observed >10x); trimming them keeps small backends focused
        if scores and cfg.min_score_frac > 0:
            # shift-invariant: ubiquitous terms/tags add a uniform floor to all
            # blocks, so measure relevance as height above the floor
            floor = min(scores)
            span = scores[0] - floor
            # only trim when a clear relevance cliff exists; near-uniform
            # scores mean uniform relevance (or none) -> keep the full budget
            if span > 0.5 * max(abs(scores[0]), 1e-9):
                keep = max(cfg.min_keep, sum(
                    1 for s in scores if (s - floor) >= cfg.min_score_frac * span))
                ranked, scores = ranked[:keep], scores[:keep]
        intervals = _merge_intervals(ranked, cfg.radius, cfg.block_size, len(context_ids))
        replay_tokens = sum(hi - lo for lo, hi in intervals)
        # enforce the hard budget by dropping lowest-ranked blocks
        while replay_tokens > cfg.max_replay_tokens and len(ranked) > 1:
            ranked = ranked[:-1]
            intervals = _merge_intervals(ranked, cfg.radius, cfg.block_size, len(context_ids))
            replay_tokens = sum(hi - lo for lo, hi in intervals)
        return ReplayResult(
            intervals=intervals,
            ranked_blocks=ranked,
            scores=scores,
            replay_tokens=replay_tokens,
            gate_used_tags=used_tags,
            n_blocks=len(blocks),
        )
