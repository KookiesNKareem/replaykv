# SPDX-License-Identifier: Apache-2.0
"""Compact concept-tag expansion for streaming replay selectors.

The sparse selector is intentionally lexical and cheap. Concept tags add a
small, ingest-time semantic channel by appending synthetic concept token IDs to
queries and blocks before sparse scoring. The default groups are a development
prototype for synthetic/RULER-style evidence recovery; production groups should
be learned or generated from evidence-grounded calibration data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from collections.abc import Iterable, Sequence


DEFAULT_CONCEPT_GROUPS: list[tuple[str, tuple[str, ...]]] = [
    ("access_identifier", (
        "access", "code", "credential", "identifier", "passcode", "pin", "secret",
    )),
    ("record_key", (
        "audit", "key", "project", "record", "entry", "field",
    )),
    ("assignment_value", (
        "assigned", "assignment", "set", "value", "token", "reads",
    )),
    ("latest_current", (
        "current", "latest", "final", "after", "update", "overrides",
    )),
    ("outcome_result", (
        "outcome", "result", "observed", "linked", "mechanism",
    )),
    ("preparation_phase", (
        "preparation", "phase", "uses", "during", "compound",
    )),
]


@dataclass
class ConceptExpander:
    concept_ids: list[int]
    concept_names: list[str]
    token_sets: list[set[int]]
    bytes_per_block: int

    def expand_query(self, query_ids: list[int]) -> list[int]:
        return append_concepts(query_ids, self)

    def expand_blocks(self, block_iter: Iterable[Sequence[int]]):
        for block in block_iter:
            yield append_concepts(list(block), self)


def _tokenize(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def _clean_decoded_piece(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "", text).lower()


def build_concept_expander(
    tok,
    concept_groups: list[tuple[str, tuple[str, ...]]] | None = None,
) -> ConceptExpander:
    groups = concept_groups or DEFAULT_CONCEPT_GROUPS
    vocab_size = len(tok)
    concept_ids: list[int] = []
    concept_names: list[str] = []
    token_sets: list[set[int]] = []
    for i, (name, terms) in enumerate(groups):
        ids: set[int] = set()
        for term in terms:
            for variant in (term, f" {term}", term.capitalize(), f" {term.capitalize()}"):
                for tid in _tokenize(tok, variant):
                    try:
                        piece = tok.decode([tid])
                    except Exception:
                        piece = ""
                    if _clean_decoded_piece(piece):
                        ids.add(int(tid))
        if not ids:
            raise RuntimeError(f"concept group {name} produced no token ids")
        concept_ids.append(vocab_size + 1000 + i)
        concept_names.append(name)
        token_sets.append(ids)
    return ConceptExpander(
        concept_ids=concept_ids,
        concept_names=concept_names,
        token_sets=token_sets,
        bytes_per_block=max(1, math.ceil(len(groups) / 8)),
    )


def append_concepts(ids: list[int], expander: ConceptExpander) -> list[int]:
    present = set(int(x) for x in ids)
    concepts = [
        cid for cid, token_set in zip(expander.concept_ids, expander.token_sets)
        if present.intersection(token_set)
    ]
    if not concepts:
        return ids
    return ids + concepts
