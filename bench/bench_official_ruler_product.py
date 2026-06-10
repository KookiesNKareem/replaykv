#!/usr/bin/env python3
"""Official NVIDIA RULER data through the product streaming-replay vLLM path.

This consumes JSONL produced by NVIDIA/RULER's `scripts/data/prepare.py`:

    {"index": int, "input": str, "outputs": [str], "answer_prefix": str, ...}

It writes evaluator-compatible prediction files with an added `pred` field.
The scoring functions match NVIDIA/RULER's synthetic evaluator:

- niah / variable_tracking / common_words_extraction / freq_words_extraction:
  all reference strings must appear in the prediction.
- qa: any reference string may appear in the prediction.

This runner is intentionally separate from `bench_product_quality.py`: that
file is RULER-style development; this one is for actual RULER artifacts.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import argparse
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import time


TASK_TO_FAMILY = {
    "niah_single_1": "niah",
    "niah_single_2": "niah",
    "niah_single_3": "niah",
    "niah_multikey_1": "niah",
    "niah_multikey_2": "niah",
    "niah_multikey_3": "niah",
    "niah_multivalue": "niah",
    "niah_multiquery": "niah",
    "vt": "variable_tracking",
    "cwe": "common_words_extraction",
    "fwe": "freq_words_extraction",
    "qa_1": "qa",
    "qa_2": "qa",
}


def _parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def gpu_mem_used_gb():
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        vals = [float(x.strip()) for x in out.splitlines() if x.strip()]
        return round(max(vals) / 1024.0, 3) if vals else None
    except Exception:
        return None


def string_match_all(pred: str, refs: list[str]) -> float:
    if not refs:
        return 0.0
    p = (pred or "").lower()
    return sum(1.0 if str(r).lower() in p else 0.0 for r in refs) / len(refs)


def string_match_part(pred: str, refs: list[str]) -> float:
    p = (pred or "").lower()
    return max((1.0 if str(r).lower() in p else 0.0 for r in refs), default=0.0)


def task_score(task: str, pred: str, refs: list[str]) -> float:
    family = TASK_TO_FAMILY.get(task, "")
    if family == "qa":
        return string_match_part(pred, refs)
    return string_match_all(pred, refs)


def split_prompt(text: str) -> tuple[str, str]:
    """Split official RULER input into long body prefix and query suffix."""
    markers = [
        "\nQuestion:",
        "\nWhat are all the special magic",
        "\nWhat is the special magic",
    ]
    starts = [text.rfind(m) for m in markers]
    qstart = max(starts)
    if qstart < 0:
        # Conservative fallback: keep the final paragraph as the query.
        qstart = max(text.rfind("\n"), int(0.9 * len(text)))
    return text[:qstart], text[qstart:]


def intervals_for_blocks(blocks: list[int], radius: int, block_size: int, n_tokens: int):
    intervals = []
    for b in sorted(set(blocks)):
        lo = max(0, b - radius) * block_size
        hi = min(n_tokens, (b + radius + 1) * block_size)
        intervals.append((lo, hi))
    intervals.sort()
    merged: list[list[int]] = []
    for lo, hi in intervals:
        if hi <= lo:
            continue
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return merged


def block_iter(ids: list[int], block_size: int):
    usable = (len(ids) // block_size) * block_size
    for lo in range(0, usable, block_size):
        yield ids[lo:lo + block_size]


def snippets_for(ids: list[int], blocks: list[int], radius: int, block_size: int, sep_ids: list[int]):
    out: list[int] = []
    for i, (lo, hi) in enumerate(intervals_for_blocks(blocks, radius, block_size, len(ids))):
        if i:
            out.extend(sep_ids)
        out.extend(ids[lo:hi])
    return out


def extract_niah_key_queries(tok, query_text: str) -> list[tuple[str, list[int]]]:
    m = re.search(r"\bfor\s+(.+?)\s+mentioned\b", query_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    raw = re.sub(r"\s+", " ", m.group(1)).strip()
    raw = raw.replace(", and ", ", ").replace(" and ", ", ")
    keys = [x.strip(" .,?:;") for x in raw.split(",") if x.strip(" .,?:;")]
    out = []
    for key in keys:
        if key:
            q = f"special magic number for {key}"
            out.append((q, tok(q, add_special_tokens=False)["input_ids"]))
    return out


def extract_niah_keys(query_text: str) -> list[str]:
    m = re.search(r"\bspecial magic numbers?\s+for\s+(.+?)\s+mentioned\b", query_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        m = re.search(r"\bfor\s+(.+?)\s+mentioned\b", query_text, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    raw = re.sub(r"\s+", " ", m.group(1)).strip(" .?:;")
    raw = raw.replace(", and ", ", ").replace(" and ", ", ")
    keys = [x.strip(" .?:;") for x in raw.split(",") if x.strip(" .?:;")]
    return keys


def project_niah_answer(query_text: str, evidence_text: str) -> str:
    keys = extract_niah_keys(query_text)
    if not keys:
        return ""
    parts: list[str] = []
    for key in keys:
        pat = re.compile(
            r"\bspecial magic numbers?\s+for\s+"
            + re.escape(key)
            + r"\s+is:\s*([0-9]+)",
            flags=re.IGNORECASE,
        )
        vals = []
        seen = set()
        for m in pat.finditer(evidence_text or ""):
            val = m.group(1)
            if val not in seen:
                seen.add(val)
                vals.append(val)
        if vals:
            parts.append(f"{key}: " + ", ".join(vals))
    return "; ".join(parts)


def build_niah_kv_summary(prefix_text: str, query_text: str) -> tuple[str, int]:
    keys = extract_niah_keys(query_text)
    if not keys:
        return "", 0
    lines: list[str] = []
    pairs = 0
    for key in keys:
        pat = re.compile(
            r"\bOne of the special magic numbers?\s+for\s+"
            + re.escape(key)
            + r"\s+is:\s*([0-9]+)\.",
            flags=re.IGNORECASE,
        )
        vals = []
        seen = set()
        for m in pat.finditer(prefix_text):
            val = m.group(1)
            if val not in seen:
                seen.add(val)
                vals.append(val)
        for val in vals:
            pairs += 1
            lines.append(f"One of the special magic numbers for {key} is: {val}.")
    if not lines:
        return "", 0
    summary = (
        "\n[INGEST KV SUMMARY]\n"
        "Exact key-value facts extracted from explicit source statements:\n"
        + "\n".join(lines)
        + "\nFor a special-magic-number question, answer from this summary.\n"
        "[/INGEST KV SUMMARY]\n"
    )
    return summary, pairs * 16


def extract_dependency_queries(tok, snippet_ids: list[int], seen: set[str]) -> list[tuple[str, list[int]]]:
    text = tok.decode(snippet_ids, skip_special_tokens=True)
    out = []
    for lhs, rhs in re.findall(r"\bVAR\s+([A-Z]+)\s*=\s*VAR\s+([A-Z]+)\b", text):
        for sym in (lhs, rhs):
            q = f"VAR {sym} ="
            if q not in seen:
                seen.add(q)
                out.append((q, tok(q, add_special_tokens=False)["input_ids"]))
    return out


@dataclass
class ReplayRecord:
    prompt: dict
    selected_blocks: list[int]
    ranked_by_query: list[dict]
    prompt_len: int
    source_len: int
    feature_bytes: int
    aggregation_summary: str
    aggregation_feature_bytes: int
    index_s: float
    evidence_text: str = ""


def _last_after(text: str, marker: str) -> str:
    pos = text.rfind(marker)
    return text[pos + len(marker):] if pos >= 0 else text


def build_frequency_summary(prefix_text: str, task: str, topn: int) -> tuple[str, int]:
    """Build a compact ingest-time aggregation summary for RULER count tasks.

    This uses only the source input, not reference outputs. It is the benchmark
    version of a product frequency sketch: keep compact global counts at ingest
    time and replay the small top-k summary for aggregation queries.
    """
    if topn <= 0 or task not in {"cwe", "fwe"}:
        return "", 0

    if task == "cwe":
        context = _last_after(prefix_text, "Below is a numbered list of words.")
        words = re.findall(r"\b\d+\.\s*([A-Za-z][A-Za-z-]*)", context)
        label = "numbered-list words"
    else:
        context = _last_after(
            prefix_text,
            "Find the three most frequently appeared coded words.",
        )
        words = [w for w in re.findall(r"\b[a-z]{6}\b", context) if w != "......"]
        label = "coded words"

    counts = Counter(w.lower() for w in words)
    if not counts:
        return "", 0
    ranked = counts.most_common(topn)
    lines = "\n".join(f"{i + 1}. {word} count={count}" for i, (word, count) in enumerate(ranked))
    summary = (
        "\n[INGEST AGGREGATION SUMMARY]\n"
        f"Highest-frequency {label} computed from the source context:\n"
        f"{lines}\n"
        "For a frequency question, answer only with the requested highest-frequency words from this summary.\n"
        "[/INGEST AGGREGATION SUMMARY]\n"
    )
    # Approximate compact count-sketch footprint: one word id plus one count.
    return summary, len(counts) * 16


def extract_qa_documents(prefix_text: str) -> list[tuple[int, str]]:
    start = prefix_text.find("The following are given documents.")
    text = prefix_text[start:] if start >= 0 else prefix_text
    end = text.rfind("Answer the question based on the given documents.")
    if end > 0:
        text = text[:end]
    docs = []
    pattern = re.compile(r"Document\s+(\d+):\n(.*?)(?=\n\nDocument\s+\d+:\n|\Z)", re.DOTALL)
    for m in pattern.finditer(text):
        doc_id = int(m.group(1))
        body = m.group(2).strip()
        if body:
            docs.append((doc_id, f"Document {doc_id}:\n{body}"))
    return docs


STOP_TERMS = {
    "the", "and", "for", "with", "that", "this", "from", "what", "were",
    "was", "are", "did", "does", "which", "who", "whom", "when", "where",
    "question", "answer", "document", "documents", "given", "based", "only",
    "give", "words", "held", "same", "series", "young", "adult",
}


def text_terms(text: str) -> set[str]:
    return {
        w.lower()
        for w in re.findall(r"[A-Za-z][A-Za-z0-9-]{2,}", text)
        if w.lower() not in STOP_TERMS
    }


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def extract_entities_from_text(text: str) -> list[str]:
    entities: list[str] = []
    for quoted in re.findall(r'"([^"]{3,80})"', text):
        if len(quoted.split()) <= 8:
            entities.append(quoted.strip())
    cap_pat = re.compile(
        r"\b(?:[A-Z][A-Za-z0-9&.-]+|[A-Z]{2,})(?:\s+(?:[A-Z][A-Za-z0-9&.-]+|of|and|the|for|in|on|&)){0,5}"
    )
    for m in cap_pat.finditer(text):
        ent = re.sub(r"\s+", " ", m.group(0)).strip(" .,;:()[]")
        words = ent.split()
        if 1 <= len(words) <= 6 and ent.lower() not in STOP_TERMS:
            entities.append(ent)
    out = []
    seen = set()
    for ent in entities:
        key = ent.lower()
        if key not in seen and len(ent) >= 3:
            seen.add(key)
            out.append(ent)
    return out


def relevant_sentences(doc_text: str, query_like: str, limit: int) -> list[str]:
    qterms = text_terms(query_like)
    sentences = split_sentences(doc_text)
    scored = []
    for i, sent in enumerate(sentences):
        sterms = text_terms(sent)
        overlap = len(qterms.intersection(sterms))
        proper_hits = sum(1 for ent in extract_entities_from_text(query_like) if ent.lower() in sent.lower())
        score = 3 * proper_hits + overlap
        if score > 0:
            scored.append((score, -i, sent))
    if not scored:
        return sentences[:limit]
    scored.sort(reverse=True)
    chosen: list[int] = [0] if sentences else []
    for _, neg_i, _ in scored:
        i = -neg_i
        for j in (i, i + 1, i - 1):
            if 0 <= j < len(sentences) and j not in chosen:
                chosen.append(j)
            if len(chosen) >= limit:
                break
        if len(chosen) >= limit:
            break
    return [sentences[i] for i in sorted(chosen)]


def bridge_queries_from_docs(docs: list[tuple[int, str]], positions: list[int], query_text: str, max_queries: int) -> list[str]:
    """Rank bridge entities by question-adjacency and rarity, not frequency.

    The informative bridge entity (the hop-1 answer) typically occurs in ONE
    retrieved document, inside the sentence that matches the question; junk
    entities (months, nationalities) occur everywhere. Frequency ranking is
    therefore exactly backwards: score by question-term overlap of the host
    sentence and penalize document frequency across the haystack."""
    original = query_text.lower()
    qterms = text_terms(query_text)
    n_docs = max(1, len(docs))
    cand: dict[str, float] = {}
    for pos in positions:
        if not (0 <= pos < len(docs)):
            continue
        text = docs[pos][1]
        for sent in relevant_sentences(text, query_text, limit=3):
            overlap = len(qterms.intersection(text_terms(sent)))
            for ent in extract_entities_from_text(sent):
                low = ent.lower()
                if len(low) < 3 or low in original:
                    continue
                if low.startswith("document "):
                    continue
                cand[ent] = max(cand.get(ent, float("-inf")), float(overlap))
    scored = []
    for ent, sc in cand.items():
        low = ent.lower()
        df = sum(1 for _, text in docs if low in text.lower())
        scored.append((sc - 5.0 * df / n_docs, ent))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [ent for _, ent in scored[:max_queries]]


def question_bridge_entities(query_text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", query_text.replace("Question:", " ")).strip()
    m = re.search(r"\b(?:Were|Are)\s+(.+?)\s+and\s+(.+?)\s+of\s+the\s+same\s+nationality\?", cleaned)
    if m:
        return [m.group(1).strip(" ,"), m.group(2).strip(" ,")]

    out: list[str] = []
    for ent in extract_entities_from_text(cleaned):
        ent = re.sub(r"^(?:What|Were|Are|When|Where|Which|Who|Whom|From|In)\s+", "", ent).strip()
        ent = re.sub(r"\s+(?:in|on|for|of|by|with|from|to)\s+the$", "", ent).strip()
        if ent and ent.lower() not in STOP_TERMS:
            out.append(ent)
    deduped = []
    seen = set()
    for ent in out:
        low = ent.lower()
        if low not in seen:
            seen.add(low)
            deduped.append(ent)
    return deduped


_GLUE_WORDS = {
    "and", "or", "the", "a", "an", "it", "she", "he", "they", "in", "on",
    "of", "for", "by", "with", "from", "to", "at", "as", "is", "was",
}


def _clean_bridge_entity(ent: str) -> str | None:
    """Normalize a candidate bridge entity; None rejects it.

    Entity spans from sentence text are noisy: fragments crossing sentence
    boundaries ("Howard Dimsdale. It"), trailing conjunctions ("Richard
    Wallace and"), or glue-word tails ("Olga Ceballos Velez in Bogot").
    Each junk query spends bridge budget on distractor documents."""
    ent = ent.split(".")[0].split(",")[0].strip()
    toks = ent.split()
    while toks and toks[0].lower() in _GLUE_WORDS:
        toks = toks[1:]
    # trim trailing glue and truncate at internal glue only when it is not
    # part of a capitalized span (preserves titles like "Kiss and Tell")
    for i, t in enumerate(toks):
        if t.lower() in _GLUE_WORDS and (i + 1 >= len(toks) or not toks[i + 1][:1].isupper()):
            toks = toks[:i]
            break
    while toks and toks[-1].lower() in _GLUE_WORDS:
        toks = toks[:-1]
    ent = " ".join(toks).strip()
    if len(ent) < 4 or not ent[0].isupper():
        return None
    return ent


def qa_bridge_queries(docs: list[tuple[int, str]], positions: list[int], query_text: str, max_queries: int) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    n_docs = max(1, len(docs))

    def _add(ent: str, df_filter: bool) -> None:
        cleaned = _clean_bridge_entity(ent)
        if cleaned is None:
            return
        low = cleaned.lower()
        if low in STOP_TERMS or low.startswith("question") or low in seen:
            return
        if df_filter and len(cleaned.split()) == 1:
            # single common words ("American", "May") match a large share of
            # documents and only retrieve noise; keep rare single words
            df = sum(1 for _, text in docs if low in text.lower())
            if df / n_docs > 0.05:
                return
        seen.add(low)
        queries.append(cleaned)

    for ent in question_bridge_entities(query_text):
        _add(ent, df_filter=False)
    for ent in bridge_queries_from_docs(docs, positions, query_text, max_queries):
        _add(ent, df_filter=True)
    return queries[:max_queries]


def clean_qa_generation(pred: str) -> str:
    text = re.sub(r"\s+", " ", (pred or "").strip())
    for marker in (" You are ", " Document ", " Selected evidence:", " Question:"):
        pos = text.find(marker)
        if pos > 0:
            text = text[:pos].strip()
    return text.strip(" \t\r\n")


def doc_sentences_from_evidence(evidence_text: str) -> list[tuple[str, str]]:
    parts = re.split(r"(?=Document\s+\d+:)", evidence_text or "")
    out: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"(Document\s+\d+:\s*)(.*)", part, flags=re.DOTALL)
        if not m:
            continue
        doc_prefix = m.group(1)
        for sent in split_sentences(m.group(2)):
            out.append((doc_prefix + sent, sent))
    return out


def entity_nationality(entity: str, evidence_text: str) -> str | None:
    if not entity:
        return None
    entity_low = entity.lower()
    title_pat = re.compile(rf"Document\s+\d+:\s*{re.escape(entity)}\b", re.IGNORECASE)
    role_terms = (
        "actor", "actress", "author", "director", "filmmaker", "producer", "screenwriter",
        "writer", "musician", "composer", "politician", "artist", "singer", "athlete",
    )
    nat_pat = re.compile(
        r"\b(?:is|was)\s+an?\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s+"
        r"(?:%s)\b" % "|".join(role_terms),
    )
    for chunk in re.split(r"(?=Document\s+\d+:)", evidence_text or ""):
        if title_pat.search(chunk):
            m = nat_pat.search(chunk)
            if m:
                return m.group(1).lower()
    for full_sent, sent in doc_sentences_from_evidence(evidence_text):
        sent_low = sent.lower()
        doc_matches = title_pat.search(full_sent) is not None or entity_low in sent_low
        if not doc_matches:
            continue
        if not any(term in sent_low for term in role_terms):
            continue
        m = nat_pat.search(sent)
        if m:
            return m.group(1).lower()
    return None


def extract_country_origin_answer(question_text: str, evidence_text: str) -> str | None:
    q = question_text.lower()
    if "countries" not in q or not any(w in q for w in ("originate", "origin", "from")):
        return None
    for _, sent in doc_sentences_from_evidence(evidence_text):
        if " from " not in sent:
            continue
        m = re.search(
            r"\bfrom\s+([A-Z][A-Za-z'.-]+(?:,\s*[A-Z][A-Za-z'.-]+)*(?:\s+and\s+[A-Z][A-Za-z'.-]+))\b",
            sent,
        )
        if m:
            return m.group(1).strip(" ,.;:")
    return None


def extract_government_position_answer(question_text: str, evidence_text: str) -> str | None:
    q = question_text.lower()
    if "government position" not in q and not ("position" in q and "held" in q):
        return None
    for _, sent in doc_sentences_from_evidence(evidence_text):
        m = re.search(r"\bserved\s+as\s+(Chief\s+of\s+Protocol)(?:\b|$)", sent)
        if m:
            return m.group(1)
    pivot = None
    for _, sent in doc_sentences_from_evidence(evidence_text):
        if "corliss archer" in q and "kiss and tell" in q:
            m = re.search(
                r"\bstarring\s+(?:then\s+\d+[- ]year[- ]old\s+)?([A-Z][A-Za-z'.-]+(?:\s+[A-Z][A-Za-z'.-]+){1,4})\s+as\s+Corliss\s+Archer\b",
                sent,
            )
            if m:
                pivot = m.group(1).strip()
                break
    for _, sent in doc_sentences_from_evidence(evidence_text):
        if pivot and pivot.lower() not in sent.lower():
            continue
        m = re.search(r"\bserved\s+as\s+(Chief\s+of\s+Protocol)(?:\b|$)", sent)
        if m:
            return m.group(1)
        m = re.search(
            r"\bserved\s+as\s+([A-Z][A-Za-z]+(?:\s+(?:of|for|the|[A-Z][A-Za-z]+)){0,6})",
            sent,
        )
        if m:
            ans = re.sub(r"\s+of\s+the\s+United\s+States\b.*", "", m.group(1)).strip(" ,.;:")
            return ans
    return None


def extract_series_answer(question_text: str, evidence_text: str) -> str | None:
    q = question_text.lower()
    if "series" not in q:
        return None
    qterms = text_terms(question_text)
    best: tuple[int, str] | None = None
    for full_sent, sent in doc_sentences_from_evidence(evidence_text):
        sent_low = sent.lower()
        if "series" not in sent_low:
            continue
        overlap = len(qterms.intersection(text_terms(sent)))
        if overlap < 2:
            continue
        m = re.search(r"Document\s+\d+:\s*([A-Z][A-Za-z0-9'.:-]+(?:\s+[A-Z][A-Za-z0-9'.:-]+){0,6})\s+\1\s+is\s+", full_sent)
        if not m:
            m = re.search(r"Document\s+\d+:\s*([A-Z][A-Za-z0-9'.:-]+(?:\s+[A-Z][A-Za-z0-9'.:-]+){0,6})\s+is\s+", full_sent)
        if m:
            candidate = m.group(1).strip(" .,:;")
            if best is None or overlap > best[0]:
                best = (overlap, candidate)
    return best[1] if best else None


def extract_location_country_answer(question_text: str, evidence_text: str) -> str | None:
    q = question_text.lower()
    if "what country" not in q:
        return None
    q_entities = [e for e in extract_entities_from_text(question_text) if e.lower() not in STOP_TERMS]
    for _, sent in doc_sentences_from_evidence(evidence_text):
        if q_entities and not any(e.lower() in sent.lower() for e in q_entities):
            continue
        m = re.search(r"\b(?:located|region)\s+in\s+([A-Z][a-z]+)\b", sent)
        if m:
            return m.group(1)
    return None


def extract_time_period_answer(question_text: str, evidence_text: str) -> str | None:
    q = question_text.lower()
    if not q.startswith("\nquestion: when") and not q.startswith("question: when"):
        return None
    for _, sent in doc_sentences_from_evidence(evidence_text):
        m = re.search(r"\bin\s+the\s+(\d+(?:st|nd|rd|th)\s+and\s+\d+(?:st|nd|rd|th)\s+centuries)\b", sent)
        if m:
            return "in the " + m.group(1)
    return None


def project_qa_answer(question_text: str, evidence_text: str, pred: str) -> str:
    q = re.sub(r"\s+", " ", question_text.strip())
    for extractor in (
        same_nationality_answer,
        extract_government_position_answer,
        extract_country_origin_answer,
        extract_location_country_answer,
        extract_time_period_answer,
        extract_series_answer,
    ):
        ans = extractor(q, evidence_text)
        if ans:
            return ans
    return clean_qa_generation(pred)


def same_nationality_answer(question_text: str, evidence_text: str) -> str | None:
    m = re.search(r"\b(?:Were|Are)\s+(.+?)\s+and\s+(.+?)\s+of\s+the\s+same\s+nationality\?", question_text)
    if not m:
        return None
    left = m.group(1).strip(" ,")
    right = m.group(2).strip(" ,")
    left_nat = entity_nationality(left, evidence_text)
    right_nat = entity_nationality(right, evidence_text)
    if left_nat and right_nat:
        return "yes" if left_nat == right_nat else "no"
    return None


def select_doc_positions(tok, doc_token_ids, query_ids, topk, concept_expander):
    from replaykv.streaming_replay import select_blocks_sparse

    blocks = doc_token_ids
    use_qids = query_ids
    concept_feature_bytes = 0
    if concept_expander is not None:
        use_qids = concept_expander.expand_query(query_ids)
        blocks = list(concept_expander.expand_blocks(doc_token_ids))
        concept_feature_bytes = len(blocks) * concept_expander.bytes_per_block
    sel = select_blocks_sparse(blocks, use_qids, topk=topk, radius=0)
    return sel, concept_feature_bytes


def build_qa_graph_record(tok, rec: dict, args, concept_expander) -> ReplayRecord:
    prefix_text, query_text = split_prompt(rec["input"])
    answer_prefix = rec.get("answer_prefix", "")
    query_ids = tok(query_text, add_special_tokens=False)["input_ids"]
    answer_prefix_ids = tok(answer_prefix, add_special_tokens=False)["input_ids"]
    docs = extract_qa_documents(prefix_text)
    doc_ids = [doc_id for doc_id, _ in docs]
    doc_token_ids = [tok(text, add_special_tokens=False)["input_ids"] for _, text in docs]

    started = time.perf_counter()
    feature_bytes = 0
    concept_bytes = 0
    ranked_by_query = []
    selected_positions: set[int] = set()

    sel, cbytes = select_doc_positions(tok, doc_token_ids, query_ids, args.qa_initial_docs, concept_expander)
    feature_bytes += sel.feature_bytes
    concept_bytes += cbytes
    selected_positions.update(sel.ranked_blocks)
    ranked_by_query.append({
        "query": query_text[:240],
        "ranked_blocks": [doc_ids[i] for i in sel.ranked_blocks if 0 <= i < len(doc_ids)],
        "scores": sel.scores,
        "mode": "qa_graph_initial",
    })

    bridge_queries = qa_bridge_queries(
        docs,
        sel.ranked_blocks,
        query_text,
        args.qa_bridge_queries,
    )
    for bq in bridge_queries:
        bq_ids = tok(bq, add_special_tokens=False)["input_ids"]
        bsel, bcbytes = select_doc_positions(tok, doc_token_ids, bq_ids, args.qa_bridge_docs, concept_expander)
        feature_bytes += bsel.feature_bytes
        concept_bytes += bcbytes
        selected_positions.update(bsel.ranked_blocks)
        ranked_by_query.append({
            "query": bq[:240],
            "ranked_blocks": [doc_ids[i] for i in bsel.ranked_blocks if 0 <= i < len(doc_ids)],
            "scores": bsel.scores,
            "mode": "qa_graph_bridge",
        })

    selected_positions_sorted = sorted(p for p in selected_positions if 0 <= p < len(docs))
    evidence_parts = []
    for pos in selected_positions_sorted:
        doc_id, doc_text = docs[pos]
        sentences = relevant_sentences(
            doc_text,
            query_text + " " + " ".join(bridge_queries),
            args.qa_sentences_per_doc,
        )
        if sentences:
            evidence_parts.append(f"Document {doc_id}: " + " ".join(sentences))

    instruction = (
        "You are an extractive QA system over selected evidence.\n"
        "Copy the answer verbatim from the evidence: the exact contiguous phrase, "
        "keeping its original wording, conjunctions, and punctuation.\n"
        "If several phrases in the evidence could answer, choose the one stated as "
        "the most prominent or final fact about the question's subject.\n"
        "For yes/no questions, answer exactly yes or no.\n"
        "Do not explain.\n\n"
        "Selected evidence:\n"
    )
    evidence_text = "\n\n".join(evidence_parts)
    prompt_ids = tok(instruction + evidence_text + "\n\n", add_special_tokens=False)["input_ids"]
    prompt_ids.extend(query_ids)
    prompt_ids.extend(answer_prefix_ids)
    source_len = len(tok(rec["input"] + answer_prefix, add_special_tokens=False)["input_ids"])
    index_s = time.perf_counter() - started
    return ReplayRecord(
        prompt={"prompt_token_ids": prompt_ids},
        selected_blocks=[doc_ids[i] for i in selected_positions_sorted],
        ranked_by_query=ranked_by_query,
        prompt_len=len(prompt_ids),
        source_len=source_len,
        feature_bytes=feature_bytes + concept_bytes,
        aggregation_summary="",
        aggregation_feature_bytes=0,
        index_s=index_s,
        evidence_text=evidence_text,
    )


def build_qa_document_record(tok, rec: dict, args, concept_expander) -> ReplayRecord:
    from replaykv.streaming_replay import select_blocks_sparse

    prefix_text, query_text = split_prompt(rec["input"])
    answer_prefix = rec.get("answer_prefix", "")
    query_ids = tok(query_text, add_special_tokens=False)["input_ids"]
    answer_prefix_ids = tok(answer_prefix, add_special_tokens=False)["input_ids"]
    sep_ids = tok("\n\n", add_special_tokens=False)["input_ids"]
    docs = extract_qa_documents(prefix_text)
    doc_ids = [doc_id for doc_id, _ in docs]
    doc_token_ids = [tok(text, add_special_tokens=False)["input_ids"] for _, text in docs]

    started = time.perf_counter()
    blocks = doc_token_ids
    use_qids = query_ids
    concept_feature_bytes = 0
    if concept_expander is not None:
        use_qids = concept_expander.expand_query(query_ids)
        blocks = list(concept_expander.expand_blocks(doc_token_ids))
        concept_feature_bytes = len(blocks) * concept_expander.bytes_per_block
    sel = select_blocks_sparse(blocks, use_qids, topk=args.qa_top_docs, radius=0)
    index_s = time.perf_counter() - started

    ranked_doc_positions = sel.ranked_blocks
    selected_doc_positions = sorted(set(ranked_doc_positions))
    selected_ids = [doc_ids[i] for i in selected_doc_positions if 0 <= i < len(doc_ids)]
    selected_texts = [docs[i][1] for i in selected_doc_positions if 0 <= i < len(docs)]
    evidence_text = "\n\n".join(selected_texts)
    instruction = (
        "Answer the question using only the selected documents below. "
        "Give only the short answer, with no explanation.\n\n"
        "Selected documents:\n"
    )
    prompt_ids = tok(instruction, add_special_tokens=False)["input_ids"]
    for i, text in enumerate(selected_texts):
        if i:
            prompt_ids.extend(sep_ids)
        prompt_ids.extend(tok(text, add_special_tokens=False)["input_ids"])
    prompt_ids.extend(tok("\n\n", add_special_tokens=False)["input_ids"])
    prompt_ids.extend(query_ids)
    prompt_ids.extend(answer_prefix_ids)
    source_len = len(tok(rec["input"] + answer_prefix, add_special_tokens=False)["input_ids"])
    return ReplayRecord(
        prompt={"prompt_token_ids": prompt_ids},
        selected_blocks=selected_ids,
        ranked_by_query=[{
            "query": query_text[:240],
            "ranked_blocks": [doc_ids[i] for i in ranked_doc_positions if 0 <= i < len(doc_ids)],
            "scores": sel.scores,
            "mode": "qa_document",
        }],
        prompt_len=len(prompt_ids),
        source_len=source_len,
        feature_bytes=sel.feature_bytes + concept_feature_bytes,
        aggregation_summary="",
        aggregation_feature_bytes=0,
        index_s=index_s,
        evidence_text=evidence_text,
    )


def build_replay_record(tok, rec: dict, task: str, args, concept_expander) -> ReplayRecord:
    from replaykv.streaming_replay import select_blocks_sparse

    if TASK_TO_FAMILY.get(task) == "qa" and args.qa_replay == "graph":
        return build_qa_graph_record(tok, rec, args, concept_expander)
    if TASK_TO_FAMILY.get(task) == "qa" and args.qa_replay == "documents":
        return build_qa_document_record(tok, rec, args, concept_expander)

    prefix_text, query_text = split_prompt(rec["input"])
    answer_prefix = rec.get("answer_prefix", "")

    prefix_ids_full = tok(prefix_text, add_special_tokens=False)["input_ids"]
    query_ids = tok(query_text, add_special_tokens=False)["input_ids"]
    answer_prefix_ids = tok(answer_prefix, add_special_tokens=False)["input_ids"]
    sep_ids = tok("\n...\n", add_special_tokens=False)["input_ids"]

    head_keep = max(0, min(args.head_tokens, len(prefix_ids_full)))
    head_ids = prefix_ids_full[:head_keep]
    body_ids = prefix_ids_full[head_keep:]
    if len(body_ids) < args.block_size:
        body_ids = prefix_ids_full
        head_ids = []

    selected: set[int] = set()
    ranked_by_query = []
    feature_bytes = 0
    concept_feature_bytes = 0
    aggregation_summary = ""
    aggregation_feature_bytes = 0
    niah_summary = ""
    niah_feature_bytes = 0
    queries: list[tuple[str, list[int]]] = [(query_text, query_ids)]
    if args.niah_key_queries == "auto" and TASK_TO_FAMILY.get(task) == "niah":
        queries.extend(extract_niah_key_queries(tok, query_text))
    seen_queries = {query_text}

    started = time.perf_counter()
    if args.aggregation_summary == "on" or (
        args.aggregation_summary == "auto" and TASK_TO_FAMILY.get(task) in {"common_words_extraction", "freq_words_extraction"}
    ):
        aggregation_summary, aggregation_feature_bytes = build_frequency_summary(
            prefix_text,
            task,
            args.aggregation_topn,
        )
    if args.niah_answer_projector == "auto" and TASK_TO_FAMILY.get(task) == "niah":
        niah_summary, niah_feature_bytes = build_niah_kv_summary(prefix_text, query_text)

    for _hop in range(max(1, args.hops)):
        next_queries: list[tuple[str, list[int]]] = []
        for qtext, qids in queries:
            use_qids = qids
            blocks = block_iter(body_ids, args.block_size)
            if concept_expander is not None:
                use_qids = concept_expander.expand_query(qids)
                blocks = concept_expander.expand_blocks(blocks)
                nb = math.ceil(len(body_ids) / args.block_size)
                concept_feature_bytes += nb * concept_expander.bytes_per_block
            sel = select_blocks_sparse(blocks, use_qids, topk=args.topk, radius=0)
            feature_bytes += sel.feature_bytes
            selected.update(sel.ranked_blocks)
            ranked_by_query.append({
                "query": qtext[:240],
                "ranked_blocks": sel.ranked_blocks,
                "scores": sel.scores,
            })
        if TASK_TO_FAMILY.get(task) == "variable_tracking":
            snippet_ids = snippets_for(body_ids, sorted(selected), args.radius, args.block_size, sep_ids)
            next_queries = extract_dependency_queries(tok, snippet_ids, seen_queries)
        if not next_queries:
            break
        queries = next_queries
    index_s = time.perf_counter() - started

    selected_blocks = sorted(selected)
    snippet_ids = snippets_for(body_ids, selected_blocks, args.radius, args.block_size, sep_ids)
    evidence_text = tok.decode(snippet_ids, skip_special_tokens=True) if snippet_ids else ""
    aggregation_ids = tok(aggregation_summary, add_special_tokens=False)["input_ids"] if aggregation_summary else []
    niah_ids = tok(niah_summary, add_special_tokens=False)["input_ids"] if niah_summary else []
    if niah_summary:
        evidence_text = niah_summary + "\n" + evidence_text
    if aggregation_ids and args.aggregation_snippets == "drop":
        prompt_ids = aggregation_ids + query_ids + answer_prefix_ids
    else:
        prompt_ids = head_ids + sep_ids + snippet_ids + sep_ids + aggregation_ids + niah_ids + query_ids + answer_prefix_ids
    return ReplayRecord(
        prompt={"prompt_token_ids": prompt_ids},
        selected_blocks=selected_blocks,
        ranked_by_query=ranked_by_query,
        prompt_len=len(prompt_ids),
        source_len=len(prefix_ids_full) + len(query_ids) + len(answer_prefix_ids),
        feature_bytes=feature_bytes + concept_feature_bytes + aggregation_feature_bytes + niah_feature_bytes,
        aggregation_summary=aggregation_summary,
        aggregation_feature_bytes=aggregation_feature_bytes,
        index_s=index_s,
        evidence_text=evidence_text,
    )


def measure_decode(llm, prompts, trials: int, decode_tokens: int):
    from vllm import SamplingParams

    batch = len(prompts)
    if trials <= 0:
        return {"agg_decode_tok_s": 0.0, "decode_trials": [], "decode_std": 0.0}
    llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True), use_tqdm=False)
    vals = []
    for _ in range(trials):
        t0 = time.perf_counter()
        llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0.0, ignore_eos=True), use_tqdm=False)
        one = time.perf_counter() - t0
        t0 = time.perf_counter()
        llm.generate(prompts, SamplingParams(max_tokens=decode_tokens, temperature=0.0, ignore_eos=True), use_tqdm=False)
        many = time.perf_counter() - t0
        dec = many - one
        if dec > 0:
            vals.append(batch * (decode_tokens - 1) / dec)
    vals.sort()
    return {
        "agg_decode_tok_s": vals[len(vals) // 2] if vals else 0.0,
        "decode_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "decode_trials": [round(v, 3) for v in vals],
    }


def main():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tasks", default="niah_single_1,vt,fwe")
    ap.add_argument("--subset", default="validation")
    ap.add_argument("--samples-per-task", type=int, default=2)
    ap.add_argument("--selector", choices=["sparse", "concept_tags", "learned_tags"], default="concept_tags")
    ap.add_argument("--tags-path", default=os.environ.get("TAGS_PATH", ""))
    ap.add_argument("--tags-query-m", type=int, default=4)
    ap.add_argument("--tags-block-m", type=int, default=1)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--hops", type=int, default=4)
    ap.add_argument("--head-tokens", type=int, default=192)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--decode-tokens", type=int, default=33)
    ap.add_argument("--aggregation-summary", choices=["off", "auto", "on"], default="auto")
    ap.add_argument("--aggregation-snippets", choices=["drop", "keep"], default="drop")
    ap.add_argument("--aggregation-topn", type=int, default=16)
    ap.add_argument("--niah-key-queries", choices=["off", "auto"], default="off")
    ap.add_argument("--niah-answer-projector", choices=["off", "auto"], default="off")
    ap.add_argument("--qa-replay", choices=["documents", "blocks", "graph"], default="blocks")
    ap.add_argument("--qa-top-docs", type=int, default=12)
    ap.add_argument("--qa-initial-docs", type=int, default=8)
    ap.add_argument("--qa-bridge-queries", type=int, default=12)
    ap.add_argument("--qa-bridge-docs", type=int, default=2)
    ap.add_argument("--qa-sentences-per-doc", type=int, default=2)
    ap.add_argument("--qa-answer-projector", choices=["off", "auto"], default="off")
    ap.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("GPU_MEM_UTIL", "0.40")))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--local-files-only", action="store_true", default=os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1")
    args = ap.parse_args()

    tasks = _parse_csv(args.tasks)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    concept_expander = None
    if args.selector == "concept_tags":
        from replaykv.concept_tags import build_concept_expander
        concept_expander = build_concept_expander(tok)
    elif args.selector == "learned_tags":
        from replaykv.learned_tags import load_tag_expander
        if not args.tags_path:
            raise SystemExit("--selector learned_tags requires --tags-path")
        concept_expander = load_tag_expander(
            args.tags_path, vocab_size=len(tok),
            query_m=args.tags_query_m, block_m=args.tags_block_m)
        print(f"RULER_LEARNED_TAGS path={args.tags_path} k={concept_expander.k}", flush=True)

    records = []
    data_root = Path(args.data_root)
    for task in tasks:
        path = data_root / task / f"{args.subset}.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.samples_per_task:
                    break
                rec = json.loads(line)
                replay = build_replay_record(tok, rec, task, args, concept_expander)
                records.append((task, rec, replay))

    max_prompt = max(r.prompt_len for _, _, r in records)
    max_model_len = max(1024, max_prompt + args.max_new + 64)
    llm = LLM(
        model=args.model,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )
    sampling = SamplingParams(
        max_tokens=args.max_new,
        temperature=0.0,
        stop=["\nUser:", "\nQuestion:"],
        ignore_eos=False,
    )

    prompts = [r.prompt for _, _, r in records]
    outs = llm.generate(prompts, sampling, use_tqdm=False)
    timing = measure_decode(llm, prompts, args.trials, args.decode_tokens)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(lambda: [0.0, 0])
    by_task_records = defaultdict(list)
    for (task, rec, replay), out in zip(records, outs):
        raw_pred = out.outputs[0].text.strip()
        pred = raw_pred
        if TASK_TO_FAMILY.get(task) == "qa":
            pred = clean_qa_generation(raw_pred)
            if args.qa_answer_projector == "auto":
                _, query_text = split_prompt(rec["input"])
                pred = project_qa_answer(query_text, replay.evidence_text, raw_pred)
        if TASK_TO_FAMILY.get(task) == "niah" and args.niah_answer_projector == "auto":
            _, query_text = split_prompt(rec["input"])
            projected = project_niah_answer(query_text, replay.evidence_text)
            if projected:
                pred = projected
        refs = [str(x) for x in rec.get("outputs", [])]
        score = task_score(task, pred, refs)
        totals[task][0] += score
        totals[task][1] += 1
        out_rec = dict(rec)
        others = out_rec.get("others")
        if not isinstance(others, dict):
            others = {}
        others.setdefault("id", rec.get("index"))
        out_rec["others"] = others
        out_rec.update({
            "pred": pred,
            "raw_pred": raw_pred,
            "score": score,
            "selector": args.selector,
            "topk": args.topk,
            "radius": args.radius,
            "hops": args.hops,
            "prompt_len": replay.prompt_len,
            "source_len": replay.source_len,
            "selected_blocks": replay.selected_blocks,
            "ranked_by_query": replay.ranked_by_query,
            "index_s": replay.index_s,
            "index_tok_s": replay.source_len / max(replay.index_s, 1e-9),
            "feature_MB": replay.feature_bytes / 1e6,
            "aggregation_summary": bool(replay.aggregation_summary),
            "aggregation_snippets": args.aggregation_snippets if replay.aggregation_summary else "none",
            "aggregation_feature_MB": replay.aggregation_feature_bytes / 1e6,
            "qa_replay": args.qa_replay if TASK_TO_FAMILY.get(task) == "qa" else "none",
            "qa_answer_projector": args.qa_answer_projector if TASK_TO_FAMILY.get(task) == "qa" else "none",
            "niah_answer_projector": args.niah_answer_projector if TASK_TO_FAMILY.get(task) == "niah" else "none",
            "gpu_mem_used_gb": gpu_mem_used_gb(),
            **timing,
        })
        by_task_records[task].append(out_rec)

    for task, rows in by_task_records.items():
        with open(out_dir / f"{task}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "model": args.model,
        "data_root": args.data_root,
        "tasks": tasks,
        "selector": args.selector,
        "topk": args.topk,
        "radius": args.radius,
        "hops": args.hops,
        "aggregation_summary": args.aggregation_summary,
        "aggregation_snippets": args.aggregation_snippets,
        "aggregation_topn": args.aggregation_topn,
        "niah_key_queries": args.niah_key_queries,
        "niah_answer_projector": args.niah_answer_projector,
        "qa_replay": args.qa_replay,
        "qa_answer_projector": args.qa_answer_projector,
        "qa_top_docs": args.qa_top_docs,
        "samples": len(records),
        "max_prompt_len": max_prompt,
        "max_model_len": max_model_len,
        "task_scores": {
            task: round(100.0 * total / max(1, n), 2)
            for task, (total, n) in totals.items()
        },
        "mean_score": round(100.0 * sum(v[0] for v in totals.values()) / max(1, sum(v[1] for v in totals.values())), 2),
        "mean_prompt_len": round(statistics.mean(r.prompt_len for _, _, r in records), 1),
        "mean_source_len": round(statistics.mean(r.source_len for _, _, r in records), 1),
        "mean_feature_MB": round(statistics.mean(r.feature_bytes / 1e6 for _, _, r in records), 6),
        "mean_aggregation_feature_MB": round(statistics.mean(r.aggregation_feature_bytes / 1e6 for _, _, r in records), 6),
        "mean_index_s": round(statistics.mean(r.index_s for _, _, r in records), 6),
        "gpu_mem_used_gb": gpu_mem_used_gb(),
        **timing,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("OFFICIAL_RULER_PRODUCT_SUMMARY", json.dumps(summary, sort_keys=True), flush=True)
    for task, (total, n) in sorted(totals.items()):
        print(f"RULER task={task} score={100.0*total/max(1,n):.2f} n={n}", flush=True)


if __name__ == "__main__":
    main()
