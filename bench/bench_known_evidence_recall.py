#!/usr/bin/env python3
"""Known-evidence selector benchmark for the product replay path.

This is the cheap quality-development loop we should use before LongBench-v2
or vLLM generation. It creates synthetic/RULER-style long contexts where the
supporting block(s) are known, runs the streaming selector, and reports:

- support-block recall under a fixed replay budget
- distractor pressure around the selected blocks
- selected prompt-token budget implied by top-k/radius
- selector indexing throughput and feature footprint

It deliberately does not call vLLM. Passing this harness is a prerequisite for
spending GPU time on product-quality runs.
"""
from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import math
import os
import statistics
import time

from replaykv.concept_tags import build_concept_expander


FILLER_TEXT = (
    "The weather today is mild and the river flows gently past the old mill. "
    "A small archive records routine observations about buildings, roads, and supplies. "
)


@dataclass
class EvidenceCase:
    task: str
    ctx_tokens: int
    block_size: int
    query_ids: list[int]
    support_blocks: set[int]
    evidence_texts: list[str]
    filler_ids: list[int]
    evidence_ids: list[list[int]]
    insert_positions: list[int]

    @property
    def body_len(self) -> int:
        return self.ctx_tokens

    def _extend_filler(self, out: list[int], start: int, length: int):
        nf = len(self.filler_ids)
        for j in range(length):
            out.append(self.filler_ids[(start + j) % nf])

    def body_slice(self, lo: int, hi: int) -> list[int]:
        lo = max(0, lo)
        hi = min(hi, self.body_len)
        if hi <= lo:
            return []
        spans = sorted(
            (pos, pos + len(ids), ids)
            for pos, ids in zip(self.insert_positions, self.evidence_ids)
        )
        out: list[int] = []
        cur = lo
        for ev_lo, ev_hi, ids in spans:
            if ev_hi <= lo or ev_lo >= hi:
                continue
            fill_hi = min(hi, ev_lo)
            if fill_hi > cur:
                self._extend_filler(out, cur, fill_hi - cur)
                cur = fill_hi
            take_lo = max(lo, ev_lo)
            take_hi = min(hi, ev_hi)
            if take_hi > take_lo:
                out.extend(ids[take_lo - ev_lo:take_hi - ev_lo])
                cur = take_hi
        if hi > cur:
            self._extend_filler(out, cur, hi - cur)
        return out

    def block_iter(self):
        usable = (self.body_len // self.block_size) * self.block_size
        for lo in range(0, usable, self.block_size):
            yield self.body_slice(lo, lo + self.block_size)


def _parse_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def _parse_floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def _tokenize(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def _base(tok, ctx_tokens: int, block_size: int, task: str, query: str,
          evidence: list[str], depths: list[float]) -> EvidenceCase:
    filler_ids = _tokenize(tok, FILLER_TEXT)
    evidence_ids = [_tokenize(tok, x) for x in evidence]
    if not filler_ids:
        raise RuntimeError("empty filler tokenization")
    usable = (ctx_tokens // block_size) * block_size
    positions = []
    for depth, ids in zip(depths, evidence_ids):
        pos = int(depth * max(1, usable - len(ids)))
        positions.append(max(0, min(pos, usable - len(ids))))
    support = {
        max(0, min((usable // block_size) - 1, pos // block_size))
        for pos in positions
    }
    return EvidenceCase(
        task=task,
        ctx_tokens=usable,
        block_size=block_size,
        query_ids=_tokenize(tok, query),
        support_blocks=support,
        evidence_texts=evidence,
        filler_ids=filler_ids,
        evidence_ids=evidence_ids,
        insert_positions=positions,
    )


def make_case(tok, task: str, ctx_tokens: int, block_size: int,
              depth: float, case_idx: int, supports: int) -> EvidenceCase:
    if task == "single_needle":
        code = f"{7000 + 137 * case_idx:04d}"
        return _base(
            tok,
            ctx_tokens,
            block_size,
            task,
            "Question: What is the magic access code? Answer with just the number.",
            [f"The magic access code is {code}. Remember it carefully."],
            [depth],
        )
    if task == "alias_needle":
        code = f"{7100 + 137 * case_idx:04d}"
        return _base(
            tok,
            ctx_tokens,
            block_size,
            task,
            "Question: What is the magic access code? Answer with just the number.",
            [f"The amber credential reads {code}. Preserve this identifier for later use."],
            [depth],
        )
    if task == "distractor_needle":
        code = f"{7200 + 137 * case_idx:04d}"
        distractors = [
            f"The magic access code is {8000 + case_idx + j}. This entry is obsolete and should be ignored."
            for j in range(max(1, supports - 1))
        ]
        evidence = distractors + [
            f"The current magic access code is {code}. This current entry overrides all obsolete entries."
        ]
        depths = [min(0.98, max(0.02, depth + 0.06 * (j - len(evidence) // 2))) for j in range(len(evidence))]
        return _base(
            tok,
            ctx_tokens,
            block_size,
            task,
            "Question: What is the current magic access code? Answer with just the number.",
            evidence,
            depths,
        )
    if task == "multi_key":
        key = f"project_{case_idx % 17}"
        value = f"token_{9000 + case_idx}"
        distractors = [
            f"The audit key project_{(case_idx + j + 1) % 17} has value token_{8000 + case_idx + j}."
            for j in range(max(0, supports - 1))
        ]
        evidence = distractors + [f"The audit key {key} has value {value}."]
        depths = [min(0.98, max(0.02, depth + 0.07 * (j - len(evidence) // 2))) for j in range(len(evidence))]
        return _base(
            tok,
            ctx_tokens,
            block_size,
            task,
            f"Question: What value is assigned to audit key {key}?",
            evidence,
            depths,
        )
    if task == "variable_tracking":
        var = f"var_{case_idx % 23}"
        old = f"value_{3000 + case_idx}"
        final = f"value_{6000 + case_idx}"
        evidence = [
            f"Initial assignment: {var} is set to {old}.",
            f"Final update: after review, {var} is set to {final}.",
        ]
        depths = [max(0.02, depth - 0.18), min(0.98, depth + 0.18)]
        return _base(
            tok,
            ctx_tokens,
            block_size,
            task,
            f"Question: What is the final value of {var}?",
            evidence,
            depths,
        )
    if task == "multi_support_qa":
        subject = f"compound_{case_idx % 31}"
        mechanism = f"mechanism_{4000 + case_idx}"
        outcome = f"outcome_{5000 + case_idx}"
        evidence = [
            f"Record A: {subject} uses {mechanism} during the preparation phase.",
            f"Record B: the observed result of {mechanism} is {outcome}.",
        ]
        depths = [max(0.02, depth - 0.22), min(0.98, depth + 0.22)]
        return _base(
            tok,
            ctx_tokens,
            block_size,
            task,
            f"Question: What outcome is linked to {subject}'s preparation mechanism?",
            evidence,
            depths,
        )
    raise ValueError(f"unknown task: {task}")


def diversify_ranked_blocks(
    ranked_blocks: list[int],
    scores: list[float],
    topk: int,
    radius: int,
    nms_gap: int,
) -> tuple[list[int], list[float]]:
    gap = nms_gap if nms_gap >= 0 else max(0, 2 * radius)
    chosen: list[int] = []
    chosen_scores: list[float] = []
    for block, score in zip(ranked_blocks, scores):
        if all(abs(block - old) > gap for old in chosen):
            chosen.append(block)
            chosen_scores.append(score)
            if len(chosen) >= topk:
                return chosen, chosen_scores
    seen = set(chosen)
    for block, score in zip(ranked_blocks, scores):
        if block in seen:
            continue
        chosen.append(block)
        chosen_scores.append(score)
        seen.add(block)
        if len(chosen) >= topk:
            break
    return chosen, chosen_scores


def replay_tokens_for(blocks: list[int], radius: int, block_size: int, ctx_tokens: int) -> int:
    intervals = []
    for b in sorted(set(blocks)):
        lo = max(0, b - radius) * block_size
        hi = min(ctx_tokens, (b + radius + 1) * block_size)
        intervals.append((lo, hi))
    merged: list[list[int]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return sum(hi - lo for lo, hi in merged)


def rank_case(case: EvidenceCase, max_candidates: int, selector: str,
              concept_expander, rag_ranker, tok) -> dict:
    """Rank all blocks once per case; topk/radius slicing happens per config."""
    from replaykv.streaming_replay import select_blocks_sparse

    concept_feature_bytes = 0
    concept_names: list[str] = []
    if selector in ("sparse", "concept_tags", "learned_tags"):
        query_ids = case.query_ids
        block_iter = case.block_iter()
        if selector in ("concept_tags", "learned_tags"):
            if concept_expander is None:
                raise RuntimeError(f"{selector} selector requires an expander")
            query_ids = concept_expander.expand_query(case.query_ids)
            block_iter = concept_expander.expand_blocks(block_iter)
            nb = math.ceil(case.ctx_tokens / case.block_size)
            concept_feature_bytes = nb * concept_expander.bytes_per_block
            concept_names = getattr(concept_expander, "concept_names", [])
        t0 = time.perf_counter()
        sel = select_blocks_sparse(
            block_iter,
            query_ids,
            topk=max_candidates,
            radius=0,
        )
        index_s = time.perf_counter() - t0
        ranked_all, scores_all = sel.ranked_blocks, sel.scores
        feature_bytes = sel.feature_bytes
    elif selector in ("bm25", "embed"):
        if rag_ranker is None:
            raise RuntimeError(f"{selector} selector requires a RAG ranker")
        block_texts = [tok.decode(list(block)) for block in case.block_iter()]
        query_text = tok.decode(case.query_ids)
        res = rag_ranker.rank(block_texts, query_text)
        ranked_all, scores_all = res.ranked, res.scores
        index_s, feature_bytes = res.index_s, res.index_bytes
    else:
        raise ValueError(f"unknown selector: {selector}")
    return {
        "ranked_all": ranked_all,
        "scores_all": scores_all,
        "index_s": index_s,
        "feature_bytes": feature_bytes,
        "concept_feature_bytes": concept_feature_bytes,
        "concept_names": concept_names,
    }


def score_case(case: EvidenceCase, ranking: dict, topk: int, radius: int,
               candidate_multiplier: int, nms_gap: int, selector: str):
    candidate_topk = max(topk, topk * max(1, candidate_multiplier))
    index_s = ranking["index_s"]
    concept_feature_bytes = ranking["concept_feature_bytes"]
    concept_names = ranking["concept_names"]
    ranked, scores = diversify_ranked_blocks(
        ranking["ranked_all"][:candidate_topk],
        ranking["scores_all"][:candidate_topk],
        topk,
        radius,
        nms_gap,
    )
    support_hits = {
        s for s in case.support_blocks
        if any(abs(b - s) <= radius for b in ranked)
    }
    replay_tokens = replay_tokens_for(ranked, radius, case.block_size, case.ctx_tokens)
    selected_window_blocks = {
        bb
        for b in ranked
        for bb in range(max(0, b - radius), min(math.ceil(case.ctx_tokens / case.block_size), b + radius + 1))
    }
    distractor_blocks = max(0, len(selected_window_blocks - case.support_blocks))
    return {
        "task": case.task,
        "ctx": case.ctx_tokens,
        "topk": topk,
        "radius": radius,
        "support_blocks": sorted(case.support_blocks),
        "ranked_blocks": ranked,
        "scores": scores,
        "support_hit": len(support_hits),
        "support_total": len(case.support_blocks),
        "all_support_hit": len(support_hits) == len(case.support_blocks),
        "any_support_hit": bool(support_hits),
        "distractor_blocks": distractor_blocks,
        "replay_tokens": replay_tokens,
        "replay_ratio": replay_tokens / max(1, case.ctx_tokens),
        "index_s": index_s,
        "index_tok_s": case.ctx_tokens / max(index_s, 1e-9),
        "candidate_topk": candidate_topk,
        "selector": selector,
        "concept_feature_MB": concept_feature_bytes / 1e6,
        "concept_names": concept_names,
        "feature_MB": (ranking["feature_bytes"] + concept_feature_bytes) / 1e6,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--tasks", default=os.environ.get(
        "EVIDENCE_TASKS",
        (
            "single_needle,multi_key,variable_tracking,multi_support_qa,"
            "alias_needle,distractor_needle"
        ),
    ))
    ap.add_argument("--ctxs", default=os.environ.get("EVIDENCE_CTXS", "32768,131072"))
    ap.add_argument("--depths", default=os.environ.get("EVIDENCE_DEPTHS", "0.1,0.5,0.9"))
    ap.add_argument("--topks", default=os.environ.get("EVIDENCE_TOPKS", "8,16,32,48"))
    ap.add_argument("--radii", default=os.environ.get("EVIDENCE_RADII", "1,4"))
    ap.add_argument("--cases-per-depth", type=int, default=int(os.environ.get("EVIDENCE_CASES_PER_DEPTH", "3")))
    ap.add_argument("--supports", type=int, default=int(os.environ.get("EVIDENCE_SUPPORTS", "4")))
    ap.add_argument("--candidate-multiplier", type=int, default=int(os.environ.get("EVIDENCE_CANDIDATE_MULTIPLIER", "6")))
    ap.add_argument("--nms-gap", type=int, default=int(os.environ.get("EVIDENCE_NMS_GAP", "-1")))
    ap.add_argument("--block-size", type=int, default=int(os.environ.get("BLOCK_SIZE", "64")))
    ap.add_argument("--selector", choices=["sparse", "concept_tags", "learned_tags", "bm25", "embed"],
                    default=os.environ.get("EVIDENCE_SELECTOR", "sparse"),
                    help=("Selector development mode. sparse is the locked lexical baseline; "
                          "concept_tags adds a tiny prototype ingest-time concept channel; "
                          "learned_tags uses embedding-cluster tags (see replaykv/learned_tags.py); "
                          "bm25/embed are external-retrieval (RAG) baselines over the same blocks."))
    ap.add_argument("--tags-path", default=os.environ.get("EVIDENCE_TAGS_PATH", ""),
                    help="token->cluster tags .pt for --selector learned_tags")
    ap.add_argument("--tags-query-m", type=int, default=int(os.environ.get("EVIDENCE_TAGS_QUERY_M", "4")))
    ap.add_argument("--tags-block-m", type=int, default=int(os.environ.get("EVIDENCE_TAGS_BLOCK_M", "1")))
    ap.add_argument("--embed-model", default=os.environ.get("EVIDENCE_EMBED_MODEL", "BAAI/bge-large-en-v1.5"))
    ap.add_argument("--embed-batch-size", type=int, default=int(os.environ.get("EVIDENCE_EMBED_BATCH", "256")))
    ap.add_argument("--embed-device", default=os.environ.get("EVIDENCE_EMBED_DEVICE", "cuda"))
    ap.add_argument("--out", default=os.environ.get("EVIDENCE_OUT", "results/product_quality/known_evidence_recall.jsonl"))
    ap.add_argument("--local-files-only", action="store_true", default=os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    ctxs = _parse_ints(args.ctxs)
    depths = _parse_floats(args.depths)
    topks = _parse_ints(args.topks)
    radii = _parse_ints(args.radii)
    concept_expander = build_concept_expander(tok) if args.selector == "concept_tags" else None
    if args.selector == "learned_tags":
        if not args.tags_path:
            raise SystemExit("--selector learned_tags requires --tags-path")
        from replaykv.learned_tags import load_tag_expander

        concept_expander = load_tag_expander(
            args.tags_path, vocab_size=len(tok), block_size=args.block_size,
            query_m=args.tags_query_m, block_m=args.tags_block_m)
        print(f"LEARNED_TAGS path={args.tags_path} k={concept_expander.k} "
              f"query_m={args.tags_query_m} block_m={args.tags_block_m}", flush=True)
    rag_ranker = None
    if args.selector in ("bm25", "embed"):
        from replaykv.rag_baselines import build_ranker

        rag_ranker = build_ranker(
            args.selector,
            embed_model=args.embed_model,
            embed_batch_size=args.embed_batch_size,
            embed_device=args.embed_device,
        )
    max_candidates = max(max(topks), max(topks) * max(1, args.candidate_multiplier))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    case_idx = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for task in tasks:
            for ctx in ctxs:
                for depth in depths:
                    for rep in range(args.cases_per_depth):
                        case = make_case(
                            tok,
                            task,
                            ctx,
                            args.block_size,
                            depth,
                            case_idx,
                            args.supports,
                        )
                        case_idx += 1
                        ranking = rank_case(
                            case,
                            max_candidates,
                            args.selector,
                            concept_expander,
                            rag_ranker,
                            tok,
                        )
                        for topk in topks:
                            for radius in radii:
                                rec = score_case(
                                    case,
                                    ranking,
                                    topk,
                                    radius,
                                    args.candidate_multiplier,
                                    args.nms_gap,
                                    args.selector,
                                )
                                rec.update({"depth": depth, "rep": rep})
                                rows.append(rec)
                                f.write(json.dumps(rec, sort_keys=True) + "\n")

    print(f"KNOWN_EVIDENCE_RECALL rows={len(rows)} out={args.out}")
    by = {}
    for task in tasks:
        task_rows = [r for r in rows if r["task"] == task]
        if not task_rows:
            continue
        by[task] = task_rows
        any_rate = sum(r["any_support_hit"] for r in task_rows) / len(task_rows)
        all_rate = sum(r["all_support_hit"] for r in task_rows) / len(task_rows)
        avg_replay = statistics.mean(r["replay_tokens"] for r in task_rows)
        avg_ratio = statistics.mean(r["replay_ratio"] for r in task_rows)
        avg_distr = statistics.mean(r["distractor_blocks"] for r in task_rows)
        avg_index_s = statistics.mean(r["index_s"] for r in task_rows)
        avg_feat_mb = statistics.mean(r["feature_MB"] for r in task_rows)
        print(
            f"TASK {task} any={100*any_rate:.2f} all={100*all_rate:.2f} "
            f"replay_tokens={avg_replay:.0f} replay_ratio={100*avg_ratio:.2f}% "
            f"distractor_blocks={avg_distr:.1f} "
            f"index_s={avg_index_s:.4f} feature_MB={avg_feat_mb:.4f}"
        )
    for topk in topks:
        for radius in radii:
            sub = [r for r in rows if r["topk"] == topk and r["radius"] == radius]
            if not sub:
                continue
            print(
                f"CONFIG topk={topk} radius={radius} "
                f"any={100*sum(r['any_support_hit'] for r in sub)/len(sub):.2f} "
                f"all={100*sum(r['all_support_hit'] for r in sub)/len(sub):.2f} "
                f"replay={statistics.mean(r['replay_tokens'] for r in sub):.0f}"
            )


if __name__ == "__main__":
    main()
