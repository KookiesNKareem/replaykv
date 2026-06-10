#!/usr/bin/env python3
"""LongBench-v2 through the product streaming-replay path.

This is a standardized quality harness for the current product-shaped method:
stream over the long context, retrieve a small set of evidence blocks, replay
those snippets densely with vLLM, and score with LongBench-v2's multiple-choice
answer extraction.

It intentionally does not benchmark the older compressed-KV backend.
"""
from __future__ import annotations

from collections import defaultdict
import argparse
import json
import os
import re
import statistics
import subprocess
import time


def extract_answer(response: str | None) -> str | None:
    if not response:
        return None
    up = response.upper()
    patterns = [
        r"(?:CORRECT\s+)?ANSWER\s+IS\s+\(([ABCD])\)",
        r"(?:CORRECT\s+)?ANSWER\s*[:：]\s*\(([ABCD])\)",
        r"(?:CORRECT\s+)?ANSWER\s+IS\s+([ABCD])\b",
        r"(?:CORRECT\s+)?ANSWER\s*[:：]\s*([ABCD])\b",
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, up))
        if matches:
            return matches[0].group(1)
    matches = list(re.finditer(r"\(([ABCD])\)", up))
    if matches:
        return matches[-1].group(1)
    matches = list(re.finditer(r"\b([ABCD])\b", up[len(up) // 2:]))
    return matches[-1].group(1) if matches else None


def build_prompt(sample: dict, context: str) -> str:
    return (
        "Please read the following text and answer the question below.\n\n"
        f"<text>\n{context}\n</text>\n\n"
        f"What is the correct answer to this question: {sample.get('question', '')}\n"
        "Choices:\n"
        f"(A) {sample.get('choice_A', '')}\n"
        f"(B) {sample.get('choice_B', '')}\n"
        f"(C) {sample.get('choice_C', '')}\n"
        f"(D) {sample.get('choice_D', '')}\n\n"
        'Format your response as follows: "The correct answer is (insert answer here)".'
    )


def build_query(sample: dict) -> str:
    return (
        f" {sample.get('question', '')} "
        f" {sample.get('choice_A', '')} {sample.get('choice_B', '')} "
        f" {sample.get('choice_C', '')} {sample.get('choice_D', '')}"
    )


def build_question_query(sample: dict) -> str:
    return f" {sample.get('question', '')} "


def build_choice_query(sample: dict, letter: str) -> str:
    return f" {sample.get('question', '')} {sample.get(f'choice_{letter}', '')} "


def load_samples(args):
    from datasets import load_dataset

    print(f"LBV2_LOAD dataset={args.dataset_name} split={args.split_name}", flush=True)
    ds = load_dataset(args.dataset_name, split=args.split_name)
    print(f"LBV2_LOAD raw_rows={len(ds)}", flush=True)
    if args.length:
        ds = ds.from_list([x for x in ds if x.get("length") == args.length])
        print(f"LBV2_LOAD after_length={args.length} rows={len(ds)}", flush=True)
    if args.domain:
        ds = ds.from_list([x for x in ds if x.get("domain") == args.domain])
        print(f"LBV2_LOAD after_domain={args.domain} rows={len(ds)}", flush=True)
    if args.difficulty:
        ds = ds.from_list([x for x in ds if x.get("difficulty") == args.difficulty])
        print(f"LBV2_LOAD after_difficulty={args.difficulty} rows={len(ds)}", flush=True)
    if args.num_samples > 0:
        end = min(args.start_idx + args.num_samples, len(ds))
        ds = ds.select(range(args.start_idx, end))
    elif args.start_idx:
        ds = ds.select(range(args.start_idx, len(ds)))
    return list(ds)


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


def block_iter(ids: list[int], block_size: int):
    usable = (len(ids) // block_size) * block_size
    for lo in range(0, usable, block_size):
        yield ids[lo:lo + block_size]


def merge_intervals(blocks: list[int], radius: int, block_size: int, n_tokens: int):
    intervals = []
    for b in sorted(set(blocks)):
        lo = max(0, b - radius) * block_size
        hi = min((b + radius + 1) * block_size, n_tokens)
        intervals.append((lo, hi))
    intervals.sort()
    merged: list[list[int]] = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)
    return merged


def diversify_ranked_blocks(
    ranked_blocks: list[int],
    scores: list[float],
    topk: int,
    radius: int,
    min_gap: int,
) -> tuple[list[int], list[float]]:
    """Keep high-scoring block centers while avoiding replay-window overlap."""
    if topk <= 0:
        return [], []
    gap = min_gap if min_gap >= 0 else max(0, 2 * radius)
    chosen: list[int] = []
    chosen_scores: list[float] = []
    for block, score in zip(ranked_blocks, scores):
        if all(abs(block - old) > gap for old in chosen):
            chosen.append(block)
            chosen_scores.append(score)
            if len(chosen) >= topk:
                return chosen, chosen_scores

    # If the context is short or the score mass is genuinely clustered, fill the
    # remaining slots with the next best unused centers rather than shrinking k.
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


def choose_routed_candidate(sample: dict, candidates: list[dict], args) -> tuple[int, str]:
    """Pick a product operating point from precomputed replay candidates."""
    if not candidates:
        raise ValueError("route requested with no candidates")

    query = build_query(sample).lower()
    question = str(sample.get("question", "")).lower().strip()
    domain = str(sample.get("domain", ""))

    # Code-repo questions are usually discrete symbol/class/file lookups. The
    # compact replay profile keeps the option-answer prompt tighter and avoids
    # the wide profile's tendency to produce open-ended explanatory text.
    force_compact = domain == "Code Repository Understanding"

    sequential_patterns = (
        " player",
        " players",
        " game",
        " log",
        " round",
        " utility",
        " tokens",
        " steps",
    )
    causal_prefixes = (
        "why does ",
        "why did ",
    )
    synthesis_patterns = (
        "multifaceted",
        "reconcile",
        "implications",
        "societal norms",
        "interplay",
    )
    wants_wide = (
        not force_compact
        and (
            any(pat in query for pat in sequential_patterns)
            or any(question.startswith(prefix) for prefix in causal_prefixes)
            or any(pat in question for pat in synthesis_patterns)
        )
    )
    target = (
        (args.route_wide_topk, args.route_wide_radius)
        if wants_wide else
        (args.route_compact_topk, args.route_compact_radius)
    )
    for i, cand in enumerate(candidates):
        if (cand["topk"], cand["radius"]) == target:
            if force_compact:
                return i, "compact_code_lookup"
            return i, "wide_targeted" if wants_wide else "compact_default"

    # Keep the product bounded if the exact target was skipped by the prompt
    # guard: choose the closest prompt-size side of the intended route.
    if wants_wide:
        best = max(range(len(candidates)), key=lambda i: candidates[i]["prompt_tokens"])
        return best, "wide_fallback_largest"
    best = min(range(len(candidates)), key=lambda i: candidates[i]["prompt_tokens"])
    return best, "compact_fallback_smallest"


_EMBED_RANKER = None
_EMBED_CACHE: dict = {}
_TAG_EXPANDER = None


def _tag_expander(tok, args):
    global _TAG_EXPANDER
    if _TAG_EXPANDER is None:
        from replaykv.learned_tags import load_tag_expander

        _TAG_EXPANDER = load_tag_expander(
            args.tags_path, vocab_size=len(tok), block_size=args.block_size,
            query_m=args.tags_query_m, block_m=args.tags_block_m)
        print(f"LBV2_LEARNED_TAGS path={args.tags_path} k={_TAG_EXPANDER.k} "
              f"query_m={args.tags_query_m} block_m={args.tags_block_m}", flush=True)
    return _TAG_EXPANDER


def _embed_rank(tok, sample, context_ids: list[int], args) -> dict:
    """Rank blocks with the external embedding retriever, cached per context.

    Ranking is independent of topk/radius, so candidate configs within one
    sample reuse the same embedding pass (as a real RAG system would)."""
    global _EMBED_RANKER
    key = id(context_ids)
    cached = _EMBED_CACHE.get(key)
    if cached is not None:
        return cached
    if _EMBED_RANKER is None:
        from replaykv.rag_baselines import EmbedRanker

        _EMBED_RANKER = EmbedRanker(args.embed_model, batch_size=args.embed_batch_size)
        print(f"LBV2_EMBED model={args.embed_model}", flush=True)
    usable = (len(context_ids) // args.block_size) * args.block_size
    block_texts = [
        tok.decode(context_ids[lo:lo + args.block_size])
        for lo in range(0, usable, args.block_size)
    ]
    res = _EMBED_RANKER.rank(block_texts, build_query(sample))
    out = {
        "ranked": res.ranked,
        "scores": res.scores,
        "index_s": res.index_s,
        "feature_MB": res.index_bytes / 1e6,
    }
    _EMBED_CACHE.clear()
    _EMBED_CACHE[key] = out
    return out


def _embed_release():
    global _EMBED_RANKER
    _EMBED_CACHE.clear()
    if _EMBED_RANKER is not None:
        torch = _EMBED_RANKER.torch
        _EMBED_RANKER.model = None
        _EMBED_RANKER = None
        torch.cuda.empty_cache()


def select_snippets(tok, sample: dict, context_ids: list[int], args):
    from replaykv.streaming_replay import select_blocks_sparse

    selector_mode = args.selector_mode
    selector_route = selector_mode
    if args.selector_mode == "sparse_domain_hybrid_nms":
        if (
            sample.get("domain") == "Multi-Document QA"
            and sample.get("sub_domain") == "Academic"
        ):
            selector_mode = "sparse_multiquery_nms"
            selector_route = "multiquery_multidoc_academic"
        else:
            selector_mode = "sparse_nms"
            selector_route = "sparse_default"

    candidate_topk = args._active_topk
    if selector_mode in ("sparse_nms", "embed_nms", "learned_tags_nms", "bm25_nms", "hybrid_nms"):
        candidate_topk = max(
            args._active_topk,
            args._active_topk * max(1, args.candidate_multiplier),
        )
    elif selector_mode == "sparse_multiquery_nms":
        # More candidate blocks are cheap relative to vLLM replay. The final
        # prompt budget is still controlled by active top-k, radius, and the
        # max-prompt guard. This is an experimental selector-development mode,
        # not the locked product baseline: on the 2026-06-09 LongBench-v2 n=100
        # slice it dropped from 45/98 to 41/98 by adding distractors.
        candidate_topk = max(
            args._active_topk,
            args._active_topk * max(1, args.candidate_multiplier),
        )

    t0 = time.perf_counter()
    if selector_mode == "sparse_multiquery_nms":
        query_texts = [build_question_query(sample), build_query(sample)]
        query_texts.extend(build_choice_query(sample, letter) for letter in "ABCD")
        merged: dict[int, float] = {}
        feature_bytes = 0
        for q_idx, query_text in enumerate(query_texts):
            query_ids = tok(query_text, add_special_tokens=False)["input_ids"]
            sel = select_blocks_sparse(
                block_iter(context_ids, args.block_size),
                query_ids,
                topk=candidate_topk,
                radius=0,
            )
            feature_bytes += sel.feature_bytes
            if not sel.scores:
                continue
            denom = max(abs(float(sel.scores[0])), 1e-6)
            weight = 1.15 if q_idx == 0 else 1.0
            for block, score in zip(sel.ranked_blocks, sel.scores):
                # Normalize within each query view so an answer-choice view can
                # add high-recall evidence without dominating by raw score scale.
                val = weight * float(score) / denom
                merged[block] = max(merged.get(block, float("-inf")), val)
        ranked_scored = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
        ranked_blocks = [int(b) for b, _ in ranked_scored]
        scores = [round(float(s), 6) for _, s in ranked_scored]
        feature_MB = feature_bytes / 1e6
    elif selector_mode == "embed_nms":
        ranking = _embed_rank(tok, sample, context_ids, args)
        ranked_blocks = ranking["ranked"][:candidate_topk]
        scores = ranking["scores"][:candidate_topk]
        feature_MB = ranking["feature_MB"]
    elif selector_mode in ("bm25_nms", "hybrid_nms"):
        from replaykv.rag_baselines import Bm25Ranker

        usable = (len(context_ids) // args.block_size) * args.block_size
        texts = [tok.decode(context_ids[lo:lo + args.block_size])
                 for lo in range(0, usable, args.block_size)]
        bm = Bm25Ranker().rank(texts, build_query(sample))
        if selector_mode == "bm25_nms":
            ranked_blocks = bm.ranked[:candidate_topk]
            scores = bm.scores[:candidate_topk]
            feature_MB = bm.index_bytes / 1e6
        else:
            emb = _embed_rank(tok, sample, context_ids, args)
            nb = len(texts)

            def _norm(ranked, sc):
                v = [0.0] * nb
                if sc:
                    mn, mx = min(sc), max(sc)
                    rng = (mx - mn) or 1.0
                    for b, x in zip(ranked, sc):
                        v[b] = (x - mn) / rng
                return v

            vb = _norm(bm.ranked, bm.scores)
            ve = _norm(emb["ranked"], emb["scores"])
            order = sorted(range(nb), key=lambda b: -(vb[b] + ve[b]))
            ranked_blocks = order[:candidate_topk]
            scores = [round(vb[b] + ve[b], 6) for b in ranked_blocks]
            feature_MB = bm.index_bytes / 1e6 + emb["feature_MB"]
    elif selector_mode == "learned_tags_nms":
        expander = _tag_expander(tok, args)
        query_ids = tok(build_query(sample), add_special_tokens=False)["input_ids"]
        nb = max(1, len(context_ids) // args.block_size)
        sel = None
        if args.tags_gate_score > 0:
            # confidence gate: strong lexical anchors -> pure sparse ranking;
            # semantic expansion only when the lexical channel is weak
            sel = select_blocks_sparse(
                block_iter(context_ids, args.block_size),
                query_ids,
                topk=candidate_topk,
                radius=0,
            )
            if not sel.scores or sel.scores[0] < args.tags_gate_score:
                sel = None
                selector_route = "learned_tags_gated_on"
            else:
                selector_route = "learned_tags_gated_off"
        if sel is None:
            sel = select_blocks_sparse(
                expander.expand_blocks(block_iter(context_ids, args.block_size)),
                expander.expand_query(query_ids),
                topk=candidate_topk,
                radius=0,
            )
        ranked_blocks = sel.ranked_blocks
        scores = sel.scores
        feature_MB = (sel.feature_bytes + nb * expander.bytes_per_block) / 1e6
    else:
        query_ids = tok(build_query(sample), add_special_tokens=False)["input_ids"]
        sel = select_blocks_sparse(
            block_iter(context_ids, args.block_size),
            query_ids,
            topk=candidate_topk,
            radius=0,
        )
        ranked_blocks = sel.ranked_blocks
        scores = sel.scores
        feature_MB = sel.feature_bytes / 1e6
    index_s = time.perf_counter() - t0
    if selector_mode == "embed_nms":
        # one real embedding pass per context; cached for sibling configs
        index_s = ranking["index_s"]
    if selector_mode in {"sparse_nms", "sparse_multiquery_nms", "embed_nms", "learned_tags_nms", "bm25_nms", "hybrid_nms"}:
        ranked_blocks, scores = diversify_ranked_blocks(
            ranked_blocks,
            scores,
            args._active_topk,
            args._active_radius,
            args.nms_gap,
        )
    intervals = merge_intervals(ranked_blocks, args._active_radius, args.block_size, len(context_ids))
    sep = tok("\n...\n", add_special_tokens=False)["input_ids"]
    snippet_ids: list[int] = []
    for i, (lo, hi) in enumerate(intervals):
        if i:
            snippet_ids.extend(sep)
        snippet_ids.extend(context_ids[lo:hi])
    snippet_text = tok.decode(snippet_ids, skip_special_tokens=True)
    return {
        "snippet_text": snippet_text,
        "snippet_tokens": len(snippet_ids),
        "ranked_blocks": ranked_blocks,
        "selected_intervals": intervals,
        "scores": scores,
        "feature_MB": feature_MB,
        "index_s": index_s,
        "selector_mode": selector_mode,
        "selector_policy": args.selector_mode,
        "selector_route": selector_route,
        "candidate_topk": candidate_topk,
    }


def measure_decode(llm, prompts, trials: int, decode_tokens: int):
    from vllm import SamplingParams

    if not prompts:
        return {"agg_decode_tok_s": 0.0, "decode_trials": [], "decode_std": 0.0}
    batch = len(prompts)
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
        "decode_trials": [round(x, 3) for x in vals],
        "decode_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
    }


def main():
    print("LBV2_IMPORT transformers/vllm", flush=True)
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--dataset-name", default=os.environ.get("DATASET_NAME", "zai-org/LongBench-v2"))
    ap.add_argument("--split-name", default=os.environ.get("SPLIT_NAME", "train"))
    ap.add_argument("--num-samples", type=int, default=int(os.environ.get("N", "30")))
    ap.add_argument("--start-idx", type=int, default=int(os.environ.get("START_IDX", "0")))
    ap.add_argument("--length", default=os.environ.get("LENGTH", "short"))
    ap.add_argument("--domain", default=os.environ.get("DOMAIN", ""))
    ap.add_argument("--difficulty", default=os.environ.get("DIFFICULTY", ""))
    ap.add_argument("--block-size", type=int, default=int(os.environ.get("BLOCK_SIZE", "64")))
    ap.add_argument("--topk", type=int, default=int(os.environ.get("STREAM_TOPK", "8")))
    ap.add_argument("--radius", type=int, default=int(os.environ.get("STREAM_RADIUS", "4")))
    ap.add_argument("--topks", default=os.environ.get("STREAM_TOPKS", ""))
    ap.add_argument("--radii", default=os.environ.get("STREAM_RADII", ""))
    ap.add_argument("--selector-mode", choices=[
        "sparse",
        "sparse_nms",
        "sparse_multiquery_nms",
        "sparse_domain_hybrid_nms",
        "embed_nms",
        "learned_tags_nms",
        "bm25_nms",
        "hybrid_nms",
    ],
                    default=os.environ.get("STREAM_SELECTOR_MODE", "sparse"),
                    help=(
                        "Product baseline uses sparse_nms with the targeted route controller. "
                        "sparse_multiquery_nms and sparse_domain_hybrid_nms are explicit "
                        "selector-development experiments, not validated defaults."
                    ))
    ap.add_argument("--candidate-multiplier", type=int,
                    default=int(os.environ.get("STREAM_CANDIDATE_MULTIPLIER", "4")))
    ap.add_argument("--nms-gap", type=int, default=int(os.environ.get("STREAM_NMS_GAP", "-1")),
                    help="Minimum gap between selected block centers; negative uses 2*radius.")
    ap.add_argument("--max-context-tokens", type=int, default=int(os.environ.get("MAX_CONTEXT_TOKENS", "131072")))
    ap.add_argument("--max-prompt-tokens", type=int, default=int(os.environ.get("MAX_PROMPT_TOKENS", "32768")),
                    help="Skip replay prompts above this limit; <=0 disables the guard.")
    ap.add_argument("--max-new", type=int, default=int(os.environ.get("MAX_NEW", "32")))
    ap.add_argument("--trials", type=int, default=int(os.environ.get("TRIALS", "3")))
    ap.add_argument("--decode-tokens", type=int, default=int(os.environ.get("DECODE_TOKENS", "33")))
    ap.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("GPU_MEM_UTIL", "0.25")))
    ap.add_argument("--embed-model", default=os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5"))
    ap.add_argument("--embed-batch-size", type=int, default=int(os.environ.get("EMBED_BATCH", "256")))
    ap.add_argument("--tags-path", default=os.environ.get("TAGS_PATH", ""))
    ap.add_argument("--tags-query-m", type=int, default=int(os.environ.get("TAGS_QUERY_M", "4")))
    ap.add_argument("--tags-block-m", type=int, default=int(os.environ.get("TAGS_BLOCK_M", "1")))
    ap.add_argument("--tags-gate-score", type=float, default=float(os.environ.get("TAGS_GATE_SCORE", "0")),
                    help="if >0, use tag expansion only when the sparse top-1 score is below this")
    ap.add_argument("--out", default=os.environ.get("JSON_OUT", "/workspace/akv/results/product_longbench_v2.jsonl"))
    ap.add_argument("--local-files-only", action="store_true", default=os.environ.get("HF_LOCAL_FILES_ONLY", "0") == "1")
    ap.add_argument("--policy-path", default=os.environ.get("REPLAY_POLICY_PATH", ""),
                    help="Optional learned replay policy checkpoint. If set, only one candidate is generated per sample.")
    ap.add_argument("--policy-cost-weight", type=float, default=float(os.environ.get("REPLAY_POLICY_COST_WEIGHT", "-1")),
                    help="Override policy cost weight; negative uses checkpoint value.")
    ap.add_argument("--route-mode", choices=["none", "query_heuristic"],
                    default=os.environ.get("REPLAY_ROUTE_MODE", "none"),
                    help="Optional product router over candidate replay configs.")
    ap.add_argument("--route-compact-topk", type=int, default=int(os.environ.get("ROUTE_COMPACT_TOPK", "48")))
    ap.add_argument("--route-compact-radius", type=int, default=int(os.environ.get("ROUTE_COMPACT_RADIUS", "1")))
    ap.add_argument("--route-wide-topk", type=int, default=int(os.environ.get("ROUTE_WIDE_TOPK", "32")))
    ap.add_argument("--route-wide-radius", type=int, default=int(os.environ.get("ROUTE_WIDE_RADIUS", "4")))
    args = ap.parse_args()
    topks = [int(x) for x in args.topks.split(",") if x.strip()] or [args.topk]
    radii = [int(x) for x in args.radii.split(",") if x.strip()] or [args.radius]

    policy = None
    if args.policy_path:
        from replaykv.replay_policy import ReplayPolicy

        policy = ReplayPolicy.load(args.policy_path, device="cpu")
        if args.policy_cost_weight >= 0:
            policy.cost_weight = args.policy_cost_weight
        print(
            f"LBV2_POLICY path={args.policy_path} cost_weight={policy.cost_weight} "
            f"candidate_topks={topks} candidate_radii={radii}",
            flush=True,
        )

    print(f"LBV2_TOKENIZER model={args.model} local={args.local_files_only}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    samples = load_samples(args)
    print(f"LBV2_PREP samples={len(samples)}", flush=True)
    choose_one_candidate = policy is not None or args.route_mode != "none"

    prepared = []
    skipped = 0
    prompt_skipped = 0
    for idx, sample in enumerate(samples):
        print(f"LBV2_PREP sample={idx} id={sample.get('_id', idx)}", flush=True)
        context = sample.get("context", "")
        context_ids = tok(context, add_special_tokens=False)["input_ids"]
        print(f"LBV2_PREP sample={idx} context_tokens={len(context_ids)}", flush=True)
        if len(context_ids) > args.max_context_tokens:
            skipped += 1
            continue
        sample_candidates = []
        sample_added = False
        for topk in topks:
            for radius in radii:
                args._active_topk = topk
                args._active_radius = radius
                evidence = select_snippets(tok, sample, context_ids, args)
                print(
                    f"LBV2_PREP sample={idx} topk={topk} radius={radius} "
                    f"selected={evidence['ranked_blocks']} "
                    f"snippet_tokens={evidence['snippet_tokens']} index_s={evidence['index_s']:.4f}",
                    flush=True,
                )
                prompt = build_prompt(sample, evidence["snippet_text"])
                prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                if args.max_prompt_tokens > 0 and len(prompt_ids) > args.max_prompt_tokens:
                    prompt_skipped += 1
                    print(
                        f"LBV2_SKIP_PROMPT sample={idx} topk={topk} radius={radius} "
                        f"prompt_tokens={len(prompt_ids)} max_prompt_tokens={args.max_prompt_tokens}",
                        flush=True,
                    )
                    continue
                cand = {
                    "idx": args.start_idx + idx,
                    "sample": sample,
                    "context_tokens": len(context_ids),
                    "topk": topk,
                    "radius": radius,
                    "prompt": {"prompt_token_ids": prompt_ids},
                    "prompt_tokens": len(prompt_ids),
                    **evidence,
                }
                if not choose_one_candidate:
                    prepared.append(cand)
                    sample_added = True
                else:
                    sample_candidates.append(cand)
        if choose_one_candidate and sample_candidates:
            route_reason = ""
            if policy is not None:
                pick = policy.choose(sample_candidates)
                route_reason = "policy"
            else:
                pick, route_reason = choose_routed_candidate(sample, sample_candidates, args)
            chosen = sample_candidates[pick]
            chosen["candidate_count"] = len(sample_candidates)
            chosen["chosen_index"] = pick
            chosen["route_mode"] = args.route_mode if policy is None else "policy"
            chosen["route_reason"] = route_reason
            if policy is not None:
                chosen["policy_cost_weight"] = policy.cost_weight
            chosen["candidates"] = [
                {
                    "topk": c["topk"],
                    "radius": c["radius"],
                    "snippet_tokens": c["snippet_tokens"],
                    "prompt_tokens": c["prompt_tokens"],
                    "scores": c["scores"],
                }
                for c in sample_candidates
            ]
            prepared.append(chosen)
            sample_added = True
            print(
                f"LBV2_ROUTE_CHOICE sample={idx} mode={chosen['route_mode']} "
                f"reason={route_reason} topk={chosen['topk']} "
                f"radius={chosen['radius']} prompt_tokens={chosen['prompt_tokens']} "
                f"snippet_tokens={chosen['snippet_tokens']}",
                flush=True,
            )
        if not sample_added:
            skipped += 1
            print(f"LBV2_SKIP sample={idx} reason=no_prompt_under_limit", flush=True)

    _embed_release()
    max_prompt = max([p["prompt_tokens"] for p in prepared] + [512])
    print(
        f"LBV2_LLM max_prompt={max_prompt} prepared={len(prepared)} "
        f"skipped={skipped} prompt_skipped={prompt_skipped}",
        flush=True,
    )
    llm = LLM(
        model=args.model,
        max_model_len=max(1024, max_prompt + args.max_new + 64),
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )
    print("LBV2_GENERATE", flush=True)
    qwen_like = "qwen" in args.model.lower()
    sp = SamplingParams(
        max_tokens=args.max_new,
        temperature=0.0,
        # "\n\n" instantly truncates models that open with a blank line
        # (Llama does); keep the legacy stops only for Qwen comparability
        stop=["\n\n", "<|im_end|>"] if qwen_like else None,
        ignore_eos=False,
    )
    t0 = time.perf_counter()
    outs = llm.generate([p["prompt"] for p in prepared], sp, use_tqdm=False)
    wall_s = time.perf_counter() - t0
    if args.trials > 0:
        timing = measure_decode(llm, [p["prompt"] for p in prepared], args.trials, args.decode_tokens)
    else:
        timing = {"agg_decode_tok_s": 0.0, "decode_trials": [], "decode_std": 0.0}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    correct = 0
    by_domain = defaultdict(lambda: [0, 0])
    by_config = defaultdict(lambda: [0, 0])
    with open(args.out, "w", encoding="utf-8") as f:
        for p, out in zip(prepared, outs):
            text = out.outputs[0].text
            pred = extract_answer(text)
            gold = str(p["sample"].get("answer", "")).strip().upper()
            ok = pred == gold
            correct += int(ok)
            key = p["sample"].get("domain", "unknown")
            by_domain[key][0] += int(ok)
            by_domain[key][1] += 1
            by_config[(p["topk"], p["radius"])][0] += int(ok)
            by_config[(p["topk"], p["radius"])][1] += 1
            rec = {
                "idx": p["idx"],
                "sample_id": p["sample"].get("_id", p["idx"]),
                "domain": p["sample"].get("domain", ""),
                "sub_domain": p["sample"].get("sub_domain", ""),
                "difficulty": p["sample"].get("difficulty", ""),
                "length": p["sample"].get("length", ""),
                "context_tokens": p["context_tokens"],
                "prompt_tokens": p["prompt_tokens"],
                "snippet_tokens": p["snippet_tokens"],
                "topk": p["topk"],
                "radius": p["radius"],
                "ranked_blocks": p["ranked_blocks"],
                "selected_intervals": p["selected_intervals"],
                "scores": p["scores"],
                "selector_mode": p["selector_mode"],
                "selector_policy": p.get("selector_policy", p["selector_mode"]),
                "selector_route": p.get("selector_route", p["selector_mode"]),
                "candidate_topk": p["candidate_topk"],
                "gold": gold,
                "pred": pred,
                "ok": ok,
                "response": text,
                "index_s": p["index_s"],
                "index_tok_s": p["context_tokens"] / max(p["index_s"], 1e-9),
                "feature_MB": p["feature_MB"],
                "gpu_mem_used_gb": gpu_mem_used_gb(),
                "generate_wall_s": wall_s,
                **timing,
            }
            if "candidate_count" in p:
                rec.update({
                    "candidate_count": p["candidate_count"],
                    "chosen_index": p["chosen_index"],
                    "route_mode": p["route_mode"],
                    "route_reason": p["route_reason"],
                    "candidates": p["candidates"],
                })
                if "policy_cost_weight" in p:
                    rec["policy_cost_weight"] = p["policy_cost_weight"]
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")

    n = len(prepared)
    print(
        f"PRODUCT_LONGBENCH_V2_RESULT n={n} skipped={skipped} "
        f"prompt_skipped={prompt_skipped} "
        f"correct={correct}/{n} acc={(100 * correct / max(1, n)):.2f} "
        f"topks={topks} radii={radii} out={args.out}",
        flush=True,
    )
    for (topk, radius), (hit, total) in sorted(by_config.items()):
        print(f"CONFIG topk={topk} radius={radius} {hit}/{total} {(100 * hit / max(1, total)):.2f}", flush=True)
    for domain, (hit, total) in sorted(by_domain.items()):
        print(f"DOMAIN {domain} {hit}/{total} {(100 * hit / max(1, total)):.2f}", flush=True)


if __name__ == "__main__":
    main()
