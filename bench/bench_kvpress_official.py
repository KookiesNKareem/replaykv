#!/usr/bin/env python3
"""KVPress baselines (SnapKV/PyramidKV/StreamingLLM) on our official RULER data.

Same JSONL inputs and scoring as bench_official_ruler_product.py, so the rows
are directly comparable at the real 128K context (the KVPress bundled RULER
mirror caps at 16K)."""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from bench_official_ruler_product import task_score, gpu_mem_used_gb, split_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--subset", default="validation")
    ap.add_argument("--samples-per-task", type=int, default=2)
    ap.add_argument("--press", required=True,
                    choices=["snapkv", "pyramidkv", "streaming_llm"])
    ap.add_argument("--compression-ratio", type=float, required=True)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    import torch
    from transformers import pipeline

    from kvpress import PyramidKVPress, SnapKVPress, StreamingLLMPress

    press_cls = {"snapkv": SnapKVPress, "pyramidkv": PyramidKVPress,
                 "streaming_llm": StreamingLLMPress}[args.press]
    press = press_cls(compression_ratio=args.compression_ratio)
    pipe = pipeline("kv-press-text-generation", model=args.model,
                    device="cuda", torch_dtype=torch.bfloat16,
                    model_kwargs={"attn_implementation": "sdpa"})

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    by_task: dict[str, list[float]] = {}
    t_start = time.perf_counter()
    for task in tasks:
        rows = [json.loads(l) for l in
                open(Path(args.data_root) / task / f"{args.subset}.jsonl")][:args.samples_per_task]
        for r in rows:
            context, query = split_prompt(r["input"])
            query = query + r.get("answer_prefix", "")
            t0 = time.perf_counter()
            with torch.no_grad():
                pred = pipe(context, question=query, press=press,
                            max_new_tokens=args.max_new)["answer"]
            score = task_score(task, pred, r.get("outputs", []))
            by_task.setdefault(task, []).append(100.0 * score)
            with open(Path(args.out_dir) / f"{task}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"index": r.get("index"), "pred": pred,
                                    "outputs": r.get("outputs"), "score": score,
                                    "gen_s": round(time.perf_counter() - t0, 1)},
                                   ensure_ascii=False) + "\n")
            print(f"KVP {args.press}@{args.compression_ratio} {task} "
                  f"score={score:.2f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    task_scores = {t: round(statistics.mean(v), 2) for t, v in sorted(by_task.items())}
    summary = {"model": args.model, "press": args.press,
               "compression_ratio": args.compression_ratio,
               "samples": sum(len(v) for v in by_task.values()),
               "wall_s": round(time.perf_counter() - t_start, 1),
               "gpu_mem_used_gb": gpu_mem_used_gb(),
               "task_scores": task_scores,
               "mean_score": round(statistics.mean(task_scores.values()), 2)}
    json.dump(summary, open(Path(args.out_dir) / "summary.json", "w"), indent=1)
    print("KVPRESS_OFFICIAL_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
