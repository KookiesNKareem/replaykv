#!/usr/bin/env python3
"""Dense full-context baseline on official NVIDIA RULER data via vLLM.

Same data, same scoring as bench_official_ruler_product.py (imported), but the
ENTIRE context goes to the model — the apples-to-apples full-attention row for
the replay-path comparison. Needs a GPU that fits weights + full-context KV.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from bench_official_ruler_product import task_score, gpu_mem_used_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("MODEL", "Qwen/Qwen2.5-7B-Instruct"))
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--subset", default="validation")
    ap.add_argument("--samples-per-task", type=int, default=5)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=131072)
    ap.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("GPU_MEM_UTIL", "0.92")))
    ap.add_argument("--yarn-factor", type=float, default=float(os.environ.get("YARN_FACTOR", "4")))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    records = []
    for task in tasks:
        path = Path(args.data_root) / task / f"{args.subset}.jsonl"
        rows = [json.loads(l) for l in open(path, encoding="utf-8")][:args.samples_per_task]
        for r in rows:
            r["_task"] = task
            records.append(r)
    print(f"RULER_FULL model={args.model} tasks={len(tasks)} records={len(records)}", flush=True)

    hf_overrides = None
    if args.yarn_factor > 1:
        # Qwen2.5 native ctx is 32K; 128K dense needs YaRN (repo standard: factor 4)
        hf_overrides = {"rope_scaling": {
            "rope_type": "yarn", "factor": float(args.yarn_factor),
            "original_max_position_embeddings": 32768,
        }}
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, trust_remote_code=True,
              enable_prefix_caching=False, hf_overrides=hf_overrides)
    sp = SamplingParams(max_tokens=args.max_new, temperature=0.0)
    prompts = [r["input"] + r.get("answer_prefix", "") for r in records]
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=True)
    wall = time.perf_counter() - t0

    os.makedirs(args.out_dir, exist_ok=True)
    by_task: dict[str, list[float]] = {}
    for r, o in zip(records, outs):
        pred = o.outputs[0].text
        score = task_score(r["_task"], pred, r.get("outputs", []))
        by_task.setdefault(r["_task"], []).append(100.0 * score)
        with open(Path(args.out_dir) / f"{r['_task']}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**{k: v for k, v in r.items() if k != "input"},
                                "pred": pred, "score": score}, ensure_ascii=False) + "\n")
    task_scores = {t: round(statistics.mean(v), 2) for t, v in sorted(by_task.items())}
    summary = {
        "model": args.model, "mode": "dense_full_context",
        "samples": len(records), "wall_s": round(wall, 1),
        "gpu_mem_used_gb": gpu_mem_used_gb(),
        "task_scores": task_scores,
        "mean_score": round(statistics.mean(task_scores.values()), 2),
    }
    with open(Path(args.out_dir) / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("RULER_FULL_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
