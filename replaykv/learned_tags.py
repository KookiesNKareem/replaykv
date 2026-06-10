# SPDX-License-Identifier: Apache-2.0
"""Learned concept tags derived from the serving model's own embedding geometry.

Replaces the hand-specified DEFAULT_CONCEPT_GROUPS prototype in concept_tags.py
with a fully automatic semantic channel: k-means clusters over the model's
input token-embedding matrix give a static token_id -> cluster_id lookup.
At ingest, each block appends the distinct cluster ids of its tokens as
synthetic concept tokens; the query does the same. Paraphrased evidence then
shares cluster tokens with the query even with zero lexical overlap, and the
existing IDF-weighted sparse scorer automatically downweights clusters that
appear everywhere.

No task data, no supervision, no second model: the lookup is read off the
weights the serving engine already holds, built once per model.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Sequence

import torch


def _load_embedding_matrix(model_name: str) -> torch.Tensor:
    """Load only the input-embedding weight from a HF checkpoint (no full model)."""
    import json
    import os

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    path = snapshot_download(model_name, allow_patterns=["*.safetensors*", "*.json"])
    candidates = [
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "embed_tokens.weight",
    ]
    index_path = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]
        for name in candidates:
            if name in weight_map:
                shard = os.path.join(path, weight_map[name])
                with safe_open(shard, framework="pt") as f:
                    return f.get_tensor(name)
        raise KeyError(f"no embedding weight among {candidates} in {index_path}")
    single = os.path.join(path, "model.safetensors")
    with safe_open(single, framework="pt") as f:
        keys = set(f.keys())
        for name in candidates:
            if name in keys:
                return f.get_tensor(name)
    raise KeyError(f"no embedding weight among {candidates} in {single}")


@torch.no_grad()
def _kmeans_topm(emb: torch.Tensor, k: int, iters: int, seed: int,
                 chunk: int = 16384, topm: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    vocab = emb.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    centers = emb[torch.randperm(vocab, generator=g)[:k].to(emb.device)].clone()
    assign = torch.zeros(vocab, dtype=torch.long, device=emb.device)
    for it in range(iters):
        for lo in range(0, vocab, chunk):
            hi = min(vocab, lo + chunk)
            sims = emb[lo:hi] @ centers.T
            assign[lo:hi] = sims.argmax(dim=1)
        new_centers = torch.zeros_like(centers)
        counts = torch.zeros(k, device=emb.device)
        new_centers.index_add_(0, assign, emb)
        counts.index_add_(0, assign, torch.ones(vocab, device=emb.device))
        empty = counts == 0
        new_centers[empty] = centers[empty]
        new_centers[~empty] /= counts[~empty, None]
        centers = torch.nn.functional.normalize(new_centers, dim=-1)
        if it % 5 == 0 or it == iters - 1:
            occupied = int((~empty).sum())
            print(f"TAGS_KMEANS iter={it} occupied={occupied}/{k}", flush=True)
    tm = torch.zeros(vocab, topm, dtype=torch.int32, device=emb.device)
    for lo in range(0, vocab, chunk):
        hi = min(vocab, lo + chunk)
        sims = emb[lo:hi] @ centers.T
        tm[lo:hi] = sims.topk(topm, dim=1).indices.to(torch.int32)
    return assign, tm


@torch.no_grad()
def build_token_clusters(model_name: str, k: int = 4096, iters: int = 25,
                         device: str = "cuda", seed: int = 0,
                         chunk: int = 16384, source: str = "distilled",
                         embed_model: str = "BAAI/bge-large-en-v1.5",
                         embed_batch: int = 512, topm: int = 4) -> dict:
    """Build a static token_id -> cluster lookup.

    source="input_emb": cluster the LLM's input-embedding matrix. NEGATIVE
    RESULT (2026-06-09): that space is surface-form geometry (cos(code,
    credential)=0.03), it cannot bridge paraphrase — kept only for ablations.

    source="distilled": encode every vocab piece's text with a strong
    embedder ONCE OFFLINE, mean-center + renormalize (anisotropy fix), then
    k-means. Serving keeps a free CPU lookup; the embedder never runs at
    ingest or query time.
    """
    from transformers import AutoTokenizer

    if source == "input_emb":
        emb = _load_embedding_matrix(model_name).to(device=device, dtype=torch.float32)
        vocab, dim = emb.shape
        emb = torch.nn.functional.normalize(emb, dim=-1)
    elif source == "distilled":
        from replaykv.rag_baselines import EmbedRanker

        tok = AutoTokenizer.from_pretrained(model_name)
        vocab = len(tok)
        texts = []
        for tid in range(vocab):
            try:
                piece = tok.decode([tid]).strip()
            except Exception:
                piece = ""
            texts.append(piece if piece else "<unk>")
        ranker = EmbedRanker(embed_model, device=device, batch_size=embed_batch)
        emb = ranker._encode(texts).to(torch.float32)
        dim = emb.shape[1]
        emb = emb - emb.mean(dim=0, keepdim=True)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        del ranker
        torch.cuda.empty_cache()
    else:
        raise ValueError(f"unknown source: {source}")
    assign, tm = _kmeans_topm(emb, k=k, iters=iters, seed=seed, chunk=chunk, topm=topm)
    return {
        "model": model_name,
        "source": source,
        "embed_model": embed_model if source == "distilled" else "",
        "k": k,
        "vocab": vocab,
        "dim": dim,
        "cluster_of": assign.to("cpu", torch.int32),
        "topm": tm.to("cpu"),
    }


@dataclass
class LearnedTagExpander:
    """Drop-in replacement for ConceptExpander backed by distilled clusters.

    Asymmetric soft assignment: blocks store each token's single nearest
    cluster (precision, tiny storage); the query expands each token to its
    top-m clusters (recall). IDF in the sparse scorer downweights clusters
    that occur in many blocks, so no stoplist is needed."""

    topm: torch.Tensor  # int32 [vocab, M]
    k: int
    base_id: int
    bytes_per_block: int
    query_m: int = 4
    block_m: int = 1

    def _clusters(self, ids: Sequence[int], m: int) -> list[int]:
        n = self.topm.shape[0]
        m = max(1, min(m, self.topm.shape[1]))
        valid = [int(t) for t in ids if 0 <= int(t) < n]
        if not valid:
            return []
        rows = self.topm[torch.tensor(valid, dtype=torch.long), :m]
        return torch.unique(rows).tolist()

    def expand_query(self, query_ids: list[int]) -> list[int]:
        return list(query_ids) + [
            self.base_id + c for c in self._clusters(query_ids, self.query_m)
        ]

    def expand_blocks(self, block_iter: Iterable[Sequence[int]]):
        for block in block_iter:
            ids = list(block)
            yield ids + [
                self.base_id + c for c in self._clusters(ids, self.block_m)
            ]


def save_token_clusters(payload: dict, path: str) -> None:
    torch.save(payload, path)


def load_tag_expander(path: str, vocab_size: int | None = None,
                      block_size: int = 64, query_m: int = 4,
                      block_m: int = 1) -> LearnedTagExpander:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    topm = payload["topm"]
    if topm.ndim != 2:
        raise ValueError("tags payload missing [vocab, M] topm assignments")
    vocab = int(payload["vocab"])
    base = (vocab_size if vocab_size is not None else vocab) + 1000
    # per-block storage: distinct nearest-cluster ids, <= block_size entries, 2B each
    bytes_per_block = 2 * min(block_size, int(payload["k"]))
    return LearnedTagExpander(
        topm=topm,
        k=int(payload["k"]),
        base_id=base,
        bytes_per_block=bytes_per_block,
        query_m=query_m,
        block_m=block_m,
    )


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build token->cluster concept tags from model embeddings")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--source", choices=["distilled", "input_emb"], default="distilled")
    ap.add_argument("--embed-model", default="BAAI/bge-large-en-v1.5")
    ap.add_argument("--embed-batch", type=int, default=512)
    ap.add_argument("--topm", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    payload = build_token_clusters(args.model, k=args.k, iters=args.iters,
                                   device=args.device, seed=args.seed,
                                   source=args.source, embed_model=args.embed_model,
                                   embed_batch=args.embed_batch, topm=args.topm)
    save_token_clusters(payload, args.out)
    print(f"TAGS_SAVED out={args.out} model={args.model} source={args.source} "
          f"k={args.k} vocab={payload['vocab']} dim={payload['dim']}", flush=True)


if __name__ == "__main__":
    main()
