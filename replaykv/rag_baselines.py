# SPDX-License-Identifier: Apache-2.0
"""External-retrieval (RAG) baselines for the known-evidence selector harness.

These represent the "just use retrieval" alternative to the product's
KV/ingest-native selectors: a word-level Okapi BM25 index and a strong
off-the-shelf dense embedding retriever. Both rank the same fixed token
blocks as the sparse/concept selectors, so support recall is directly
comparable at matched topk/radius replay budgets. Index cost (seconds,
bytes) is reported alongside recall because the product claim is
recall-at-budget AND index-cost, not recall alone.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time

_WORD_RE = re.compile(r"[0-9A-Za-z_]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


@dataclass
class RankResult:
    ranked: list[int]
    scores: list[float]
    index_s: float
    index_bytes: int


class Bm25Ranker:
    """Okapi BM25 over word-level block texts (classic lexical RAG)."""

    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def rank(self, block_texts: list[str], query_text: str) -> RankResult:
        t0 = time.perf_counter()
        docs = [_words(t) for t in block_texts]
        nb = len(docs)
        if nb == 0:
            raise ValueError("no blocks to rank")
        df: dict[str, int] = {}
        tfs: list[dict[str, int]] = []
        for d in docs:
            tf: dict[str, int] = {}
            for w in d:
                tf[w] = tf.get(w, 0) + 1
            tfs.append(tf)
            for w in tf:
                df[w] = df.get(w, 0) + 1
        avgdl = sum(len(d) for d in docs) / nb
        scores = [0.0] * nb
        for w in set(_words(query_text)):
            n = df.get(w)
            if not n:
                continue
            idf = math.log(1.0 + (nb - n + 0.5) / (n + 0.5))
            for i, tf in enumerate(tfs):
                f = tf.get(w)
                if not f:
                    continue
                dl = len(docs[i])
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / max(avgdl, 1e-9))
                scores[i] += idf * f * (self.k1 + 1.0) / denom
        index_s = time.perf_counter() - t0
        order = sorted(range(nb), key=lambda i: (-scores[i], i))
        postings = sum(len(tf) for tf in tfs)
        return RankResult(
            ranked=order,
            scores=[round(scores[i], 6) for i in order],
            index_s=index_s,
            index_bytes=postings * 8,
        )


class EmbedRanker:
    """Dense retrieval with a strong off-the-shelf embedder (default BGE-large).

    CLS pooling + L2 normalization, query instruction prefix per the BGE
    recipe. Embeddings are computed on GPU in fp16; index_bytes counts the
    stored fp16 block-embedding matrix.
    """

    name = "embed"

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5",
                 device: str = "cuda", batch_size: int = 256,
                 query_prefix: str = "Represent this sentence for searching relevant passages: "):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.model_name = model_name
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=dtype).to(device).eval()

    def _encode(self, texts: list[str]):
        torch = self.torch
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tok(
                    texts[i:i + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                h = self.model(**batch).last_hidden_state[:, 0]
                outs.append(torch.nn.functional.normalize(h, dim=-1))
        return torch.cat(outs, dim=0)

    def rank(self, block_texts: list[str], query_text: str) -> RankResult:
        torch = self.torch
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        emb = self._encode(block_texts)
        q = self._encode([self.query_prefix + query_text])[0]
        sims = (emb @ q).float().tolist()
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        index_s = time.perf_counter() - t0
        nb = len(block_texts)
        order = sorted(range(nb), key=lambda i: (-sims[i], i))
        return RankResult(
            ranked=order,
            scores=[round(sims[i], 6) for i in order],
            index_s=index_s,
            index_bytes=emb.shape[0] * emb.shape[1] * 2,
        )


def build_ranker(selector: str, embed_model: str = "BAAI/bge-large-en-v1.5",
                 embed_batch_size: int = 256, embed_device: str = "cuda"):
    if selector == "bm25":
        return Bm25Ranker()
    if selector == "embed":
        return EmbedRanker(embed_model, device=embed_device, batch_size=embed_batch_size)
    raise ValueError(f"no RAG ranker for selector: {selector}")
