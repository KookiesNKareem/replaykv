# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible bounded-replay proxy.

Sits in front of any OpenAI-compatible backend (Ollama, llama.cpp server,
LM Studio, vLLM). Requests whose context exceeds a threshold are indexed and
rewritten to a bounded replay prompt; the backend only ever sees ~2K tokens,
so consumer GPUs can serve million-token contexts.
"""
from __future__ import annotations

import hashlib
import json
import time

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .replay import ReplayConfig, ReplayEngine

REPLAY_SYSTEM = (
    "You are answering from selected excerpts of a longer document. "
    "The excerpts below were retrieved for the user's question; non-contiguous "
    "excerpts are separated by '...'. Answer from the excerpts; quote exact "
    "values verbatim."
)


class ProxyState:
    def __init__(self, backend_url: str, cfg: ReplayConfig,
                 min_context_tokens: int, passthrough_model: str | None,
                 fixed_tokenizer=None, fixed_engine: ReplayEngine | None = None):
        self.backend_url = backend_url.rstrip("/")
        # bare backend root for native-API and passthrough routes
        self.backend_root = (self.backend_url[:-3]
                             if self.backend_url.endswith("/v1")
                             else self.backend_url)
        self.cfg = cfg
        self.min_context_tokens = min_context_tokens
        self.passthrough_model = passthrough_model
        self.client = httpx.AsyncClient(timeout=600.0)
        self._cache: dict[str, list[int]] = {}  # context sha -> token ids
        self._fixed = (fixed_tokenizer, fixed_engine) if fixed_tokenizer else None
        self._resolved: dict[str, tuple] = {}   # model name -> (tok, engine)

    def resolve(self, model_name: str):
        """(tokenizer, engine) for this request — fixed at startup or inferred
        from the model name with bundled tags tables (zero-config mode)."""
        if self._fixed is not None:
            return self._fixed
        hit = self._resolved.get(model_name)
        if hit is not None:
            return hit
        from transformers import AutoTokenizer

        from .registry import resolve_model

        tok_id, table_path = resolve_model(model_name)
        tok = AutoTokenizer.from_pretrained(tok_id)
        tag_table = None
        if table_path:
            from .tags import load_tag_table

            tag_table = load_tag_table(table_path)
            tag_table.bind_vocab(len(tok))
        engine = ReplayEngine(self.cfg, tag_table)
        print(f"[replaykv-proxy] resolved '{model_name}' -> {tok_id} "
              f"(tags={'yes' if tag_table else 'lexical-only'})")
        self._resolved[model_name] = (tok, engine)
        return tok, engine

    def _tokenize_cached(self, tok, text: str) -> list[int]:
        key = hashlib.sha256(text.encode()).hexdigest()
        ids = self._cache.get(key)
        if ids is None:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(self._cache) > 8:
                self._cache.clear()
            self._cache[key] = ids
        return ids


def _split_context_and_query(messages: list[dict]) -> tuple[str, str, list[dict]]:
    """Largest message body is the context; the last user message is the query."""
    query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            query = msg.get("content") or ""
            break
    bodies = [(len(m.get("content") or ""), i, m) for i, m in enumerate(messages)]
    bodies.sort(reverse=True)
    _, ctx_idx, ctx_msg = bodies[0]
    context = ctx_msg.get("content") or ""
    if context == query:
        # single huge user message: split a trailing question heuristically
        tail = context.rfind("\n", 0, len(context))
        qpos = max(context.rfind("Question:"), context.rfind("question:"), tail)
        if qpos > 0.5 * len(context):
            return context[:qpos], context[qpos:], messages
    return context, query, messages


def create_app(state: ProxyState) -> FastAPI:
    app = FastAPI(title="replaykv-proxy")

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "backend": state.backend_url}

    def _rewrite(body: dict) -> tuple[dict, dict | None]:
        messages = body.get("messages") or []
        context, query, _ = _split_context_and_query(messages)
        t0 = time.perf_counter()
        tok, engine = state.resolve(body.get("model") or state.passthrough_model or "")
        ctx_ids = state._tokenize_cached(tok, context) if context else []
        if len(ctx_ids) < state.min_context_tokens or not query:
            return body, None
        q_ids = tok(query, add_special_tokens=False)["input_ids"]
        sel = engine.select(ctx_ids, q_ids)
        snippets = "\n...\n".join(
            tok.decode(ctx_ids[lo:hi], skip_special_tokens=True)
            for lo, hi in sel.intervals
        )
        body = dict(body)
        body["messages"] = [
            {"role": "system", "content": REPLAY_SYSTEM},
            {"role": "user", "content": f"{snippets}\n\n{query}"},
        ]
        meta = {
            "context_tokens": len(ctx_ids),
            "replay_tokens": sel.replay_tokens,
            "blocks": len(sel.ranked_blocks),
            "gate_used_tags": sel.gate_used_tags,
            "index_s": round(time.perf_counter() - t0, 4),
        }
        return body, meta

    async def _forward(path: str, body: dict, meta: dict | None,
                       stream_default: bool):
        if state.passthrough_model:
            body["model"] = state.passthrough_model
        url = f"{state.backend_root}{path}"
        if body.get("stream", stream_default):
            body.setdefault("stream", True)
            upstream = state.client.stream("POST", url, json=body)

            async def gen():
                async with upstream as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk

            return StreamingResponse(gen(), media_type="application/x-ndjson"
                                     if stream_default else "text/event-stream")
        r = await state.client.post(url, json=body)
        out = r.json()
        if meta is not None and isinstance(out, dict):
            out["replaykv"] = meta
        return JSONResponse(out, status_code=r.status_code)

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body, meta = _rewrite(await request.json())
        return await _forward("/v1/chat/completions", body, meta,
                              stream_default=False)

    @app.post("/api/chat")
    async def ollama_chat(request: Request):
        # Ollama native chat API (streams NDJSON by default)
        body, meta = _rewrite(await request.json())
        return await _forward("/api/chat", body, meta, stream_default=True)

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "HEAD", "PATCH"])
    async def passthrough(request: Request, path: str):
        # transparent relay for every other endpoint (/api/tags, /api/pull,
        # /api/embeddings, /v1/models, ...), so takeover mode is invisible
        url = f"{state.backend_root}/{path}"
        headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length")}
        upstream = state.client.stream(
            request.method, url, headers=headers,
            params=dict(request.query_params), content=await request.body())

        async def gen():
            async with upstream as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

        return StreamingResponse(gen())

    return app
