# replaykv-proxy

Serve **million-token contexts on consumer GPUs** by putting a bounded-replay
sidecar in front of any OpenAI-compatible backend — Ollama, llama.cpp server,
LM Studio, or vLLM.

The proxy ingests your long context into a tiny CPU-built index (lexical
features + concept tags distilled from an embedder into a static lookup table),
selects the few blocks your question actually needs, and forwards a ~1.5K-token
prompt to the backend. The model never sees the full context, so KV memory and
decode cost stay **flat in context length** — a 12GB GPU that cannot hold 128K
of KV at any quantization can answer questions over 1M+ tokens.

Backed by the ReplayKV paper (official NVIDIA RULER at 128K: 95.4% with a
7B model; 96.7% at 1M tokens at 33GB on one A100 — see `../paper/`).

## Quickstart (Ollama)

```bash
pip install replaykv-proxy
ollama pull qwen2.5:7b
replaykv-proxy        # that's it — defaults to local Ollama, zero config
```

Or fully **drop-in** — existing apps need no changes at all, not even a
`base_url`:

```bash
brew services stop ollama   # free port 11434 (sudo systemctl stop ollama on Linux)
replaykv-proxy --takeover
```

Takeover mode binds Ollama's own port (11434), starts the real Ollama on
11435 automatically, and transparently relays every endpoint — native
`/api/*` and OpenAI `/v1/*` alike — rewriting only long-context chat
requests. Every Ollama client on your machine gets bounded-replay long
context without knowing the proxy exists.

The proxy resolves the tokenizer and the bundled concept-tag table from each
request's model name (Qwen2.5 and Llama 3.x families ship built in; other
models run lexical-only). Explicit `--tokenizer`/`--tags-table`/`--backend-url`
flags override for vLLM, llama.cpp server, LM Studio, or custom tables.

Then point any OpenAI client at the proxy:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8800/v1", api_key="unused")
resp = client.chat.completions.create(
    model="qwen2.5:7b",
    messages=[
        {"role": "system", "content": open("million_token_doc.txt").read()},
        {"role": "user", "content": "What is the access code mentioned for project Aurora?"},
    ],
)
print(resp.choices[0].message.content)
```

Responses include an `replaykv` field with replay stats
(`context_tokens`, `replay_tokens`, `index_s`, `gate_used_tags`).

## How it routes

- Requests under `--min-context-tokens` (default 8192) pass through untouched.
- Above it, the largest message body is treated as context and the last user
  message as the query; the rewritten request carries only the selected
  excerpts.
- The semantic (concept-tag) channel activates only when lexical evidence is
  weak (`--gate-score`), so exact-identifier lookups stay precise while
  paraphrased questions still find their evidence.

## Tag tables

The wheel bundles prebuilt tables (`replaykv_proxy/tables/`):
`qwen2.5_k1024.npz` (all Qwen2.5 sizes share the tokenizer) and
`llama3.1_k512.npz` (Llama 3.x). Build one for another
tokenizer with the research repo's
`python -m replaykv.learned_tags --source distilled --k 1024 --model <hf-id>`
then `tools/convert_tags.py`. Without `--tags-table` the proxy runs
lexical-only (still strong on exact-term queries).

## Honest limits

Bounded replay answers evidence-shaped questions (lookups, multi-key, variable
tracking, multi-hop with lexical anchors). Tasks needing diffuse attention over
the whole context — e.g., full-document summarization — should go directly to
the backend; the proxy is a router, not a replacement for attention.
