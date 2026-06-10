# SPDX-License-Identifier: Apache-2.0
"""Zero-config resolution: model name -> (tokenizer id, bundled tags table).

Patterns are matched against the request's model string (lowercased), so
'qwen2.5:7b', 'qwen2.5-coder:14b-instruct-q4_K_M' and 'Qwen/Qwen2.5-32B'
all resolve to the same tokenizer family and table. Unknown models fall back
to the Llama tokenizer in lexical-only mode (block splitting only needs a
consistent tokenization; the replay is decoded back to text)."""
from __future__ import annotations

from importlib import resources


# (substring pattern, HF tokenizer id, bundled table filename or None)
MODEL_REGISTRY: list[tuple[str, str, str | None]] = [
    ("qwen2.5", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5_k1024.npz"),
    ("qwen2",   "Qwen/Qwen2.5-7B-Instruct", "qwen2.5_k1024.npz"),
    ("llama3",  "NousResearch/Meta-Llama-3.1-8B-Instruct", "llama3.1_k512.npz"),
    ("llama-3", "NousResearch/Meta-Llama-3.1-8B-Instruct", "llama3.1_k512.npz"),
]

FALLBACK = ("NousResearch/Meta-Llama-3.1-8B-Instruct", None)


def resolve_model(model_name: str) -> tuple[str, str | None]:
    """Return (tokenizer_id, table_path_or_None) for a served model name."""
    low = (model_name or "").lower()
    for pat, tok_id, table in MODEL_REGISTRY:
        if pat in low:
            return tok_id, bundled_table_path(table) if table else None
    return FALLBACK[0], None


def bundled_table_path(filename: str) -> str:
    ref = resources.files("replaykv_proxy").joinpath("tables").joinpath(filename)
    return str(ref)
