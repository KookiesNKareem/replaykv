# SPDX-License-Identifier: Apache-2.0
"""replaykv-proxy entrypoint."""
from __future__ import annotations

import argparse
import os


def main():
    ap = argparse.ArgumentParser(
        description="Bounded-replay proxy: serve million-token contexts on any "
                    "OpenAI-compatible backend (Ollama, llama.cpp, LM Studio, vLLM).")
    ap.add_argument("--backend-url", default=os.environ.get(
        "AKV_BACKEND", "http://localhost:11434/v1"),
        help="OpenAI-compatible backend base URL (default: local Ollama)")
    ap.add_argument("--tokenizer", default="",
                    help="HF tokenizer id matching the served model. Omit to "
                         "auto-resolve per request from the model name "
                         "(Qwen2.5/Llama3 families ship bundled tag tables)")
    ap.add_argument("--tags-table", default="",
                    help=".npz concept-tag table (only with --tokenizer; "
                         "omit both for zero-config mode)")
    ap.add_argument("--model", default="",
                    help="force this model name on backend requests (e.g. the "
                         "Ollama tag like 'qwen2.5:7b')")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--takeover", action="store_true",
                    help="drop-in mode: bind Ollama's port 11434 and run the "
                         "real Ollama on 11435 (started automatically if "
                         "needed) — existing apps need NO changes at all")
    ap.add_argument("--min-context-tokens", type=int, default=8192,
                    help="requests below this pass through untouched")
    ap.add_argument("--topk", type=int, default=48)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--gate-score", type=float, default=1.0)
    ap.add_argument("--max-replay-tokens", type=int, default=12000)
    args = ap.parse_args()

    import uvicorn

    from .replay import ReplayConfig, ReplayEngine
    from .server import ProxyState, create_app

    if args.takeover:
        import socket
        import subprocess
        import time as _time

        def _listening(port: int) -> bool:
            with socket.socket() as s:
                return s.connect_ex(("127.0.0.1", port)) == 0

        if _listening(11434):
            raise SystemExit(
                "[replaykv-proxy] port 11434 is already taken (Ollama is "
                "running there). Stop it first, e.g.:\n"
                "  brew services stop ollama   # macOS\n"
                "  sudo systemctl stop ollama  # Linux\n"
                "then re-run with --takeover (the proxy starts Ollama on 11435 itself).")
        if not _listening(11435):
            print("[replaykv-proxy] starting ollama on 127.0.0.1:11435")
            subprocess.Popen(["ollama", "serve"],
                             env={**os.environ, "OLLAMA_HOST": "127.0.0.1:11435"},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            for _ in range(30):
                if _listening(11435):
                    break
                _time.sleep(0.5)
        args.backend_url = "http://127.0.0.1:11435/v1"
        args.port = 11434

    cfg = ReplayConfig(topk=args.topk, radius=args.radius,
                       gate_score=args.gate_score,
                       max_replay_tokens=args.max_replay_tokens)
    fixed_tok = fixed_engine = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        fixed_tok = AutoTokenizer.from_pretrained(args.tokenizer)
        tag_table = None
        if args.tags_table:
            from .tags import load_tag_table

            tag_table = load_tag_table(args.tags_table)
            tag_table.bind_vocab(len(fixed_tok))
            print(f"[replaykv-proxy] tags: {args.tags_table} (K={tag_table.k})")
        fixed_engine = ReplayEngine(cfg, tag_table)
    state = ProxyState(args.backend_url, cfg, args.min_context_tokens,
                       args.model or None, fixed_tokenizer=fixed_tok,
                       fixed_engine=fixed_engine)
    mode = args.tokenizer or "zero-config (per-request resolution)"
    print(f"[replaykv-proxy] {args.host}:{args.port} -> {args.backend_url} "
          f"(min_context={args.min_context_tokens}, topk={args.topk}, mode={mode})")
    uvicorn.run(create_app(state), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
