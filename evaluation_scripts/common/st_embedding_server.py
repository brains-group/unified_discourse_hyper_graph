#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/embeddings server backed by sentence-transformers.

Fallback for when vLLM cannot host the embedding model (voyageai/voyage-4-nano ships a custom
Qwen3-bidirectional module loaded with trust_remote_code). HippoRAG 2, MemGraphRAG and
MS GraphRAG all talk to an OpenAI-style embeddings endpoint, so pointing them at this server gives
them exactly the vectors LogicRAG gets from SentenceTransformer directly (plain encode, no prompt).

    python evaluation_scripts/common/st_embedding_server.py --model voyageai/voyage-4-nano --port 8001
    # HippoRAG 2 / MemGraphRAG: http://localhost:8001/v1/embeddings    MS GraphRAG: http://localhost:8001/v1
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentence_transformers import SentenceTransformer


def make_handler(model, model_name, lock, batch_size):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/").endswith("/models"):
                self._send(200, {"object": "list", "data": [{"id": model_name, "object": "model"}]})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            if not self.path.rstrip("/").endswith("/embeddings"):
                return self._send(404, {"error": "not found"})
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = req["input"]
            if isinstance(texts, str):
                texts = [texts]
            if texts and not isinstance(texts[0], str):
                return self._send(400, {"error": "token-id input is not supported; send strings"})
            with lock:
                vecs = model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
            self._send(200, {
                "object": "list", "model": model_name,
                "data": [{"object": "embedding", "index": i, "embedding": v.tolist()} for i, v in enumerate(vecs)],
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            })

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="voyageai/voyage-4-nano")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()
    model = SentenceTransformer(args.model, trust_remote_code=True, device=args.device)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, args.model, threading.Lock(), args.batch_size))
    print(f"serving {args.model} (dim={model.get_sentence_embedding_dimension()}) on http://{args.host}:{args.port}/v1/embeddings", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
