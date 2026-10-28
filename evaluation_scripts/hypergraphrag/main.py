#!/usr/bin/env python3
"""HyperGraphRAG (LHRLAB/HyperGraphRAG, NeurIPS 2025) on evaluation_data/<dataset>.

Same three steps as upstream evaluation/README.md, each writing to outputs/<dataset>/:

  index     = upstream Step 1 (script_insert.py): rag.insert(passages)             -> index/
  retrieve  = upstream Step 2 (script_hypergraphrag.py): aquery(q, only_need_context) -> knowledge.json
  answer    = evaluation_scripts/common/answer.py --system hypergraphrag (the reader shared by all systems)
  qa-upstream (optional) = upstream Step 3 (get_generation.py) verbatim   -> predictions_upstream_reader.json

Then score with evaluation_scripts/common/score.py --system hypergraphrag (run.sh does all of this).

What is upstream and what is ours:
- The package in ./hypergraphrag is upstream's evaluation/hypergraphrag (the copy the paper's numbers
  came from) and is not modified. All defaults are upstream's: chunk 1200/100 tiktoken(gpt-4o-mini)
  tokens, entity_extract_max_gleaning=2, entity_summary_to_max_tokens=500, QueryParam() = hybrid mode,
  top_k=60, 4000-token limits for entities / relations / sources, NanoVectorDB cosine threshold 0.2.
  Index concurrency 32/32 (script_insert.py), retrieval with constructor defaults 16/16 and 32
  questions in flight (script_hypergraphrag.py).
- Ingestion: one document per corpus passage, "title\\ntext", like HippoRAG 2. HyperGraphRAG chunks
  inside each document, so no chunk mixes passages.
- LLM: an OpenAI-compatible endpoint (LLM_MODEL / LLM_BASE_URL) in place of gpt-4o-mini. As upstream,
  no temperature or max_tokens is sent for index/retrieval calls, except the max_tokens=500 upstream
  passes for entity summaries. Retry policy = upstream's (tenacity, 3 attempts on rate-limit /
  connection / timeout errors). DeepSeek thinking is disabled unless LLM_DISABLE_THINKING=0.
- Embeddings: an OpenAI-compatible /v1/embeddings server (EMBEDDING_MODEL / EMBEDDING_BASE_URL) in place
  of text-embedding-3-small; the dimension is read from the server.
- Added for crash safety (does not change results): every LLM call is cached in
  outputs/<ds>/llm_cache.sqlite, so a rerun after a crash replays finished calls for free. Upstream
  caches nothing during indexing.
- Reader: the shared common/answer.py reads knowledge.json (whole context as one document). The
  optional `qa-upstream` step answers with get_generation.py's <think>/<answer> prompt verbatim.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import logging

import numpy as np
from openai import APIConnectionError, AsyncOpenAI, OpenAI, RateLimitError, Timeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

HERE = Path(__file__).resolve().parent
EVAL_ROOT = HERE.parent
DATA_ROOT = EVAL_ROOT.parent / "evaluation_data"
# The repo root has another (non-evaluation) copy of the package; make sure ours wins.
sys.path.insert(0, str(HERE))

from hypergraphrag import HyperGraphRAG, QueryParam  # noqa: E402
from hypergraphrag.prompt import PROMPTS  # noqa: E402
from hypergraphrag.utils import EmbeddingFunc  # noqa: E402

logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request otherwise
assert Path(sys.modules["hypergraphrag"].__file__).resolve().parent == HERE / "hypergraphrag", sys.modules["hypergraphrag"].__file__


# ---------------------------------------------------------------- settings
def env(name, default=None):
    value = os.environ.get(name, default)
    if value is None:
        sys.exit(f"export {name} first (source evaluation_scripts/common/env.sh)")
    return value


LLM_MODEL = env("LLM_MODEL", "deepseek-v4-flash")
LLM_BASE_URL = env("LLM_BASE_URL", "https://api.deepseek.com")
LLM_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL = env("EMBEDDING_MODEL", "voyageai/voyage-4-nano")
EMBEDDING_BASE_URL = env("EMBEDDING_BASE_URL", "http://localhost:8001/v1")
EMBEDDING_API_KEY = os.environ.get("EMBEDDING_API_KEY", "EMPTY")
DISABLE_THINKING = os.environ.get("LLM_DISABLE_THINKING", "1") != "0"
# [unh_v1] client timeout per LLM call (default 300 s); a local server under load needs longer (env_qwen.sh: 1800),
# since a timed-out extraction is retried from scratch
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "300"))
EXTRA_BODY = {"thinking": {"type": "disabled"}} if DISABLE_THINKING else None

# upstream llm.py: @retry on openai_complete_if_cache / openai_embedding
UPSTREAM_RETRY = dict(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, Timeout)),
    reraise=True,
)

# upstream evaluation/get_generation.py, verbatim
UPSTREAM_READER_PROMPT = """---Role---

You are a helpful assistant responding to questions based on given knowledge.

---Knowledge---

{knowledge}

---Goal---

Answer the given question.
You must first conduct reasoning inside <think>...</think>.
When you have the final answer, you can output the answer inside <answer>...</answer>.

Output format for answer:
<think>
...
</think>
<answer>
...
</answer>

---Question---

{question}
"""


# ---------------------------------------------------------------- LLM call cache + usage
class CallCache:
    """sqlite cache of LLM responses, keyed on the full request. Safe across threads."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)")
        self.conn.commit()
        self.lock = threading.Lock()
        self.usage = {"calls": 0, "cache_hits": 0, "prompt_tokens": 0, "completion_tokens": 0, "empty_responses": 0}

    @staticmethod
    def key(request: dict) -> str:
        return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def get(self, key):
        with self.lock:
            row = self.conn.execute("SELECT response FROM cache WHERE key = ?", (key,)).fetchone()
            if row is not None:
                self.usage["cache_hits"] += 1
            return None if row is None else row[0]

    def put(self, key, response, usage):
        with self.lock:
            self.usage["calls"] += 1
            if usage is not None:
                self.usage["prompt_tokens"] += usage.prompt_tokens or 0
                self.usage["completion_tokens"] += usage.completion_tokens or 0
            if not response:
                self.usage["empty_responses"] += 1
                return  # never cache an empty answer; a rerun retries it
            self.conn.execute("INSERT OR REPLACE INTO cache (key, response) VALUES (?, ?)", (key, response))
            self.conn.commit()


def make_llm_func(cache: CallCache, temperature):
    """Drop-in for upstream gpt_4o_mini_complete / openai_complete_if_cache."""
    client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), **UPSTREAM_RETRY)
    async def _create(**params):
        return await client.chat.completions.create(**params)

    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        kwargs.pop("hashing_kv", None)
        kwargs.pop("keyword_extraction", None)  # upstream's JSON-mode path is never reached by HyperGraphRAG's operate.py
        if kwargs.pop("stream", False):
            raise ValueError("streaming is not used by this harness")
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})
        params = {"model": LLM_MODEL, "messages": messages, **kwargs}
        if temperature is not None:
            params["temperature"] = temperature
        if EXTRA_BODY:
            params["extra_body"] = EXTRA_BODY
        key = cache.key(params)
        hit = cache.get(key)
        if hit is not None:
            return hit
        response = await _create(**params)
        content = response.choices[0].message.content or ""
        cache.put(key, content, response.usage)
        return content

    return llm_model_func


async def make_embedding_func(max_async: int) -> EmbeddingFunc:
    client = AsyncOpenAI(base_url=EMBEDDING_BASE_URL, api_key=EMBEDDING_API_KEY, timeout=300)

    @retry(wait=wait_exponential(multiplier=1, min=4, max=60), **UPSTREAM_RETRY)
    async def embed(texts: list[str]) -> np.ndarray:
        response = await client.embeddings.create(model=EMBEDDING_MODEL, input=texts, encoding_format="float")
        return np.array([d.embedding for d in response.data])

    dim = (await embed(["dimension probe"])).shape[1]
    # upstream openai_embedding: embedding_dim=1536 (text-embedding-3-small), max_token_size=8192
    return EmbeddingFunc(embedding_dim=dim, max_token_size=8192, func=embed, concurrent_limit=max_async)


async def make_rag(out_dir: Path, cache: CallCache, args, llm_max_async: int, embedding_max_async: int) -> HyperGraphRAG:
    # HyperGraphRAG writes hypergraphrag.log into the cwd; keep it with the outputs.
    os.chdir(out_dir)
    return HyperGraphRAG(
        working_dir=str(out_dir / "index"),
        llm_model_func=make_llm_func(cache, args.temperature),
        llm_model_name=LLM_MODEL,
        embedding_func=await make_embedding_func(embedding_max_async),
        llm_model_max_async=llm_max_async,
        embedding_func_max_async=embedding_max_async,
    )


# ---------------------------------------------------------------- steps
def load_dataset(data_root: Path, dataset: str, limit: int, quality_inputs: Path = None):
    if dataset == "quality":
        # [unh_v1] QuALITY: one document per dev article {title, text} (indexed as "title\ntext", chunked by
        # HyperGraphRAG itself, 1200/100 per document); questions carry their four options.
        corpus = json.load(open(quality_inputs / "documents.json", encoding="utf-8"))
        samples = json.load(open(quality_inputs / "questions.json", encoding="utf-8"))
        return corpus, samples[:limit] if limit else samples
    corpus = json.load(open(data_root / dataset / f"{dataset}_corpus.json", encoding="utf-8"))
    samples = json.load(open(data_root / dataset / f"{dataset}.json", encoding="utf-8"))
    return corpus, samples[:limit] if limit else samples


async def step_index(args, out_dir: Path, cache: CallCache, corpus):
    docs = [f"{d['title']}\n{d['text']}" for d in corpus]
    batch = args.insert_batch_size or len(docs)
    started = time.time()
    for start in range(0, len(docs), batch):
        part = docs[start:start + batch]
        # upstream script_insert.py: up to 10 attempts, 10 s apart. A fresh instance per attempt, because
        # upstream's limit_async_func_call never releases a slot when a call raises (it would hang).
        for attempt in range(1, args.insert_retries + 1):
            rag = await make_rag(out_dir, cache, args, args.index_llm_max_async, args.index_embedding_max_async)
            try:
                await rag.ainsert(part)
                break
            except Exception as exc:  # noqa: BLE001 - mirror upstream's catch-all retry loop
                print(f"[index] docs {start}-{start + len(part)}: attempt {attempt}/{args.insert_retries} failed: {exc!r}", flush=True)
                # upstream's as_completed leaves the other extraction tasks running; stop them so the
                # retry does not pay for the same calls twice (finished ones are already cached)
                orphans = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                for t in orphans:
                    t.cancel()
                await asyncio.gather(*orphans, return_exceptions=True)
                if attempt == args.insert_retries:
                    raise
                await asyncio.sleep(10)
    rag = await make_rag(out_dir, cache, args, args.index_llm_max_async, args.index_embedding_max_async)
    graph = rag.chunk_entity_relation_graph._graph
    stats = {
        "passages": len(docs),
        "unique_docs_indexed": len(await rag.full_docs.all_keys()),
        "chunks": len(await rag.text_chunks.all_keys()),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "hyperedges": sum(1 for n in graph.nodes if str(n).startswith("<hyperedge>")),
        "seconds": round(time.time() - started, 1),
        "llm_usage_this_run": cache.usage,
        "llm_model": LLM_MODEL, "thinking_disabled": DISABLE_THINKING, "embedding_model": EMBEDDING_MODEL,
    }
    json.dump(stats, open(out_dir / "index_stats.json", "w"), indent=2)
    print(json.dumps(stats, indent=2))
    if stats["unique_docs_indexed"] < len(set(d.strip() for d in docs)):
        sys.exit("[index] some passages are not in full_docs yet; rerun (finished LLM calls are cached)")


async def step_retrieve(args, out_dir: Path, cache: CallCache, samples):
    rag = await make_rag(out_dir, cache, args, args.retrieve_llm_max_async, args.retrieve_embedding_max_async)
    sem = asyncio.Semaphore(args.retrieve_concurrency)

    async def one(question):
        async with sem:
            try:
                # a fresh QueryParam per call: _build_query_context mutates param.mode
                return await rag.aquery(question, QueryParam(only_need_context=True)), None
            except Exception as exc:  # noqa: BLE001
                return None, repr(exc)

    outputs = await tqdm_asyncio.gather(*[one(s["question"]) for s in samples], desc="retrieve")
    rows = []
    for s, (knowledge, error) in zip(samples, outputs):
        status = "error" if error else "no_context" if knowledge is None else \
            "fail_response" if knowledge == PROMPTS["fail_response"] else "ok"
        rows.append({"id": s.get("id", s.get("_id")), "question": s["question"], "knowledge": knowledge,
                     "status": status, "error": error})
    json.dump(rows, open(out_dir / "knowledge.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    counts = {k: sum(r["status"] == k for r in rows) for k in ("ok", "fail_response", "no_context", "error")}
    print(f"[retrieve] wrote {out_dir / 'knowledge.json'}: {counts}")
    if counts["error"]:
        sys.exit("[retrieve] some questions failed; rerun this step (finished LLM calls are cached)")


def step_qa(args, out_dir: Path, cache: CallCache, samples):
    rows = json.load(open(out_dir / "knowledge.json", encoding="utf-8"))
    rows = rows[: len(samples)]
    for r, s in zip(rows, samples):
        assert r["question"] == s["question"], "knowledge.json does not match the dataset order"
    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, timeout=LLM_TIMEOUT)

    def call(params):
        if EXTRA_BODY:
            params["extra_body"] = EXTRA_BODY
        key = cache.key(params)
        hit = cache.get(key)
        if hit is not None:
            return hit
        response = client.chat.completions.create(**params)
        content = response.choices[0].message.content or ""
        cache.put(key, content, response.usage)
        return content

    def answer(row):
        knowledge = row["knowledge"] or ""  # upstream passes whatever aquery returned
        prompt = UPSTREAM_READER_PROMPT.format(knowledge=knowledge, question=row["question"])
        raw = call({"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}]})  # upstream: no other params
        found = re.findall(r"<answer>(.*?)</answer>", raw, flags=re.S)
        return raw, (found[-1].strip() if found else raw)

    with ThreadPoolExecutor(max_workers=args.qa_workers) as ex:
        outputs = list(tqdm(ex.map(answer, rows), total=len(rows), desc="qa (upstream reader)"))

    predictions = [{"id": r["id"], "question": r["question"], "pred_answer": pred, "raw_response": raw,
                    "retrieval_status": r["status"], "retrieved_docs": [r["knowledge"] or ""]}
                   for r, (raw, pred) in zip(rows, outputs)]
    name = "predictions_upstream_reader.json"
    json.dump(predictions, open(out_dir / name, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    empty = sum(not p["pred_answer"] for p in predictions)
    print(f"[qa] wrote {out_dir / name} ({len(predictions)} predictions, {empty} empty); llm usage this run: {cache.usage}")


# ---------------------------------------------------------------- main
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "quality"])
    ap.add_argument("--quality-inputs", default=os.environ.get("QUALITY_INPUTS", str(DATA_ROOT / "quality" / "inputs")),
                    help="--dataset quality: dir with documents.json / questions.json (evaluation_data/build_quality_inputs.py)")
    ap.add_argument("--step", default="all", choices=["index", "retrieve", "qa-upstream", "all"],
                    help="all = index + retrieve; answers come from common/answer.py")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (retrieve/qa); 0 = all")
    ap.add_argument("--data-root", default=str(DATA_ROOT), help="directory holding <dataset>/<dataset>{,_corpus}.json")
    ap.add_argument("--out-dir", default=None, help="default: evaluation_scripts/hypergraphrag/outputs/<dataset>")
    ap.add_argument("--temperature", type=float,
                    default=float(os.environ["LLM_TEMPERATURE"]) if os.environ.get("LLM_TEMPERATURE") else None,
                    help="index/retrieval calls; default: not sent, like upstream (provider default)")
    # index (upstream script_insert.py)
    ap.add_argument("--index-llm-max-async", type=int, default=32)
    ap.add_argument("--index-embedding-max-async", type=int, default=32)
    ap.add_argument("--insert-retries", type=int, default=10)
    ap.add_argument("--insert-batch-size", type=int, default=0,
                    help="passages per rag.insert call; 0 = all at once, like upstream")
    # retrieve (upstream script_hypergraphrag.py)
    ap.add_argument("--retrieve-llm-max-async", type=int, default=16)
    ap.add_argument("--retrieve-embedding-max-async", type=int, default=16)
    ap.add_argument("--retrieve-concurrency", type=int, default=32)
    # qa-upstream
    ap.add_argument("--qa-workers", type=int, default=16)
    return ap.parse_args()


def main():
    args = parse_args()
    if not LLM_API_KEY:
        sys.exit("export DEEPSEEK_API_KEY (or LLM_API_KEY) first")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else HERE / "outputs" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = CallCache(out_dir / "llm_cache.sqlite")
    corpus, samples = load_dataset(Path(args.data_root).resolve(), args.dataset, args.limit, Path(args.quality_inputs).resolve())
    print(f"[hypergraphrag] {args.dataset}: {len(corpus)} documents, {len(samples)} questions, llm={LLM_MODEL} "
          f"(thinking {'disabled' if DISABLE_THINKING else 'provider default'}), embeddings={EMBEDDING_MODEL}", flush=True)
    if args.step in ("index", "all"):
        asyncio.run(step_index(args, out_dir, cache, corpus))
    if args.step in ("retrieve", "all"):
        asyncio.run(step_retrieve(args, out_dir, cache, samples))
    if args.step == "qa-upstream":
        step_qa(args, out_dir, cache, samples)


if __name__ == "__main__":
    main()
