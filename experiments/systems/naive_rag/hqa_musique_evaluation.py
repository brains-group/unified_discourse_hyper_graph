#!/usr/bin/env python3
"""
naive_rag_eval.py  (experiments/systems/naive_rag/hqa_musique_evaluation.py)
=============================================================================
Naive dense-retrieval RAG baseline for HotpotQA / MuSiQue.

All paragraphs in the corpus are embedded via a locally-running Ollama model
and stored in a FAISS flat index.  At query time the question is embedded the
same way and the top-k nearest paragraphs are retrieved by cosine similarity.
Their full original text is concatenated and passed to the LLM for answer
generation using the same DSPy signatures as the KGGen and HyperGraphRAG
harnesses so results are directly comparable.

Embedding is done by calling Ollama's OpenAI-compatible /v1/embeddings
endpoint via litellm, with ThreadPoolExecutor parallelism.  No local GPU
inference or SentenceTransformer is involved.

TWO MODES (--mode)
------------------
  index       Embed the full corpus via Ollama and save a FAISS flat index +
              paragraph ID list + build metadata to run_dir/.
  evaluation  Load the saved FAISS index, embed all questions via Ollama,
              retrieve top-k paragraphs per question, and score EM / F1.

EMBEDDING
---------
  --embed-model      Ollama model tag (default: embeddinggemma:300m)
  --embed-api-base   Ollama base URL   (default: http://127.0.0.1:11434/v1)
  --embed-concurrency  Parallel requests to Ollama (default: 8)

Each paragraph / question is sent as a single embedding request.  Concurrency
is achieved via ThreadPoolExecutor so Ollama can pipeline requests.

PROVIDERS (--provider)  [evaluation mode only]
-----------------------------------------------
  local_vllm (default)
      model    : Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
      api-base : http://localhost:8000/v1
      api-key  : EMPTY
  deepseek
      model    : deepseek-v4-flash
      api-base : https://api.deepseek.com
      api-key  : DEEPSEEK_API_KEY env var  or  --api-key

DIRECTORY LAYOUT
----------------
  run_dir/
      faiss_index.bin             <- FAISS IndexFlatIP (L2-normalised vectors)
      paragraph_ids.json          <- [pid_0, pid_1, ...] maps index pos -> id
      index_metadata.json         <- embed model, corpus path, build stats
      evaluation_results.json     <- first eval run
      evaluation_results_v1.json  <- second eval run (non-destructive)
      evaluation_results_v2.json  ...

EXAMPLES
--------
  # Build FAISS index via Ollama
  python -m experiments.systems.naive_rag.hqa_musique_evaluation \\
      --mode index --dataset hotpotqa \\
      --corpus-file ./data/hotpot_corpus_1k.jsonl \\
      --run-dir ./results/hotpotqa/naive_rag/run1 \\
      --embed-concurrency 8

  # Evaluate with deepseek, top-10 retrieval, 2000-token context
  python -m experiments.systems.naive_rag.hqa_musique_evaluation \\
      --mode evaluation --dataset hotpotqa --provider deepseek \\
      --questions-file ./data/hotpot_questions_1k.jsonl \\
      --run-dir ./results/hotpotqa/naive_rag/run1 \\
      --retrieval-k 10 --max-context-tokens 2000 --eval-rpm 50

  # Re-evaluate with different k (writes evaluation_results_v1.json)
  python -m experiments.systems.naive_rag.hqa_musique_evaluation \\
      --mode evaluation --dataset hotpotqa --provider deepseek \\
      --questions-file ./data/hotpot_questions_1k.jsonl \\
      --run-dir ./results/hotpotqa/naive_rag/run1 \\
      --retrieval-k 20 --max-context-tokens 4000 --eval-rpm 50
"""

import os, sys, json, time, uuid, threading, argparse, traceback
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss
import numpy as np
import litellm
from transformers import AutoTokenizer

from experiments.evaluate_mine import configure_dspy, ask_llm_query, ask_llm_query_short
from experiments.utils.multihop_utils import (
    HOTPOTQA, MUSIQUE, SUPPORTED_DATASETS,
    load_corpus_jsonl, load_questions_jsonl,
    score_em, score_f1, truncate_tokens, safe_div,
    next_eval_results_path,
)

# Suppress litellm's verbose logging — we only want our own progress output.
litellm.suppress_debug_info = True
os.environ.setdefault("LITELLM_LOG", "ERROR")


# =============================================================================
# PROVIDER CONFIGS  (LLM for answer generation; not used for embedding)
# =============================================================================

@dataclass
class ProviderConfig:
    provider: str
    model:    str
    api_base: str
    api_key:  str


PROVIDER_DEFAULTS = {
    "local_vllm": ProviderConfig(
        provider="local_vllm",
        model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        api_base="http://localhost:8000/v1",
        api_key="EMPTY",
    ),
    "deepseek": ProviderConfig(
        provider="deepseek",
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com",
        api_key="",
    ),
}


def resolve_provider(args) -> ProviderConfig:
    base     = PROVIDER_DEFAULTS[args.provider]
    model    = args.model    or base.model
    api_base = args.api_base or base.api_base
    api_key  = args.api_key  or base.api_key
    if args.provider == "deepseek" and not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if not api_key:
            print("[WARN] provider=deepseek: no api-key found. "
                  "Set DEEPSEEK_API_KEY env var or pass --api-key.")
    return ProviderConfig(provider=args.provider, model=model,
                          api_base=api_base, api_key=api_key)


def litellm_model_string(cfg: ProviderConfig) -> str:
    if cfg.provider == "local_vllm":
        return f"openai/{cfg.model}"
    if cfg.provider == "deepseek":
        return f"deepseek/{cfg.model}"
    return cfg.model


# =============================================================================
# RATE LIMITER  (thread-safe token bucket, for LLM eval calls)
# =============================================================================

class RateLimiter:
    """Thread-safe token-bucket rate limiter.  rpm=0 disables entirely."""
    def __init__(self, rpm: int = 0):
        self.rpm      = rpm
        self.disabled = rpm <= 0
        if not self.disabled:
            self._cap  = float(rpm)
            self._tok  = float(rpm)
            self._rate = rpm / 60.0
            self._last = time.monotonic()
            self._lock = threading.Lock()

    def acquire(self):
        if self.disabled:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._tok  = min(self._cap, self._tok + (now - self._last) * self._rate)
                self._last = now
                if self._tok >= 1.0:
                    self._tok -= 1.0
                    return
            time.sleep(1.0 / self._rate)

    def __enter__(self):  self.acquire(); return self
    def __exit__(self, *_): pass
    def __repr__(self):
        return ("RateLimiter(disabled)" if self.disabled
                else f"RateLimiter(rpm={self.rpm})")


# =============================================================================
# OLLAMA EMBEDDING
# =============================================================================

def _embed_texts(
    texts: list,
    embed_model: str,
    embed_api_base: str,
    concurrency: int = 8,
    label: str = "",
    max_retries: int = 3,
) -> np.ndarray:
    """
    Embed `texts` via Ollama's OpenAI-compatible /v1/embeddings endpoint.

    Each text is sent as a separate litellm.embedding() call.  `concurrency`
    controls how many requests are in flight at once via ThreadPoolExecutor.
    Vectors are L2-normalised so that inner product in the FAISS IndexFlatIP
    equals cosine similarity.

    Returns float32 array of shape (len(texts), embedding_dim).
    Failed embeddings are replaced with a zero vector and a warning is printed.
    """
    model_str = f"openai/{embed_model}"
    n         = len(texts)
    results   = [None] * n
    err_count = [0]
    done_cnt  = [0]
    lock      = threading.Lock()
    prefix    = f"[Embed{' ' + label if label else ''}]"

    def _one(i: int, text: str):
        for attempt in range(max_retries):
            try:
                resp = litellm.embedding(
                    model=model_str,
                    input=[text],
                    api_base=embed_api_base,
                    api_key="ollama",
                )
                results[i] = resp.data[0]["embedding"]
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    with lock:
                        err_count[0] += 1
                    print(f"{prefix} ERROR idx={i}: {e}")
                else:
                    time.sleep(1.0 * (attempt + 1))

    def _tracked(i: int, text: str):
        _one(i, text)
        with lock:
            done_cnt[0] += 1
            d = done_cnt[0]
        if d % max(1, n // 20) == 0 or d == n:
            print(f"{prefix} {d:,}/{n:,}  ({d/n*100:.0f}%)")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_tracked, i, t) for i, t in enumerate(texts)]
        for f in as_completed(futs):
            f.result()  # surface any uncaught exceptions

    elapsed = round(time.time() - t0, 2)
    rate    = round(n / elapsed, 1) if elapsed > 0 else 0
    print(f"{prefix} Done: {n:,} texts in {elapsed}s  ({rate} texts/s)  "
          f"errors={err_count[0]}")

    # Determine embedding dim from first successful result
    dim = next((len(v) for v in results if v is not None), 0)
    if dim == 0:
        raise RuntimeError(f"{prefix} All {n} embedding requests failed.")

    if err_count[0]:
        print(f"{prefix} WARNING: {err_count[0]} failed — replaced with zero vectors.")

    arr   = np.array(
        [v if v is not None else [0.0] * dim for v in results],
        dtype=np.float32,
    )
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
    return arr / norms


def _warmup_embed(embed_model: str, embed_api_base: str) -> int:
    """
    Embed one test string, confirm Ollama is reachable, and return the
    embedding dimension.  Raises on connection failure.
    """
    resp = litellm.embedding(
        model=f"openai/{embed_model}",
        input=["warmup"],
        api_base=embed_api_base,
        api_key="ollama",
    )
    return len(resp.data[0]["embedding"])


# =============================================================================
# FAISS INDEX FILE NAMES
# =============================================================================

FAISS_INDEX_FILE    = "faiss_index.bin"
PARAGRAPH_IDS_FILE  = "paragraph_ids.json"
INDEX_METADATA_FILE = "index_metadata.json"


def _index_paths(run_dir: str):
    return (
        os.path.join(run_dir, FAISS_INDEX_FILE),
        os.path.join(run_dir, PARAGRAPH_IDS_FILE),
        os.path.join(run_dir, INDEX_METADATA_FILE),
    )


# =============================================================================
# INDEX BUILDING
# =============================================================================

def run_index(args):
    """
    Embeds all corpus paragraphs via Ollama and saves a FAISS IndexFlatIP.

    Truncation of paragraphs longer than the embed model's context window is
    handled server-side by Ollama — the LLM prompt at evaluation time always
    uses the full untruncated text from the corpus JSONL.
    """
    os.makedirs(args.run_dir, exist_ok=True)
    index_path, pids_path, meta_path = _index_paths(args.run_dir)

    print(f"[Index] Loading corpus: {args.corpus_file}")
    corpus = load_corpus_jsonl(args.corpus_file)
    pids_ordered = list(corpus.keys())
    texts = [corpus[pid]["full_context"] for pid in pids_ordered]
    print(f"[Index]   {len(texts):,} paragraphs")

    # Confirm Ollama is up and get embedding dimension before the big run.
    print(f"[Index] Warming up embed model: {args.embed_model} "
          f"@ {args.embed_api_base} ...")
    try:
        dim = _warmup_embed(args.embed_model, args.embed_api_base)
    except Exception as e:
        print(f"[Index FATAL] Ollama warmup failed: {e}")
        print(f"  Make sure Ollama is running and '{args.embed_model}' is pulled.")
        sys.exit(1)
    print(f"[Index]   embedding dim={dim}  concurrency={args.embed_concurrency}")

    # Embed all paragraphs
    print(f"[Index] Embedding {len(texts):,} paragraphs ...")
    t_enc = time.time()
    embs = _embed_texts(
        texts,
        embed_model=args.embed_model,
        embed_api_base=args.embed_api_base,
        concurrency=args.embed_concurrency,
        label="corpus",
    )
    enc_time = round(time.time() - t_enc, 2)
    print(f"[Index]   shape={embs.shape}  time={enc_time}s")

    # Build FAISS flat index (inner product on L2-normalised vecs = cosine sim)
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    print(f"[Index]   FAISS index built: {index.ntotal:,} vectors  dim={dim}")

    # Save everything
    faiss.write_index(index, index_path)
    with open(pids_path, "w", encoding="utf-8") as f:
        json.dump(pids_ordered, f, indent=2)
    meta = {
        "timestamp":       datetime.now().isoformat(),
        "dataset":         args.dataset,
        "corpus_file":     os.path.abspath(args.corpus_file),
        "embed_model":     args.embed_model,
        "embed_api_base":  args.embed_api_base,
        "embed_backend":   "ollama",
        "n_paragraphs":    len(texts),
        "embedding_dim":   dim,
        "encode_time_sec": enc_time,
        "embed_concurrency": args.embed_concurrency,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Index] Saved:")
    print(f"  {index_path}")
    print(f"  {pids_path}")
    print(f"  {meta_path}")


# =============================================================================
# EVALUATION
# =============================================================================

def _agg(results, key):
    g = defaultdict(list)
    for r in results:
        if r["status"] == "SUCCESS" and r.get(key) is not None:
            g[r[key]].append(r)
    return {str(k): {
        "count":     len(rs),
        "em":        safe_div(sum(r["em"]        for r in rs), len(rs)),
        "f1":        safe_div(sum(r["f1"]        for r in rs), len(rs)),
        "precision": safe_div(sum(r["precision"] for r in rs), len(rs)),
        "recall":    safe_div(sum(r["recall"]    for r in rs), len(rs)),
    } for k, rs in g.items()}


def run_evaluation(args, cfg: ProviderConfig):
    """
    Loads the FAISS index, embeds all questions via Ollama, retrieves top-k
    paragraphs per question, and generates answers via DSPy.

    Embedding and LLM calls are both parallelised with ThreadPoolExecutor.
    Evaluation results are versioned (never overwrite existing files).
    """
    index_path, pids_path, meta_path = _index_paths(args.run_dir)
    for p in (index_path, pids_path, meta_path):
        if not os.path.exists(p):
            print(f"[Evaluation FATAL] Missing index file: {p}")
            print("Run --mode index first.")
            sys.exit(1)

    with open(meta_path, "r", encoding="utf-8") as f:
        index_meta = json.load(f)
    with open(pids_path, "r", encoding="utf-8") as f:
        pid_list = json.load(f)

    print(f"[Evaluation] Loading FAISS index: {index_path}")
    index = faiss.read_index(index_path)
    print(f"[Evaluation]   {index.ntotal:,} vectors  dim={index.d}  "
          f"embed_model={index_meta.get('embed_model', '?')}")

    # Resolve corpus path
    corpus_file = index_meta.get("corpus_file", "")
    if not os.path.exists(corpus_file):
        corpus_file = args.corpus_file or ""
    if not corpus_file or not os.path.exists(corpus_file):
        print(f"[Evaluation FATAL] Cannot find corpus file. "
              f"Tried index_metadata ({index_meta.get('corpus_file', 'N/A')!r}) "
              f"and --corpus-file ({args.corpus_file!r}).")
        sys.exit(1)
    print(f"[Evaluation] Loading corpus: {corpus_file}")
    corpus = load_corpus_jsonl(corpus_file)

    print(f"[Evaluation] Loading questions: {args.questions_file}")
    questions = load_questions_jsonl(args.questions_file)
    print(f"[Evaluation]   {len(questions):,} questions")

    # Resolve embed endpoint (prefer what's stored in metadata)
    embed_model    = index_meta.get("embed_model",    args.embed_model)
    embed_api_base = index_meta.get("embed_api_base", args.embed_api_base)

    # Embed all questions up front
    print(f"[Evaluation] Embedding {len(questions):,} questions via Ollama ...")
    t_qemb = time.time()
    q_embs = _embed_texts(
        [q["question"] for q in questions],
        embed_model=embed_model,
        embed_api_base=embed_api_base,
        concurrency=args.embed_concurrency,
        label="questions",
    )
    print(f"[Evaluation]   Questions embedded in {round(time.time()-t_qemb, 2)}s")

    # Tokenizer for LLM context truncation
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        print(f"[Evaluation] Tokenizer loaded: {cfg.model}")
    except Exception:
        import tiktoken

        class _TiktokenWrapper:
            def __init__(self, enc): self._enc = enc
            def encode(self, text, **kw): return self._enc.encode(text)
            def decode(self, ids, **kw): return self._enc.decode(ids)

        tokenizer = _TiktokenWrapper(tiktoken.get_encoding("cl100k_base"))
        print("[Evaluation] HuggingFace tokenizer unavailable; "
              "falling back to tiktoken cl100k_base.")

    # Configure DSPy
    configure_dspy(
        model=litellm_model_string(cfg),
        api_base=cfg.api_base,
        api_key=cfg.api_key,
    )
    if args.short_answer:
        ans_label = "ShortAnswer (span-style, EM/F1 optimised)"
    elif args.chain_of_thought:
        ans_label = "ChainOfThought (sentence-style)"
    else:
        ans_label = "Predict (sentence-style)"
    print(f"[Evaluation] DSPy configured  model={litellm_model_string(cfg)}  "
          f"answer_module={ans_label}")

    limiter = RateLimiter(rpm=args.eval_rpm)
    print(f"[Evaluation] {limiter}  max_workers={args.max_workers}")
    print(f"[Evaluation] retrieval_k={args.retrieval_k}  "
          f"max_context_tokens={args.max_context_tokens}")

    def _eval_one(i, q, q_emb):
        qid      = q.get("question_id", "")
        question = q["question"]
        gold     = q["answer"]
        aliases  = q.get("answer_aliases", [])
        qtype    = q.get("type", "unknown")
        qlevel   = q.get("level", None)

        retrieved_pids = []
        llm_ans = raw_ctx = trunc_ctx = ""
        em = f1 = prec = rec = 0.0
        ctx_tok = 0
        status = "SUCCESS"
        err    = ""

        try:
            # FAISS retrieval (thread-safe: read-only index)
            _D, I = index.search(q_emb.reshape(1, -1), args.retrieval_k)
            retrieved_pids = [
                pid_list[idx] for idx in I[0]
                if 0 <= idx < len(pid_list)
            ]

            # Assemble context from full (untruncated) corpus text
            parts   = [corpus[pid]["full_context"]
                       for pid in retrieved_pids if pid in corpus]
            raw_ctx = "\n\n---\n\n".join(parts)

            # Truncate to LLM token budget
            trunc_ctx, ctx_tok = truncate_tokens(raw_ctx, tokenizer,
                                                 args.max_context_tokens)

            # Generate answer via DSPy (same signatures as KGGen / HGRAG)
            with limiter:
                if args.short_answer:
                    llm_ans = ask_llm_query_short(question=question,
                                                  context=trunc_ctx)
                else:
                    llm_ans = ask_llm_query(question=question,
                                            context=trunc_ctx,
                                            use_chain_of_thought=args.chain_of_thought)

            em            = score_em(llm_ans, gold, aliases=aliases, dataset=args.dataset)
            f1, prec, rec = score_f1(llm_ans, gold, aliases=aliases, dataset=args.dataset)

        except Exception as e:
            status = "ERROR"
            err    = str(e)
            print(f"[Eval ERROR] {qid}: {e}")

        return {
            "question_id":              qid,
            "question":                 question,
            "expected_answer":          gold,
            "expected_answer_aliases":  aliases,
            "type":                     qtype,
            "level":                    qlevel,
            "llm_answer":               llm_ans,
            "em":                       float(em),
            "f1":                       float(f1),
            "precision":                float(prec),
            "recall":                   float(rec),
            "retrieved_paragraph_ids":  retrieved_pids,
            "retrieved_context":        trunc_ctx,
            "raw_context_length_chars": len(raw_ctx),
            "context_tokens":           ctx_tok,
            "max_context_tokens":       args.max_context_tokens,
            "retrieval_k":              args.retrieval_k,
            "status":                   status,
            **({"error": err} if err else {}),
        }

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        fmap = {
            ex.submit(_eval_one, i, q, q_embs[i]): i
            for i, q in enumerate(questions)
        }
        done = 0
        for future in as_completed(fmap):
            r = future.result()
            results.append(r)
            done += 1
            if done % max(1, len(questions) // 20) == 0 or done == len(questions):
                ok   = [x for x in results if x["status"] == "SUCCESS"]
                n_ok = len(ok)
                print(f"[Evaluation] {done}/{len(questions)}  "
                      f"EM={safe_div(sum(x['em'] for x in ok), n_ok):.4f}  "
                      f"F1={safe_div(sum(x['f1'] for x in ok), n_ok):.4f}")

    eval_t = round(time.time() - t0, 2)
    ok  = [r for r in results if r["status"] == "SUCCESS"]
    bad = [r for r in results if r["status"] != "SUCCESS"]
    n   = len(ok)

    overall = {
        "em":                    safe_div(sum(r["em"]                    for r in ok), n),
        "f1":                    safe_div(sum(r["f1"]                    for r in ok), n),
        "precision":             safe_div(sum(r["precision"]             for r in ok), n),
        "recall":                safe_div(sum(r["recall"]                for r in ok), n),
        "avg_context_tokens":    safe_div(sum(r["context_tokens"]        for r in ok), n),
        "avg_raw_context_chars": safe_div(sum(r["raw_context_length_chars"] for r in ok), n),
    }
    by_type  = _agg(ok, "type")
    by_level = _agg(ok, "level") if args.dataset == HOTPOTQA else None

    output = {
        "run_id":    f"naive_rag_eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "system":    "NAIVE_RAG",
        "dataset":   args.dataset,
        "metadata": {
            "questions_file":     os.path.abspath(args.questions_file),
            "corpus_file":        os.path.abspath(corpus_file),
            "run_dir":            os.path.abspath(args.run_dir),
            "provider":           cfg.provider,
            "model":              cfg.model,
            "api_base":           cfg.api_base,
            "embed_model":        embed_model,
            "embed_api_base":     embed_api_base,
            "embed_backend":      "ollama",
            "retrieval_params":   {"k": args.retrieval_k},
            "max_context_tokens": args.max_context_tokens,
            "answer_module":      ans_label,
            "short_answer":       args.short_answer,
            "chain_of_thought":   args.chain_of_thought,
            "max_workers":        args.max_workers,
            "eval_rpm":           args.eval_rpm,
        },
        "index": index_meta,
        "evaluation": {
            "total_questions":     len(questions),
            "successful":          n,
            "errored":             len(bad),
            "evaluation_time_sec": eval_t,
            "overall":             overall,
            "by_type":             by_type,
            **({"by_level": by_level} if by_level else {}),
            "results":             results,
        },
    }

    out_path = next_eval_results_path(args.run_dir)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[Evaluation] DONE in {eval_t}s  ->  {out_path}")
    print(f"[Evaluation]   EM={overall['em']:.4f}  F1={overall['f1']:.4f}  "
          f"n={n}  errors={len(bad)}")
    for k, v in sorted(by_type.items()):
        print(f"     type {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    if by_level:
        for k, v in sorted(by_level.items()):
            print(f"    level {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Naive dense-retrieval RAG baseline for HotpotQA / MuSiQue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--mode",    required=True, choices=["index", "evaluation"])
    p.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS))
    p.add_argument("--run-dir", required=True, type=str,
                   help="Directory to hold the FAISS index and evaluation results.")
    p.add_argument("--corpus-file", type=str, default=None,
                   help="Path to corpus JSONL (required for --mode index; "
                        "optional for --mode evaluation if stored in index_metadata).")
    p.add_argument("--questions-file", type=str, default=None,
                   help="Path to questions JSONL (required for --mode evaluation).")

    # ---- Embedding (Ollama) ----
    p.add_argument("--embed-model", type=str, default="embeddinggemma:300m",
                   help="Ollama model tag for embeddings. "
                        "Default: embeddinggemma:300m. "
                        "Must match the model used when the index was built.")
    p.add_argument("--embed-api-base", type=str,
                   default="http://127.0.0.1:11434/v1",
                   help="Ollama base URL. Default: http://127.0.0.1:11434/v1.")
    p.add_argument("--embed-concurrency", type=int, default=8,
                   help="Number of parallel embedding requests to Ollama. "
                        "Default: 8. Raise if Ollama is fast; lower if you "
                        "see connection errors.")

    # ---- LLM provider (evaluation only) ----
    p.add_argument("--provider", default="local_vllm",
                   choices=list(PROVIDER_DEFAULTS),
                   help="LLM provider for answer generation (evaluation mode only). "
                        "local_vllm (default) or deepseek.")
    p.add_argument("--model",    type=str, default="",
                   help="Override provider default model name.")
    p.add_argument("--api-base", type=str, default="",
                   help="Override provider default API base URL.")
    p.add_argument("--api-key",  type=str, default="",
                   help="Override provider default API key.")

    # ---- Retrieval (evaluation only) ----
    p.add_argument("--retrieval-k", type=int, default=10,
                   help="Number of paragraphs to retrieve per question. "
                        "Default: 10.")
    p.add_argument("--max-context-tokens", type=int, default=2000,
                   help="Token budget for context passed to the answer LLM. "
                        "Default: 2000.")

    # ---- Answer generation (evaluation only) ----
    p.add_argument("--short-answer", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use ShortAnswerQuestion (span-style) for answer generation "
                        "(default: on). Pass --no-short-answer for sentence-style.")
    p.add_argument("--chain-of-thought", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use dspy.ChainOfThought (default: on). Only applies when "
                        "--no-short-answer is set.")

    # ---- Rate limiting / concurrency (evaluation only) ----
    p.add_argument("--eval-rpm", type=int, default=0, metavar="N",
                   help="Max LLM calls/minute during evaluation. 0=unlimited. "
                        "Recommended for DeepSeek free tier: 50.")
    p.add_argument("--max-workers", type=int, default=10,
                   help="Thread pool size for parallel LLM answer generation. "
                        "Default: 10.")

    args = p.parse_args()

    # ---- Validate ----
    if args.mode == "index" and not args.corpus_file:
        p.error("--mode index requires --corpus-file")
    if args.mode == "evaluation" and not args.questions_file:
        p.error("--mode evaluation requires --questions-file")

    if args.mode == "index":
        ans_mode = "N/A (index mode)"
    elif args.short_answer:
        ans_mode = "ShortAnswer (span-style, EM/F1 optimised)"
    elif args.chain_of_thought:
        ans_mode = "ChainOfThought (sentence-style)"
    else:
        ans_mode = "Predict (sentence-style)"

    print(f"\n{'='*65}")
    print(f"  NAIVE RAG {args.mode.upper()} -- {args.dataset.upper()}")
    print(f"  embed-model   : {args.embed_model}")
    print(f"  embed-api-base: {args.embed_api_base}")
    print(f"  embed-concur  : {args.embed_concurrency}")
    print(f"  run-dir       : {args.run_dir}")
    if args.mode == "index":
        print(f"  corpus        : {args.corpus_file}")
    else:
        cfg = resolve_provider(args)
        print(f"  provider      : {cfg.provider}")
        print(f"  model         : {cfg.model}")
        print(f"  api-base      : {cfg.api_base}")
        print(f"  retrieval-k   : {args.retrieval_k}")
        print(f"  max-ctx-tok   : {args.max_context_tokens}")
        print(f"  answer gen    : {ans_mode}")
        rpm_str = str(args.eval_rpm) if args.eval_rpm > 0 else "unlimited"
        print(f"  eval RPM      : {rpm_str}")
    print(f"{'='*65}\n")

    if args.mode == "index":
        run_index(args)
    else:
        try:
            run_evaluation(args, cfg)
        except KeyboardInterrupt:
            print("\n[Evaluation] Interrupted.")
        except Exception as e:
            print(f"\n[Evaluation FATAL] {e}")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
