#!/usr/bin/env python3
"""
kggen_multihop_eval.py
======================
KGGen evaluation harness for HotpotQA and MuSiQue.

PROVIDERS  (--provider)
-----------------------
  local_vllm (default)
      model    : Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
      api-base : http://localhost:8000/v1
      api-key  : EMPTY
  deepseek
      model    : deepseek-v4-flash  (current prod model, May 2026)
      api-base : https://api.deepseek.com
      api-key  : DEEPSEEK_API_KEY env var  (or --api-key)
      Note: deepseek-v4-flash uses OpenAI-compatible /chat/completions.
            The legacy alias deepseek-chat still works but retires 2026-07-24.
  Override any provider field: --model / --api-base / --api-key

RATE LIMITING  (--construction-rpm / --eval-rpm)
------------------------------------------------
KGGen has NO native rate limiting. It routes calls via DSPy then LiteLLM
with zero throttling built in; without a limiter you will get 429s from
any external API almost immediately.

This script wraps every LLM call in a thread-safe token-bucket limiter.
  --construction-rpm N   0 = disabled (default; fine for local vLLM)
  --eval-rpm N           0 = disabled (default)

KGGen makes ~2 LLM calls per paragraph (entity pass + relation pass):
  50 RPM => ~25 paragraphs/minute => 1000-para corpus ~40 min
Recommended for DeepSeek free/tier-1: --construction-rpm 50 --eval-rpm 50

EXAMPLES
--------
  # Local vLLM, no rate limit
  python kggen_multihop_eval.py \\
      --mode construction --provider local_vllm \\
      --dataset hotpotqa --corpus-file ./data/hotpot_corpus_1k.jsonl \\
      --run-dir ./results/hotpotqa/kggen/run1 \\
      --checkpoint-interval 100 --max-workers 10

  # DeepSeek v4 flash, 50 RPM
  python kggen_multihop_eval.py \\
      --mode construction --provider deepseek \\
      --dataset hotpotqa --corpus-file ./data/hotpot_corpus_1k.jsonl \\
      --run-dir ./results/hotpotqa/kggen/run1_ds \\
      --checkpoint-interval 50 --max-workers 5 --construction-rpm 50

  # Evaluation (either provider)
  python kggen_multihop_eval.py \\
      --mode evaluation --provider deepseek \\
      --dataset hotpotqa --questions-file ./data/hotpot_questions_1k.jsonl \\
      --run-dir ./results/hotpotqa/kggen/run1_ds --eval-rpm 50
"""

import os, sys, json, time, uuid, threading, argparse, traceback, shutil
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from transformers import AutoTokenizer

from .wrappers import CustomKGGen

from experiments.utils.multihop_utils import (
    HOTPOTQA, MUSIQUE, SUPPORTED_DATASETS,
    load_corpus_jsonl, load_questions_jsonl,
    find_all_checkpoints, find_latest_checkpoint, find_final_checkpoint,
    next_checkpoint_num, save_checkpoint, load_checkpoint,
    FINAL_CHECKPOINT_NAME,
    GlobalTokenAccumulator, attach_token_tracker, detach_token_tracker,
    add_token_counters,
    score_em, score_f1, truncate_tokens, safe_div,
    next_eval_results_path,
)
from experiments.evaluate_mine import ask_llm_query, ask_llm_query_short, configure_dspy


# =============================================================================
# PROVIDER CONFIGS
# =============================================================================

@dataclass
class ProviderConfig:
    """Fully-resolved connection parameters for one LLM provider."""
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
        # deepseek-chat = DeepSeek-V3: non-reasoning model, no thinking tokens.
        # Use deepseek-v4-flash only if reasoning output is acceptable (cannot be disabled via API).
        provider="deepseek",
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com",
        api_key="",
    ),
}


def resolve_provider(args) -> ProviderConfig:
    """Merge provider defaults with any explicit CLI overrides."""
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


def kg_model_string(cfg: ProviderConfig) -> str:
    """
    LiteLLM routing string for KGGen's model= argument.
    local_vllm: needs 'openai/<model>' prefix for non-OpenAI compat endpoints.
    deepseek:   needs 'deepseek/<model>' prefix so LiteLLM identifies the provider.
    """
    if cfg.provider == "local_vllm":
        return f"openai/{cfg.model}"
    if cfg.provider == "deepseek":
        return f"deepseek/{cfg.model}"
    return cfg.model



# =============================================================================
# THREAD-SAFE TOKEN-BUCKET RATE LIMITER
# =============================================================================

class RateLimiter:
    """
    Thread-safe token-bucket rate limiter.  rpm=0 disables entirely.

    Tokens refill at rpm/60 per second, capped at rpm.
    Each acquire() blocks until a token is available, then consumes one.
    Multiple threads run in parallel up to the RPM ceiling.

    Usage:  with limiter: api_call()
    """
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
        return "RateLimiter(disabled)" if self.disabled else f"RateLimiter(rpm={self.rpm})"


# ====================================================================
# CONSTRUCTION
#===================================================================

def run_construction(args, cfg):
    os.makedirs(args.run_dir, exist_ok=True)
    if find_final_checkpoint(args.run_dir):
        print("[!] final-checkpoint already exists. Delete it to rebuild.")
        return

    print(f"[Construction] Loading corpus: {args.corpus_file}")
    corpus = load_corpus_jsonl(args.corpus_file)
    print(f"[Construction]   {len(corpus):,} paragraphs")

    latest = find_latest_checkpoint(args.run_dir)
    if latest:
        print(f"[Construction] Resuming from: {latest}")
        prev_meta, absorbed_ids, base_graph = load_checkpoint(latest)
        cum_tokens = prev_meta.get("cumulative_construction_tokens",
            {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"api_calls":0})
        cum_time   = float(prev_meta.get("cumulative_construction_time_sec", 0.0))
        print(f"[Construction]   {len(absorbed_ids):,} absorbed, "
              f"{len(base_graph.entities):,} entities in base graph")
    else:
        print("[Construction] Starting fresh.")
        absorbed_ids = []
        base_graph   = None
        cum_tokens   = {"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,"api_calls":0}
        cum_time     = 0.0

    absorbed_set = set(absorbed_ids)
    remaining    = [p for pid, p in sorted(corpus.items()) if pid not in absorbed_set]
    print(f"[Construction] {len(remaining):,} remaining "
          f"({len(absorbed_set):,}/{len(corpus):,} already absorbed)")

    os.environ["OPENAI_API_BASE"] = cfg.api_base
    os.environ["OPENAI_API_KEY"]  = cfg.api_key
    kg = CustomKGGen(model=kg_model_string(cfg), api_key=cfg.api_key,
                     api_base=cfg.api_base, retrieval_model=args.embed_model)

    limiter   = RateLimiter(rpm=args.construction_rpm)
    para_rate = (f"~{args.construction_rpm // 2} paras/min"
                 if args.construction_rpm > 0 else "unlimited")
    print(f"[Construction] {limiter}  ({para_rate})")

    def _gen(para):
        pid, title, text = para["id"], para["title"], para["full_context"]
        try:
            with limiter:
                g = kg.generate(input_data=text,
                                context=f"Wikipedia paragraph about: {title}")
            return pid, g
        except Exception as e:
            print(f"[Construction WARN] {pid}: {e}")
            return pid, None

    ckpt_num = next_checkpoint_num(args.run_dir)

    while remaining:
        block     = remaining[:args.checkpoint_interval]
        remaining = remaining[args.checkpoint_interval:]
        print(f"\n[Construction] --- checkpoint-{ckpt_num}: {len(block)} paras ---")
        t0 = time.time()

        blk_tracker = GlobalTokenAccumulator()
        attach_token_tracker(blk_tracker)
        new_subs, new_abs = [], []
        try:
            with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
                fmap = {ex.submit(_gen, p): p["id"] for p in block}
                done = 0
                for future in as_completed(fmap):
                    pid, g = future.result(); done += 1
                    if g is not None: new_subs.append(g); new_abs.append(pid)
                    if done % max(1, len(block)//5) == 0 or done == len(block):
                        print(f"[Construction]   {done}/{len(block)}")
        finally:
            detach_token_tracker()

        to_merge = ([base_graph] if base_graph else []) + new_subs
        if not to_merge:
            print("[Construction WARN] Nothing to aggregate."); ckpt_num += 1; continue

        print(f"[Construction] Aggregating {len(to_merge)} graphs...")
        agg_t0 = time.time()
        agg_tracker = GlobalTokenAccumulator()
        attach_token_tracker(agg_tracker)
        try:
            aggregated = kg.aggregate(to_merge)
        finally:
            detach_token_tracker()
        agg_tokens = agg_tracker.snapshot()

        blk_time   = round(time.time() - t0, 2)
        blk_tokens = add_token_counters(blk_tracker.snapshot(), agg_tokens)
        absorbed_ids.extend(new_abs)
        cum_tokens = add_token_counters(cum_tokens, blk_tokens)
        cum_time   = round(cum_time + blk_time, 2)

        ents  = len(aggregated.entities)
        edges = len(getattr(aggregated, "edges", getattr(aggregated, "relations",
                    getattr(aggregated, "triples", []))))
        meta = {
            "checkpoint_id": f"checkpoint-{ckpt_num}", "checkpoint_num": ckpt_num,
            "timestamp": datetime.now().isoformat(), "dataset": args.dataset,
            "provider": cfg.provider, "model": cfg.model,
            "embedding_model": args.embed_model, "api_base": cfg.api_base,
            "corpus_file": os.path.abspath(args.corpus_file),
            "total_paragraphs_in_corpus": len(corpus),
            "paragraphs_absorbed_so_far": len(absorbed_ids),
            "paragraphs_in_this_block": len(new_abs),
            "paragraphs_failed_in_block": len(block) - len(new_abs),
            "cumulative_graph_entities": ents, "cumulative_graph_edges": edges,
            "cumulative_construction_tokens":   cum_tokens,
            "this_block_tokens":                blk_tokens,
            "cumulative_construction_time_sec": cum_time,
            "this_block_time_sec":              blk_time,
            "construction_params": {
                "checkpoint_interval": args.checkpoint_interval,
                "max_workers": args.max_workers,
                "construction_rpm": args.construction_rpm,
                "delete_old_checkpoints": args.delete_old_checkpoints,
            },
        }
        cp = save_checkpoint(run_dir=args.run_dir, checkpoint_num=ckpt_num,
                             metadata=meta, absorbed_paragraph_ids=absorbed_ids,
                             aggregated_graph=aggregated, kg_instance=kg, final=False)
        print(f"[Construction] Saved {cp}")
        print(f"[Construction]   absorbed={len(absorbed_ids):,}/{len(corpus):,}  "
              f"entities={ents:,}  tokens={cum_tokens['total_tokens']:,}  time={cum_time}s")

        # ---- Optionally delete previous checkpoint-N folders ----
        # We never touch final-checkpoint (it doesn't exist yet at this point,
        # but guard defensively). The newly-saved checkpoint is ckpt_num, so
        # we delete every checkpoint with num < ckpt_num.
        if args.delete_old_checkpoints:
            for old_num, old_path in find_all_checkpoints(args.run_dir):
                if old_num < ckpt_num:
                    shutil.rmtree(old_path, ignore_errors=True)
                    print(f"[Construction] Deleted old checkpoint: {old_path}")

        base_graph = aggregated; ckpt_num += 1

    if base_graph is None:
        print("[Construction FATAL] No graph produced."); return

    latest = find_latest_checkpoint(args.run_dir)
    last_meta, last_abs, _ = load_checkpoint(latest)
    final_meta = {**last_meta,
        "checkpoint_id": FINAL_CHECKPOINT_NAME, "checkpoint_num": None,
        "timestamp": datetime.now().isoformat(), "is_final": True,
        "note": "Embeddings recomputed from graph.json at eval time (no LLM cost)."}
    final_cp = save_checkpoint(run_dir=args.run_dir, checkpoint_num=0,
                               metadata=final_meta, absorbed_paragraph_ids=last_abs,
                               aggregated_graph=base_graph, kg_instance=kg, final=True)
    print(f"\n[Construction] FINAL -> {final_cp}")
    print(f"[Construction]   entities={len(base_graph.entities):,}  "
          f"tokens={cum_tokens['total_tokens']:,}  time={cum_time}s")


# =============================================================================
# EVALUATION
#=============================================================================

def _eval_one(q, kg, nx_g, embs, tok, args, limiter):
    qid     = q.get("question_id", "")
    question= q["question"]; gold = q["answer"]
    aliases = q.get("answer_aliases", [])
    qtype   = q.get("type", "unknown"); qlevel = q.get("level", None)
    llm_ans = raw_ctx = trunc_ctx = ""
    em = f1 = prec = rec = 0.0; ctx_tok = 0
    status = "SUCCESS"; err = ""
    try:
        _, _, raw_ctx = kg.retrieve(
            query=question, node_embeddings=embs, graph=nx_g,
            k=args.retrieval_k, depth=args.retrieval_depth,
            verbose=args.retrieval_verbose,
        )
        trunc_ctx, ctx_tok = truncate_tokens(raw_ctx, tok, args.max_context_tokens)
        with limiter:
            if args.short_answer:
                llm_ans = ask_llm_query_short(question=question, context=trunc_ctx)
            else:
                llm_ans = ask_llm_query(
                    question=question, context=trunc_ctx,
                    use_chain_of_thought=args.chain_of_thought,
                )
        em = score_em(llm_ans, gold, aliases=aliases, dataset=args.dataset)
        f1, prec, rec = score_f1(llm_ans, gold, aliases=aliases, dataset=args.dataset)
    except Exception as e:
        status = "ERROR"; err = str(e); print(f"[Eval ERROR] {qid}: {e}")
    return {
        "question_id": qid, "question": question,
        "expected_answer": gold, "expected_answer_aliases": aliases,
        "type": qtype, "level": qlevel, "llm_answer": llm_ans,
        "em": float(em), "f1": float(f1), "precision": float(prec), "recall": float(rec),
        "retrieved_context": trunc_ctx,
        "raw_context_length_chars": len(raw_ctx), "context_tokens": ctx_tok,
        "max_context_tokens": args.max_context_tokens, "status": status,
        **({"error": err} if err else {}),
    }


def _agg(results, key):
    g = defaultdict(list)
    for r in results:
        if r["status"] == "SUCCESS" and r.get(key) is not None:
            g[r[key]].append(r)
    return {str(k): {
        "count": len(rs),
        "em":        safe_div(sum(r["em"]        for r in rs), len(rs)),
        "f1":        safe_div(sum(r["f1"]        for r in rs), len(rs)),
        "precision": safe_div(sum(r["precision"] for r in rs), len(rs)),
        "recall":    safe_div(sum(r["recall"]    for r in rs), len(rs)),
    } for k, rs in g.items()}


def run_evaluation(args, cfg):
    final_cp = find_final_checkpoint(args.run_dir)
    latest_cp = find_latest_checkpoint(args.run_dir)

    if final_cp:
        cp_to_load = final_cp
    elif latest_cp:
        print(f"[Evaluation WARN] No final-checkpoint found. "
              f"Using latest intermediate checkpoint: {latest_cp}")
        cp_to_load = latest_cp
    else:
        print("[Evaluation FATAL] No checkpoint found in run-dir."); sys.exit(1)

    print(f"[Evaluation] Loading: {cp_to_load}")
    final_meta, abs_ids, combined = load_checkpoint(cp_to_load)
    print(f"[Evaluation]   {len(combined.entities):,} entities, {len(abs_ids):,} paras")

    os.environ["OPENAI_API_BASE"] = cfg.api_base
    os.environ["OPENAI_API_KEY"]  = cfg.api_key
    kg   = CustomKGGen(model=kg_model_string(cfg), api_key=cfg.api_key,
                       api_base=cfg.api_base, retrieval_model=args.embed_model)
    nx_g = kg.to_nx(combined)

    print("[Evaluation] Recomputing embeddings (no LLM calls)...")
    t0 = time.time()
    embs, _ = kg.generate_embeddings(combined)
    emb_t   = round(time.time() - t0, 2)
    print(f"[Evaluation]   {len(embs):,} nodes in {emb_t}s")

    questions = load_questions_jsonl(args.questions_file)
    print(f"[Evaluation] {len(questions):,} questions")

    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.model)
        print(f"[Evaluation] Tokenizer loaded from HuggingFace: {cfg.model}")
    except Exception:
        import tiktoken

        class _TiktokenWrapper:
            def __init__(self, enc):
                self._enc = enc
            def encode(self, text, **kwargs):
                return self._enc.encode(text)
            def decode(self, ids, **kwargs):
                return self._enc.decode(ids)

        tokenizer = _TiktokenWrapper(tiktoken.get_encoding("cl100k_base"))
        print("[Evaluation] HuggingFace tokenizer unavailable for this model; "
              "falling back to tiktoken cl100k_base for context truncation.")

    # Configure DSPy globally so ask_llm_query() knows which LLM to call.
    configure_dspy(
        model=kg_model_string(cfg),
        api_base=cfg.api_base,
        api_key=cfg.api_key,
    )
    if args.short_answer:
        ans_label = "Predict[ShortAnswer]"
    else:
        ans_label = "ChainOfThought" if args.chain_of_thought else "Predict"
    print(f"[Evaluation] DSPy configured  model={kg_model_string(cfg)}  "
          f"answer_module={ans_label}")

    limiter   = RateLimiter(rpm=args.eval_rpm)
    print(f"[Evaluation] {limiter}")

    tracker = GlobalTokenAccumulator()
    attach_token_tracker(tracker)
    t0 = time.time(); results = []
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            fmap = {ex.submit(_eval_one, q, kg, nx_g, embs,
                              tokenizer, args, limiter): i
                    for i, q in enumerate(questions)}
            done = 0
            for future in as_completed(fmap):
                r = future.result(); results.append(r); done += 1
                if done % max(1, len(questions)//20) == 0 or done == len(questions):
                    ok = [x for x in results if x["status"] == "SUCCESS"]; n = len(ok)
                    print(f"[Evaluation] {done}/{len(questions)}  "
                          f"EM={safe_div(sum(x['em'] for x in ok),n):.4f}  "
                          f"F1={safe_div(sum(x['f1'] for x in ok),n):.4f}")
    finally:
        detach_token_tracker()

    eval_t  = round(time.time() - t0, 2)
    eval_tk = tracker.snapshot()
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
        "run_id":    f"kggen_eval_{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now().isoformat(),
        "system": "KGGEN", "dataset": args.dataset,
        "metadata": {
            "corpus_file":             final_meta.get("corpus_file"),
            "questions_file":          os.path.abspath(args.questions_file),
            "run_dir":                 os.path.abspath(args.run_dir),
            "checkpoint_path":         os.path.abspath(cp_to_load),
            "is_final_checkpoint":     final_cp is not None,
            "provider":                cfg.provider, "model": cfg.model,
            "embedding_model":         args.embed_model, "api_base": cfg.api_base,
            "retrieval_params":        {
                "k":       args.retrieval_k,
                "depth":   args.retrieval_depth,
                "verbose": args.retrieval_verbose,
            },
            "answer_module":           ans_label,
            "max_context_tokens":      args.max_context_tokens,
            "max_workers":             args.max_workers,
            "eval_rpm":                args.eval_rpm,
            "embedding_recompute_time_sec": emb_t,
        },
        "construction": final_meta,
        "evaluation": {
            "total_questions": len(questions), "successful": n, "errored": len(bad),
            "evaluation_time_sec": eval_t, "evaluation_tokens": eval_tk,
            "overall": overall, "by_type": by_type,
            **({"by_level": by_level} if by_level else {}),
            "results": results,
        },
        "totals": {
            "all_tokens": add_token_counters(
                final_meta.get("cumulative_construction_tokens", {}), eval_tk),
            "all_time_sec": round(
                float(final_meta.get("cumulative_construction_time_sec", 0.0)) + eval_t, 2),
        },
    }
    path = next_eval_results_path(args.run_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n[Evaluation] DONE {eval_t}s  ->  {path}")
    print(f"[Evaluation]   EM={overall['em']}  F1={overall['f1']}  n={n}  err={len(bad)}")
    for k, v in sorted(by_type.items()):
        print(f"     type {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    if by_level:
        for k, v in sorted(by_level.items()):
            print(f"    level {k:>12}: EM={v['em']:.4f}  F1={v['f1']:.4f}  (n={v['count']})")
    print(f"[Evaluation]   tokens={eval_tk['total_tokens']:,}  calls={eval_tk['api_calls']:,}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="KGGen evaluation on HotpotQA / MuSiQue",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode",    required=True, choices=["construction","evaluation"])
    p.add_argument("--dataset", required=True, choices=list(SUPPORTED_DATASETS))
    p.add_argument("--provider", default="local_vllm", choices=list(PROVIDER_DEFAULTS),
                   help=("LLM provider. local_vllm (default) or deepseek. "
                         "Override defaults with --model/--api-base/--api-key."))
    p.add_argument("--corpus-file",    type=str)
    p.add_argument("--questions-file", type=str)
    p.add_argument("--run-dir",        type=str, required=True)
    p.add_argument("--model",     type=str, default="",
                   help="Override provider default model name.")
    p.add_argument("--embed-model", type=str,
                   default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Sentence-transformer for node embeddings (always local).")
    p.add_argument("--api-base",  type=str, default="",
                   help="Override provider default API base URL.")
    p.add_argument("--api-key",   type=str, default="",
                   help="Override provider default API key.")
    p.add_argument("--construction-rpm", type=int, default=0, metavar="N",
                   help=("Max RPM during KG construction. 0=unlimited (default). "
                         "KGGen makes ~2 calls/para => N RPM ~ N/2 paras/min. "
                         "Recommended for DeepSeek free tier: 50."))
    p.add_argument("--eval-rpm", type=int, default=0, metavar="N",
                   help="Max RPM during evaluation answer-gen. 0=unlimited.")
    p.add_argument("--checkpoint-interval", type=int, default=100,
                   help=("Save a checkpoint every N paragraphs processed. "
                         "E.g. --checkpoint-interval 50 saves after every 50 paragraphs. "
                         "Smaller values = more frequent saves = safer against crashes, "
                         "but more disk I/O. Default: 100."))
    p.add_argument("--delete-old-checkpoints", action="store_true",
                   help=("After saving checkpoint-N, delete all previous checkpoint-M "
                         "folders (M < N). Keeps only the latest checkpoint on disk, "
                         "saving storage. The final-checkpoint is never deleted. "
                         "Safe to use: resume still works because only the latest "
                         "checkpoint is needed. Default: off (keep all checkpoints)."))
    p.add_argument("--max-workers",         type=int, default=10,
                   help="Thread pool size (construction + evaluation).")

    # ---- Retrieval parameters (evaluation) ----
    p.add_argument("--retrieval-k",       type=int, default=10,
                   help="Number of seed nodes retrieved from the entity embedding index "
                        "before graph traversal. Higher k = broader initial coverage, "
                        "more context, slower retrieval. Default: 10.")
    p.add_argument("--retrieval-depth",   type=int, default=2,
                   help="Graph traversal depth from each seed node. depth=1 returns only "
                        "direct neighbours; depth=2 returns neighbours-of-neighbours, etc. "
                        "Higher depth = more context but also more noise. Default: 2.")
    p.add_argument("--retrieval-verbose", action="store_true",
                   help="Print per-node context during retrieval (debug; very noisy).")

    # ---- Answer generation ----
    p.add_argument("--short-answer", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use ShortAnswerQuestion (span-style, no explanation) for answer "
                        "generation (default: on). Designed for EM/F1 benchmarks where the "
                        "gold answer is a short phrase like 'Paris' or '1984'. "
                        "Pass --no-short-answer to fall back to the sentence-style AnswerQuestion "
                        "prompt (controlled by --chain-of-thought).")
    p.add_argument("--chain-of-thought", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Use dspy.ChainOfThought for answer generation (default: on). "
                        "Only applies when --no-short-answer is set. "
                        "Pass --no-chain-of-thought to use dspy.Predict instead "
                        "(single forward pass, faster, no reasoning trace).")
    p.add_argument("--max-context-tokens",  type=int, default=2000,
                   help="Token budget for context passed to the answer-generation LLM.")

    args = p.parse_args()

    if args.mode == "construction" and not args.corpus_file:
        p.error("--mode construction requires --corpus-file")
    if args.mode == "evaluation" and not args.questions_file:
        p.error("--mode evaluation requires --questions-file")

    cfg     = resolve_provider(args)
    rpm_val = args.construction_rpm if args.mode == "construction" else args.eval_rpm
    rpm_str = str(rpm_val) if rpm_val > 0 else "unlimited"

    print(f"\n{'='*65}")
    print(f"  KGGEN {args.mode.upper()} -- {args.dataset.upper()}")
    print(f"  provider : {cfg.provider}")
    print(f"  model    : {cfg.model}")
    print(f"  api-base : {cfg.api_base}")
    print(f"  run-dir  : {args.run_dir}")
    print(f"  RPM      : {rpm_str} ({args.mode})")
    if args.mode == "construction":
        print(f"  ckpt every : {args.checkpoint_interval} paragraphs")
        print(f"  delete old : {'yes' if args.delete_old_checkpoints else 'no'}")
    else:
        if args.short_answer:
            ans_mode = "ShortAnswer (span-style, EM/F1 optimised)"
        elif args.chain_of_thought:
            ans_mode = "ChainOfThought (sentence-style)"
        else:
            ans_mode = "Predict (sentence-style)"
        print(f"  retrieval  : k={args.retrieval_k}  depth={args.retrieval_depth}"
              f"  verbose={'yes' if args.retrieval_verbose else 'no'}")
        print(f"  answer gen : {ans_mode}")
    print(f"{'='*65}\n")

    if args.mode == "construction":
        try:
            run_construction(args, cfg)
        except KeyboardInterrupt:
            print("\n[Construction] Interrupted. Checkpoint preserved -- rerun to resume.")
        except Exception as e:
            print(f"\n[Construction FATAL] {e}"); traceback.print_exc(); sys.exit(1)
    else:
        run_evaluation(args, cfg)


if __name__ == "__main__":
    main()