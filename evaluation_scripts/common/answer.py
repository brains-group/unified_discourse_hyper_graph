#!/usr/bin/env python3
"""Shared answer step: one function turns every system's retrieved context into an answer.

    python evaluation_scripts/common/answer.py --dataset hotpotqa --system hipporag2 \
        --retrieval evaluation_scripts/hipporag2/outputs/hotpotqa/predictions.json

Each system keeps its own retrieval and decides its own context (how many documents, how long). This
script only replaces the final answer-generation call, so that every system is answered by the same
code with the same prompt and the same generation settings:

  prompt       HippoRAG 2 `rag_qa_musique` one-shot CoT (prompts.build_rag_qa_messages); each context
               document becomes one "Wikipedia Title: ..." block. HippoRAG 2 and MemGraphRAG already use
               this prompt.
  answer       text after the first "Answer:" (prompts.extract_answer), as in HippoRAG 2 / MemGraphRAG
  settings     max_tokens 2048 (HippoRAG 2's reader), no temperature sent (DeepSeek default) unless LLM_TEMPERATURE,
               DeepSeek thinking disabled unless LLM_DISABLE_THINKING=0

The context each system hands to the reader (what its own final-answer call reads):
  hipporag2      predictions.json      retrieved_docs[:qa_top_k]  (HippoRAG.qa: docs[:qa_top_k], qa_top_k=5)
  memgraphrag    qa_results.json       retrieved_docs[:qa_top_k]  (MemGraphRAG.qa, run.sh --qa-top-k 5)
  logicrag       result/logicrag_*.json info_summary               (LogicRAG.generate_answer reads only this)
  hypergraphrag  knowledge.json        knowledge                  (aquery(only_need_context=True))
  ms_graphrag    local_search_contexts.json context_text           (LocalSearch.search's build_context output)
  hygrag         contexts.json         context_text               (HKGraphTreeQuery._retrieve_relevant_contexts)
  unh            contexts.json         context_text               (nkg Retriever.retrieve, evaluation_scripts/unh/main.py)

--subset <ids file> (common/make_subset.py): only those questions, in dataset order; the retrieval file must hold
exactly them, in that order (UNH retrieves only the subset; for the others, run on all and filter at scoring).

QuALITY (--dataset quality, see evaluation_scripts/QUALITY_EVAL_PLAN.md): the same contexts, but the shared
multiple-choice prompt (prompts.build_mcq_messages: context + question with options, reply with the letter
only); pred_answer = the letter (prompts.parse_option_letter) or "" when the reply is not a bare letter, which
score_quality.py then hands to HyGRAG's option extractor. Questions come from --questions (default
evaluation_data/quality/inputs/questions.json, built by evaluation_data/build_quality_inputs.py), where
`question` is the question with its options, the string every system was queried with.

G-Medical (--dataset gmedical, see evaluation_scripts/GMEDICAL_EVAL_PLAN.md): open-ended questions, answered with
LinearRAG's answer prompt (prompts.build_linearrag_messages: the HippoRAG 2 / MemGraphRAG system instruction, zero-shot,
no Wikipedia example or labels; LinearRAG runs it on G-Medical) and max_tokens 4096; questions from --questions (default $GMEDICAL_INPUTS/questions.json or
evaluation_data/gmedical/inputs/questions.json, built by evaluation_data/build_gmedical_inputs.py). Score with
common/score_gmedical.py (LLM-Acc, per question type).

Insurance contracts: common/contracts_score.py uses this Reader with prompt="contracts" (prompts.build_contract_messages).

Output: answers_<system>_<dataset>.json next to the retrieval file (list of
{id, question, pred_answer, raw_response, finish_reason, retrieved_docs, usage}) and answers_<system>_<dataset>_usage.json
(answer-step calls and prompt / completion / prompt-cache-hit tokens, totals and per-question means); score it with
common/score.py --system <system> --predictions <that file>. Every call is cached in
reader_cache.sqlite next to it, so a rerun only pays for questions that did not finish.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import (build_contract_messages, build_linearrag_messages, build_mcq_messages, build_rag_qa_messages,  # noqa: E402
                     build_unh_short_messages, extract_answer, extract_contract_answer, parse_option_letter)

DATA_ROOT = Path(__file__).resolve().parents[2] / "evaluation_data"
SYSTEMS = ("hipporag2", "memgraphrag", "logicrag", "hypergraphrag", "ms_graphrag", "hygrag", "unh")


def load_contexts(system: str, path: str, qa_top_k: int):
    """Return [(question, [context docs])] in dataset order."""
    data = json.load(open(path, encoding="utf-8"))
    if system == "hipporag2":
        return [(r["question"], r["retrieved_docs"][:qa_top_k]) for r in data]
    if system == "memgraphrag":
        return [(r["question"], r["retrieved_docs"][:qa_top_k]) for r in data["results"]]
    if system == "logicrag":
        missing = [i for i, r in enumerate(data["results"]) if r.get("info_summary") is None]
        if missing:
            sys.exit(f"{len(missing)} LogicRAG results have no info_summary (first: index {missing[0]}); "
                     "they come from a run before the [unh_v1] patch that saves it")
        return [(r["question"], [r["info_summary"]]) for r in data["results"]]
    if system == "hypergraphrag":
        return [(r["question"], [r["knowledge"] or ""]) for r in data]
    if system in ("ms_graphrag", "hygrag", "unh"):
        return [(r["question"], [r["context_text"]]) for r in data]
    raise ValueError(system)


class Reader:
    """The single answer-generation function shared by all systems."""

    def __init__(self, model, base_url, api_key, max_tokens, temperature, cache_path: Path, mcq: bool = False,
                 prompt: str = "hipporag2"):
        self.mcq = mcq
        self.prompt = prompt
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=300, max_retries=5)
        self.params = {"model": model, "max_tokens": max_tokens}
        if temperature is not None:
            self.params["temperature"] = temperature
        if os.environ.get("LLM_DISABLE_THINKING", "1") != "0":
            self.params["extra_body"] = {"thinking": {"type": "disabled"}}
        self.conn = sqlite3.connect(str(cache_path), check_same_thread=False, timeout=120)  # prompts may share the cache concurrently
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT, finish_reason TEXT)")
        if "usage" not in [r[1] for r in self.conn.execute("PRAGMA table_info(cache)")]:  # caches from before usage logging
            self.conn.execute("ALTER TABLE cache ADD COLUMN usage TEXT")
        self.conn.commit()
        self.api_calls_this_run = 0
        self.lock = threading.Lock()

    def __call__(self, question: str, docs: list):
        if self.mcq:
            messages = build_mcq_messages(question, docs)
        elif self.prompt == "unh_short":
            messages = build_unh_short_messages(question, docs)
        elif self.prompt == "contracts":  # common/contracts_score.py: the shared contract prompt (prompts.py)
            messages = build_contract_messages(question, docs)
        elif self.prompt == "linearrag":  # G-Medical: LinearRAG's zero-shot answer prompt (prompts.py)
            messages = build_linearrag_messages(question, docs)
        else:
            messages = build_rag_qa_messages(question, docs, top_k=len(docs))
        request = {**self.params, "messages": messages}
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        with self.lock:
            row = self.conn.execute("SELECT response, finish_reason, usage FROM cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            raw, finish, usage = row[0], row[1], json.loads(row[2]) if row[2] else None
        else:
            response = self.client.chat.completions.create(**request)
            raw, finish = response.choices[0].message.content or "", response.choices[0].finish_reason
            usage = usage_record(response.usage)
            with self.lock:
                self.api_calls_this_run += 1
                if raw:  # never cache an empty response; a rerun retries it
                    self.conn.execute("INSERT OR REPLACE INTO cache (key, response, finish_reason, usage) VALUES (?, ?, ?, ?)",
                                      (key, raw, finish, json.dumps(usage)))
                    self.conn.commit()
        if self.mcq:
            return raw, parse_option_letter(raw) or "", finish, usage
        if self.prompt == "unh_short":  # no 'Answer:' tag expected: the whole reply is the answer
            return raw, raw.strip(), finish, usage
        if self.prompt == "contracts":
            return raw, extract_contract_answer(raw), finish, usage
        return raw, extract_answer(raw), finish, usage


def usage_record(u) -> dict:
    """prompt / completion / prompt-cache-hit tokens of one response (cache hit: DeepSeek prompt_cache_hit_tokens,
    else OpenAI-style prompt_tokens_details.cached_tokens)."""
    d = u.model_dump() if hasattr(u, "model_dump") else dict(u or {})
    hit = d.get("prompt_cache_hit_tokens")
    if hit is None:
        hit = (d.get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"prompt_tokens": int(d.get("prompt_tokens") or 0), "completion_tokens": int(d.get("completion_tokens") or 0),
            "prompt_cache_hit_tokens": int(hit or 0)}


def usage_summary(answers: list, api_calls_this_run: int) -> dict:
    """Answer-step token totals over every question (from the reader cache, so a resumed run still counts all)."""
    known = [a["usage"] for a in answers if a.get("usage")]
    tot = {k: sum(u[k] for u in known) for k in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens")}
    n = max(1, len(known))
    return {"questions": len(answers), "questions_with_usage": len(known),
            "api_calls_total": len(known), "api_calls_this_run": api_calls_this_run, **tot,
            "mean_prompt_tokens_per_question": round(tot["prompt_tokens"] / n, 1),
            "mean_completion_tokens_per_question": round(tot["completion_tokens"] / n, 1),
            "note": "prompt tokens = the shared answer prompt + this system's context; questions_with_usage < questions "
                    "means some answers came from a reader cache written before usage logging"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "quality", "gmedical"])
    ap.add_argument("--questions", default=None,
                    help="quality / gmedical: questions.json (default: $QUALITY_INPUTS/questions.json or "
                         "evaluation_data/quality/inputs/; $GMEDICAL_INPUTS/questions.json or evaluation_data/gmedical/inputs/)")
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--retrieval", required=True, help="the system's own output file (see the table above)")
    ap.add_argument("--out", default=None, help="default: <retrieval dir>/answers_<system>_<dataset>.json")
    ap.add_argument("--subset", default=None, help="ids file from common/make_subset.py (hotpotqa / musique)")
    ap.add_argument("--qa-top-k", type=int, default=5, help="docs passed to the reader for hipporag2 / memgraphrag")
    ap.add_argument("--max-tokens", type=int, default=None, help="default 2048; gmedical 4096 (user decision 2026-10-11)")
    ap.add_argument("--prompt", default=None, choices=["hipporag2", "unh_short", "linearrag"],
                    help="default: hipporag2 = the shared HippoRAG 2 one-shot CoT prompt (gmedical: linearrag = LinearRAG's "
                         "zero-shot version of it, no Wikipedia example); unh_short: UNH's own August prompt "
                         "(prompts.build_unh_short_messages: direct short answer, no CoT). Same settings otherwise.")
    ap.add_argument("--temperature", default=os.environ.get("LLM_TEMPERATURE") or "none",
                    help='default: LLM_TEMPERATURE if set, else "none" = not sent (provider default)')
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    args = ap.parse_args()

    mcq = args.dataset == "quality"
    if args.prompt is None:
        args.prompt = "linearrag" if args.dataset == "gmedical" else "hipporag2"
    if args.max_tokens is None:
        args.max_tokens = 4096 if args.dataset == "gmedical" else 2048
    if args.dataset in ("quality", "gmedical"):
        env = "QUALITY_INPUTS" if mcq else "GMEDICAL_INPUTS"
        qpath = args.questions or Path(os.environ.get(env, DATA_ROOT / args.dataset / "inputs")) / "questions.json"
        samples = json.load(open(qpath, encoding="utf-8"))
    else:
        samples = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
        if args.subset:
            keep = set(json.load(open(args.subset))["ids"])
            samples = [s for s in samples if s.get("id", s.get("_id")) in keep]
            if len(samples) != len(keep):
                sys.exit(f"{len(keep) - len(samples)} subset ids are not in {args.dataset}.json")
    rows = load_contexts(args.system, args.retrieval, args.qa_top_k)
    if len(rows) > len(samples):
        sys.exit(f"{len(rows)} rows but only {len(samples)} questions")
    for i, ((question, _), s) in enumerate(zip(rows, samples)):
        if question.strip() != s["question"].strip():
            sys.exit(f"question mismatch at index {i}: {question!r} vs {s['question']!r}")

    out = Path(args.out) if args.out else Path(args.retrieval).resolve().parent / f"answers_{args.system}_{args.dataset}.json"
    temperature = None if args.temperature.lower() == "none" else float(args.temperature)
    reader = Reader(args.model, args.base_url, os.environ[args.api_key_env], args.max_tokens, temperature,
                    out.parent / "reader_cache.sqlite", mcq=mcq, prompt=args.prompt)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(lambda r: reader(*r), rows), total=len(rows), desc=f"answer {args.system}"))

    answers = [{"id": s.get("id", s.get("_id")), "question": q, "pred_answer": pred, "raw_response": raw,
                "finish_reason": finish, "retrieved_docs": docs, "usage": usage}
               for s, (q, docs), (raw, pred, finish, usage) in zip(samples, rows, results)]
    json.dump(answers, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    usage_out = out.with_name(out.stem + "_usage.json")
    json.dump(usage_summary(answers, reader.api_calls_this_run), open(usage_out, "w"), indent=2)
    print(f"answer-step token usage -> {usage_out}")
    empty = sum(not a["raw_response"] for a in answers)
    truncated = sum(a["finish_reason"] == "length" for a in answers)
    if mcq:
        unparsed = sum(not a["pred_answer"] for a in answers)
        print(f"wrote {out}: {len(answers)} answers, {empty} empty, {truncated} hit max_tokens, "
              f"{unparsed} not a bare letter (score_quality.py sends these to HyGRAG's extractor)")
        return
    no_answer_tag = sum("Answer:" not in a["raw_response"] for a in answers)
    print(f"wrote {out}: {len(answers)} answers, {empty} empty, {truncated} hit max_tokens, "
          f"{no_answer_tag} without 'Answer:' (whole response used)")


if __name__ == "__main__":
    main()
