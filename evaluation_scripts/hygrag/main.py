#!/usr/bin/env python3
"""HyGRAG (vendored in HyGRAG/, method config config/HKGraphTreeLSH.yaml) on one dataset: index -> context per question.

    python main.py --dataset hotpotqa        # reads data/hotpotqa (prepare_data.py), writes outputs/hotpotqa/

1. Index: HyGRAG's own `GraphRAG.insert(corpus)`, with the corpus loaded by its own `RAGQueryDataset.get_corpus()`.
2. Checks: HyGRAG catches every error inside the graph build, logs it and carries on, and it replaces failed
   embeddings and summaries with random vectors or placeholder text. This runner counts all of those log
   messages and verifies the built index (hierarchy present; chunk / entity / community FAISS indexes non-empty,
   with the embedding size; one chunk vector per chunk). Any failure aborts the run before querying.
3. Context: for each question, `HKGraphTreeQuery._retrieve_relevant_contexts(question)`, which is exactly the
   context HyGRAG's own answer call (`generation_qa`) reads. That answer call is not run; common/answer.py answers
   from this context. No LLM call happens at query time (retrieval is FAISS + embeddings only).
   Questions that hit HyGRAG's fallback retrieval or a failed search are flagged and counted; by default any such
   question makes the run exit non-zero (after writing everything) so it can be inspected before answering.

Only housekeeping is set here (models, keys, paths, concurrency); every method setting comes from the YAML.
Outputs in outputs/<ds>/: contexts.json, run_report.json, hygrag.log, llm_cache.sqlite, work/ (graph + FAISS).
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
HYGRAG_DIR = HERE / "HyGRAG"
DATA_ROOT = HERE.parents[1] / "evaluation_data"

# Log messages that mean HyGRAG silently degraded something. Fatal ones abort before querying.
FATAL_BUILD = {
    "build_error": r"Error building HK graph tree",
    "embedding_failed_random_vector": r"Failed to generate embedding for node",
    "no_text_random_vector": r"No text content found for node .* using random initialization",
    "faiss_build_failed": r"Failed to build FAISS indexes|Failed to create FAISS index",
    "hierarchy_save_failed": r"Failed to save hierarchy data|Failed to save FAISS indexes",
    "summary_failed_placeholder": r"Failed to generate enhanced summary",
    "summary_empty_placeholder": r"Empty summary generated for",
    "summary_missing_member_mean": r"No valid summary for",
    "summary_embedding_failed": r"Failed to generate text-based embedding for|Zero norm embedding for",
}
COUNTED = {  # upstream behaviour, kept: an unparseable NER / OpenIE reply means no entities / triples
    "json_decode_failed": r"JSON decoding failed",
    "json_no_values": r"No values could be extracted from the string",
    "llm_retry_exhausted": r"Retry attempts exhausted",
}
QUERY = {
    "fallback_retrieval": r"Using fallback retrieval method",
    "search_failed": r"Failed to search .* index|Flat search failed|FAISS search failed",
    "context_retrieval_failed": r"HKGraphTree context retrieval failed",
}
PATTERNS = {name: re.compile(p) for group in (FATAL_BUILD, COUNTED, QUERY) for name, p in group.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "quality"])
    ap.add_argument("--questions", default=None, help="default: evaluation_data/<ds>/<ds>.json; quality: "
                    "$QUALITY_INPUTS/questions.json or evaluation_data/quality/inputs/questions.json")
    ap.add_argument("--data-dir", default=None, help="default: data/<dataset> (prepare_data.py)")
    ap.add_argument("--out-dir", default=None, help="default: outputs/<dataset>")
    ap.add_argument("--config", default=str(HERE / "config" / "HKGraphTreeLSH.yaml"))
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("HYGRAG_CONCURRENCY", "500")))
    ap.add_argument("--limit-q", type=int, default=0, help="first N questions only (smoke tests)")
    ap.add_argument("--force-rebuild", action="store_true", help="rebuild the graph (LLM calls replay from the cache)")
    ap.add_argument("--allow-fallback", action="store_true", help="do not exit non-zero on fallback / failed retrievals")
    args = ap.parse_args()

    data_dir = Path(args.data_dir).resolve() if args.data_dir else HERE / "data" / args.dataset
    out = Path(args.out_dir).resolve() if args.out_dir else HERE / "outputs" / args.dataset
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    os.environ["HYGRAG_LLM_CACHE"] = str(out / "llm_cache.sqlite")

    # HyGRAG resolves Option/Config2.yaml and other paths from the working directory (Core/Common/Constants.py)
    os.chdir(HYGRAG_DIR)
    sys.path.insert(0, str(HYGRAG_DIR))
    from loguru import logger as loguru_logger
    from Config.EmbConfig import EmbeddingConfig
    from Option.Config2 import Config
    from Core.GraphRAG import GraphRAG
    from Core.Provider.ResponseCache import get_cache
    from Data.QueryDataset import RAGQueryDataset

    counts: Counter = Counter()

    def count_sink(message):
        text = message.record["message"]
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                counts[name] += 1

    loguru_logger.add(count_sink, level="DEBUG")
    loguru_logger.add(str(out / "hygrag.log"), level="INFO", enqueue=True)

    # ---- config: the method YAML + HyGRAG's Option/Config2.yaml, then our models / paths (housekeeping only)
    opt = Config.parse(Path(args.config), dataset_name=args.dataset)
    opt.llm.api_key = os.environ["DEEPSEEK_API_KEY"]
    opt.llm.base_url = os.environ.get("LLM_BASE_URL", "https://api.deepseek.com")
    opt.llm.model = os.environ.get("LLM_MODEL", "deepseek-flash")
    opt.llm.max_concurrent = args.concurrency
    # DeepSeek's default is 1.0; HyGRAG always sends a temperature, so send that default unless LLM_TEMPERATURE is set
    # [unh_v1] any other endpoint (e.g. local Qwen for the contracts): not sent, so the server's default applies
    if os.environ.get("LLM_TEMPERATURE"):
        opt.llm.temperature = float(os.environ["LLM_TEMPERATURE"])
    elif "deepseek.com" in opt.llm.base_url:
        opt.llm.temperature = 1.0
    else:
        opt.llm.temperature = None
    # upstream Config2 uses a local bge-m3 (hf, 1024) and sends `dimensions`; our vLLM voyage-4-nano server rejects
    # `dimensions`, so it is left unset (native 2048, matching cleora_dim in the YAML)
    opt.embedding = EmbeddingConfig(api_type="openai", api_key="EMPTY",
                                    base_url=os.environ.get("EMBEDDING_BASE_URL", "http://localhost:8001/v1"),
                                    model=os.environ.get("EMBEDDING_MODEL", "voyageai/voyage-4-nano"),
                                    embed_batch_size=16)
    opt.working_dir = str(work)
    opt.data_root = str(data_dir.parent)
    opt.graph.faiss_index_path = str(work / "faiss_index")  # upstream default is shared across datasets
    if args.force_rebuild:
        opt.graph.force = True

    report = {"dataset": args.dataset, "config": args.config, "concurrency": args.concurrency,
              "llm_model": opt.llm.model, "temperature": opt.llm.temperature, "embedding_model": opt.embedding.model}

    async def run():
        digimon = GraphRAG(config=opt)
        corpus = RAGQueryDataset(data_dir=str(data_dir)).get_corpus()
        t0 = time.time()
        await digimon.insert(corpus)
        report["index_seconds"] = round(time.time() - t0, 1)

        # ---- post-build checks
        g = digimon.graph
        n_chunks = await digimon.doc_chunk.size  # an async property upstream
        probe_dim = len(g.embedding_model._get_text_embedding("probe"))
        checks = {"corpus_docs": len(corpus), "chunks": n_chunks, "embedding_dim": probe_dim,
                  "cleora_dim": g.cleora_dim, "hierarchy_levels": len(g.hierarchy_levels or []),
                  "communities": len(g.community_summaries), "faiss": {}}
        problems = []
        for node_type in ("chunk", "entity", "community"):
            index = g.faiss_indexes.get(node_type)
            checks["faiss"][node_type] = None if index is None else {"ntotal": index.ntotal, "dim": index.d}
            if index is None or index.ntotal == 0:
                problems.append(f"{node_type} FAISS index missing or empty")
            elif index.d != probe_dim:
                problems.append(f"{node_type} FAISS index is {index.d}-dim but queries are {probe_dim}-dim")
        if checks["faiss"]["chunk"] and checks["faiss"]["chunk"]["ntotal"] != n_chunks:
            problems.append(f"chunk FAISS index has {checks['faiss']['chunk']['ntotal']} vectors for {n_chunks} chunks")
        if not checks["hierarchy_levels"]:
            problems.append("no hierarchy levels built")
        problems += [f"{name}: {counts[name]}" for name in FATAL_BUILD if counts[name]]
        report["build_checks"] = checks
        report["build_log_counts"] = {k: counts[k] for k in list(FATAL_BUILD) + list(COUNTED)}
        cache = get_cache()
        report["llm_cache"] = {"hits": cache.hits, "misses": cache.misses} if cache else None
        if problems:
            report["build_problems"] = problems
            return 3

        # ---- one context per question
        if args.questions:
            qpath = Path(args.questions)
        elif args.dataset == "quality":  # `question` = the question with its four options
            qpath = Path(os.environ.get("QUALITY_INPUTS", DATA_ROOT / "quality" / "inputs")) / "questions.json"
        else:
            qpath = DATA_ROOT / args.dataset / f"{args.dataset}.json"
        questions = json.load(open(qpath, encoding="utf-8"))
        if args.limit_q:
            questions = questions[: args.limit_q]
        querier = digimon._querier
        rows, flagged = [], 0
        for i, q in enumerate(questions):
            before = Counter({k: counts[k] for k in QUERY})
            context = await querier._retrieve_relevant_contexts(q["question"])
            flags = {k: counts[k] - before[k] for k in QUERY if counts[k] - before[k]}
            if context.startswith("Context retrieval failed"):
                flags["context_retrieval_failed"] = flags.get("context_retrieval_failed", 0) or 1
            flagged += bool(flags)
            rows.append({"id": q.get("id", q.get("_id")), "question": q["question"], "context_text": context,
                         "context_chars": len(context), "flags": flags})
            if (i + 1) % 100 == 0 or i + 1 == len(questions):
                print(f"[hygrag] {i + 1}/{len(questions)} contexts, {flagged} flagged", flush=True)
        json.dump(rows, open(out / "contexts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        report["questions"] = len(rows)
        report["questions_flagged"] = flagged
        report["query_log_counts"] = {k: counts[k] for k in QUERY}
        report["empty_contexts"] = sum(not r["context_text"].strip() for r in rows)
        return 4 if flagged and not args.allow_fallback else 0

    code = asyncio.run(run())
    json.dump(report, open(out / "run_report.json", "w"), indent=2)
    print(json.dumps(report, indent=2))
    if code == 3:
        sys.exit(f"[hygrag] build checks failed: {report['build_problems']} (see {out / 'hygrag.log'})")
    if code == 4:
        sys.exit(f"[hygrag] {report['questions_flagged']} questions used fallback / failed retrieval (see contexts.json flags)")


if __name__ == "__main__":
    main()
