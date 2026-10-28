#!/usr/bin/env python3
"""UNH (nkg) on the HippoRAG 2 HotpotQA / MuSiQue data, evaluated like every other system in evaluation_scripts/:
own index + own retrieval here, then the shared answer step (common/answer.py --system unh) and common/score.py.
launch_unh.sh runs everything (both datasets, both views); this script runs one dataset's steps.

    python evaluation_scripts/unh/main.py --dataset hotpotqa --subset evaluation_data/hotpotqa/subsets/hotpotqa_250_seed0.json --step construct
    python evaluation_scripts/unh/main.py --dataset hotpotqa --subset ... --step retrieve --view full
    python evaluation_scripts/unh/main.py --dataset hotpotqa --subset ... --step retrieve --view hypergraph

Steps (outputs/<run name>/, run name = the subset name, e.g. hotpotqa_250_seed0, or <ds>_fullcorpus):
  prepare    the corpus -> data/corpus.jsonl in UNH's corpus format ({id, title, paragraph}; ids p000000.. keep
             order). --corpus subset (default): <subset>_corpus.json, the passages of the subset's own questions
             (gold + distractors; common/make_subset.py). --corpus full: evaluation_data/<ds>/<ds>_corpus.json.
  construct  UNH's own construction CLI: experiments/systems/nkg/hqa_musique_evaluation.py --mode construction
             --input-mode prepared_chunks (one passage = one chunk). UNH's current defaults except, by default here:
             --no-clustered-round3, --no-edge-descriptions (label-only fact->fact edge prompts), fact dedup on,
             --dedup-max-cluster-size 72, --dedup-similarity-threshold 0.5, --same-chunk-min-facts 3,
             --fact-dedup-top-k 20, --fact-dedup-cluster-size 30, --clustered-max-facts-per-prompt 50.
             DeepSeek, no temperature, 2,300 workers / 2,300 sub-workers. Resumable from outputs/<run>/checkpoint/.
  shorten    only for graphs built WITH edge descriptions: kg.graphml -> kg_short.graphml (description := label).
             A label-only graph already stores each label as the description and is used as is.
  retrieve   old engine (nkg.retrieval.engine.Retriever, seed weights as in hqa_musique_evaluation.build_retriever),
             DEFAULT_RETRIEVAL_ARGS_OLD with, by default here, top_k_seeds 12, max_depth 4, beam_width 6,
             beam MMR 0.75 (final_top_k 20, final MMR 0.6, beam_search, deduplicate_paths unchanged), neutral_layout off,
             no context cap, no source passages, empty-name entities dropped; the --subset questions, or all without one.
             --view full: mode "all" (every edge type) -> contexts.json
             --view hypergraph: mode "hypergraph" (entity<->fact edges only; entity seeds) -> contexts_hypergraph.json
             --view discourse: mode "discourse" (fact->fact edges only) -> contexts_discourse.json
  all        prepare + construct + shorten + retrieve (--view)

Insurance contracts (CONTRACTS_EVAL_PLAN.md; run_contracts.sh drives this): --dataset contracts --inputs
evaluation_data/contracts/<set>/<contract> --out-dir <dir>. The contract is ONE document (contract.txt) that UNH chunks
itself (--input-mode concat_file, UNH's default 600/50 tokens); the queries are <inputs>/questions.json (QA questions,
then facts verbatim). The LLM is whatever LLM_BASE_URL serves: DeepSeek (provider deepseek) or a local vLLM
(provider local_vllm, e.g. Qwen via common/env_qwen.sh).

QuALITY (QUALITY_EVAL_PLAN.md, QUALITY_POOLING_FAIRNESS.md; unh/run_quality.sh drives this): --dataset quality
[--inputs <quality inputs dir>, default evaluation_data/quality/inputs] --out-dir <dir>. ONE pooled graph over all
articles of documents.json (as every baseline), each article ("Title: ..." + text) chunked on its own by UNH's chunker
(--input-mode per_document, 600/50), so no chunk crosses articles; the queries are <inputs>/questions.json (question
with its options, in that order); the shared multiple-choice answer step reads the contexts unchanged.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = REPO / "evaluation_data"
sys.path.insert(0, str(REPO))

VIEWS = {"full": ("all", "contexts.json"), "hypergraph": ("hypergraph", "contexts_hypergraph.json"),
         "discourse": ("discourse", "contexts_discourse.json")}


def is_deepseek(base_url: str) -> bool:
    return "deepseek.com" in (base_url or "")


def qid(sample):
    return sample.get("id", sample.get("_id"))


def corpus_path(args) -> Path:
    if args.dataset == "contracts":
        return Path(args.inputs) / "contract.txt"
    if args.dataset == "quality":
        return Path(args.inputs) / "documents.json"
    if args.corpus == "full":
        return DATA_ROOT / args.dataset / f"{args.dataset}_corpus.json"
    sub = json.load(open(args.subset))
    return Path(args.subset).resolve().parent / sub["corpus_file"]


def step_prepare(args, out: Path):
    src = corpus_path(args)
    if args.dataset == "contracts":  # one contract = one document, chunked by UNH itself at construction
        corpus = [{"title": "", "text": src.read_text(encoding="utf-8")}]
    elif args.dataset == "quality":  # one article = one document, chunked on its own by UNH at construction
        corpus = [{"title": d["title"], "text": d["text"]} for d in json.load(open(src, encoding="utf-8"))]
    else:
        corpus = json.load(open(src, encoding="utf-8"))
    if args.limit_docs:
        corpus = corpus[: args.limit_docs]
    path = out / "data" / "corpus.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(corpus):
            f.write(json.dumps({"id": f"p{i:06d}", "title": doc["title"], "paragraph": doc["text"]}, ensure_ascii=False) + "\n")
    print(f"[prepare] {path}: {len(corpus):,} passages from {src.name}"
          f"{' (UNH chunks it, 600/50)' if args.dataset in ('contracts', 'quality') else ' (one chunk each)'}", flush=True)
    return path


def built_with_edge_descriptions(out: Path) -> bool:
    meta = json.load(open(out / "construction_results.json"))["metadata"]["construction_args"]
    return meta.get("edge_descriptions", True)


def step_construct(args, out: Path):
    corpus_file = out / "data" / "corpus.jsonl"
    if not corpus_file.exists():
        corpus_file = step_prepare(args, out)
    flag = lambda on, name: f"--{name}" if on else f"--no-{name}"  # noqa: E731
    cmd = [sys.executable, "-m", "experiments.systems.nkg.hqa_musique_evaluation", "--mode", "construction",
           "--dataset", args.dataset, "--corpus-file", str(corpus_file), "--run-dir", str(out),
           *(["--input-mode", "per_document", "--chunk-size", "600", "--chunk-overlap", "50"] if args.dataset == "quality"
             else ["--input-mode", "concat_file", "--chunk-size", "600", "--chunk-overlap", "50"] if args.dataset == "contracts"
             else ["--input-mode", "prepared_chunks"]),
           "--provider", "deepseek" if is_deepseek(args.base_url) else "local_vllm", "--model", args.model,
           "--api-base", args.base_url, "--temperature", args.temperature,
           flag(args.round3, "clustered-round3"), flag(args.fact_dedup, "fact-dedup"),
           flag(args.edge_descriptions, "edge-descriptions"),
           "--dedup-max-cluster-size", str(args.dedup_max_cluster_size),
           "--dedup-similarity-threshold", str(args.dedup_similarity_threshold),
           "--same-chunk-min-facts", str(args.same_chunk_min_facts),
           "--fact-dedup-top-k", str(args.fact_dedup_top_k),
           "--fact-dedup-cluster-size", str(args.fact_dedup_cluster_size),
           "--clustered-max-facts-per-prompt", str(args.clustered_max_facts_per_prompt),
           "--edge-mode", args.edge_mode,
           *(["--fast-cluster-embed-loader", args.fast_cluster_embed_loader,
              "--fast-cluster-leiden-levels", args.fast_cluster_leiden_levels,
              "--fast-cluster-leiden-passes", str(args.fast_cluster_leiden_passes),
              *(["--fast-cluster-min-size", str(args.fast_cluster_min_size)] if args.fast_cluster_min_size else [])]
             if args.edge_mode == "clustered_fast" else []),
           # only when not the default, so existing runs' saved construction configs stay unchanged
           *(["--fact-prompt", args.fact_prompt] if args.fact_prompt != "default" else []),
           *(["--fact-edge-prompt", args.fact_edge_prompt] if args.fact_edge_prompt != "default" else []),
           *(["--global-fact-edge-prompt", args.global_fact_edge_prompt] if args.global_fact_edge_prompt else []),
           "--max-workers", str(args.max_workers or args.concurrency),
           "--max-sub-workers", str(args.max_sub_workers or args.concurrency), "-y"]
    # the API key reaches the child through DEEPSEEK_API_KEY (never on the command line)
    print("[construct] " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, cwd=REPO, check=True)
    if not (out / "construction_results.json").exists():
        sys.exit("[construct] construction_results.json not written (interrupted?); rerun to resume from the checkpoint")
    print(f"[construct] done in {time.time() - t0:.0f}s", flush=True)


def retrieval_graph(out: Path) -> Path:
    return out / ("kg_short.graphml" if built_with_edge_descriptions(out) else "kg.graphml")


def step_shorten(args, out: Path):
    src = out / "kg.graphml"
    if not src.exists():
        sys.exit(f"[shorten] {src} missing: run --step construct first")
    if not built_with_edge_descriptions(out):
        print("[shorten] graph built label-only (edge_descriptions=False): retrieval uses kg.graphml as is", flush=True)
        return
    from experiments.systems.nkg.short_description_ablation import make_short_label_graph
    dst = out / "kg_short.graphml"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        print(f"[shorten] {dst} is up to date", flush=True)
        return
    info = make_short_label_graph(str(src), str(dst))
    json.dump(info, open(out / "kg_short_info.json", "w"), indent=2)
    print(f"[shorten] {dst}: {info['fact_fact_edges_changed']:,} fact->fact descriptions replaced by their labels", flush=True)


PPR_RETRIEVERS = ("ppr", "ppr_clean")  # LLM-free retrievers taking --ppr-args
_RETRIEVERS = {}


def step_retrieve(args, out: Path):
    import experiments.systems.nkg.hqa_musique_evaluation as H
    from experiments.utils.multihop_utils import GlobalTokenAccumulator, attach_token_tracker, detach_token_tracker
    from sentence_transformers import CrossEncoder, SentenceTransformer

    kg = retrieval_graph(out)
    if not kg.exists():
        sys.exit(f"[retrieve] {kg} missing: run --step construct (and shorten) first")
    if args.dataset in ("contracts", "quality"):
        samples = json.load(open(Path(args.inputs) / "questions.json", encoding="utf-8"))
    else:
        samples = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
    if args.subset:
        keep = set(json.load(open(args.subset))["ids"])
        samples = [s for s in samples if qid(s) in keep]
        if len(samples) != len(keep):
            sys.exit(f"{len(keep) - len(samples)} subset ids are not in {args.dataset}.json")
    if args.limit_q:
        samples = samples[: args.limit_q]

    mode, ctx_name = VIEWS[args.view]
    retrieval_args = dict(H.DEFAULT_RETRIEVAL_ARGS_OLD)
    retrieval_args["neutral_layout"] = False  # short labels inline; the graph holds no long descriptions
    retrieval_args["mode"] = mode
    if args.final_top_k is not None:
        retrieval_args["final_top_k"] = args.final_top_k
    if args.top_k_seeds is not None:
        retrieval_args["top_k_seeds"] = args.top_k_seeds
    if args.max_depth is not None:
        retrieval_args["max_depth"] = args.max_depth
    if args.beam_width is not None:
        retrieval_args["beam_width"] = args.beam_width
    if args.source_passages:
        retrieval_args["source_passages"] = args.source_passages
    if args.beam_mmr_lambda is not None:
        retrieval_args["beam_mmr_lambda"] = args.beam_mmr_lambda
    report_name = f"retrieval_report_{args.view}.json"
    if args.tag:  # a retrieval variant: its own contexts / report next to the official ones
        ctx_name = ctx_name.replace(".json", f"_{args.tag}.json")
        report_name = f"retrieval_report_{args.view}_{args.tag}.json"

    path = out / ctx_name
    done = {}
    if path.exists():
        done = {r["id"]: r for r in json.load(open(path, encoding="utf-8")) if r["status"] == "ok"}
    todo = [s for s in samples if qid(s) not in done]
    if args.ppr_args and args.retriever in PPR_RETRIEVERS:  # merged before the log line, so it shows the arguments actually used
        retrieval_args.update(json.loads(args.ppr_args))  # per-call knobs too (e.g. ppr_passages, path_score)
    print(f"[retrieve:{args.view}] {len(samples)} questions ({len(done)} already done), graph={kg.name}, "
          f"retriever={args.retriever}, args={retrieval_args}"
          + (" (PPR: beam_mmr_lambda / search_mode are old-engine arguments, accepted and ignored)"
             if args.retriever in PPR_RETRIEVERS else ""), flush=True)

    key = (str(kg), args.drop_empty_entities, args.retriever)
    if key not in _RETRIEVERS:  # one embedding pass per process: --view full,hypergraph reuses it
        bi = SentenceTransformer(args.retrieval_model, trust_remote_code=True)
        ce = CrossEncoder(args.cross_encoder_model)
        # same seed weights as hqa_musique_evaluation.build_retriever("old", ...)
        common = dict(retrieval_model=bi, cross_encoder=ce, graph_path=str(kg),
                      entity_weights={"name": 0.6, "role": 0.4, "anchors": 0.0},
                      fact_weights={"sentence": 0.8, "questions": 0.0, "chunk": 0.2},
                      drop_empty_entities=args.drop_empty_entities)
        if args.retriever in PPR_RETRIEVERS:
            # no LLM at query time: HippoRAG 2-style PPR seeds + UNH paths (nkg/retrieval/ppr_engine.py)
            # ppr_clean: the same engine plus ppr_clean_engine.CleanPPRRetriever's options (hub_gamma hub penalty,
            # force_mode, passages_first), passed per call in --ppr-args; with none of them set it is the same retrieval
            if args.retriever == "ppr_clean":
                from nkg.retrieval.ppr_clean_engine import CleanPPRRetriever as PPRRetriever
            else:
                from nkg.retrieval.ppr_engine import PPRRetriever
            slug = args.retrieval_model.replace("/", "__")
            cache = kg.parent / f"{kg.stem}.{slug}{'.dropempty' if args.drop_empty_entities else ''}.retriever_emb.npz"
            _RETRIEVERS[key] = PPRRetriever(**common, embedding_cache_path=str(cache), model_name=args.retrieval_model,
                                            chunk_cache_path=str(kg.parent / f"{kg.stem}.{slug}.chunk_emb.npz"),
                                            ppr_args=json.loads(args.ppr_args) if args.ppr_args else None)
        elif args.retriever == "efficient":
            # same retrieval, low memory (nkg/retrieval/efficient_engine.py); embeddings cached next to the graph,
            # so the other views' processes load them instead of re-encoding
            from nkg.retrieval.efficient_engine import EfficientRetriever
            slug = args.retrieval_model.replace("/", "__")
            cache = kg.parent / f"{kg.stem}.{slug}{'.dropempty' if args.drop_empty_entities else ''}.retriever_emb.npz"
            _RETRIEVERS[key] = EfficientRetriever(**common, embedding_cache_path=str(cache),
                                                  model_name=args.retrieval_model)
        else:
            from nkg.retrieval.engine import Retriever
            _RETRIEVERS[key] = Retriever(**common)
    retriever = _RETRIEVERS[key]
    if args.drop_empty_entities:
        print(f"[retrieve:{args.view}] dropped {retriever.dropped_empty_entities} empty-name entities", flush=True)
    model_string = f"deepseek/{args.model}" if is_deepseek(args.base_url) else f"openai/{args.model}"
    if args.retriever not in PPR_RETRIEVERS:  # the PPR retriever makes no LLM call
        H.configure_dspy(model=model_string, api_base=args.base_url, api_key=os.environ["DEEPSEEK_API_KEY"],
                         max_tokens=args.query_max_tokens, num_retries=1,
                         temperature=H.resolve_temperature(args.temperature))
    tracker = GlobalTokenAccumulator()
    attach_token_tracker(tracker)

    tok = None
    if args.retriever in PPR_RETRIEVERS:  # context tokens counted with the PPR budget's tokenizer
        tok = retriever._tokenizer(retriever.ppr_args["budget_tokenizer"])

    def one(s):
        t0 = time.time()
        try:
            extra = {}
            if args.retriever in PPR_RETRIEVERS:
                ctx, paths = retriever.retrieve(query=s["question"], return_raw_paths=True, **retrieval_args)
                # distinct edges of the final paths by type (an entity hop = fact_entity + entity_fact = 2 edges)
                seen = {}
                for p in paths:
                    for u, et, _, v in p.history:
                        seen.setdefault(frozenset((u, v)), et)
                extra = {"edges_discourse": sum(et == "fact_fact" for et in seen.values()),
                         "edges_entity": sum(et in ("fact_entity", "entity_fact") for et in seen.values()),
                         "context_tokens": len(tok(ctx)["input_ids"])}
            else:
                ctx = retriever.retrieve(query=s["question"], **retrieval_args)
            ctx = ctx if isinstance(ctx, str) else str(ctx)
            return {"id": qid(s), "question": s["question"], "context_text": ctx, "context_chars": len(ctx),
                    "status": "ok", "error": None, "seconds": round(time.time() - t0, 2), **extra}
        except Exception as exc:  # noqa: BLE001 - recorded; the run exits non-zero below
            return {"id": qid(s), "question": s["question"], "context_text": "", "context_chars": 0,
                    "status": "error", "error": repr(exc), "seconds": round(time.time() - t0, 2)}

    t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(one, s) for s in todo]
            for i, f in enumerate(as_completed(futures), 1):
                r = f.result()
                done[r["id"]] = r
                if i % 10 == 0 or i == len(todo):
                    print(f"[retrieve:{args.view}] {i}/{len(todo)}  errors={sum(x['status'] != 'ok' for x in done.values())}", flush=True)
    finally:
        detach_token_tracker()
    rows = [done[qid(s)] for s in samples if qid(s) in done]
    json.dump(rows, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    report = {"dataset": args.dataset, "view": args.view, "subset": args.subset, "corpus": args.corpus,
              "questions": len(rows), "kg": str(kg), "engine": "old", "retriever": args.retriever,
              "retrieval_args": retrieval_args,
              "context_cap": None, "errors": sum(r["status"] != "ok" for r in rows),
              "empty_contexts": sum(not r["context_text"].strip() for r in rows),
              "mean_context_chars": round(sum(r["context_chars"] for r in rows) / max(1, len(rows))),
              **ppr_path_stats(rows),
              "query_llm_usage_this_run": tracker.snapshot(), "seconds_this_run": round(time.time() - t0, 1),
              "llm_model": args.model, "temperature": args.temperature, "tag": args.tag,
              "drop_empty_entities": args.drop_empty_entities,
              "dropped_empty_entities": retriever.dropped_empty_entities}
    json.dump(report, open(out / report_name, "w"), indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "retrieval_args"}, indent=2), flush=True)
    if report["errors"]:
        sys.exit(f"[retrieve:{args.view}] {report['errors']} questions failed (see {path.name}); rerun to retry only those")


def ppr_path_stats(rows) -> dict:
    """PPR retriever rows: mean context tokens and the mean per-question share of discourse (fact->fact) vs entity
    (fact<->entity) edges among the distinct path edges, and of hops (an entity hop = 2 edges = 1 hop)."""
    rows = [r for r in rows if "edges_discourse" in r]
    if not rows:
        return {}
    e, h = [], []
    for r in rows:
        ff, en = r["edges_discourse"], r["edges_entity"]
        if ff + en:
            e.append(ff / (ff + en))
            h.append(ff / (ff + en / 2))
    pct = lambda x: round(100 * sum(x) / len(x), 1) if x else None  # noqa: E731
    return {"mean_context_tokens": round(sum(r["context_tokens"] for r in rows) / len(rows)),
            "max_context_tokens": max(r["context_tokens"] for r in rows),
            "pct_edges_discourse": pct(e), "pct_edges_entity": None if not e else round(100 - pct(e), 1),
            "pct_hops_discourse": pct(h), "pct_hops_entity": None if not h else round(100 - pct(h), 1),
            "mean_edges_per_question": round(sum(r["edges_discourse"] + r["edges_entity"] for r in rows) / len(rows), 1)}


def run_dir(args) -> Path:
    if args.out_dir:
        return Path(args.out_dir).resolve()
    if args.corpus == "full":
        return HERE / "outputs" / f"{args.dataset}_fullcorpus"
    return HERE / "outputs" / Path(args.subset).stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "contracts", "quality"])
    ap.add_argument("--inputs", default=None, help="--dataset contracts: evaluation_data/contracts/<set>/<contract>; "
                    "--dataset quality: QuALITY inputs dir (default evaluation_data/quality/inputs)")
    ap.add_argument("--step", default="all", choices=["prepare", "construct", "shorten", "retrieve", "all"])
    ap.add_argument("--view", default="full", help="retrieve: full | hypergraph | discourse, or a comma list (one graph embedding)")
    ap.add_argument("--subset", default=None, help="ids file from common/make_subset.py (questions; and the corpus "
                    "with --corpus subset)")
    ap.add_argument("--corpus", default="subset", choices=["subset", "full"],
                    help="subset (default): the subset questions' own passages; full: the whole <ds>_corpus.json")
    ap.add_argument("--out-dir", default=None, help="default: outputs/<subset name> or outputs/<ds>_fullcorpus")
    ap.add_argument("--round3", action=argparse.BooleanOptionalAction, default=False,
                    help="clustered round 3 (middle-k) cross-chunk edges (default: off)")
    ap.add_argument("--fact-dedup", action=argparse.BooleanOptionalAction, default=True,
                    help="fact deduplication during construction (default: on)")
    ap.add_argument("--edge-descriptions", action=argparse.BooleanOptionalAction, default=False,
                    help="fact->fact edge description sentences (default: off = label-only edge prompts)")
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-flash"))
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--temperature", default=os.environ.get("LLM_TEMPERATURE") or "none",
                    help="'none' (default) = not sent, as for every other system")
    ap.add_argument("--dedup-max-cluster-size", type=int, default=72,
                    help="entity dedup: max entities per LLM dedup cluster (UNH default 64; earlier runs 72)")
    ap.add_argument("--dedup-similarity-threshold", type=float, default=0.5,
                    help="entity dedup: min combined name/role/type similarity for two entities to be dedup "
                         "candidates (UNH default 0.75; earlier runs 0.35)")
    ap.add_argument("--same-chunk-min-facts", type=int, default=3,
                    help="a chunk needs at least this many facts for its same-chunk fact->fact edge call "
                         "(UNH default 2; earlier runs 5)")
    ap.add_argument("--fact-dedup-top-k", type=int, default=20,
                    help="fact dedup: nearest neighbours per fact in its similarity graph (UNH default 12)")
    ap.add_argument("--fact-dedup-cluster-size", type=int, default=30,
                    help="fact dedup: max facts per LLM dedup call (UNH default 8)")
    ap.add_argument("--clustered-max-facts-per-prompt", type=int, default=50,
                    help="cross-chunk fact->fact edges: max facts per clustered LLM prompt (UNH default 40)")
    ap.add_argument("--edge-mode", default="clustered", choices=["clustered", "clustered_fast"],
                    help="cross-chunk fact->fact edge clustering: 'clustered' (default, the original code) or "
                         "'clustered_fast' (nkg/index/construction/fast_fact_clustering.py: batched GPU similarity "
                         "graph + cached embeddings + the --fast-cluster-* options)")
    ap.add_argument("--fast-cluster-embed-loader", default="remote_code", choices=["remote_code", "legacy"],
                    help="clustered_fast: embedding model loaded with trust_remote_code (remote_code, the real "
                         "voyage-4-nano) or as the original code loads it (legacy)")
    ap.add_argument("--fast-cluster-leiden-levels", default="final", choices=["final", "all"],
                    help="clustered_fast: Leiden's final clusters only (final) or every hierarchy level (all, original)")
    ap.add_argument("--fact-prompt", default="default", choices=["default", "detailed"],
                    help="construction: fact-sentence extraction prompt; default (ChunkFacts) or detailed "
                         "(ChunkFactsDetailed: keeps the text's wording and every time / cause / attitude / attribution / "
                         "condition cue inside each fact)")
    ap.add_argument("--fact-edge-prompt", default="default", choices=["default", "general", "event"],
                    help="construction: fact->fact edge prompt; default (label-only, Bulk*FactEdgesLabelOnly) or general "
                         "(Bulk*FactEdgesGeneral: one relation vocabulary for narrative and expository text, links only "
                         "where one fact explains, answers or follows from the other)")
    ap.add_argument("--global-fact-edge-prompt", default=None, choices=["default", "general", "event"],
                    help="construction: global cross-chunk fact->fact edge prompt (default: --fact-edge-prompt); event = "
                         "BulkGlobalFactEdgesEvent (RED / Hong et al. relation families + WILDCARD, specific labels)")
    ap.add_argument("--fast-cluster-min-size", type=int, default=0,
                    help="clustered_fast: merge clusters below this many facts into their most strongly connected "
                         "cluster while staying <= --clustered-max-facts-per-prompt (0 = off, the plain Leiden partition)")
    ap.add_argument("--fast-cluster-leiden-passes", type=int, default=1,
                    help="clustered_fast: Leiden passes (seeds 42, 43, ...), final clusters pooled")
    ap.add_argument("--concurrency", type=int, default=int(os.environ.get("UNH_CONCURRENCY", "2300")),
                    help="construction --max-workers and --max-sub-workers (default 2300). The HTTP pool is sized by "
                         "LLM_MAX_CONNECTIONS (nkg.utils.config.configure_dspy, default 2500)")
    ap.add_argument("--max-workers", type=int, default=int(os.environ.get("UNH_MAX_WORKERS") or 0) or None,
                    help="construction --max-workers: passages extracted at once (default: --concurrency; env UNH_MAX_WORKERS)")
    ap.add_argument("--max-sub-workers", type=int, default=int(os.environ.get("UNH_MAX_SUB_WORKERS") or 0) or None,
                    help="construction --max-sub-workers: LLM calls at once in dedup / cross-chunk edge stages "
                         "(default: --concurrency; env UNH_MAX_SUB_WORKERS)")
    ap.add_argument("--retriever", default=os.environ.get("UNH_RETRIEVER", "efficient"),
                    choices=["efficient", "standard", "ppr", "ppr_clean"],
                    help="efficient (default): nkg/retrieval/efficient_engine.EfficientRetriever, the same retrieval "
                         "as standard (nkg.retrieval.engine.Retriever) without its N x N MMR matrix (~97 GB per "
                         "question on the full HotpotQA graph), one vector per distinct string, embeddings cached "
                         "next to the graph, vectorised seed scores; ppr: nkg/retrieval/ppr_engine.PPRRetriever, no "
                         "LLM at query time (HippoRAG 2-style PPR seeds, UNH paths). env UNH_RETRIEVER")
    ap.add_argument("--ppr-args", default=os.environ.get("UNH_PPR_ARGS", ""),
                    help="--retriever ppr: JSON of DEFAULT_PPR_ARGS overrides, e.g. '{\"ppr_passages\": 5}'. "
                         "env UNH_PPR_ARGS")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("UNH_RETRIEVAL_WORKERS", "16")),
                    help="questions retrieved in parallel")
    ap.add_argument("--retrieval-model", default=os.environ.get("EMBEDDING_MODEL", "voyageai/voyage-4-nano"))
    ap.add_argument("--cross-encoder-model", default="cross-encoder/ms-marco-MiniLM-L6-v2")
    ap.add_argument("--query-max-tokens", type=int, default=8000,
                    help="max_tokens for query-time LLM calls (hqa_musique_evaluation.py --judge-max-tokens default)")
    ap.add_argument("--final-top-k", type=int, default=None, help="retrieve: final_top_k (default 20)")
    ap.add_argument("--top-k-seeds", type=int, default=12, help="retrieve: top_k_seeds (default 12; old engine default 9)")
    ap.add_argument("--max-depth", type=int, default=4, help="retrieve: max_depth (default 4; old engine default 3)")
    ap.add_argument("--beam-width", type=int, default=6, help="retrieve: beam_width (default 6, the old engine default; earlier runs here used 10)")
    ap.add_argument("--beam-mmr-lambda", type=float, default=0.75,
                    help="retrieve: beam_mmr_lambda (default 0.75; old engine default 0.6; higher = more relevance, "
                         "less diversity)")
    ap.add_argument("--source-passages", type=int, default=0,
                    help="retrieve: append the source text of up to N chunks of the facts in the final paths")
    ap.add_argument("--drop-empty-entities", action=argparse.BooleanOptionalAction, default=True,
                    help="retrieve: drop entities with an empty name and role at query time")
    ap.add_argument("--tag", default="", help="retrieve: variant tag -> contexts[_<view>]_<tag>.json, "
                    "retrieval_report_<view>_<tag>.json (official files untouched)")
    ap.add_argument("--limit-docs", type=int, default=0, help="smoke tests: first N passages only")
    ap.add_argument("--limit-q", type=int, default=0, help="smoke tests: first N questions only")
    args = ap.parse_args()
    if any(v not in VIEWS for v in args.view.split(",")):
        ap.error(f"--view: each of {args.view!r} must be one of {list(VIEWS)}")
    if not os.environ.get("DEEPSEEK_API_KEY") and not (args.step == "retrieve" and args.retriever in PPR_RETRIEVERS):
        sys.exit("export DEEPSEEK_API_KEY first (source evaluation_scripts/common/env.sh)")
    if args.dataset == "contracts" and not (args.inputs and args.out_dir):
        sys.exit("--dataset contracts needs --inputs <contract inputs dir> and --out-dir")
    if args.dataset == "quality":
        args.inputs = args.inputs or str(DATA_ROOT / "quality" / "inputs")
        if not args.out_dir:
            sys.exit("--dataset quality needs --out-dir")
    if args.dataset not in ("contracts", "quality") and args.corpus == "subset" and not args.subset:
        sys.exit("--corpus subset needs --subset (or pass --corpus full)")

    out = run_dir(args)
    out.mkdir(parents=True, exist_ok=True)
    if args.step in ("prepare", "all"):
        step_prepare(args, out)
    if args.step in ("construct", "all"):
        step_construct(args, out)
    if args.step in ("shorten", "all"):
        step_shorten(args, out)
    if args.step in ("retrieve", "all"):
        failed = []
        for view in args.view.split(","):  # e.g. --view full,hypergraph: one graph embedding, both views
            args.view = view
            try:
                step_retrieve(args, out)
            except SystemExit as exc:
                if exc.code not in (None, 0):
                    print(exc.code, flush=True)
                    failed.append(view)
        if failed:
            sys.exit(f"[retrieve] views with failures: {failed} (rerun resumes)")


if __name__ == "__main__":
    main()
