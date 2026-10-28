#!/usr/bin/env python3
"""Local-search context for every question, from a finished `graphrag index` workspace.

    python query.py --dataset hotpotqa        # reads caches/hotpotqa, writes outputs/hotpotqa/local_search_contexts.json

The engine is assembled exactly as graphrag 3.2's `graphrag query --method local` builds it
(graphrag.cli.query.run_local_search -> graphrag.api.query.local_search_streaming): same index tables,
same community level (CLI default 2), same stock prompt and the local_search settings from settings.yaml.
For each question it then runs the first half of LocalSearch.search():

    context_builder.build_context(query=question, conversation_history=None, **context_builder_params)

and saves `context_chunks`: the exact context text LocalSearch.search() formats into its system prompt.
The second half of search() (graphrag's own answer) is not run; common/answer.py answers from this text,
as it does for the other systems. An empty context (no searchable entity matched, see build_engine) is kept
as is and counted: the reader then answers with no context, as graphrag's own answer call would. No LLM is called here, only the embedding endpoint (one query embedding
per question).
"""
import argparse
import json
import os
import time
from pathlib import Path

from graphrag.cli.query import _resolve_output_files
from graphrag.config.embeddings import entity_description_embedding
from graphrag.config.load_config import load_config
from graphrag.query.factory import get_local_search_engine
from graphrag.query.indexer_adapters import (
    read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.utils.api import get_embedding_store, load_search_prompt

HERE = Path(__file__).resolve().parent
DATA_ROOT = HERE.parents[1] / "evaluation_data"


def build_engine(root: Path, community_level: int):
    """Same calls, in the same order, as graphrag.cli.query.run_local_search + api.local_search_streaming."""
    config = load_config(root_dir=root)
    dfs = _resolve_output_files(
        config=config,
        output_list=["communities", "community_reports", "text_units", "relationships", "entities"],
        optional_list=["covariates"],
    )
    description_embedding_store = get_embedding_store(config=config.vector_store, embedding_name=entity_description_embedding)
    entities_ = read_indexer_entities(dfs["entities"], dfs["communities"], community_level)
    covariates_ = read_indexer_covariates(dfs["covariates"]) if dfs["covariates"] is not None else []
    prompt = load_search_prompt(config.local_search.prompt)
    engine = get_local_search_engine(
        config=config,
        reports=read_indexer_reports(dfs["community_reports"], dfs["communities"], community_level),
        text_units=read_indexer_text_units(dfs["text_units"]),
        entities=entities_,
        relationships=read_indexer_relationships(dfs["relationships"]),
        covariates={"claims": covariates_},
        description_embedding_store=description_embedding_store,
        response_type="multiple paragraphs",  # CLI default; unused, no answer is generated here
        system_prompt=prompt,
    )
    sizes = {k: (0 if v is None else len(v)) for k, v in dfs.items()}
    # Local search only sees entities that belong to a community at level <= community_level. With the stock
    # cluster_graph.use_lcc: true, only the largest connected component is clustered, so entities outside it
    # can never be retrieved; a question that only matches those gets an empty context.
    sizes["entities_searchable"] = len(entities_)
    return engine, sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "musique", "quality"])
    ap.add_argument("--questions", default=None,
                    help="quality: questions.json (default $QUALITY_INPUTS/questions.json or evaluation_data/quality/inputs/)")
    ap.add_argument("--root", default=None, help="graphrag workspace (default: caches/<dataset>)")
    ap.add_argument("--out", default=None, help="default: outputs/<dataset>/local_search_contexts.json")
    ap.add_argument("--community-level", type=int, default=2, help="graphrag query CLI default: 2")
    ap.add_argument("--limit", type=int, default=0, help="first N questions only (smoke tests)")
    args = ap.parse_args()

    root = Path(args.root) if args.root else HERE / "caches" / args.dataset
    out = Path(args.out) if args.out else HERE / "outputs" / args.dataset / "local_search_contexts.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.dataset == "quality":
        # [unh_v1] QuALITY: `question` = the question with its four options (evaluation_data/build_quality_inputs.py)
        qpath = args.questions or Path(os.environ.get("QUALITY_INPUTS", DATA_ROOT / "quality" / "inputs")) / "questions.json"
        samples = json.load(open(qpath, encoding="utf-8"))
    else:
        samples = json.load(open(DATA_ROOT / args.dataset / f"{args.dataset}.json", encoding="utf-8"))
    if args.limit:
        samples = samples[: args.limit]

    engine, sizes = build_engine(root, args.community_level)
    print(f"[query] index tables: {sizes}", flush=True)

    rows, empty, t0 = [], 0, time.time()
    for i, s in enumerate(samples):
        result = engine.context_builder.build_context(
            query=s["question"], conversation_history=None, **engine.context_builder_params
        )
        chunks = result.context_chunks
        text = chunks if isinstance(chunks, str) else "\n\n".join(str(c) for c in chunks)
        empty += not text.strip()
        rows.append({"id": s.get("id", s.get("_id")), "question": s["question"], "context_text": text,
                     "context_tokens": len(engine.tokenizer.encode(text))})
        if (i + 1) % 100 == 0 or i + 1 == len(samples):
            print(f"[query] {i + 1}/{len(samples)} contexts  {time.time() - t0:.0f}s", flush=True)

    json.dump(rows, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    toks = sorted(r["context_tokens"] for r in rows)
    print(f"[query] wrote {out}: {len(rows)} contexts, {empty} empty, "
          f"tokens median {toks[len(toks) // 2]} max {toks[-1]}", flush=True)


if __name__ == "__main__":
    main()
