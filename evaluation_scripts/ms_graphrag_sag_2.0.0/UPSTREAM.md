# Upstream provenance

- Upstream: https://github.com/microsoft/graphrag
- Integration form: released dependency plus SAG-Benchmark reproduce scripts
- Pinned engine: `graphrag==2.0.0`
- Vendored commit: not applicable; the GraphRAG package source is not vendored here
- License: upstream MIT; dependency license remains authoritative

Local ownership covers dataset preparation, indexing/retrieval/QA orchestration, artifact
paths, adapters, and benchmark documentation. Do not copy package internals into this
directory; update the pinned dependency and lockfile in a reviewed change.

---

# [unh_v1] How this copy was made

- Source: https://github.com/Zleap-AI/SAG-Benchmark @ 942f498fb20001ade7492756a9002b5f6ad750d8 (MIT, `licenses/SAG-Benchmark-LICENSE`; GraphRAG is MIT, `licenses/GraphRAG-LICENSE`).
- Why this repo: MemGraphRAG, HippoRAG 2, LogicRAG and LinearRAG publish no Microsoft GraphRAG code. Their GraphRAG numbers have no released scripts. SAG-Benchmark is the only public harness found that runs `graphrag` on the HippoRAG 2 HotpotQA/MuSiQue files. Its `dataset/README.md` requires those files unchanged, which is what `evaluation_data/` holds.
- Copied: `external/graphrag/{graphrag_benchmark,reproduce,src,pyproject.toml,uv.lock,UPSTREAM.md}`, plus `pipeline/evaluation/utils/{load_utils,sample_identity,reproduce_dataset}.py` verbatim with an empty package `__init__`.

## Local changes

1. **Paths:** `REPOSITORY_ROOT` now points at this folder, which contains the vendored `pipeline/`. `.env` is read from here. `--dataset-root` defaults to `<repo>/evaluation_data/<ds>/`. Touched files: `reproduce/_common.py`, `reproduce/Step_0_prepare_dataset.py`, `graphrag_benchmark/dataset.py`, `graphrag_benchmark/retrieval.py`.
2. **Step 3 replaced** (`reproduce/Step_3_qa_eval.py`; the original is kept as `Step_3_qa_eval.sag_upstream.py.txt`). Upstream shelled out to SAG's `scripts/run_qa_benchmark.py`, which needs SAG's whole `pipeline` package and uses SAG's own zero-shot reader prompt. The replacement reads the local-search context with the HippoRAG 2 `rag_qa_musique` prompt (`common/prompts.py`) and writes `outputs/<ds>/predictions.json` for `common/score.py`.

Upstream settings kept:
- local search at community level 2, and the whole context passed as one doc;
- chunks of 1000/100 tokens grouped by passage id (so no chunk mixes passages);
- `extract_graph` with `max_gleanings=1` and entity types organization/person/geo/event;
- claims off.

For DeepSeek, keep `GRAPHRAG_DISABLE_THINKING=0`. The value 1 sends vLLM-only `chat_template_kwargs`.
