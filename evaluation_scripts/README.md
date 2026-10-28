# Evaluation scripts: HotpotQA and MuSiQue

This folder holds the published systems, vendored from their official code with only minimal patches. All of them run on the same data (`evaluation_data/<ds>/`, HippoRAG 2 release), the same LLM, and the same embedding model, and all are scored by one shared scorer.

| Folder | System | Upstream | Ingestion unit |
|---|---|---|---|
| `hipporag2/` | HippoRAG 2 (`main.py`) | OSU-NLP-Group/HippoRAG @ 1438aba | one doc per passage (`title\ntext`) |
| `logicrag/` | LogicRAG (`run.py` + its `RAGEvaluator`) | chensyCN/LogicRAG @ 72a298c | one doc per passage (`Title: … Content: …`) |
| `memgraphrag/` | MemGraphRAG (`index.py`, `retrieval_dataset_test.py`) | XMUDeepLIT/MemGraphRAG @ cd6fabd | **256-token windows over the whole `.txt`**, crossing passage boundaries (upstream behaviour) |
| `ms_graphrag/` | Microsoft GraphRAG 3.2.0 (stock `graphrag init` settings), local-search context → shared reader | PyPI `graphrag==3.2.0`, unmodified; DeepSeek via `deepseek_proxy.py` | **the whole `<ds>_scrambled.txt` as one document**, stock 1200/100-token chunks across passage boundaries |
| `hypergraphrag/` | HyperGraphRAG (`main.py`: index → hybrid context → reader) | LHRLAB/HyperGraphRAG @ d587cdf `evaluation/hypergraphrag` | one doc per passage (`title\ntext`), chunked within the doc |
| `hygrag/` | HyGRAG (`HKGraphTreeLSH`: index → `_retrieve_relevant_contexts` → shared reader) | zjunet/HyGRAG @ fa1a57e (DIGIMON fork), vendored in `hygrag/HyGRAG/` | one doc per passage (`title\ntext`, DIGIMON `Corpus.json`), never chunked across passages |
| `common/` | shared reader prompt, judge prompt, metrics, scorer, data check, ST embedding server | — | — |

Each system folder has an `UPSTREAM.md` that lists every local change. Every patched line in the code is tagged `[unh_v1]`.



## UNH (our system), same pipeline

- UNH runs through the same pipeline as the baselines: the full corpus with one passage per chunk, UNH's current construction defaults, and old-engine retrieval with short labels.
- It answers 250-question stratified subsets (`common/make_subset.py`) through the shared answer step and judge.
- Run it with `bash evaluation_scripts/launch_all.sh unh` or `bash evaluation_scripts/unh/run.sh <ds>`.
- `python evaluation_scripts/common/collect_results.py --subset evaluation_data/{hotpotqa,musique}/subsets/*_250_seed0.json` scores all seven systems on the same questions, with no reruns of the baselines.
- See [unh/NOTES.md](unh/NOTES.md).

## QuALITY (multiple choice, pooled dev articles)

- Every baseline also runs on QuALITY's official dev split: 115 articles, 2,086 questions, one pooled index per system, a shared letter-only answer prompt, and HyGRAG's accuracy metric.
- Inputs: `python evaluation_data/build_quality_inputs.py`.
- Run all six systems: `bash evaluation_scripts/launch_quality.sh`.
- Run one system: `bash evaluation_scripts/<system>/run_quality.sh`.
- Scores: `common/score_quality.py`; `common/collect_results.py` prints them.
- Details and per-system ingestion: [QUALITY_IMPLEMENTATION.md](QUALITY_IMPLEMENTATION.md). Decisions: [QUALITY_EVAL_PLAN.md](QUALITY_EVAL_PLAN.md).

## Insurance contracts (one graph per contract, local Qwen)

- Two sets: **short** (`ten_contracts_dataset`, 10 contracts, 361 QA, 731 facts) and **long** (`long_form_contracts`, 5 contracts, 445 QA, 2,497 facts).
- Each system builds one graph per contract and retrieves a context for every QA question and every fact (the fact verbatim is the query).
- QA goes through one shared answer step (written reasoning, then `Answer:`), then a judge that scores 1 / 0.5 / 0.
- Fact recall: a judge decides whether the fact can be inferred from the context.
- Every LLM call, judges included, goes to local Qwen3-30B-A3B-Instruct-2507 (`common/env_qwen.sh`).
- Inputs: `python evaluation_data/build_contract_inputs.py`.
- Run: `bash evaluation_scripts/launch_contracts.sh <short|long|smoke> [systems...]`.
- Tables: `python evaluation_scripts/common/collect_contracts.py`.
- Details: [CONTRACTS_EVAL_PLAN.md](CONTRACTS_EVAL_PLAN.md).

## Models

- **LLM:** DeepSeek V4.1 Flash (`LLM_MODEL`, default `deepseek-flash`). Thinking is disabled in every client (`LLM_DISABLE_THINKING=1`); DeepSeek turns it on by default. No client sends a temperature, so every call uses DeepSeek's default (1.0). That covers each system's internal calls, the shared answer step and the judge. `LLM_TEMPERATURE=<x>` sends x everywhere instead. Upstream sends 0 in HippoRAG 2, MemGraphRAG and LogicRAG; `[unh_v1]` patches change that.
- **Embeddings:** `voyageai/voyage-4-nano`, the model the nkg/GraphRAG/HyperGraphRAG harnesses already use.
  - LogicRAG loads it directly with SentenceTransformer (`trust_remote_code=True`).
  - The others call one OpenAI-compatible `/v1/embeddings` endpoint, `EMBEDDING_BASE_URL` (default `http://localhost:8001/v1`). The vLLM 0.19 command is in `common/env.sh`. It needs `--runner pooling --convert embed --enforce-eager --hf-overrides '{"architectures": ["VoyageQwen3BidirectionalEmbedModel"]}'`; its vectors match SentenceTransformer's (cosine mean 0.9997, top-5 retrieval overlap 0.98). `common/st_embedding_server.py` is a slower fallback.
  - None of the four sends voyage's `query:` / `document:` prompts; that is upstream behaviour everywhere.

All defaults live in `common/env.sh`.

## Run

Python env: `~/miniconda3/envs/unh_eval` (a clone of `contracts` plus python-igraph, backoff, ratelimit and gritlm) imports all four systems.

```bash
bash evaluation_scripts/launch_all.sh     # all four systems x both datasets in parallel, logs in run_logs/ (GPU=0 default)
bash evaluation_scripts/stop_all.sh       # stop them (LLM calls are cached, so a relaunch resumes cheaply)
```

or one at a time:

```bash
export DEEPSEEK_API_KEY=...
python evaluation_scripts/common/verify_data.py          # byte-check the datasets
bash evaluation_scripts/hipporag2/run.sh   musique       # add JUDGE=1 for LLM-Acc
bash evaluation_scripts/logicrag/run.sh    musique       # optional 2nd arg: question limit
bash evaluation_scripts/memgraphrag/run.sh musique
bash evaluation_scripts/ms_graphrag/run.sh musique       # first: conda env ms_graphrag (ms_graphrag/requirements.txt)
bash evaluation_scripts/hygrag/run.sh musique            # first: conda env hygrag (hygrag/requirements.unh.txt)
bash evaluation_scripts/hypergraphrag/run.sh musique     # optional 2nd arg: question limit; READER=upstream adds its own reader
```

`python evaluation_scripts/common/collect_results.py` gathers every `scores_*.json` into one Markdown table plus `results_table.csv` (add `--include-native` for the systems' own answers).

Each runner runs its system unchanged (its own index, retrieval and context), then the **shared answer step** `common/answer.py`, then `common/score.py` on those answers. The system's own answers are also scored, as `scores_native_*.json` (EM/F1 only), for reference. `score.py` writes `scores_<system>_<ds>.json` next to the answers, with per-question rows and these totals:

| Metric | Definition | Source code |
|---|---|---|
| EM, F1 | gold = `answer` + `answer_aliases`, max over golds | HippoRAG 2 `evaluation/qa_eval.py` (same in MemGraphRAG) |
| Contain-Acc | normalized gold ⊂ normalized prediction (the MemGraphRAG table's "Str-Acc") | LinearRAG `src/evaluate.py` |
| LLM-Acc (`JUDGE=1`) | "correct"/"incorrect" judge; the papers used gpt-4o-mini, so record which judge you used | LinearRAG `src/evaluate.py`, verbatim prompt |
| Recall@k (off; `--recall-k 2 5`) | gold passages in top-k retrieved | HippoRAG 2 `evaluation/retrieval_eval.py`; only for native hipporag2 / logicrag outputs |

## Answer step (shared)

`common/answer.py` is the only code that produces the reported answers. For every system it uses:
- the same prompt: HippoRAG 2's `rag_qa_musique` one-shot CoT, built by `common/prompts.py`, identical to the messages HippoRAG's `qa()` sends;
- the same answer extraction (text after "Answer:");
- the same settings: max_tokens 2048, no temperature sent (DeepSeek default), thinking disabled.

Only the context differs, and each system chooses its own:

| System | Context passed to the reader | Where it comes from |
|---|---|---|
| hipporag2 | top-5 passages | `predictions.json` `retrieved_docs` (= `HippoRAG.qa()` input) |
| memgraphrag | top-5 token windows | `qa_results.json` `retrieved_docs[:5]` (= `MemGraphRAG.qa()` input with `--qa-top-k 5`) |
| logicrag | its final information summary | `info_summary`, saved by an `[unh_v1]` patch (= `LogicRAG.generate_answer()` input) |
| hypergraphrag | the hybrid-mode entities/relations/sources context | `knowledge.json` (= `aquery(only_need_context=True)`) |

Each context document becomes one `Wikipedia Title: …` block. That is the same formatting the prompt's one-shot example uses, and the same way `ms_graphrag` Step 3 passes GraphRAG's context. The LLM judge (`score.py --llm-judge`) sees only the predicted answer and the gold answer, never the context.

## Verified

2026-09-30, real DeepSeek (`deepseek-flash`, thinking off) plus the voyage-4-nano server, on the first 3 HotpotQA questions and their 30 passages: hipporag2, logicrag, memgraphrag and hypergraphrag each ran their own pipeline → `common/answer.py` → `score.py` (the judge was checked on hipporag2). None had empty or truncated answers.

2026-09-23, local mock LLM and the real voyage-4-nano embeddings:

Each pipeline was run end to end on the first 3 HotpotQA questions:

- hipporag2: index → QA → `predictions.json`. `score.py` Recall@2/5 = HippoRAG's own numbers.
- logicrag: `run.sh` → its evaluator → `score.py` incl. judge.
- memgraphrag: index (memory, schema, conflicts, graph) → QA → `score.py`.
- ms_graphrag: that 2.0.0 harness is archived in `ms_graphrag_sag_2.0.0/`. Its replacement, graphrag 3.2.0, was smoke-tested on 2026-09-30 against the real DeepSeek API: 200 passages, 10 questions, index → context → answer → score incl. judge (see `ms_graphrag/UPSTREAM.md`).

