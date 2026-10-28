# Insurance contracts on every system: plan

Status 2026-10-09: decisions taken (§6). Nothing is implemented and nothing has run.

## 1. The data: two contract sets, reported as two tables

**Short contracts** (`evaluation_data/ten_contracts_dataset/`): 10 synthetic life-insurance policies.

| Property | Value |
|---|---|
| Contract size | 1.3k–3.1k tokens each, 21k in total |
| QA pairs | 361 |
| Facts | 731 |
| Format | the same as the long set: `qa_pairs/qa_<c>.json`, `facts/facts_<c>.json`, `contracts/<c>.txt` |

QA by category:

| Category | Count |
|---|---|
| CAT1 | 145 |
| CAT2 | 56 |
| CAT3 | 74 |
| CAT4 | 35 |
| CAT5 | 51 |

**Long contracts** (`evaluation_data/long_form_contracts/`):

| Contract | Tokens (cl100k) | QA pairs | Facts |
|---|---|---|---|
| 2025_pdp_contract | 6,000 | 64 | 241 |
| td_contract | 10,700 | 100 | 428 |
| all_state_auto_insurance | 13,700 | 85 | 423 |
| progressive_maine_auto_policy | 15,800 | 85 | 509 |
| capital_health_plan | 48,700 | 111 | 896 |
| **Total** | **95,000** | **445** | **2,497** |

- **QA** (`qa_pairs/qa_<contract>.json`): `{question, answer, evidence, category}`, in five categories.

  | Category | Count |
  |---|---|
  | CAT1 atomic lookup | 131 |
  | CAT2 relational lookup | 100 |
  | CAT3 conditional reasoning | 98 |
  | CAT4 multi-hop reasoning | 82 |
  | CAT5 unanswerable | 34 |

  Gold answers are short sentences or phrases, not spans. Every CAT5 gold answer is the literal string `"Not specified in this policy."`.
- **Facts** (`facts/<contract>.json`): `{id, fact}`, one atomic sentence each.
- **Text** (`contracts/<contract>.txt`): Markdown converted from PDF. It contains tables and `<!-- image -->` markers. `capital_health_plan` alone holds 51% of the tokens and 36% of the facts.

The corpus is about 7× smaller than QuALITY dev and about 100× smaller than the HotpotQA corpus. The cost is in the queries: 2,942 queries per system row, each one a retrieval plus a judge call.

## 2. The protocol in one paragraph

- **One graph per contract per system:** 5 builds × 7 systems.
- **Queries:**
  - each QA question is retrieved against its own contract's graph only;
  - each fact is used verbatim as the retrieval query against its own contract's graph.
- **Retrieval:** each system keeps its own retrieval and context size, exactly as on HotpotQA, MuSiQue and QuALITY.
- **Answering:** QA contexts go through one shared answer step, the same prompt for every system.
- **Judging:**
  - **QA:** an LLM judge compares the answer to the gold answer.
  - **Fact recall:** the judge sees only the retrieved context and the fact, and decides whether the fact is stated in or can be inferred from the context. There is no answer step for facts.
- **Models:**
  - **LLM:** every call (construction, retrieval-time calls, answer step, judge) goes to the local **Qwen3-30B-A3B-Instruct-2507-FP8** on vLLM `:8005`. This model has no thinking mode, so chain of thought is off by construction.
  - **Embeddings:** voyage-4-nano on `:8001`, as before.
- **Settings:** all systems use the same default settings as in the HotpotQA, MuSiQue and QuALITY runs.

## 3. Ingestion per system (same rule as QuALITY: a system that has its own chunker uses it)

The text is the contract `.txt` with `<!-- image -->` lines removed (they are PDF artifacts). It is otherwise byte-identical for every system, and the first line is the contract title.

| System | What it indexes per contract | Who chunks | Query | Context passed on |
|---|---|---|---|---|
| HyGRAG | `Corpus.json`, 1 row (`title\ntext`) | HyGRAG, 1200/100 | question / fact | `_retrieve_relevant_contexts` |
| HyperGraphRAG | `rag.insert([title\ntext])` | HyperGraphRAG, 1200/100 | question / fact | `aquery(only_need_context=True)`, hybrid |
| MS GraphRAG | `input/<contract>.txt` | graphrag, 1200/100 (o200k) | question / fact | local-search `build_context` |
| MemGraphRAG | `<contract>.txt` | MemGraphRAG, 256/32 windows | question / fact | top-5 windows |
| HippoRAG 2 | pre-cut chunks | HyGRAG's chunker, 1200/100 (it has none) | question / fact | top-5 passages |
| LogicRAG | pre-cut chunks | HyGRAG's chunker, 1200/100 (it has none) | question / fact | `info_summary` (QA and facts) |
| UNH, full view | the contract | UNH's chunker (see D2) | question / fact | `contexts.json`, mode `all` |
| UNH, hypergraph view | the same graph | — | question / fact | mode `hypergraph` |

UNH construction defaults are the same as in the HotpotQA 250 run:
- fact dedup on;
- `dedup_similarity_threshold` 0.5;
- `dedup_max_cluster_size` 72;
- `same_chunk_min_facts` 3;
- no clustered round 3;
- label-only edges.

The only thing that changes is the chunking, because a contract is one long document rather than a list of passages.

**What this means for small contracts.** The PDP contract cuts into about 5 chunks of 1,200 tokens, so a top-5 retriever (HippoRAG 2, LogicRAG) returns almost the whole contract. On the small contracts, fact recall for chunk-based systems will sit near the full-contract ceiling. The short set is affected the same way (see §6).

## 4. Shared answer step and judges

### 4.1 Answer prompt (QA only), `common/prompts.py build_contract_messages`

The HippoRAG 2 one-shot prompt is a Wikipedia multi-hop example with "Wikipedia Title:" blocks, which is a poor fit for contracts. The proposal is one plain prompt for every system. Its wording comes from the existing insurance `AnswerQuestion` signature (`experiments/evaluate_mine.py:79`), so the results stay comparable with the earlier insurance runs:

```
system: You are a precise question-answering system grounded strictly in the context you are given.
user:   Context:
        <the system's context>

        Question: <question>

        Use ONLY the context; do not use outside knowledge. First reason briefly after "Thought:",
        then give the final answer in 1-2 concise sentences after "Answer:".
        If the context does not contain the answer, the final answer is exactly: Not specified in this policy.
```

- **Settings:** the same as the HotpotQA reader:
  - `max_tokens` 2048;
  - no temperature sent (Qwen's `generation_config` defaults apply, see D9);
  - cached in `reader_cache.sqlite`;
  - usage recorded.
- **Abstention phrase:** the exact phrase lets CAT5 be judged. It also means an answerable question whose context missed the evidence gets an abstention, which is scored wrong. That is intended, since it measures the retrieval miss.

### 4.2 QA judge, `common/score_contracts.py --task qa`

The judge sees the question, the gold answer and the predicted answer, never the context. The rubric is the existing insurance `EvaluateQA` (`evaluate_mine.py:98`), sent as a plain prompt:
- 1.0 fully correct;
- 0.5 partially correct;
- 0.0 wrong.

**CAT5 rule:** a prediction that abstains (says it is not specified, not stated or not in the context) is correct. Any concrete answer to a CAT5 question is wrong.

**Reported:**
- mean score and strict accuracy (share scored 1.0), both overall excluding CAT5 and including CAT5;
- per category;
- per contract;
- macro over contracts;
- abstention rate on answerable questions (CAT1–4);
- mean context tokens;
- token usage.

### 4.3 Fact-recall judge, `common/score_contracts.py --task facts`

The prompt is the existing insurance `EvaluateFact` (`evaluate_mine.py:66`), sent as a plain prompt. It asks for 1 if the fact is stated, paraphrased or directly inferable from the context, else 0, with no outside knowledge. The judge sees **only the context and the fact**.

**Reported:**
- recall (share = 1), overall and per contract;
- macro over contracts;
- mean context tokens;
- empty-context count;
- usage.

**Leakage check:** the fact is also the retrieval query. A context written by an LLM that was shown the query can restate the fact, which would be a false recall. Of the 7 systems, only LogicRAG's `info_summary` is LLM-written from the query. It is used anyway (D6) and stored so it can be checked; the other systems pass retrieved text or graph content.

## 5. Running on local Qwen

- **No code change for most systems.** Every runner already reads its endpoint from `common/env.sh`. A new `common/env_qwen.sh`, sourced after it, sets:

  ```
  LLM_BASE_URL=http://localhost:8005/v1
  LLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507-FP8
  LLM_API_KEY=EMPTY        (and DEEPSEEK_API_KEY=EMPTY, which env.sh / runners require)
  LLM_DISABLE_THINKING=0   (the DeepSeek "thinking" field is meaningless to vLLM; don't send it)
  ```

  Per system:
  - **MS GraphRAG:** `settings.yaml` already takes `${LLM_MODEL}`. `deepseek_proxy.py` forwards to `LLM_BASE_URL`.
  - **HyGRAG:** its config comes from our harness. It gets a Qwen variant of the LLM block.
  - **UNH:** `configure_dspy` with `openai/Qwen/...` and the `:8005` base, as the old insurance runs did.
- **vLLM server requirements:**
  - `--max-model-len` of at least 65,536. The full-contract reference row needs about 50k tokens for capital_health_plan, and HyperGraphRAG contexts reach about 17k tokens.
  - `--enable-prefix-caching` (the default), so the many fact queries against the same contract reuse cached prefixes.
  - `--enable-prompt-tokens-details`, so cached tokens are reported like DeepSeek's cache hits.
- **The server is not running now.** `:8005` did not answer on 2026-10-09.
- **Concurrency:** one local server is the bottleneck, so the DeepSeek-era settings are far too high. MS GraphRAG 1,200 and UNH 900 would only queue and hit timeouts.
  - Run builds with `CONCURRENCY_MULTIPLIER` around 0.25, and UNH at about 64/64 workers.
  - Run one or two systems at a time, not all seven.
  - Raise client timeouts where a system has them (HyperGraphRAG 300 s).

## 6. Decisions (taken 2026-10-09)

| # | Decision |
|---|---|
| D1 | **No context cap.** Each system's native context. Context tokens are stored for every query |
| D2 | **UNH uses its own default chunking** (600/50) |
| D3 | **HippoRAG 2 and LogicRAG** (no chunker of their own) get pre-cut 1200/100 chunks. Every other system gets the whole contract as one document and chunks it itself |
| D4 | **A plain contract answer prompt** (§4.1), shared by every system. Written reasoning ("Thought:") before "Answer:", as in the HotpotQA reader; the answer is the text after "Answer:". Qwen3-Instruct-2507 has no hidden thinking mode, so the model's reasoning is this written step |
| D5 | **QA judge:** the old 1 / 0.5 / 0 scale (old `EvaluateQA` wording). **Fact judge:** 1 / 0 (old `EvaluateFact` wording). Both judges reason in writing first, then end with a parseable `Score:` line; the reasoning is stored. Plain OpenAI-client prompts through the shared harness, not DSPy. A contract's score is the mean over its questions, shown as a percentage. Appendix: the same judge outputs with 0.5 counted as 0 (no extra calls) |
| D6 | **LogicRAG:** the final context it returns (`info_summary`) is used for QA and for fact recall |
| D7 | **No reference rows** |
| D8 | **Report both** pooled (micro) and mean of per-contract scores (macro), per category, for each contract set separately |
| D9 | **No temperature sent** |
| D10 | **Qwen for everything**, judges included |

### Result tables (one per set: short, long)

- **QA table:** rows are the systems (UNH appears twice, once per view). Columns:
  - CAT1, CAT2, CAT3, CAT4, CAT5;
  - average over all questions;
  - average excluding CAT5;
  - macro average over contracts;
  - mean context tokens.
- **Fact-recall table:** rows are the systems. Columns:
  - recall pooled;
  - recall macro over contracts;
  - one column per contract (long set) or the min/max contract (short set);
  - mean context tokens.
- **Per-contract tables:** the same columns for each contract.

### What is stored, per system, per set

`outputs/contracts_<set>/<contract>/`:

- `contexts_qa.json` and `contexts_facts.json`: for each query, the id, the query, the full retrieved context, its token count, and retrieval time.
- `answers_<system>.json`: for each question, the question, gold answer, category, evidence, the full context, the reader's raw reply, the answer, and reader usage.
- `scores_<system>_qa.json`:
  - per question: score (1 / 0.5 / 0), the judge's raw reply, category;
  - the contract's mean, per category, with and without CAT5.
- `scores_<system>_facts.json`:
  - per fact: id, fact, score (1 / 0), the judge's raw reply, the full context, context tokens;
  - the contract's recall.

`outputs/contracts_<set>/summary_<system>.json` holds the set-level pooled and macro numbers, per category, plus construction usage (calls, prompt/completion/cached tokens, time) per contract.

**Caveat to print under the short-contract tables:**
- Each short contract cuts into 2–3 chunks of 1,200 tokens, so HippoRAG 2's and LogicRAG's top-5 is the whole contract.
- On the long set, the same happens for the PDP contract (about 5 chunks).

**Caveat for LogicRAG fact recall (D6):** LogicRAG's `info_summary` is written by the LLM after it has seen the fact as the query, so it can restate the fact. The stored contexts make this checkable afterwards.

## 7. Implementation steps (after your go-ahead; no system internals touched)

1. **`evaluation_data/build_contract_inputs.py`** builds `<set dir>/inputs/<contract>/` for both sets (`ten_contracts_dataset` = short, `long_form_contracts` = long):
   - `document.txt` and `documents.json` (one row);
   - `chunks.json` and `chunks_logicrag.json` (HyGRAG's chunker);
   - `memgraphrag_corpus.txt`;
   - `ms_graphrag_input/<contract>.txt`;
   - `qa.json` with ids `<contract>/qa/<n>`;
   - `facts.json` with ids `<contract>/fact/<id>`;
   - `manifest.json` with SHA-256 of every file.

   It also writes a smoke subset: the PDP contract, the first 10 QA and 20 facts.
2. **`common/prompts.py`:** `build_contract_messages`, the D4 prompt. **`common/answer.py`:** `--dataset contracts` reads every contract's QA contexts, with the same caching and usage records.
3. **`common/score_contracts.py`:** the QA judge and fact judge (D5), with caches, usage, and the reports listed in §4.
4. **Each system, `<system>/run_contracts.sh`,** loops over the 5 contracts:
   - build the index into `outputs/contracts/<contract>/`;
   - retrieve contexts for the QA questions into `contexts_qa.json`;
   - retrieve contexts for the facts into `contexts_facts.json`.

   Loader changes: a `contracts` choice in `hipporag2/main.py`, `hypergraphrag/main.py`, `ms_graphrag/query.py` and `hygrag/{prepare_data,main}.py`; a corpus path in `memgraphrag/run_contracts.sh` and `logicrag/run.py`. Each change goes in that system's `UPSTREAM.md`.
5. **UNH:** a `--dataset contracts --contract <name>` mode in `unh/main.py` that chunks the contract with UNH's chunker (D2), builds with the HotpotQA-run defaults, and retrieves both views for QA and facts. `launch_unh.sh` gets a contracts mode.
6. **Launch scripts:**
   - `common/env_qwen.sh` and `launch_contracts.sh` (systems in sequence or two at a time, Qwen concurrency settings);
   - `stop_all.sh` also stops `*_contracts*` jobs;
   - `collect_results.py` gets a contracts table: QA mean, QA strict, QA by category, fact recall micro/macro, context tokens, construction tokens and calls.
8. **Smoke test** every system on the smoke subset (PDP contract, 10 QA, 20 facts), against Qwen once it is up. Check that:
   - every build finishes;
   - no context is empty;
   - every QA and fact id lines up;
   - parse rates are sane;
   - the abstention rate on CAT1–4 is sane.
9. **Full runs only when you say go.**

## 8. Size of the full run (local, so time rather than money)

| Item | Count |
|---|---|
| Builds | 105 (15 contracts × 7 systems) |
| Queries per row | 4,034 (806 QA + 3,228 facts) |
| Rows | 8 |
| Answer calls | 6,448 (8 × 806) |
| Judge calls | about 32,300 (8 × 4,034) |

The short set adds 10 small builds per system, 361 QA and 731 facts.

- **Construction input**, scaled from QuALITY's projections by corpus size (×0.14):
  - HippoRAG 2 about 0.7M tokens;
  - HyGRAG about 1M;
  - MS GraphRAG about 3.5M;
  - MemGraphRAG about 11M;
  - HyperGraphRAG about 12.5M;
  - UNH a few million.
- **Fact-judge input dominates:** about 2,500 facts × 8 rows × (2–15k context tokens) ≈ 50–250M prompt tokens. On one vLLM server with prefix caching, expect hours rather than days. The biggest single cost is HyperGraphRAG's long contexts.
- **Retrieval-time LLM calls on the 2,942 queries:**
  - LogicRAG: several rounds per query;
  - HippoRAG 2: one recognition-memory filter call;
  - HyperGraphRAG: one keyword call;
  - UNH: about one planner call;
  - MS GraphRAG, MemGraphRAG and HyGRAG: none or few.

## 9. Implementation (2026-10-09)

| File | What it does |
|---|---|
| `evaluation_data/build_contract_inputs.py` | Builds `evaluation_data/contracts/{short,long,smoke}/<contract>/`, one inputs dir per contract, in the QuALITY inputs layout. `questions.json` holds the QA questions, then the facts; every row has a `kind` of `qa` or `fact`. |
| `common/env_qwen.sh` | Points every runner at local Qwen. Source it instead of `env.sh`. |
| `common/contracts_lib.sh` | `for_each_contract`: builds and retrieves each contract, `CONTRACT_JOBS` at a time, with per-contract logs in `run_logs/contracts/`. |
| `<system>/run_contracts.sh <set>` | One graph per contract, then retrieval for every query, then `common/contracts_score.py`. |
| `common/contracts_score.py` | The shared answer step (`answer.Reader`, contract prompt), the QA judge and the fact judge. Writes `contracts_results/<set>/<system>/<contract>/scores_{qa,facts}.json` and `summary.json`. |
| `common/collect_contracts.py` | Writes `contracts_results/<set>/tables.md` and `qa.csv` / `facts.csv`. |
| `launch_contracts.sh <set> [systems]` | Starts the runners in the background. `stop_all.sh` stops them. |

Every runner reuses its existing inputs-dir loader (`--dataset quality`), so no system's index or retrieval code changed.

**Code changes:**
- **LogicRAG:** `topk(min(top_k, len(corpus)))`. Upstream raised an error on corpora of fewer than 5 chunks, and every retrieval came back empty.
- **HyGRAG:** no temperature is sent for non-DeepSeek endpoints.
- **HyperGraphRAG:** the client timeout reads `LLM_TIMEOUT` (1,800 s on Qwen).
- **UNH:** `unh/main.py --dataset contracts`.

Each change is listed in that system's `UPSTREAM.md`.

### Smoke test (2026-10-09)

Setup: the `smoke` set, two contracts (`contract_1_term_life`, `2025_pdp_contract`), each with 10 QA (2 per category) and 10 facts. Qwen3-30B-A3B-Instruct-2507-FP8 on :8000 and voyage-4-nano on :8001.

**Every baseline ran end to end.** On every run:
- all queries were aligned;
- 0 contexts were empty;
- 0 judge replies were unparsed;
- 0 reader replies hit `max_tokens` or lacked `Answer:`.

System-specific checks:
- **HyGRAG:** build checks passed and no questions were flagged.
- **HyperGraphRAG:** 40/40 contexts ok.
- **MS GraphRAG:** 177 of 180 entities were searchable.
- **LogicRAG:** every row has an `info_summary`.

| System | QA all | Fact recall | Ctx tokens | Index + retrieval (PDP contract) |
|---|---|---|---|---|
| HippoRAG 2 | 97.5 | 100 | 3.6k | 136 s |
| LogicRAG | 87.5 | 100 | 0.1-0.2k | 467 s for both contracts |
| MemGraphRAG | 90.0 | 90 | 1.2k | 240 s |
| HyperGraphRAG | 100 | 100 | 14k | 1,397 s |
| MS GraphRAG | 95.0 | 100 | 7.5k | 1,061 s |
| HyGRAG | 92.5 | 100 | 3.6k | 125 s |

These numbers only show that the pipeline works; with 20 questions and small contracts they mean nothing. LogicRAG's CAT5 score is 50 on 4 questions. One reply explained that the policy has no cash value, which the strict CAT5 rule scores 0; the other made up a "cannot be reinstated" rule.

**UNH: not smoke-tested to completion** (skipped on request).
- `contract_1_term_life` built in 281 s (85 nodes, 20 LLM calls).
- Retrieval runs at about 45-65 s per query on this small graph. The time goes to UNH's own path ranking (re-encoding paths on the GPU), not the LLM.

## 10. Switch to DeepSeek and UNH PPR (2026-10-11)

- **LLM:** every runner now sources `common/env_contracts.sh`, which defaults to the DeepSeek setup of every other benchmark (`env.sh`: deepseek-flash, thinking disabled, no temperature sent; HyGRAG sends DeepSeek's default 1.0 as always; voyage-4-nano on :8001). `CONTRACTS_LLM=qwen` restores the local-Qwen setup (`env_qwen.sh`). Indexing, retrieval-time calls, the shared answer step and both judges all use DeepSeek.
- **Set:** the long set only (5 contracts, 445 QA, 2,497 facts); the short set is not run.
- **UNH:** `unh/run_contracts.sh` retrieves with the PPR retriever, setting `t2000_f40_dmp06` (as HotpotQA, MuSiQue, QuALITY, G-Medical), all three views (full, hypergraph, discourse; `contracts_score.py` knows `unh_discourse`). Construction knobs are environment variables as in `run_gmedical.sh`; defaults are `unh/main.py`'s.
- **HyGRAG:** `--allow-fallback`, as on G-Medical.
- **The Qwen smoke outputs** were moved to `*_contracts_smoke_qwen` / `contracts_results/smoke_qwen`, so a DeepSeek smoke run cannot reuse Qwen-built indexes.
- Per-system concurrency per contract is unchanged (MS GraphRAG 64, HyGRAG 64, HyperGraphRAG 32, MemGraphRAG 20, HippoRAG 2 OpenIE 8, LogicRAG 12 processes, scorer 32), with 5 contracts at once per system: the peak over the six baselines is roughly 1,100 concurrent DeepSeek requests.
- **QA only (user decision 2026-10-11):** no fact recall. `build_contract_inputs.py` also writes `long_qa` and `smoke_qa`: the same contracts, documents and chunks (byte-identical), with only the QA questions (445 and 20). Runners and `contracts_score.py` take the set name unchanged; with no fact rows nothing is fact-judged, and `collect_contracts.py` drops the fact-recall table. Run: `bash evaluation_scripts/launch_contracts.sh long_qa hipporag2 memgraphrag logicrag hypergraphrag ms_graphrag hygrag`.
- **Reader and judge max_tokens 4096** (was 2048), as on G-Medical.

## 11. Results, long contracts, QA only (DeepSeek, 2026-10-11)

Pooled over 445 questions; judge 1 / 0.5 / 0, in percent. UNH variants reuse one graph per contract (detailed facts, compact event prompt for same-chunk and global edges, clustered_fast 20/45) and the PPR setting t2000_f40_dmp06; "+ N passages" = `ppr_passages` N (`unh/main.py --tag hyb_pN`, scorer systems `unh_hyb_pN`).

| Rank | System | CAT1 | CAT2 | CAT3 | CAT4 | CAT5 | Mean | Ctx tokens |
|---|---|---|---|---|---|---|---|---|
| 1 | UNH full + 5 passages | 95.04 | 94.0 | 95.92 | 79.88 | 97.06 | 92.36 | 4,754 |
| 2 | HyperGraphRAG | 98.09 | 95.5 | 94.39 | 72.56 | 100.0 | 92.13 | 18,383 |
| 3 | LogicRAG | 93.51 | 93.5 | 94.39 | 82.32 | 88.24 | 91.24 | 122 |
| 4 | UNH full + 3 passages | 94.27 | 90.0 | 94.9 | 76.22 | 100.0 | 90.56 | 3,602 |
| 5 | HippoRAG 2 | 97.33 | 91.0 | 93.88 | 71.34 | 97.06 | 90.34 | 5,796 |
| 6 | HyGRAG | 94.27 | 88.0 | 91.33 | 71.95 | 100.0 | 88.54 | 4,914 |
| 7 | UNH full (PPR only) | 89.31 | 86.0 | 94.39 | 61.59 | 100.0 | 85.39 | 1,857 |
| 8 | MemGraphRAG | 81.68 | 85.5 | 90.31 | 70.73 | 100.0 | 83.82 | 1,250 |
| 9 | UNH discourse | 88.55 | 85.5 | 89.29 | 59.76 | 100.0 | 83.6 | 1,874 |
| 10 | UNH hypergraph | 86.26 | 87.0 | 86.73 | 62.8 | 97.06 | 83.03 | 1,824 |
| 11 | MS GraphRAG | 69.85 | 86.5 | 73.47 | 64.02 | 97.06 | 75.39 | 9,107 |
