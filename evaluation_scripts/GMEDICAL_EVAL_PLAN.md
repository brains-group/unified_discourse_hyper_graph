# G-Medical: evaluation protocol for every system

Date: 2026-10-11.

G-Medical is the Medical subset of GraphRAG-Bench (Xiang et al., *When to use Graphs in RAG*, arXiv 2506.05690, ICLR 2026). The harness follows how MemGraphRAG evaluates it (Table 1, column G-Medical), using the same data files MemGraphRAG ships.

## 1. The data, as MemGraphRAG has it

Source: `MemGraphRAG/dataset/medical/`.

- **`medical.txt`:** 44 lines, one NCCN patient guideline per line, 165k words (about 221k cl100k tokens).
  - Each guideline has had its PDF layout flattened into one paragraph, with no title and no separator other than the newline.
  - 3 lines are exact duplicates (AML, CML, ALL), so there are 41 distinct guidelines.
- **`question.json`:** 2,062 questions, grouped by type.

  | Type | GraphRAG-Bench level | Count | Gold answer |
  |---|---|---|---|
  | `type1` | Fact Retrieval | 1,098 | about 11 words |
  | `type2` | Complex Reasoning | 509 | about 17 words |
  | `type3` | Contextual Summarize | 289 | about 30 words |
  | `type4` | Creative Generation | 166 | about 120 words |

  - Each question has `{id, source, question, answer, evidence}`, where `evidence` is a list of fact sentences.
  - `source` names one of 34 guidelines, so nearly every question is answered from a single guideline.

**How MemGraphRAG ingests it:** `code/index.py` reads `medical.txt` as one string and cuts 256/32-token windows straight across guideline boundaries.

**How it answers:**
- `retrieval_dataset_test.py` loads the type-grouped questions and retrieves the top 5 passages (top-k = 5 in the paper).
- It answers with its HippoRAG 2 `rag_qa_musique` one-shot CoT prompt and keeps the text after "Answer:".
- Its released code stops there: there is no answer scorer in the repo.

**Metric (paper §5):** LLM-Acc only, because "golden answers consist of lengthy descriptive statements". The judge is GPT-4o-mini with the LinearRAG correct/incorrect prompt, the same one as in our `common/score.py --llm-judge`.

## 2. The harness

| Step | File | What it does |
|---|---|---|
| Inputs | `evaluation_data/build_gmedical_inputs.py` | Writes the QuALITY input layout to `evaluation_data/gmedical/inputs/` (see below). `--smoke` writes `gmedical/smoke/inputs/`: 2 guidelines (Adrenal Tumors and Kidney Cancer, 4.3k words, 6 chunks) and their first 3 questions of each type (24 questions). |
| Index + retrieval | `<system>/run_gmedical.sh` | **Each system's QuALITY code path, unchanged**, pointed at the G-Medical inputs (`QUALITY_INPUTS=$GMEDICAL_INPUTS`), the same way the contracts harness reuses it. Same chunking, retrieval settings and context size as on QuALITY, HotpotQA and MuSiQue. |
| Answer | `common/answer.py --dataset gmedical` | One open-ended prompt for every system: LinearRAG's answer prompt (`prompts.build_linearrag_messages`, verbatim from LinearRAG `qa()`): the HippoRAG 2 / MemGraphRAG system instruction, zero-shot, no Wikipedia example or "Wikipedia Title:" labels. `pred_answer` is the text after "Answer:". DeepSeek, thinking off, temperature not sent, max_tokens 4096. |
| Score | `common/score_gmedical.py` | LLM-Acc with the LinearRAG judge prompt (verbatim). Reported overall, per level, and as a macro average over the four levels. Judge calls are cached in `judge_cache.sqlite`. |
| Launch | `launch_gmedical.sh [systems]` | All seven systems in parallel (`setsid nohup`), with logs in `run_logs/<system>_<tag>.log`. `stop_all.sh` stops them. |

**Inputs written to `gmedical/inputs/`:**
- **`questions.json`:** 2,062 rows `{id, question, answer, question_type, level, source, evidence}`, in MemGraphRAG's order.
- **`documents.json`:** ONE document, `medical.txt` verbatim (all 44 lines, duplicates and line breaks kept; byte-identical to GraphRAG-Bench's `medical.json` context). Used by HyGRAG, HyperGraphRAG and UNH.
- **`chunks.json`:** 991 passages, cut 256/32 with MemGraphRAG's own `split_text` and the voyage-4-nano tokenizer, each prefixed `<idx>:` (as MemGraphRAG and GraphRAG-Bench's `run_hipporag2.py` do). Byte-identical to the 991 passages MemGraphRAG indexes. Used by HippoRAG 2, and by LogicRAG as `chunks_logicrag.json` (neither has a chunker of its own; on QuALITY they got HyGRAG's 1200/100, the setup that benchmark mirrors).
- **`memgraphrag_corpus.txt`:** `medical.txt` itself, exactly what MemGraphRAG's `index.py` ingests.
- **`ms_graphrag_input/`:** one `.txt` file (the same document).
- `--per-line` writes the earlier alternative instead (41 distinct lines as separate documents); not used for the reported runs.

**Per-system specifics:**

| System | Indexes | Context passed to the reader |
|---|---|---|
| HippoRAG 2 | 991 passages (256/32, = MemGraphRAG's) | top-5 passages (about 1.3k tokens) |
| MemGraphRAG | `memgraphrag_corpus.txt`, its own 256/32 windows (991) | top-5 windows (`--qa-top-k 5`, about 1.3k tokens) |
| LogicRAG | 991 passages (256/32) | `info_summary`, 3 rounds, top-5 |
| HyperGraphRAG | the one document, its own 1200/100 chunks | hybrid-mode knowledge |
| MS GraphRAG | one file, its own chunker | local-search `build_context` |
| HyGRAG | the one document, its own 1200/100 chunks | 1 community + 4 chunks, 5 entities, 5 relations |
| UNH | the one document, UNH chunker 600/50, `clustered_fast` | PPR `t2000_f40_dmp06`, all three views (same as HotpotQA, MuSiQue and QuALITY) |

## 3. Decisions and differences from MemGraphRAG's paper

| | MemGraphRAG paper | Here | Why |
|---|---|---|---|
| Corpus | `medical.txt` verbatim (44 lines, 3 duplicated) | the same: `medical.txt` verbatim as one document | the benchmark defines no document boundaries (user decision, 2026-10-11) |
| Chunking | MemGraphRAG: 256/32 windows over the whole file | identical for MemGraphRAG; every other system keeps its own chunker; HippoRAG 2 / LogicRAG (no chunker) read HyGRAG's 1200/100 chunks | same rule as QuALITY |
| LLM / embeddings | GPT-4o-mini / NV-Embed-v2 | DeepSeek `deepseek-flash` (thinking off) / voyage-4-nano | same models as every other benchmark here |
| Answer prompt | MemGraphRAG's own `rag_qa_musique` (HippoRAG 2's, with a Wikipedia one-shot example) | LinearRAG's zero-shot version of it, shared by all systems | same instruction without the Wikipedia example; LinearRAG (same authors' group) uses it on G-Medical; one answer step for everyone (user decision 2026-10-11) |
| Temperature | 0 for every call | not sent (DeepSeek default 1.0) for reader and judge | same for every system; see the noise check below |
| Reader / judge max tokens | 500 (`max_new_tokens`) / 2000 | 4096 / 4096 | more room for every system: under 2048, 0.05–0.2% of G-Medical replies ran out mid-reasoning with no "Answer:" |
| Judge | GPT-4o-mini | DeepSeek (`$LLM_MODEL`), same prompt | recorded in each scores file |
| Report | one number (the paper's G-Medical LLM-Acc, comparable to our *overall*) | overall, per level, and macro | Level 1 is 53% of the questions, and GraphRAG-Bench finds plain RAG matches graph RAG there; the graph advantage shows up in Levels 2–4 |

**Known limitation:** the system instruction asks for "a concise, definitive response" after "Answer:", which may hold back long Level 3–4 answers. It applies equally to every system.

## 4. Smoke check (2026-10-11)

This checked the answer and score steps only, not any system's index:
- 24 smoke questions answered from plain dense top-5 chunks (voyage);
- **LLM-Acc 83.3%**, no empty answers, no unparseable judge replies.

## 5. How to run

```bash
python evaluation_data/build_gmedical_inputs.py && python evaluation_data/build_gmedical_inputs.py --smoke
GMEDICAL_TAG=gmedical_smoke GMEDICAL_INPUTS=evaluation_data/gmedical/smoke/inputs bash evaluation_scripts/launch_gmedical.sh
bash evaluation_scripts/launch_gmedical.sh            # full: 41 guidelines, 2,062 questions, all seven systems
```

## 6. Fairness checks on the MemGraphRAG run (2026-10-11)

- **Same as upstream:** corpus file (byte-identical), 256/32 windows, every `index.py` flag (= upstream `run_index.sh`), every retrieval flag (= upstream `run_retrieval_test.sh`: no LLM fact rerank, raw-similarity threshold 0.6), top-k 5 (paper §5.1).
- **Graph used:** 1,952 of 2,062 questions (94.7%) had facts above the threshold and went through MemGraphRAG's graph search; 110 fell back to dense retrieval (QuALITY: 77% fell back).
- **Answer step:** the shared reader's messages are byte-identical to `MemGraphRAG.qa()`'s (checked against its `sample_qa_prompts.json`). MemGraphRAG's own answers over the same top-5 windows score 55.29 overall / 53.74 macro (`scores_native_memgraphrag_gmedical.json`) vs 54.03 / 51.09 for the shared reader. 278 of 2,062 verdicts flip between the two runs: that is sampling noise (temperature not sent), not a difference in prompt or context.

**2026-10-11, after §6:** the answer prompt became LinearRAG's and the budgets 4096, so the UNH and MemGraphRAG scores in §6 and in memory (62.85 / 54.03) were produced with the old prompt; their answer and score steps must be re-run before comparing with the baselines.

## 7. Re-answered with LinearRAG's prompt, max_tokens 4096 (2026-10-11)

Same retrieval files as before; old answers and scores kept in `<system>/outputs/gmedical/oldprompt/`.

| System | Overall | Fact Retrieval | Complex Reasoning | Contextual Summarize | Creative Generation | Macro |
|---|---|---|---|---|---|---|
| UNH full | 65.28 | 67.85 | 68.57 | 61.94 | 43.98 | 60.58 |
| UNH hypergraph | 64.60 | 67.12 | 66.80 | 64.01 | 42.17 | 60.03 |
| UNH discourse | 64.26 | 65.76 | 66.60 | 64.71 | 46.39 | 60.86 |
| MemGraphRAG | 59.84 | 60.02 | 61.49 | 56.06 | 60.24 | 59.45 |

**UNH full + top-5 PPR passages** (`--tag hyb_p5`, PPR args `t2000_f40_dmp06` + `"ppr_passages": 5`, same graph, 2026-10-11):
71.29 overall | 72.04 / 72.30 / 72.32 / 61.45 per level | 69.53 macro | about 4.8k context tokens
(`unh/outputs/gmedical/scores_unh_hyb_p5_gmedical.json`).
**UNH full + top-10 PPR passages** (`--tag hyb_p10`): 74.49 overall | 73.77 / 76.62 / 77.51 / 67.47 per level | 73.84 macro | about 7.8k context tokens (`unh/outputs/gmedical/scores_unh_hyb_p10_gmedical.json`).

## 8. Baseline results (full G-Medical, 2026-10-11; LinearRAG prompt, max_tokens 4096)

| System | Overall | Fact Retrieval | Complex Reasoning | Contextual Summarize | Creative Generation | Macro | Context tokens |
|---|---|---|---|---|---|---|---|
| HyGRAG | 79.58 | 78.32 | 80.94 | 86.51 | 71.69 | 79.36 | ~5.5k |
| HyperGraphRAG | 77.64 | 77.96 | 80.55 | 77.85 | 66.27 | 75.66 | ~20.9k |
| MS GraphRAG | 75.02 | 73.68 | 77.41 | 75.43 | 75.90 | 75.61 | ~9.8k |
| HippoRAG 2 | 69.59 | 72.59 | 66.40 | 70.24 | 58.43 | 66.92 | ~1.3k |
| MemGraphRAG | 59.84 | 60.02 | 61.49 | 56.06 | 60.24 | 59.45 | ~1.3k |
| LogicRAG | 67.80 | 71.22 | 68.37 | 70.24 | 39.16 | 62.25 | ~0.2k (its summary) |
**UNH variants, 10 passages, `--retriever ppr_clean`** (2026-10-11): hub penalty (`hub_gamma` 1.0) + passages first 76.19 overall / 74.41 macro; passages first only 74.73 / 74.20; hub penalty only 74.20 / 72.70 (baseline 74.49 / 73.84). Files `unh/outputs/gmedical/scores_unh_p10_{hub_pf,pf,hub}_gmedical.json`.
**UNH, 5 PPR passages + the next chunk of each (`passage_next` 1, merged), hub penalty, passages first** (`p5next_hub_pf`): 73.81 overall / 71.63 macro, about 6.5k context tokens; significantly below 10 separate passages with the same options (76.19; 121 wins vs 170 losses, p = 0.004).
