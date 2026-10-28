# LogicRAG: upstream provenance

- Source: https://github.com/chensyCN/LogicRAG @ 72a298c95ffd75890c7f9c28fd1c094201f669a7 (MIT, see LICENSE)
- Copied: `run.py`, `setup.py`, `requirements.txt`, `README.md`, `LICENSE`, `config/`, `src/`. Not copied: `dataset/`, which is identical to `evaluation_data/` (see its README).
- Evaluation is upstream's own `src/evaluation/evaluation.py` (`RAGEvaluator`): string accuracy/precision/recall, LLM-judged accuracy, answer coverage, and token cost.

## Local changes (tagged `[unh_v1]`)

1. `config/config.py`: every hard-coded value is now read from the environment, with the upstream default kept:
   - `OPENAI_BASE_URL` is new.
   - `LOGICRAG_LLM_MODEL` (default `gpt-4o-mini`).
   - `LOGICRAG_EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`).
   - `LOGICRAG_CALLS_PER_MINUTE` (default 20).
   - `LOGICRAG_CACHE_DIR` / `LOGICRAG_RESULT_DIR` (default `cache/` and `result/`, now anchored to this folder rather than the cwd).
2. `src/utils/utils.py`: `OpenAI(api_key=…, base_url=OPENAI_BASE_URL)`, so DeepSeek works.
3. `src/models/base_rag.py`:
   - `SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)`, needed for voyage-4-nano.
   - The corpus-embedding cache file name includes the model name. Upstream keyed it on corpus size only, so switching models silently reused stale vectors.
4. `src/main.py`: `--dataset` / `--corpus` default to `evaluation_data/hotpotqa/…`.
5. `src/utils/utils.py`: sends `extra_body={"thinking": {"type": "disabled"}}` unless `LLM_DISABLE_THINKING=0`.
6. `src/models/logic_rag.py` / `src/evaluation/evaluation.py`: save `info_summary`, the only context `generate_answer()` reads, in each result. The reported answers come from `common/answer.py` over that summary. LogicRAG's own answer (its concise prompt, max_tokens 250) is scored separately as `scores_native_*`. Its internal calls keep max_tokens 250.
7. `src/utils/utils.py`: `temperature` is taken from `LLM_TEMPERATURE`. When that is unset (the default), it is not sent at all. Upstream sends 0.0.
8. JSON replies (2026-10-07). Added after the QuALITY run crashed at question 568 of 2,086.
   - **What went wrong:** DeepSeek answered a `dependency_aware_rag` prompt in markdown prose. The reply was cut off at max_tokens 250, so `fix_json_response` returned `None`. `answer_question` then failed on `analysis["can_answer"]`.
   - **Fix:** `get_response_with_retry(..., json_mode=True)` sends `response_format={"type": "json_object"}`, OpenAI-compatible JSON mode. DeepSeek requires the word "json" in the prompt, and both prompts contain it. `LOGICRAG_JSON_MODE=0` turns it off.
   - **Where it applies:** only the two JSON-object prompts, `warm_up_analysis` and `dependency_aware_rag`. `_sort_dependencies` asks for tuples, so it is left alone.
   - **Unparsable replies:** `dependency_aware_rag` retries once, then falls back to `{"can_answer": True, ...}`. That is upstream's own result when this function raises, and what `warm_up_analysis` does on a parse failure.
   - **Upstream** sends no `response_format` and returns `None` here.
   - **Consistency:** the first 565 QuALITY questions (checkpoint of 2026-10-06) ran without JSON mode; the rest of the run uses it. Unparsable replies had been 4 in 569 questions.

Upstream behaviour worth knowing:
- The LLM judge is the same model as the reader (`DEFAULT_MODEL`), with a single `gold_answer` (MuSiQue aliases are not used).
- Questions run sequentially.

## QuALITY (`run_quality.sh`, 2026-10-01; see ../QUALITY_IMPLEMENTATION.md)
- No code change; `run.py` is called with QuALITY files:
  - `--dataset quality/inputs/questions.json`: `question` holds the question with its four options, and `answer` is the option text, used only by LogicRAG's own judge.
  - `--corpus quality/inputs/chunks_logicrag.json`: LogicRAG has no chunker, so each dev article is cut into 1200/100-token chunks by HyGRAG's chunker. The title line is not in the text; instead `title` is the article title on chunk 0 and "" on later chunks ("option B"). So every unit reads "Title: {title}. Content: {text}", and the title appears once per article, as for the other systems.
- Same settings as `run.sh`: 3 rounds, top-5.
- `LOGICRAG_RESULT_DIR` and `LOGICRAG_CACHE_DIR` = `result/<tag>` and `cache/<tag>`. The embedding cache is keyed only on corpus size, so QuALITY gets its own cache directory.
- The reported answer comes from `common/answer.py --dataset quality` over `info_summary`, the same context LogicRAG's own `generate_answer` reads.

## Insurance contracts (`run_contracts.sh`, 2026-10-09; see ../CONTRACTS_EVAL_PLAN.md)
- One corpus per contract (`chunks_logicrag.json`, the 1200/100-token chunks cut by HyGRAG's chunker, title ""). Same settings as `run.sh`. LogicRAG answers one query at a time, so each contract's queries are split into shards that run as separate processes (`contracts_shards.py`, harness only) and are merged back in query order. The context passed on is `info_summary`, for QA and for facts. No code change.
- `[unh_v1]` `src/models/base_rag.py retrieve`: `topk(min(top_k, len(corpus)))`. Upstream raises "selected index k out of range" when the corpus has fewer than `top_k` units (each short contract is 2-3 chunks), and the exception made every retrieval return nothing. Unchanged for any corpus of at least `top_k` units (all earlier runs).
