# HippoRAG 2: upstream provenance

- Source: https://github.com/OSU-NLP-Group/HippoRAG @ 1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff (MIT, see LICENSE)
- Copied: `main.py`, `src/hipporag/` (entire package), `requirements.txt`, `LICENSE`. Not copied: `reproduce/dataset/`, which is byte-identical to `evaluation_data/`.

## Local changes (all tagged `[unh_v1]`)

1. The dataset directory is `<repo>/evaluation_data/<dataset>/` (`--data_root` to override) instead of `reproduce/dataset/`.
2. New `--embedding_base_url` flag. `BaseConfig.embedding_base_url` already exists upstream; the CLI just didn't expose it. The `vllm` provider POSTs to this full URL (`…/v1/embeddings`).
3. After `rag_qa`, the script writes `outputs/<ds>/predictions.json` (per-question answer, raw response, top-5 docs, gold) and `upstream_metrics.json` (HippoRAG's own Recall@k / EM / F1).
4. `src/hipporag/llm/openai_gpt.py`: sends `extra_body={"thinking": {"type": "disabled"}}` on every call unless `LLM_DISABLE_THINKING=0`. DeepSeek V4/V4.1 think by default.
5. Reported answers come from `common/answer.py`: the same prompt, over `predictions.json`'s `retrieved_docs` (the top 5 that `HippoRAG.qa()` reads). HippoRAG's own answers are scored separately as `scores_native_*`.
6. `main.py`: `temperature` is taken from `LLM_TEMPERATURE`. When that is unset (the default), it is not sent at all, so DeepSeek's default applies. Upstream sends 0.
7. OpenIE failure policy (`information_extraction/openie_openai.py`, `llm/openai_gpt.py`). Upstream aborts the whole index if the entity or triple reply of any chunk cannot be parsed. With DeepSeek this happened for 4/9,811 HotpotQA and 5/11,656 MuSiQue chunks (replies cut off at the 512-token NER cap, a refusal on a very short passage, or the wrong JSON shape). Now each failed chunk is retried once with a fresh call (`_unh_refresh=True` skips the cache lookup and overwrites the cached reply). A chunk that still fails keeps no entities or triples; it stays in the index as a passage and is listed in a warning.
8. `main.py`: `openie_ner_max_tokens=2048`. The paper-era code (Feb 2025) ran NER with the general `max_new_tokens=2048`. The dedicated 512 default was added upstream in Aug 2026 (`2f52a86`), and it cut off 6 of about 21.5k entity replies in our first run. The newer entity-name normalization (`text_processing`, which keeps accented letters and collapses spaces) is kept on purpose; the paper-era version stripped non-ASCII letters.
9. `information_extraction/openie_openai.py`: non-text entity values (e.g. a bare `1963`) are kept as text, and null or object items are dropped. Upstream's newer consistency check aborted on them. `filter_invalid_triples` already does the same for triples.
10. `embedding_model/VLLM.py`: empty strings are embedded as a single space, because vLLM rejects `""`. About 53 extracted items normalize to `""` (e.g. `£`, `€`).
11. `run.sh`: `HIPPORAG_FORCE_INDEX=true` passes `--force_index_from_scratch true`. That rebuilds the graph while reusing saved embeddings and OpenIE; it is needed to resume an index that stopped before its graph was saved.

Paper settings are unchanged: `retrieval_top_k=200`, `linking_top_k=5`, `qa_top_k=5`, temperature 0, `max_new_tokens=2048`, reader prompt `rag_qa_musique`.

## QuALITY (`run_quality.sh`, 2026-10-01; see ../QUALITY_IMPLEMENTATION.md)
- `main.py --dataset quality` reads `evaluation_data/quality/inputs/{chunks,questions}.json`.
  - HippoRAG 2 has no chunker: one document = one passage.
  - It therefore indexes the 652 chunks that HyGRAG's chunker makes from the 115 dev articles (`title\ntext`, 1200/100 tokens). Only an article's first chunk holds the title.
  - Questions are asked with their four options.
- Only `rag.retrieve` is called, which is the same retrieval `rag_qa` runs first. HippoRAG's own short-answer QA call is skipped, because `common/answer.py` answers with the shared multiple-choice prompt over `retrieved_docs[:5]`.
- No HippoRAG code is changed. `--quality_inputs` and the `quality` branch are in `main.py` only.

## Insurance contracts (`run_contracts.sh`, 2026-10-09; see ../CONTRACTS_EVAL_PLAN.md)
- One index per contract. Each contract inputs dir (`evaluation_data/contracts/<set>/<contract>/`, built by `build_contract_inputs.py`) has the QuALITY inputs layout, so `main.py --dataset quality --quality_inputs <dir>` reads it unchanged: the 1200/100-token chunks cut by HyGRAG's chunker, and the queries (QA questions, then facts verbatim). Retrieval only (top-5 passages). No code change.
