# MemGraphRAG: upstream provenance

- Source: https://github.com/XMUDeepLIT/MemGraphRAG @ cd6fabd (`code/` copied whole, plus `LICENSE` and `requirements.txt`). MIT.
- Its `dataset/` is byte-identical to `evaluation_data/` except for the `musique.json` wrapper (see `evaluation_data/README.md`). `evaluation_data/<ds>/<ds>.txt` is MemGraphRAG's own file.
- `run_index.sh` and `run_retrieval_test.sh` are the unchanged upstream templates, kept for reference. Use `run.sh`.

## Local changes (tagged `[unh_v1]`)

1. `src/embedding_model/VLLM.py` (new) is ported from HippoRAG 2's `VLLMEmbeddingModel`: an OpenAI-compatible `/v1/embeddings` client, selected with the model name prefix `VLLM/`. Upstream only loads local HF checkpoints (NV-Embed-v2 / GritLM / bge-large / contriever).
2. `src/embedding_model/__init__.py` routes `VLLM/…` names to it.
3. `src/utils/config_utils.py` adds `embedding_base_url` and `embedding_request_timeout`.
4. `index.py` and `retrieval_dataset_test.py` get a new `--embedding-base-url` flag. `index.py`'s tokenizer fallback strips the `VLLM/` prefix.
5. `src/llm/openai_gpt.py`: sends `extra_body={"thinking": {"type": "disabled"}}` on every call unless `LLM_DISABLE_THINKING=0`.
6. `run.sh` indexes `<ds>_scrambled.txt` by default (see Chunking below; `CORPUS_TXT` overrides it). Reported answers come from `common/answer.py` over `qa_results.json`'s top-5 `retrieved_docs`, which are the docs `MemGraphRAG.qa()` reads with `--qa-top-k 5`. MemGraphRAG's own answers (max_new_tokens 500) are scored separately as `scores_native_*`.
7. `src/llm/openai_gpt.py` `infer()`: `temperature` is taken from `LLM_TEMPERATURE`. When that is unset (the default), it is not sent at all. Upstream sends 0, including the explicit `temperature=0.0` in the schema and conflict calls.
8. `src/MemGraphRAG.py`: an unparseable schema-extraction reply means "no schema for this fact", the same as upstream's handling of an empty reply (1/145,694 HotpotQA and 2/147,697 MuSiQue replies in our run). An unparseable conflict-detection reply means "no conflicts found for this fact". Upstream raised on either and aborted the whole index.
9. `src/MemGraphRAG.py` conflict resolution: an unparseable reply leaves that component unresolved, so its facts stay unchanged. Upstream already does this for components over `conflict_resolution_max_component_size` and for content-filter errors, but raised on any other error and aborted the index. Cause in our run: a reply that hit upstream's 1,400-token cap mid-JSON (1 of the first 46 HotpotQA components).
10. `src/embedding_model/VLLM.py` embeds an empty string as `" "`. vLLM returns HTTP 400 for `""`, and OpenIE yields an empty entity in 311 MuSiQue and 555 HotpotQA facts. This is the same fix as HippoRAG 2's `VLLM.py`.
11. `src/information_extraction/openie_openai.py` `ner()`: keeps only string entities. The reply is parsed with `eval()`, so a literal `...` in the entity list becomes `Ellipsis`; `triple_extraction` then crashes in `json.dumps` outside any `try`, which aborted the whole G-Medical index (2026-10-11). Valid replies are unchanged.

## Upstream behaviour kept on purpose

- **Chunking:** `index.py` concatenates the whole corpus `.txt` and cuts 256-token windows with 32 overlap across passage boundaries. `<ds>.txt` is in question order, so one window touches every gold passage for 32% of HotpotQA questions (8.7% of MuSiQue). The windowing is kept, but `run.sh` feeds it `<ds>_scrambled.txt` (same passages, shuffled with seed 0), which brings that to 0.1% (0% fully inside one window).
- **No metrics:** `retrieval_dataset_test.py` computes none (it calls `rag_qa(gold_answers=None)`). `run.sh` scores its `qa_results.json` with `common/score.py`.
- **top-k:** the paper uses top-k = 5 while the CLI default `--qa-top-k` is 10. `run.sh` passes 5.
- **Indexing cost:** roughly one LLM call per passage chunk (OpenIE), plus one per unique triple (schema extraction), plus conflict checks.

## QuALITY (`run_quality.sh`, 2026-10-01; see ../QUALITY_IMPLEMENTATION.md)
- No code change. `index.py --corpus quality/inputs/memgraphrag_corpus.txt` is upstream's input format: one `.txt`, cut into 256/32-token windows by `index.py`.
- The file is the 115 dev articles (`title\ntext`) joined by a blank line. At most one window per article boundary mixes two articles; no QuALITY question spans articles.
- Index and retrieval flags are identical to `run.sh`, including `--qa-top-k 5`.
- MemGraphRAG's own QA still runs, as upstream does; it is kept for reference only. The reported answer comes from `common/answer.py --dataset quality` over the same top-5 windows.

## Insurance contracts (`run_contracts.sh`, 2026-10-09; see ../CONTRACTS_EVAL_PLAN.md)
- One index per contract: `index.py --corpus <contract>.txt` (its own 256/32-token windows), `retrieval_dataset_test.py --questions <contract queries>`; all flags as `run_quality.sh`. No code change.
