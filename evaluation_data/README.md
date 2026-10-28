# Evaluation data

## HotpotQA and MuSiQue (`hotpotqa/`, `musique/`)

These are the standard 1,000-question multi-hop benchmarks used by HippoRAG 2, MemGraphRAG,
LinearRAG and LogicRAG. They are copied unchanged from the HippoRAG 2 data release. Run
`python evaluation_scripts/common/verify_data.py` to re-check them.

| File | Contents | Source |
|---|---|---|
| `<ds>.json` | 1,000 questions (HotpotQA: raw dev-distractor records; MuSiQue: answerable dev records incl. `answer_aliases`) | [osunlp/HippoRAG_2](https://huggingface.co/datasets/osunlp/HippoRAG_2) |
| `<ds>_corpus.json` | retrieval corpus, `[{"title","text"}]`: every paragraph attached to any of the 1,000 questions (gold + distractors), deduplicated | same |
| `<ds>.txt` | the corpus as `title\ntext` blocks joined by a blank line (MemGraphRAG's `index.py` input) | [XMUDeepLIT/MemGraphRAG](https://github.com/XMUDeepLIT/MemGraphRAG) `dataset/` |
| `hotpotqa_scrambled.txt` | the same passages and format as `hotpotqa.txt`, in shuffled order (seed 0). Gold passages no longer sit next to each other, so MemGraphRAG's cross-boundary 256-token windows cover all gold passages of a question for 0.1% of questions instead of 32%. Not used by any runner yet. | `python evaluation_data/make_scrambled_txt.py [--dataset musique]` |

| | HotpotQA | MuSiQue |
|---|---|---|
| questions | 1,000 (811 bridge / 189 comparison, all `hard`) | 1,000 (518 2-hop / 316 3-hop / 166 4-hop) |
| corpus passages | 9,811 | 11,656 (only 9,838 unique titles: never deduplicate by title) |

### Byte-level provenance (checked 2026-09-23)

- HuggingFace `osunlp/HippoRAG_2` = `OSU-NLP-Group/HippoRAG` `reproduce/dataset/` @ 1438aba, byte-identical for all four `.json` files.
- `XMUDeepLIT/MemGraphRAG` @ cd6fabd:
  - `hotpotqa.json`, `hotpotqa_corpus.json` and `musique_corpus.json` are byte-identical.
  - Its `musique.json` differs in bytes only. It wraps the same 1,000 records as `[{"source":"musique","questions":{"type1":[...]}}]` with indentation; the records are deep-equal and in the same order. MemGraphRAG's loader accepts the flat file used here.
- `chensyCN/LogicRAG` @ 72a298c:
  - The HotpotQA files are byte-identical.
  - The MuSiQue files are content-identical (same records, same order) but pretty-printed.
- The `.txt` files equal `"\n\n".join(title + "\n" + text) + "\n\n"` over `<ds>_corpus.json`.

The previous files in `evaluation_data/wiki_data/` were removed. They were our own stratified 1k samples, overlapping the standard sets by only 147 (HotpotQA) and 419 (MuSiQue) questions. The MuSiQue corpus was also deduplicated by title, which dropped a gold paragraph for 407/1,000 questions.

## PopQA and QuALITY (`popqa/`, `quality/`, `subsets/`)

Built by `python evaluation_data/build_popqa_quality.py` (see its docstring); background, examples and the
case for/against using them: `POPQA_QUALITY_REPORT.md`. Token counts for every dataset: `python evaluation_data/dataset_stats.py`.

| File | Contents | Source |
|---|---|---|
| `popqa/popqa.json`, `popqa/popqa_corpus.json` | 1,399 long-tail questions (answer + all aliases, relation, subject, the question's passage block) and 33,595 retrieved Wikipedia passages | DIGIMON copy in `HyGRAG/Data/Popqa` (Mallen et al. 2023; Self-RAG long-tail subset) |
| `quality/quality.json`, `quality/quality_corpus.json` | official **dev** split: 2,086 multiple-choice questions (options, gold letter, HARD flag, article id) over 115 articles | QuALITY v1.0.1 (`quality/official/`, nyu-mll/quality) |
| `subsets/<ds>_<N>q_seed<s>/` | seeded 100/200-question subsets for quick iteration (PopQA: stratified by relation + those questions' passages; QuALITY: whole articles) in evaluation_data, UNH JSONL and DIGIMON formats | the two above |
