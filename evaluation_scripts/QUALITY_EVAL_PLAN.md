# QuALITY on every baseline: plan

Status 2026-10-01: decided and implemented; see QUALITY_IMPLEMENTATION.md. The decisions taken: D1 (a) dev only; D2 shared prompt, letter only; D3 title once per article (LogicRAG: title field on the first chunk only); D4 no reference rows, VanillaRAG later.
Data already built: `evaluation_data/quality/` (official dev split) and `evaluation_data/subsets/quality_{100,200}q_seed0/`.

## 1. How HyGRAG / DIGIMON evaluates QuALITY (verified in the code)

| Step | What HyGRAG does | Where |
|---|---|---|
| Data | `Data/quality/Corpus.json`: **265 whole articles** (QuALITY train 150 + dev 115), one row each, `{title, context}`. `Question.json`: 4,609 rows; `question` already holds the options (`"...?\nA: 7 years\nB: 10 hours\n..."`), gold in `answer_idx` (a letter). No article id, no HARD flag. | `HyGRAG/Data/quality/` |
| Index | **One shared graph over all 265 articles.** Each article is chunked on its own (tiktoken cl100k, 1200 tokens / 100 overlap; chunks never cross articles) → 1,518 chunks. The `title` field is dropped from the chunk text (the article text itself starts with its title in 101/115 dev articles). | `DocChunk.build_chunks` |
| Query | The full `question` string **including the four options** is used for retrieval and for answering. Every question searches the whole pool; nothing tells the system which article it is about. | `main.py wrapper_query` |
| Answer | HyGRAG's own prompt: "First analyze the evidence and reasoning process, then provide your answer with source attribution... Give the best full answer amongst the option... If the context does not contain sufficient information... state 'Insufficient information'." The output is a free-text paragraph. | `HKGraphTreeQuery.generation_qa` |
| Letter extraction | A second LLM call (the same LLM as the system) with `CLOSE_EXTRACT_OPTION_PORMPT`: three few-shot examples, return JSON `{"predict": "B"}`, or `"-1"` if the output picks no option. | `Core/Utils/Evaluation.py close_eval` |
| Metric | **Accuracy** = extracted letter == `answer_idx` (after SQuAD normalisation), × 100. Nothing else is reported for QuALITY. | same |
| Fallback | `run_evaluation.py` uses a regex (`answer is X`, `option X`, last standalone capital letter) if no LLM is configured. | `run_evaluation.py` |
| How many questions | `main.py` answers only the **first 500** questions (all from dev, 39 articles). The paper's table lists 4,609 questions but doesn't say which script produced the numbers. | `main.py:35` |

**Is there an LLM judge?** Not one that judges correctness. Because the gold answer is a letter, the LLM only pulls the chosen letter out of a long answer; the comparison itself is exact string match. Our HotpotQA LLM-Acc judge isn't needed for QuALITY.

**Score scale** (HyGRAG paper, Llama-3.1-8B, accuracy %): random guessing 25, zero-shot 33.5, VanillaRAG 56.4, HippoRAG 2 46.0, MS GraphRAG local 50.0, HyGRAG 59.5, EraRAG 60.8 (best). Humans: 93.5 (QuALITY paper).

**Quirks in HyGRAG's scorer** (I'd fix these in ours and report how often they fire):
1. The extraction prompt's FORMAT section asks for the key `"model_output"`, but the examples and the code use `"predict"`. A reply that follows the FORMAT section raises a KeyError, gets scored `-1`, and counts as wrong.
2. Normalisation removes the article "a", so gold `A` normalises to the empty string. Gold-`A` questions (491 of 2,086 in dev) are then also marked correct when the extractor returns an empty string.
3. The answer prompt invites "Insufficient information", which is extracted as `-1` and scored wrong. Standard QuALITY scoring is forced choice.

## 2. Answers to the specific questions

- **Does HyGRAG take each passage separately? Are they long?** It takes each **article** as one document (265 rows, about 5,700 tokens each; range 2,250–8,400). It chunks each article separately into about 6 chunks of 1,200 tokens, then puts all chunks from all articles into **one graph**.
- **Do some questions need several articles?** No. Every QuALITY question is about exactly one article. Pooling is HyGRAG's choice, and it adds a "find the right story" step on top. The catch is that about 19% of questions say "the story", "the author" or "the passage" without naming it, so they are ambiguous in a pooled corpus. This is a known property of HyGRAG's protocol; we would keep it and note it.

## 3. Can each baseline take whole articles?

| System | Has its own chunker? | What we would feed it for QuALITY | Change needed (all in our harness) |
|---|---|---|---|
| **HyGRAG** | Yes, per document, 1200/100 cl100k | 115 whole articles, one row each, `context` = article text (same as DIGIMON's copy) | `prepare_data.py` / `main.py`: accept `quality`; query = question with options |
| **HyperGraphRAG** | Yes, `rag.insert(docs)` chunks **each doc** 1200/100 (tiktoken gpt-4o) | 115 whole articles, one doc each | `main.py` loader for `quality`; query = question with options |
| **MS GraphRAG** | Yes, `input/` = one `.txt` per document, chunked per document 1200/100 (o200k) | **115 `.txt` files, one per article.** This is MS GraphRAG's native input; no scrambling needed | `run.sh`: copy the article files into `input/` instead of the scrambled corpus |
| **MemGraphRAG** | Only for one big `.txt`: `index.py` reads the whole file as one string and cuts 256/32-token windows that **cross document boundaries** | One `.txt` with the 115 articles joined by blank lines (stock input). Of about 2,990 windows, at most 114 straddle two articles (about 4%), harmless because questions never span articles | `run.sh`: point `CORPUS_TXT` at `quality.txt` (built by our data script). No code change |
| **HippoRAG 2** | **No.** One document = one passage node; OpenIE runs over the whole document | **Pre-chunked:** 642 chunks made with DIGIMON/HyGRAG's chunker (1200/100 cl100k, per article). Feeding a 6k-token article as one passage would mean one OpenIE call per article and whole-article retrieval units. In the HyGRAG paper, HippoRAG 2 also ran on DIGIMON's chunks | `main.py` loader: docs = chunk texts |
| **LogicRAG** | **No.** One corpus entry = one retrieval unit, embedded whole | The same 642 chunks as HippoRAG 2 | `run.py` corpus path = the chunk file; query = question with options |

So: HyGRAG, HyperGraphRAG and MS GraphRAG take whole articles and chunk them per article (all at 1200/100). MemGraphRAG takes one concatenated `.txt`, as it does upstream. HippoRAG 2 and LogicRAG have no chunker, so they get the same 1200/100 chunks HyGRAG builds internally. None of this touches a system's internals.

## 4. Proposed common protocol ("HyGRAG protocol, made strict")

Kept from HyGRAG:
- one pooled index per system over all articles in the corpus;
- per-article 1200/100 chunking (each system's own chunker, or DIGIMON's for the two systems without one);
- the query string = question + options;
- accuracy against the gold letter, using HyGRAG's extraction prompt verbatim.

Same as our HotpotQA/MuSiQue benchmark:
- each system keeps its own retrieval and context (top-5 or its own budget, as listed in `common/answer.py`);
- **one shared answer step** for every system: same model (deepseek-flash, thinking off), same prompt, same settings, cached.

Changed from HyGRAG on purpose:
1. **Shared MCQ answer prompt** instead of HyGRAG's verbose one. HyGRAG's prompt names its own context sections ("Entities, Key Relationships, Community Summaries"), so it can't be shared. The new prompt: context + question with options, "think briefly, then end with `Answer: <A|B|C|D>`". Forced choice, no abstention (standard QuALITY scoring).
2. **Scoring:** (a) parse `Answer: X` with a regex; (b) also run HyGRAG's LLM extractor verbatim; (c) compare letters strictly (A–D, no normalisation). Report regex accuracy as the headline, the extractor accuracy alongside, and the count of rows where they disagree. Because the prompt forces a final letter, the two should agree on nearly every row; any disagreement points to a parsing problem.
3. **All questions**, not the first 500.
4. **Extra columns** from the official data: accuracy on **HARD** and **EASY** questions, mean context tokens, and the number of empty or unparseable answers.

Reference rows (cheap, no graph):
- zero-shot (question + options only);
- **VanillaRAG** (top-5 of the 642 chunks by voyage-4-nano cosine), the baseline that beat most graph methods in the HyGRAG paper;
- full-article oracle (the gold article given directly, about 5.7k tokens), the upper bound.

## 5. Decisions for you

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | What goes into the pool | **(a)** dev only: 115 articles, 2,086 questions, 0.66M tokens<br>**(b)** HyGRAG's exact pool: 265 articles (train 150 + dev 115), 1.52M tokens. HyGRAG's file has 4,609 questions (train 2,523 + dev 2,086), but its `main.py` answers only the first 500 (all dev, from 39 articles). Under (b) we would answer the 2,086 dev questions, and the 150 train articles would only be extra haystack<br>**(c)** one graph per article (official reading-comprehension setup, like the contracts) | **(a)**. Official split with the HARD flag, same protocol shape as HyGRAG, under half the cost. Choose (b) only if matching the paper's distractor pool matters. (c) means 115 separate builds per system, which is awkward for MS GraphRAG and MemGraphRAG |
| D2 | Answer prompt | shared MCQ prompt / HyGRAG's own | shared (§4.1) |
| D3 | Title on the pre-chunked units (HippoRAG 2, LogicRAG) | chunk text only / prefix the article title | **chunk text only**, so they see exactly what HyGRAG's chunks contain. LogicRAG's corpus format needs a `title`; we would pass `""` |
| D4 | Reference rows | zero-shot, VanillaRAG, full article | all three; together they cost less than one graph build |

## 6. Cost (projected from our HotpotQA/MuSiQue construction ratios; order of magnitude)

| Pool | HippoRAG 2 | HyGRAG | MS GraphRAG | MemGraphRAG | HyperGraphRAG |
|---|---|---|---|---|---|
| (a) dev, 0.66M tokens | 5M in / 1.8M out | 7M / 2.2M | 24M / 11M | 77M / 8M | 87M / 9M |
| (b) 265 articles, 1.52M tokens | 11M / 4.3M | 17M / 5.0M | 56M / 25M | 180M / 19M | 202M / 21M |

The answer step adds 2,086 calls per system (about 3–10M input tokens depending on context size). LogicRAG has no construction cost; its cost is per question.

## 7. Implementation steps (after your go-ahead; no system internals touched)

1. `evaluation_data/build_popqa_quality.py`: also write
   - `quality/quality_chunks.json`: the 642 chunks, using DIGIMON's chunker settings;
   - `quality/quality.txt`: the articles joined, for MemGraphRAG;
   - `quality/articles/<article_id>.txt`: one file per article, for MS GraphRAG.

   The same three files go into each QuALITY subset.
2. `common/prompts.py`: an MCQ prompt and a letter parser. `common/answer.py`: `--dataset quality` uses them.
3. `common/score.py`: `--dataset quality` reports regex accuracy, HyGRAG-extractor accuracy, disagreements, and HARD/EASY accuracy.
4. Each runner accepts `quality` and loads the input from §3: `hygrag/{prepare_data,main}.py`, `hypergraphrag/main.py`, `ms_graphrag/run.sh`, `memgraphrag/run.sh` (corpus path only), `hipporag2/main.py`, `logicrag/run.py`. Each runner's `UPSTREAM.md` gets a line.
5. `common/quality_baselines.py`: the zero-shot, VanillaRAG and full-article rows.
6. Smoke test every system on `subsets/quality_100q_seed0` (6 articles, 113 questions, 32k tokens; a few cents each). Check build guards, non-empty contexts and parse rates.
7. Full dev runs only when you say go.
