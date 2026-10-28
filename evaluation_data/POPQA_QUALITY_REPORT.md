# PopQA and QuALITY: what they test, how they differ from HotpotQA / MuSiQue, and whether to use them

Prepared 2026-10-01 from the two benchmark papers (`popqa benchmark.pdf`, `quality benchmark.pdf`), the HyGRAG
WWW'26 paper, DIGIMON's data copies (`HyGRAG/Data/{Popqa,quality}`), the official QuALITY v1.0.1 release, our own
benchmark runs, and earlier UNH (nkg) runs.

Note on sources: the PopQA and QuALITY copies are in **HyGRAG's** `Data/` folder (HyGRAG is a fork of DIGIMON).
HyperGraphRAG's folder (`HyperGraphRAG-contexts&datasets/`) has its own domain sets (agriculture, cs, hypertension,
legal, mix), not these two. The HyGRAG paper is the one that reports results on PopQA and QuALITY.

---

## 1. Summary

| | **PopQA** | **QuALITY** |
|---|---|---|
| In one sentence | "Do you know a simple fact about an obscure thing?", answered from a pile of Wikipedia passages | "Did you understand this whole 5,000-word story?", as a multiple-choice question |
| What it really tests | finding the one right passage about a rare entity among many look-alikes (single-hop lookup) | reading comprehension across a long document: why, how, feelings, themes |
| Hops | 1 | usually several parts of one document, but not "hops" between documents |
| Documents | 33,595 short Wikipedia passages (~120 tokens each) | 115 long articles in dev (~5,700 tokens each) |
| Answer | short entity, many aliases (e.g. "India / Bharat / Hindustan") | pick A/B/C/D |
| Hard because | the entity is rare (the LLM doesn't know it) and its name is shared by other things | answers need the gist of the whole text; options are designed to be close |
| Closest to our data | the HotpotQA/MuSiQue *format* (many passages), but single-hop | **the insurance contracts** (one long document per graph) |

My read for UNH:
- **PopQA is a good fit** if UNH handles entity disambiguation well. It is single-hop and the facts are literally
  knowledge-graph triples.
- **QuALITY is closer to the contracts in structure**, but its questions are mostly "why / describe / interpret",
  which a fact graph alone may not capture.
- Details in §5.

---

## 2. PopQA: what it tests

**Plain words:** PopQA asks very simple factual questions ("What is X's occupation?", "Who directed Y?") about
entities, most of which are **obscure**. The point is to find what language models *don't* already know, so they
must rely on retrieved text.

**How it was made** (Mallen et al., ACL 2023):
- The authors took facts `(subject, relation, object)` from **Wikidata** and turned each into a question with a
  template, one template per relation type. There are 16 relation types, e.g. `(Kathy Saltzman, occupation,
  politician)` → "What is Kathy Saltzman's occupation?".
- Every valid object in Wikidata counts as a correct answer, hence the alias lists.
- Each question has a **popularity score**: the subject's monthly Wikipedia page views.

**Main finding of the paper:**
- LLMs know facts about popular entities but fail on rare ones. Even GPT-3 davinci-003 gets only ~19% on the 4,000
  least popular questions, and scaling the model barely helps.
- Retrieval helps a lot on rare entities, but can *hurt* on popular ones (misleading passages).
- Metric: a prediction is correct if any gold answer string appears in it ("substring accuracy").

**The version DIGIMON / HyGRAG use:** the standard **long-tail subset of 1,399 questions**: entities with fewer
than 100 monthly page views, the same subset Self-RAG and others use. In DIGIMON's copy:
- The corpus is ~24 retrieved Wikipedia passages per question, pooled into one corpus: 33,595 passages, 4.1M tokens.
- The corpus is stored in question order, in blocks.
- The gold answer string appears somewhere in the corpus for ~98% of questions, and in the question's own block for
  ~79–87%.

**Examples** (from the data):

| Relation | Question | Gold answers |
|---|---|---|
| country | In what country is Anaikudam? | India, Bharat, Hindustan, … |
| author | Who is the author of *O dia das calças roladas*? | Germano Almeida |
| sport | What sport does Marc Santo-Roman play? | chess, … |
| place of birth | In what city was Karl Bartholomaeus Heller born? | Moravia |
| director | Who was the director of *Messiah*? | William Klein |
| author | Who is the author of *Shame*? | Karin Alvtegen |

The last two show **why it is hard for retrieval**. There are many works called *Messiah* and *Shame*, and the
corpus holds passages about several of them. The HyGRAG paper's case study is the same problem: "In what city was
Paul Walker born?" with several people named Paul Walker.

**Question mix:**

| Relation | Questions |
|---|---|
| country | 274 |
| author | 198 |
| sport | 196 |
| place of birth | 173 |
| occupation | 121 |
| genre | 113 |
| director | 95 |
| composer | 48 |
| screenwriter | 42 |
| religion | 40 |
| producer | 37 |
| capital | 34 |
| father | 27 |
| mother | 1 |

## 3. QuALITY: what it tests

**Plain words:** each question is about one long article: mostly 1950s–70s science-fiction short stories from
Project Gutenberg, plus Slate magazine essays and some other non-fiction. Answering correctly requires having
*understood* the article, not just finding a sentence. The format is always 4-option multiple choice.

**How it was made** (Pang et al., NAACL 2022):
- Writers read the **whole** article (2k–8k tokens, ~5k on average) and wrote questions meant to need several parts
  of the text.
- Every question was then checked twice:
  - by people who read the full text (to make sure it is answerable and unambiguous);
  - by "speed validators" who had only **45 seconds** with the text (to check that skimming or keyword search isn't
    enough).
- Questions the speed validators mostly got wrong are marked **HARD**: about half of them (1,065 of 2,086 dev
  questions).
- 81% of HARD questions were rated as needing "at least a long paragraph or two" of context.

**Results in the paper:**
- Humans 93.5%, best model at the time 55.4%. Simply picking the option with the most word overlap gets 26.6%,
  about chance.
- Question words: what 42%, **why 25%**, how 12%.
- Reasoning needed:

  | Reasoning | Share |
  |---|---|
  | description | 33% |
  | why / reason | 31% |
  | symbolism / interpretation | 28% |
  | entity / location / object facts | under 4% each |

**Examples** (from the dev set):

| Article | Difficulty | Question (options shortened) | Answer |
|---|---|---|---|
| *Captain Chaos* | easy | What is the most likely meaning of the slang O.Q.? A: cool B: no worries C: my bad **D: O.K.** | D |
| *Captain Chaos* | easy | Why does the Skipper stop abruptly after he says "when you're running a blockade"? (4 explanations) | D |
| *The Girl in His Mind* | HARD | Why does Deirdre get so upset when Blake suggests she go to prom with the young man? (4 explanations about guilt, love, …) | B |
| *The Girl in His Mind* | HARD | Why does shame flame in Blake's cheeks when Deirdre goes to prepare Eldoria's dias? | C |
| (paper) | HARD | Which is NOT a reason why the narrator is concerned with the antichrist? | D |

**Official splits:**
- train: 150 articles / 2,523 questions;
- **dev: 115 articles / 2,086 questions** (1,065 HARD);
- test: 116 articles, answers hidden.

DIGIMON/HyGRAG pool **train + dev** (265 articles, 4,609 questions) into one corpus, drop the article ID and the
HARD flag, and their `main.py` scores only the first 500 questions. Their article texts are identical to the
official ones. About 19% of questions say "the story", "the author" or "the passage"; with 265 articles pooled and
no title in the question, those are ambiguous. **The official protocol is per article.**

## 4. How they differ from HotpotQA and MuSiQue

| | HotpotQA / MuSiQue (our runs) | PopQA | QuALITY |
|---|---|---|---|
| Task | connect facts from 2–4 passages ("Who directed the film starring X?") | look up 1 fact | understand 1 long document |
| Retrieval challenge | find *all* the pieces, then chain them | find the *right* passage among same-named entities | find the parts of a 5k-token story that matter, or keep its overall gist |
| Corpus | ~10k passages, ~1.3M tokens | 33.6k passages, 4.1M tokens (3× larger) | 115 articles, 0.66M tokens (dev); long documents |
| Questions | ~21 tokens | ~9 tokens (template) | ~59 tokens with options |
| Answer format | free short answer (EM/F1) | free short answer, many aliases (substring accuracy) | choice A–D (accuracy) |
| Does the LLM know it already? | partly (popular Wikipedia) | mostly **no** (long tail: zero-shot ~15% in HyGRAG's paper) | no (fiction) |

HyGRAG paper results (Llama-3.1-8B, pooled corpora, their substring-accuracy metric):

| Method | PopQA | QuALITY |
|---|---|---|
| Zero-shot (no retrieval) | 15.4 | 33.5 |
| VanillaRAG | 67.0 | 56.4 |
| HippoRAG 2 | 68.1 | **46.0** |
| L-GraphRAG (MS GraphRAG local) | 54.8 | 50.0 |
| RAPTOR | 66.3 | 58.1 |
| EraRAG (summary tree) | 61.4 | **60.8** (best) |
| HyGRAG | **72.3** (best) | 59.5 |

The pattern matters for UNH:
- **On PopQA, relation/entity-aware methods help** (HyGRAG +5 over vanilla RAG).
- **On QuALITY, relation-centric graphs lose to plain passage retrieval** (HippoRAG 2 −10, MS GraphRAG −6), and
  summary/hierarchy methods win (RAPTOR, EraRAG).

## 5. Could UNH do better on these than on HotpotQA / MuSiQue?

What we know about UNH so far:
- **Insurance contracts** (`long_form_contracts`, 5 contracts, 445 questions): LLM-judged score **0.86–0.94 per
  contract** (`sample_exp/nkg/insurance`). That is strong.
- **HotpotQA**, 100 questions from the old `wiki_data` sample: F1 **57.7** (EM 48; bridge 55, comparison 67), with
  deepseek-v4-flash.
- **MuSiQue**, 100 questions from the old sample: F1 **31.2** (2-hop 41, 3-hop 17, 4-hop 27), with Qwen3-30B.

For reference, the published systems in our current benchmark score F1 62–75 (HotpotQA) and 40–56 (MuSiQue), on
different question sets. UNH's weak point is visibly the **multi-hop chaining** (3- and 4-hop MuSiQue).

**PopQA: likely more favourable.**
- **Why it might go well:**
  - No multi-hop chaining, UNH's weak spot.
  - The questions are *literally* Wikidata triples, so a system that extracts entity–relation facts is in its element.
  - The answers are entities.
- **The risk is different:**
  - **Entity ambiguity and scale.** 33.6k passages, many sharing names (the *Messiah* / *Shame* / Paul Walker
    problem), and long-tail entities the LLM can't fix from memory. If UNH's entity resolution merges different
    people or works with the same name, it will answer confidently and wrongly.
  - The big corpus (4.1M tokens, 3× HotpotQA) makes full construction the most expensive of all our datasets (§6).

**QuALITY: structurally like the contracts, but a different kind of question.**
- **Like the contracts:**
  - one long document per graph (5.7k tokens vs 2k–19k for the contracts);
  - closed world (no outside knowledge);
  - the answer is usually spread across the text.
- **Unlike the contracts:**

  | | Contract questions | QuALITY questions |
  |---|---|---|
  | lookup (atomic / relational) | 52–56% | ~15% |
  | conditional / multi-hop reasoning | 30–40% | |
  | description / why / interpretation | | most (33% / 31% / 28%; categories overlap) |
  | unanswerable | 8–14% | |

  QuALITY is about narratives (characters' motives and feelings, irony, themes) and uses multiple choice with close
  distractors.
- **Why it might go well:** per-article graphs exactly match how UNH is run on contracts; documents are small; there
  is no cross-document search.
- **Why it might not:**
  - The evidence from the HyGRAG paper is that methods which turn text into entity–relation facts *lose* the
    narrative glue that "why / describe" questions need. HippoRAG 2 scored 46 vs VanillaRAG 56.
  - UNH would likely need to keep and retrieve original passages, or section summaries, alongside its facts.
  - The multiple-choice format also needs an answer step that compares the four options.

**Honest bottom line:** PopQA tests a skill (single-fact lookup with disambiguation) that UNH's design should be
good at. QuALITY tests a skill (whole-document understanding) that matches UNH's per-document setup but not
obviously its fact-centric representation. Running both on the small subsets (§7) is a cheap way to find out before
committing. If you include QuALITY, report the **HARD** subset separately: it is the part that cannot be solved by
skimming.

## 6. Size, tokens and construction cost

From `python evaluation_data/dataset_stats.py` (tiktoken cl100k_base; DeepSeek's tokenizer differs slightly):

| Dataset | Docs | Corpus tokens | Tokens/doc (mean / median / max) | Chunks @1200/100 | Questions | Tokens/question |
|---|---|---|---|---|---|---|
| HotpotQA | 9,811 | 1,299,765 | 132 / 118 / 1,902 | 9,813 | 1,000 | 20.8 |
| MuSiQue | 11,656 | 1,348,035 | 116 / 101 / 599 | 11,656 | 1,000 | 21.3 |
| **PopQA (full)** | 33,595 | **4,113,007** | 122 / 108 / 746 | 33,595 | 1,399 | 9.4 |
| **QuALITY (dev)** | 115 | **655,584** | 5,701 / 6,511 / 8,409 | 642 | 2,086 | 58.9 |
| QuALITY (DIGIMON pooled train+dev) | 265 | 1,521,395 | 5,741 / 6,525 / 8,418 | 1,494 | 4,609 | 64.6 |
| PopQA subset 100q | 2,474 | 306,028 | 124 / 110 / 642 | 2,474 | 100 | 9.1 |
| PopQA subset 200q | 4,931 | 599,316 | 122 / 106 / 689 | 4,931 | 200 | 9.2 |
| QuALITY subset ~100q | 6 | 32,361 | 5,394 / 5,851 / 7,264 | 32 | 113 | 48.8 |
| QuALITY subset ~200q | 11 | 60,609 | 5,510 / 6,510 / 8,096 | 60 | 206 | 55.2 |
| long_form_contracts | 5 | 94,963 | 18,993 / 13,695 / 48,699 | 89 | 445 | 16.9 |
| ten_contracts_dataset | 10 | 20,857 | 2,086 / 1,990 / 3,149 | 23 | 361 | 12.5 |

**Projected construction cost.** These are LLM input / output tokens in millions: corpus tokens × the
construction tokens per corpus token we *measured* on our HotpotQA + MuSiQue runs. UNH isn't in this table; its
construction usage would need to be measured the same way.

| Dataset | HippoRAG 2 | HyGRAG | MS GraphRAG | MemGraphRAG | HyperGraphRAG |
|---|---|---|---|---|---|
| ratio (in / out per corpus token) | 7.5 / 2.8 | 11.0 / 3.3 | 36.5 / 16.6 | 118 / 12.3 | 133 / 14.0 |
| PopQA full | 31 / 11 | 45 / 14 | 150 / 68 | 485 / 51 | 546 / 58 |
| QuALITY dev | 4.9 / 1.8 | 7.2 / 2.2 | 24 / 11 | 77 / 8 | 87 / 9 |
| PopQA 100q / 200q | 2.3 / 4.5 in | 3.4 / 6.6 in | 11 / 22 in | 36 / 71 in | 41 / 80 in |
| QuALITY ~100q / ~200q | 0.2 / 0.5 in | 0.4 / 0.7 in | 1.2 / 2.2 in | 3.8 / 7.1 in | 4.3 / 8.0 in |

These are order-of-magnitude guides only. Long documents amortise each call's prompt overhead differently than
130-token passages do, and MemGraphRAG's cost grows with the number of extracted facts.

The takeaway:
- **Full PopQA is ~3× HotpotQA in every system's construction cost.**
- **QuALITY dev is half of HotpotQA.**
- The subsets are 15–40× cheaper.

## 7. Subsets for quick UNH iteration (built)

Yes: subsets of 100 and 200 questions are built and reproducible. Run
`python evaluation_data/build_popqa_quality.py [--sizes 50 100 200] [--seed 0]`. Output is in
`evaluation_data/subsets/<ds>_<N>q_seed<seed>/`.

**PopQA:**
- **Sampling:** N questions, sampled **per relation type in proportion**. The file is sorted by relation, so a plain
  "first 100" would be all occupation questions.
- **Which questions are eligible:** only those whose passage block was located reliably, 1,180 of 1,399.
- **Corpus:** exactly those questions' retrieved-passage blocks:

  | Subset | Questions | Passages | Corpus tokens | Gold answer string present in corpus |
  |---|---|---|---|---|
  | 100q | 100 | 2,474 | 306k | 87% |
  | 200q | 200 | 4,931 | 599k | 87% |

**QuALITY:**
- **Sampling:** whole dev articles in seeded random order until there are at least N questions. All questions of a
  chosen article are kept, since its graph is built anyway.
- **Sizes:**

  | Subset | Questions (HARD) | Articles |
  |---|---|---|
  | ~100q | 113 (55) | 6 |
  | ~200q | 206 (98) | 11 |

**Formats in each subset folder:**
- `questions.json` + `corpus.json`: the `evaluation_data` layout used by `evaluation_scripts`.
- `unh_questions.jsonl` + `unh_corpus.jsonl`: the schema of `experiments/utils/multihop_utils.py`
  (`load_corpus_jsonl` / `load_questions_jsonl`), so UNH's `--corpus-file` / `--questions-file` can point straight
  at them.
  - For QuALITY, `question` includes the A–D options, and the records also carry `answer_letter`, `options` and
    `article_id`.
- `digimon/Corpus.json` + `Question.json`: for HyGRAG / DIGIMON.
- `manifest.json`: seed, chosen articles or relation counts, sizes.

**Caveats:**
- **Subset scores are for iteration, not for reporting.** A PopQA subset has ~2.5–5k passages instead of 33.6k, so
  there are fewer same-named distractors and retrieval is *easier* than on the full set. A QuALITY subset of 6–11
  articles has high variance (questions within one article are correlated). Use the same seed for every UNH version
  you compare, and run the full sets for final numbers.
- **QuALITY should be indexed per article** (one graph per article, its questions asked against it), like the
  contracts. Pooling 6–11 articles into one graph re-creates DIGIMON's "which story?" ambiguity.

## 8. What would still be needed to benchmark these in `evaluation_scripts`

Not done; for when you decide.

- **PopQA** fits the current pipeline. The shared answer prompt is short-answer, `score.py` already handles alias
  lists (EM/F1, Contain-Acc ≈ the paper's substring accuracy) and the LLM judge. Each runner needs `popqa` added as
  a dataset name.
  - The full corpus is 3.2× HotpotQA's tokens, so construction time and cost scale about 3×
    (~500M input tokens each for MemGraphRAG and HyperGraphRAG, §6).
- **QuALITY** needs three additions:
  1. a multiple-choice answer prompt (question + 4 options → letter);
  2. letter accuracy in `score.py`, reported overall and on HARD;
  3. a per-article indexing mode in each runner, or a documented decision to pool like DIGIMON.

## Sources

- Mallen et al., *When Not to Trust Language Models: Investigating Effectiveness of Parametric and Non-Parametric
  Memories* (ACL 2023): `popqa benchmark.pdf`; https://github.com/AlexTMallen/adaptive-retrieval
- Pang et al., *QuALITY: Question Answering with Long Input Texts, Yes!* (NAACL 2022): `quality benchmark.pdf`;
  https://github.com/nyu-mll/quality (v1.0.1, copied to `quality/official/`)
- HyGRAG, *A Unified Framework for Context-Aware and Relation-Aware Graph RAG* (WWW'26):
  `HyGRAG/WWW26_HyGRAG_camera_ready.pdf` (Table 1 results, Table 7 dataset statistics)
- Self-RAG (Asai et al., ICLR 2024), PopQA long-tail subset of 1,399 questions: https://arxiv.org/abs/2310.11511
- Our runs: `evaluation_scripts/` (results_table.csv, construction-token measurements of 2026-10-01);
  UNH: `sample_exp/nkg/{hotpotqa,musique,insurance}`
