# Prompt for Claude Design: UNH construction and retrieval figures

> Attach `paper.pdf` (the compiled paper) together with this prompt.

---

You are designing **two method figures for a research paper** about **UNH (Unified Narrative Hypergraph)**, a knowledge index for retrieval-augmented generation (RAG). The paper is attached as a PDF.

**What to read in the PDF:**
- Section 1 (Introduction).
- Section 4 (Methodology): 4.1 Index Structure, 4.2 Construction, 4.3 Retrieval.
- Appendix B (hyperparameters).

**What to ignore in the PDF:**
- The pipeline figure inside the PDF. It is outdated or a blank placeholder.
- The original "Related Work" section (Section 2). It still uses an older name ("Unified Discourse-Hypergraph", UDH) and an older design.

The specification below is authoritative. Where it differs from anything in the PDF, follow this specification.

## ⚙ Configuration switches (I set these before pasting; follow them exactly)

- **FACT_CONSOLIDATION = OFF**
  - ON → draw the optional step ⑤b in Figure A (spec in §4).
  - OFF → do not draw it, and do not mention fact deduplication anywhere.
- **SOURCE_PASSAGES = 0**
  - 5 → draw the optional source-passage add-on in Figure B, step ⑥ (spec in §5).
  - 0 → do not draw it.
- **BEAM_MMR_LAMBDA = 0.6**
  - The value printed in Figure B, step ④ (alternative: 0.75).

## 0. Deliverables

Make **two separate figures** on two separate artboards. They will be two separate figures in the paper.

| | Figure A: Construction | Figure B: Retrieval |
|---|---|---|
| Purpose | How a corpus becomes a UNH index | How a query becomes a context for the reader |
| Artboard | 1600 × 1000 px, landscape | 1600 × 900 px, landscape |
| Panel label | small "(a) Index construction", top-left | small "(b) Retrieval", top-left |

- **Export:** each figure as **PDF (vector)** and **SVG**, plus **PNG at 300 dpi**, and use **Export to Canva** so I can edit it there.
- **Work order:** build a **wireframe first** with boxes, arrows and labels only. Check the flow against the specification. Only then do the high-fidelity pass. Don't polish before the layout is right.
- **Titles and captions:** don't put a big title or a caption on the artboard; the LaTeX caption does that. Only the small panel label goes on the figure.

## 1. Print legibility is the hard constraint

- **Print size:** each figure will print about **12 cm wide** (a single LNCS column), so 1 px on the artboard ≈ 0.22 pt in print.
  - Body text **≥ 28 px** (≈ 6 pt printed).
  - Box titles **34–36 px**, semibold.
  - Phase headers **38–40 px**.
  - Nothing smaller than 26 px anywhere, including arrow labels and legend text.
- **Text per box:**
  - **title** (≤ 5 words)
  - **one line** saying what the step produces
  - **one small illustration**
  - **one badge** for who does the work (LLM / embedding / algorithm)
- **Detail goes elsewhere:** extra detail belongs in the caption, not the figure. If a box needs more than about 12 words of running text, cut words; never shrink the font.
- **Whitespace:** at least 24 px between boxes, and 20 px padding inside boxes.

## 2. Visual language (use it identically in both figures)

**Style:** clean, flat and modern, like a top-venue ML method figure (NeurIPS/ACL style).
- White background.
- No 3D, no heavy gradients, no glossy effects, no clip-art people.
- Very subtle shadow at most: 0 2 6 rgba(0,0,0,0.08). None is fine too.

**Shapes**
- Every step is a **rounded rectangle**: corner radius 16 px, 2 px border.
- Don't use circles or bubbles for steps. Circles and pills are reserved for **entities**.
- **Grouping containers:** larger rounded rectangles with a 1.5 px **dashed** grey border (#9AA0A6), a very light fill (#F7F8FA), and a header tab in the top-left corner.

**Colour: Okabe–Ito palette (colour-blind safe), with fixed meanings**

| Meaning | Colour | Where |
|---|---|---|
| **Fact** (atomic proposition) | blue `#0072B2` (fill `#E5F1F8`) | rounded-rectangle "fact cards" inside illustrations |
| **Entity** | orange `#E69F00` (fill `#FDF3E1`) | pills/circles with name + small role text |
| **Typed relation edge** (fact → fact) | bluish green `#009E73` | solid arrows carrying an UPPERCASE label |
| **Participation edge** (fact ↔ entity, i.e. hyperedge membership) | grey `#7F8C8D` | thin 2 px solid lines, no arrowheads |
| **LLM step** | vermillion accent `#D55E00` | box border plus robot badge |
| **Embedding / similarity step** | sky blue `#56B4E9` | box border plus vector badge |
| **Deterministic algorithm** (clustering, MMR, beam search, dedup) | dark grey `#4D4D4D` | box border plus gear badge |
| **Communities / clusters** | reddish purple `#CC79A7` plus the palette's other hues for different clusters | soft translucent blobs behind grouped nodes |
| Highlight of the running example's answer path | yellow `#F0E442` glow under the path | Figure B only |

- **Step box fill:** white.
- **Step box border:** the colour of whoever does the work (LLM / embedding / algorithm).

**Badges (icons): the reader must tell LLM steps from non-LLM steps at a glance**
- **LLM steps:** a small, friendly, flat **robot head** icon, two-tone vermillion and white, about 44 px, in the box's top-right corner. Next to it is a tiny pill naming the call rate, e.g. `LLM · 1 call / chunk`.
  - Use the robot **only** on LLM steps.
- **Embedding steps:** a small "vector" icon (a short column of 4 coloured dots followed by →), with the pill `Embeddings`.
- **Algorithmic steps:** a small **gear** icon, with a pill naming the algorithm (`Leiden`, `MMR`, `Beam search`).
- **Cross-encoder (Figure B, step 5):** a small **balance-scale** icon with the pill `Cross-encoder`.
  - It is a neural reranker, **not** an LLM: no robot.
- **Data / index objects:** a small **cylinder** icon (corpus, UNH index).
- **Icon style:** all icons share one outline-icon style with 2 px strokes and rounded caps. No emoji.

**Arrows**
- **Flow arrows between steps:** 3 px, dark grey `#3C4043`, filled triangular heads.
  - Straight or orthogonal (elbow) routing only; never diagonal spaghetti.
  - Arrows **never cross boxes**.
- **Arrow labels:** 26–28 px italic grey text on a white background chip, naming *what is passed along*, e.g. *facts*, *entity inventory*, *communities*.
- **Number the steps:** ①②③… in a small filled dark circle before each box title, so the order is unambiguous even in greyscale.

**Typography**
- One sans-serif family: Inter, or Helvetica/Arial as a fallback.
- Labels for relation edges and ids are in **monospace UPPERCASE**, e.g. `EXCEPTION_TO`.
- Example text quoted from documents is italic, in quotes.

## 3. The running example (use it in both figures so they tell one story)

Three clauses of one auto-insurance policy, in three different chunks:
- **Chunk A** (Coverage): *"We will pay for direct and accidental loss to your covered auto caused by collision."*
- **Chunk B** (Exclusions): *"We do not provide coverage for loss while your covered auto is operated by a driver not listed on the declarations page."*
- **Chunk C** (Endorsement): *"The listed-driver exclusion does not apply to a resident relative of the named insured."*

Extracted **facts** (blue cards):
- **f1** "The insurer pays for collision damage to the covered auto."
- **f2** "Loss is not covered while the covered auto is driven by a non-listed driver."
- **f3** "The non-listed-driver exclusion does not apply to resident relatives."

**Entities** (orange pills, name / micro-role):
- *insurer* / pays covered losses
- *covered auto* / vehicle insured by the policy
- *non-listed driver* / driver absent from declarations
- *resident relative* / household member of the insured

**Participants** (hyperedges):
- f1 = {insurer, covered auto}
- f2 = {covered auto, non-listed driver}
- f3 = {resident relative}

f3 shares **no** entity with f1 or f2. Only the typed relation f3 → f2 connects it, which is the paper's central point.

**Typed relations** (green arrows):
- f2 —`EXCEPTION_TO`→ f1, with backward label `HAS_EXCEPTION`
- f3 —`EXCEPTION_TO`→ f2

**Question:** *"Is the policyholder reimbursed for collision damage while a non-listed driver operates the insured vehicle?"*
**Answer:** *"No, unless the driver is a resident relative."*

Use shortened versions of these texts inside illustrations (e.g. "f2: not covered if non-listed driver"). Full sentences only where space allows.

## 4. Figure A: Index construction (1600 × 1000)

**Layout:** two horizontal rows, read left → right, then down, then left → right again.
- **Row 1** sits inside a dashed container whose header tab reads **"Per chunk · runs in parallel for every chunk"**.
- **Row 2** sits inside a dashed container whose header tab reads **"Corpus level"**.
- **Connecting the rows:** one elbow arrow leaves the right end of Row 1, goes down, runs left along the gap between the rows, and enters the first box of Row 2. Label it *chunk subgraphs (facts, entities, relations)*.
- **Hero:** the final output, the UNH index, sits at the far right of Row 2. It is the largest and most visually prominent element.

### Row 1: per-chunk extraction (y ≈ 70–460)

**Input: "Corpus"** (narrow box, ~130 px wide)
- **Icon:** cylinder, or a stack of 3 small page sheets.
- **Line:** "passages → chunks".
- **Under it, small:** "1 passage = 1 chunk (long docs: 600-token windows)".
- **Arrow out:** *chunk*.

**① Describe & list entities** · `LLM · 1 call / chunk` (robot badge)
- **Produces:** "1-sentence chunk description + entity inventory (name, micro-role)".
- **Illustration:** a small document card (Chunk B's text, truncated) on the left. A thin arrow leads to:
  - a grey one-line strip labelled `description`, and
  - two orange entity pills: `covered auto · vehicle insured…` and `non-listed driver · driver absent…`.
- **Footnote inside the box (26 px, grey):** "description is used later only as context for relation calls".

**② Extract atomic facts** · `LLM · 1 call / chunk` (robot)
- **Produces:** "atomic, self-contained, grounded fact sentences".
- **Illustration:** the same chunk card on the left with two highlighted spans. Arrows lead to 2–3 stacked blue fact cards (`f2`, …).
- **Three tiny check-chips under the cards:** `atomic` · `self-contained` · `grounded`.

**③ Annotate facts & link entities** · `LLM · 1 call / 15 facts` (robot), plus a gear for the linking
- **Produces:** "topics, answered Qs, follow-up Qs, participants".
- **Illustration:** one blue fact card (f2) with four small tag rows attached:
  - `topics: driver exclusions`
  - `answers: "When is collision damage not covered?"`
  - `follow-up: "Are there exceptions to this exclusion?"`
  - `participants →`
- **Participants:** grey participation lines go to the two orange entity pills. Draw f2 as a **hyperedge**: a soft translucent blue blob enclosing the fact card and both entity pills, with a mini caption "fact = hyperedge over its entities".

**④ Label intra-chunk relations** · `LLM · 1 call / chunk (≥ 3 facts)` (robot)
- **Produces:** "typed relations between facts of the same chunk".
- **Illustration:** two blue fact cards connected by a green arrow labelled `CAUSES →` above and a grey return arrow `← CAUSED_BY` below.
  - Use generic cards "fx", "fy" here; the running example's relations are cross-chunk.
- **Small chip:** `strength r ∈ [0,1]` (pairs with r = 0 dropped).
- **Small chip with a strike-through icon:** "label only, no free-text description".

### Row 2: corpus-level (y ≈ 540–950)

**⑤ Merge & resolve entities**
- **Badges:** gear `exact + SemHash`, then robot `LLM · 1 call / cluster`.
- **Produces:** "one node per real-world entity".
- **Illustration:** a funnel. Three orange pills (`Covered Auto`, `covered auto`, `your covered auto`) flow into **one** orange pill `covered auto`.
- **Under the funnel, three micro-steps in tiny type, each with its own icon:**
  1. normalize names (exact match) [gear]
  2. near-duplicate hashing [gear]
  3. similarity clusters (Leiden) → LLM confirms [gear → robot]

**⑤b (optional; only if FACT_CONSOLIDATION = ON) Consolidate facts** · `LLM · 1 call / cluster of ≤ 8 facts` (robot), plus a gear for the clustering
- **Produces:** "one fact per claim; contradictions and scope differences resolved".
- **How it works:** facts are clustered by similarity (sentence 0.7, entities 0.15, topics 0.15; threshold 0.8). The LLM sees each cluster *with its source chunks* and removes, rewrites or merges duplicate, contradictory or differently-scoped facts.
- **Illustration:**
  - Two near-identical blue fact cards plus one conflicting card (red-tinted edge) enter a small merge symbol, and one clean fact card comes out.
  - Three tiny tags: `duplicate` · `contradiction` · `scope`.
- **Layout:** Row 2 then has 6 boxes. Narrow ⑤–⑧ to about 220–240 px each and keep the hero as the widest box.

**⑥ Multi-view fact affinity** · `Embeddings` (vector badge)
- **Produces:** "affinity between every pair of facts".
- **Illustration:** two fact cards, f1 (left) and f2 (right), with four horizontal "view" bars between them, each a small similarity bar:
  - `sentence`
  - `topics`
  - `entity roles`
  - `follow-up(f1) ↔ answered(f2)`, drawn as a curved arrow from f1's follow-up question chip *"Are there exceptions to collision coverage?"* to f2's answered-question chip *"When is collision damage not covered?"*. Mark this bar as **asymmetric**.
- The bars sum (a "+" stack) to a single score `a(f1, f2)`.
- **Footnote:** "the question view links a fact to the fact that answers what it raises".

**⑦ Candidate graph → communities** · `Leiden` (gear badge)
- **Produces:** "groups of facts likely to be related".
- **Illustration, left to right:**
  - Left: a scatter of ~14 small blue dots where each dot connects to its nearest neighbours, labelled `k = 12 nearest by affinity`.
  - A small arrow.
  - Right: the same dots coloured into 3 translucent community blobs (purple, sky blue, orange), labelled `communities ≤ 40 facts`.
- **The point to show:** f1, f2 and f3 (highlighted, with their labels) land in the **same** community even though they come from **different chunks**. Give each of their dots a tiny chunk tag: A, B, C.

**⑧ Adjudicate each community** · `LLM · 1 call / community` (robot)
- **Produces:** "typed cross-chunk relations".
- **Illustration:**
  - **Input:** a "prompt card" listing three lines in monospace:
    - `1: [Context: coverage section] f1`
    - `2: [Context: exclusions section] f2`
    - `3: [Context: endorsement] f3`
  - **Output:** green arrows `f2 —EXCEPTION_TO→ f1` and `f3 —EXCEPTION_TO→ f2`.
- **Footnote:** "1 call per community, not per pair → LLM calls grow linearly".

**Output: UNH index (hero box, ~400 px wide, slightly larger, cylinder icon)**
- **Title:** "UNH index".
- **Illustration: a two-layer schematic**
  - **Bottom layer:** orange entity pills (`insurer`, `covered auto`, `non-listed driver`, `resident relative`).
  - **Top layer:** blue fact cards f1, f2, f3.
  - **Participation:** grey participation lines go from each fact down to its entities. Each fact plus its entities sits inside a faint blue blob, showing that each fact is a hyperedge.
  - **Relations:** green typed arrows run **between facts** on the top layer: f2 `EXCEPTION_TO` f1, f3 `EXCEPTION_TO` f2.
  - f3 connects only to `resident relative`, so the green arrow f3 → f2 joins two facts that share **no** entity. Add a tiny callout on that arrow: "no shared entity: reachable only via the relation". This is the key point of the paper.
  - **Provenance:** a faint, small row of grey chunk icons (A, B, C) at the very bottom, joined by dotted provenance lines. Label it `provenance only`.
- **Two-line key inside the box:**
  - grey line = "participation (hyperedge)"
  - green arrow = "typed relation"

### Legend strip (bottom, y ≈ 960–995, one line)
`[robot] LLM call` · `[vector] embedding similarity` · `[gear] algorithm` · `[blue card] fact` · `[orange pill] entity` · `[green arrow] typed relation` · `[grey line] participation`

## 5. Figure B: Retrieval (1600 × 900)

**Layout:** two rows, the same reading order as Figure A.
- **Row 1:** query understanding and seeding.
- **Row 2:** graph search, ranking and context assembly.
- **Connecting the rows:** an elbow arrow from the end of Row 1 to the start of Row 2, labelled *9 seeds (facts + entities)*.
- **Show the UNH index:** a small cylinder labelled "UNH index", top-left under the panel label, with dotted grey arrows into steps ② and ④ ("reads").
- **The only LLM call inside UNH retrieval is step ①.** Make that visually obvious: it is the only box with a robot in this figure, except the reader at the end, which sits outside UNH.

### Row 1 (y ≈ 70–420)

**Query card** (narrow, ~220 px)
- A speech-bubble-shaped rounded rectangle (rounded rectangle with a small tail is fine) holding the question text from §3, shortened.
- **Note:** "used verbatim (not rewritten)".

**① Plan the query** · `LLM · 1 call / query` (robot)
- **Produces:** "a structured query plan".
- **Illustration:** a "plan card" with five labelled rows, each with a coloured left stripe showing what it matches later:
  - `topics` (blue stripe → facts): *collision coverage; driver exclusions*
  - `named entities` (orange stripe → entity names): *(none)*. Show the empty field greyed out; it illustrates that fields stay empty rather than being invented.
  - `role anchors` (orange stripe → entity roles): *policyholder; non-listed driver*
  - `target relation labels` (green stripe → relation edges): `EXCEPTION_TO`, `CONDITION_FOR`
  - `connection phrases` (green stripe → relation edges): *exceptions to collision coverage*
- **Footnote:** "1–3 items per field; empty if not implied".

**② Score facts & entities together** · `Embeddings` (vector badge)
- **Produces:** "one ranked pool of facts and entities".
- **Illustration:** two mini horizontal bar charts side by side, merging into one list:
  - **Facts:** `0.8 · sim(query, sentence) + 0.2 · sim(topics)`. Bars for f2, f1, …
  - **Entities:** `0.6 · sim(names) + 0.4 · sim(roles)`. Bars for `non-listed driver`, `covered auto`, …
- **Footnote:** "named subjects enter via entities; described events enter via facts".

**③ Pick diverse seeds** · `MMR` (gear badge)
- **Produces:** "K = 9 seeds".
- **Illustration:** a ranked column of ~8 items (mixed blue cards and orange pills).
  - Selected items are filled.
  - A near-duplicate of an already chosen fact is hollow and struck through, with a tiny note "too similar".
- **Formula chip:** `λ = 0.6 relevance vs diversity`.

### Row 2 (y ≈ 480–850)

**④ Label-guided beam search** · `Beam search` (gear badge), with the UNH index feeding it. **Hero box, ~560 px wide.**
- **Produces:** "candidate evidence paths".
- **Illustration: a small UNH subgraph** using the same visual language as Figure A's hero (blue fact cards, orange entity pills, grey participation lines, green labelled arrows).
  - **Start** at the seed entity `non-listed driver` (thick outline + "seed" tag).
  - **Highlight the path** with the yellow glow: `[non-listed driver]` —(participates)→ `f2` —`EXCEPTION_TO`→ `f1`.
  - **Branch:** a second branch from `f2` via `HAS_EXCEPTION` to `f3`.
  - **Pruned branches:** 2–3 faint grey branches ending in a small ✕, to show beam pruning.
- **Three score chips along the edges:**
  - on a green edge: `0.3 · label↔target labels + 0.5 · label↔phrases + 0.2 · r`
  - on a fact→entity step: `role ↔ role anchors`
  - on an entity→fact step: `unlabelled (uniform)`
- **Parameter chips:** `beam B = 6` · `MMR λ = BEAM_MMR_LAMBDA` · `≤ 3 facts per path` · `≤ 6 hops` · `no revisits`.

**⑤ Rank paths** · `Cross-encoder` (scale badge), then `MMR` (gear badge)
- **Produces:** "top P = 20 diverse paths".
- **Illustration:** 3–4 thin horizontal "path strips". Each is a serialized path in monospace, e.g. `"f2" → EXCEPTION_TO → "f1"`, with a score bubble on the right (0.91, 0.84, 0.40).
  - The low-scoring strip is faded.
  - A bracket groups the kept strips: `P = 20, λ = 0.6`.

**⑥ Assemble deduplicated chains** · `Algorithm` (gear badge)
- **Produces:** "the context: fact chains joined by relation labels".
- **Illustration:**
  - Two overlapping path strips. The shared segment is greyed out ("already shown"), and only the new segment (a "burst") is coloured.
  - An arrow leads to a final **context card**:
    ```
    "Loss is not covered while … non-listed driver." → EXCEPTION_TO → "The insurer pays for collision damage …"
    "… does not apply to resident relatives." → EXCEPTION_TO → "Loss is not covered …"
    ```
    Monospace labels, italic fact text.
- **Footnote:** "every sentence is an extracted fact; no summaries, no truncation".
- **(Optional; only if SOURCE_PASSAGES = 5)** Below the context card, add a thin grey "＋ source passages" strip: 2–3 small document icons labelled `≤ 5 source chunks of the facts in the top paths (in rank order)`, with a dotted provenance line from the fact chains to the documents.

**Reader → Answer** (narrow, dashed-border box with the label `shared reader (outside UNH)`) · robot badge `LLM`
- **Inside:** a tiny "context + question" stack goes into a robot, which produces an answer bubble: *"No, unless the driver is a resident relative."*

### Legend strip (bottom, one line)
The same legend as Figure A, plus `[scale] cross-encoder (not an LLM)` and `[yellow glow] example answer path`.

## 6. Do NOT include (these are outdated or wrong)

- ❌ "Top-K", "Middle-K", "middle-k", "skip top 33%", or any second/third clustering round. There is **one** affinity-graph clustering for cross-chunk relations.
- ❌ Long free-text edge descriptions, a "description" field on edges, or any "Logical Conclusions" block. **Edges carry short UPPERCASE labels only.**
- ❌ Fact deduplication or fact merging, **unless** FACT_CONSOLIDATION = ON (then only as step ⑤b).
- ❌ Chunk–chunk edges, community *summaries*, or any LLM-generated summary text stored in the index.
- ❌ Entity *types* or "relational anchors / features" on entities. Entities are **name + micro-role** only.
- ❌ The name "UDH" or "Unified Discourse-Hypergraph". The system is **UNH (Unified Narrative Hypergraph)**.
- ❌ A robot icon on any non-LLM step, including the cross-encoder, MMR, Leiden and beam search.
- ❌ Text smaller than 26 px, paragraphs inside boxes, decorative backgrounds, or stock photos.

## 7. Final self-check before export

1. Can a reader follow ① → ⑧ (Figure A) and ① → ⑥ → Reader (Figure B) without reading the caption? Every step is numbered and connected by exactly one incoming flow arrow.
2. Count the robots:
   - Figure A has robots on ①, ②, ③, ④, ⑧, the LLM sub-step of ⑤, and ⑤b if it is on.
   - Figure B has robots only on ① and the Reader.
3. Does Figure A's output box clearly show **both** edge families on the same fact objects, including one typed relation between facts that share no entity?
4. Are colours used with the fixed meanings in §2, consistently across both figures?
5. Zoom to 25 %. Is all text still readable, and are no two elements touching?
6. Is anything from §6 present? Remove it.
