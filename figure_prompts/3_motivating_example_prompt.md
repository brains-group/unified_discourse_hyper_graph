# Figure 1: motivating example (brief)

This brief is for the full-width figure that opens the introduction: `figures/motivation.pdf`, `\label{fig:motivation}`.

Use the same visual language as the construction and retrieval figures:
- Okabe–Ito colours: fact `#0072B2`, entity `#E69F00`, relation `#009E73`, participation `#7F8C8D`, LLM-written text `#D55E00`, highlight `#F0E442`.
- IBM Plex fonts.

Facts, entities and relations must match Table 2 (`tab:running`) in paper.tex exactly.

## The story

The source is the TD 20-Year Term Life sample policy (`evaluation_data/long_form_contracts/contracts/td_contract.txt`). Label it "term-life policy", with no brand.

**Question (beneficiary):** "My father applied for this policy saying he was 58, but he was actually 61. He died six years later, long after the two-year contestability period. Will I receive the death benefit?"

**Answer:** No. The policy is treated as never in force, and the premiums are refunded.

| Fact | Policy section | Clause (abridged) | Entities |
|---|---|---|---|
| f1 | General provisions | After two years, answers given as evidence of insurability are treated as true, except for fraud | evidence of insurability, two-year contestability period |
| f2 | General provisions | The two-year rule does not apply to the insured's date of birth | two-year contestability period, **date of birth** |
| f3 | Misstatement of age | Misstated, still eligible → premium corrected, benefit reduced by the underpaid amount | **date of birth**, insured person, premium, death benefit |
| f4 | Misstatement of age | Misstated, not eligible at the correct age → never in force, premiums refunded | **date of birth**, insured person, premium |
| f5 | Eligibility (pages earlier) | Applications accepted only from ages 18 to six months past the 60th birthday | application, age limit |

**Relations:**
- f1 HAS_EXCEPTION f2 (within a chunk)
- f3 CONTRASTS_WITH f4 (within a chunk)
- f4 DEPENDS_ON f5 and f3 DEPENDS_ON f5 (across sections, no shared entity)

**Entity link:** date of birth joins f2, f3 and f4. It appears in only 3 of the policy's 251 sentences.

**Answer path (UNH):** f1 —HAS_EXCEPTION→ f2 → [date of birth] → f4 —DEPENDS_ON→ f5. It alternates relation, entity, relation.

## Layout

The question goes in a strip across the top, with the three section tabs that f1–f5 come from. Below that are five equal panels, (a)–(e). Each panel ends with a one-line verdict: a red ✕ for (a)–(d) and a green ✓ for (e).

**(a) Dense retrieval**
- Show a ranked list: f1, f4, f3, f2 above a dashed "top-k" line; f5 faded below it.
- Verdict: "cannot tell which branch applies".

**(b) Triple graph**
- Triples hang off "date of birth" (f2–f4). The eligibility triple (application, age limit) floats unconnected.
- Verdict: "the age limit is not linked to the clause it decides".

**(c) Hypergraph**
- Blobs f2, f3, f4 overlap at date of birth; the f5 blob is isolated.
- Verdict: "participants kept together; facts meet only at shared entities".

**(d) Community summaries**
- Two clusters, "General provisions" (f1–f4) and "Application & eligibility" (f5 plus declarations), each with an orange LLM-written summary card.
- Put a red "?" on the gap between them.
- Optionally show the paraphrase hazard: "misstated ages lead to a premium correction" (drops the f4 branch).
- Verdict: "the needed link lies between clusters".

**(e) UNH**
- Fact cards with entity pills.
- Green typed arrows for the relations.
- The answer path in yellow, with the entity hop through date of birth marked.
- Add the note "no shared entity" on f4→f5.
- Verdict: "the answer is a path; each clause traceable to its section".

## Before finalising

- Verify panels (a) and (d) on real indices (see the `\abid` note in the introduction).
- Keep the panel letters (a)–(e), because the caption and the introduction refer to them.
