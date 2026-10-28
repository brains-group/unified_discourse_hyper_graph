# Fact→fact edge prompt: event-relation families, compact version (draft)

This is a condensed version of [FACT_EDGE_EVENT_PROMPT.md](FACT_EDGE_EVENT_PROMPT.md). It covers the same families,
which come from RED (O'Gorman et al., 2016) and Hong et al. (2016), plus a WILDCARD family. It keeps the core test,
the label rules and the score rubric, one good example per family, and the five most important bad examples. Every
name in the examples is invented and absent from the HotpotQA, MuSiQue and QuALITY data. It is not yet wired into
`extract_relations.py`, and it assumes a `relation_family` output field.

---

```text
You link fact sentences in a knowledge graph. An edge means: "read these two facts together, because combined they
reveal something neither reveals alone."

CORE TEST: mentally merge the two facts into one sentence. Link them only if the merged sentence tells you something
new: why something happened, what it led to, what made it possible or stopped it, when it happened relative to
something else, what larger event it belongs to, who an unnamed person is, or that two reports cannot both be true.
If the merge is just "A, and also B", do not link. Sharing a person, place, topic, category or coincidental date is
NOT enough, because shared entities are already linked elsewhere. Facts from different documents or unrelated
contexts that only share a name are not linked. Fewer, meaningful edges beat many vague ones.

RELATION FAMILIES (choose the one most informative relation per pair; cause > precondition > time order):
  CONTINGENCY  cause (given A, B was inevitable); precondition/enable (without A, no B, including implicit
               conditions with no "because" in the text); prevent (A is why B did not happen); motive (a feeling,
               belief or goal in A is why someone does B); purpose (A is done in order to achieve B).
               If B merely presupposes A by definition (a "remarriage" presupposes a marriage), it is not a cause.
  INHERITANCE  same event with complementary details (only with the same modality and polarity: "planned to sell"
               is not "sold"); subevent (B is one part of the larger event A; a more specific event is usually a
               subevent, not the same event); recurrence (B repeats A); generic habit vs. one occurrence.
  EXPANSION    instance (B is a case or member of the general statement A); conjunction (two parts of one larger
               event with similar roles); disjunction (two parts of one larger event with opposing roles).
  COMPARISON   contradiction (cannot both be true); negation (B ends A: a job lost, a promise broken); competing
               accounts of the same event; concession (an expected outcome failed); a meaningful contrast.
  TEMPORALITY  before, immediately before, during/contains, begins on, ends on, overlap, simultaneous. Use only when
               the timing itself matters, and never in addition to a causal edge.
  BRIDGING     part/whole; set/member; identity claim (an unnamed person matched to a named one: say "suspect"
               unless the text confirms it); plan, threat or hope → what actually happened.
  WILDCARD     clearly and strongly related, but no family fits: a clue and what it reveals, a question and its
               answer, a statement that exposes a lie, a rule that decides someone's outcome. Invent a label naming
               exactly what is revealed. If you cannot name it, emit nothing.

LABELS: UPPERCASE_WITH_UNDERSCORES, 2-6 words: the relation word plus WHAT it is about. Never use a bare relation
name like CAUSES, MOTIVATES, CONTRADICTS, PRECEDES or ELABORATES_ON.
  CAUSES_TWO_DAY_BLACKOUT (not CAUSES), FEAR_MOTIVATES_FLIGHT (not MOTIVATES), CONTRADICTS_CAR_COLOR (not CONTRADICTS).
Use content words, not full names. label_forward reads source → target. label_backward is the same relation read
target → source and is just as specific (CAUSES_TWO_DAY_BLACKOUT ↔ BLACKOUT_CAUSED_BY_STORM); symmetric relations
may repeat the label. Never use an attribute of one entity (FOUNDED, BORN, LOCATED_AT, EMPLOYED_BY) as a label.

SCORE (0.0-1.0) = how much new understanding the combination gives:
  0.9-1.0  answers a why/how/who/what-next question neither fact answers alone
  0.6-0.8  a clear, useful gain
  0.3-0.5  weak but real
  below 0.3  do not emit

GOOD EXAMPLES
  CONTINGENCY  "The storm knocked down Brindlewick's power lines." + "Brindlewick had no electricity for two days."
               → CAUSES_TWO_DAY_BLACKOUT / BLACKOUT_CAUSED_BY_STORM (0.95)
               "Mara Okafor learned her brother had secretly sold the family farm." + "Mara Okafor did not speak to
               her brother for a year." → ANGER_MOTIVATES_YEAR_OF_SILENCE / SILENCE_MOTIVATED_BY_FARM_SALE (0.95)
  INHERITANCE  "Brindlewick held a three-day harvest festival." + "A pie contest was held on the festival's second
               day." → INCLUDES_PIE_CONTEST / SUBEVENT_OF_HARVEST_FESTIVAL (0.7)
  EXPANSION    "Every shop on Dovecote Lane closed during the strike." + "The Dovecote Lane bakery shut on Monday."
               → INSTANCE_BAKERY_CLOSES_IN_STRIKE / PART_OF_STRIKE_CLOSURES (0.75)
  COMPARISON   "Doctors expected the treatment to cure Halvard's fever." + "Halvard's fever returned a week later."
               → EXPECTED_CURE_FAILED / SHOWS_TREATMENT_FAILED (0.9)
  TEMPORALITY  "The siege of the fortress lasted six months." + "The commander's son was born in the fortress during
               the siege." → CONTAINS_SONS_BIRTH / BORN_DURING_SIEGE (0.7)
  BRIDGING     "A masked man robbed the Quenby bank." + "Police arrested Karl Reiss as a suspect in the Quenby
               robbery." → SUSPECT_FOR_MASKED_ROBBER / MASKED_ROBBER_SUSPECTED_REISS (0.9)
  WILDCARD     "The anonymous letter was written in green ink." + "Only the governor's secretary used green ink."
               → INK_POINTS_TO_SECRETARY / SECRETARY_LIKELY_WROTE_LETTER (0.95)

BAD EXAMPLES
  ✗ Shared entity only: "Mara Okafor grew up in Brindlewick." + "Mara Okafor owns a grey cat." → no edge.
  ✗ Restatement: "The bakery opened in 1990." + "The bakery has been open since 1990." → no edge (a duplicate).
  ✗ Vague label: storm → blackout labelled CAUSES. Use CAUSES_TWO_DAY_BLACKOUT.
  ✗ Subevent called the same event: "A fire spread through the old town." + "A fire destroyed the Dovecote Lane
    bakery." → not SAME_FIRE, but INCLUDES_BAKERY_DESTROYED / SUBEVENT_OF_OLD_TOWN_FIRE.
  ✗ Different modality as the same event: "Karl Reiss threatened to burn the barn." + "The barn burned down." → not
    SAME_EVENT, but THREAT_PRECEDES_BARN_FIRE (or THREAT_CARRIED_OUT only if the text says Reiss set it).

OUTPUT, one edge per qualifying pair:
  source_id, target_id (integer IDs from the list, never equal); relation_family (CONTINGENCY | INHERITANCE |
  EXPANSION | COMPARISON | TEMPORALITY | BRIDGING | WILDCARD); label_forward; label_backward; relevance_score.
  No descriptions or explanations: the two labels are the only text the edge will ever have.
```
