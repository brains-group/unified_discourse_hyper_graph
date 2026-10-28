# Fact→fact edge prompt: event-relation families (draft)

Sources: O'Gorman, Wright-Bettner & Palmer (2016), *Richer Event Description* (RED), and Hong et al. (2016),
*Building a Cross-document Event-Event Relation Corpus*. The relation families combine Hong et al.'s five types
(Inheritance, Expansion, Contingency, Comparison, Temporality) with RED's bridging relations and its definitions of
cause, precondition, prevention and modality. A WILDCARD family covers anything else. Every name and scenario in the
examples is invented; none of the example names appear in the HotpotQA, MuSiQue or QuALITY data.

Status: draft, not yet wired into `extract_relations.py`. It assumes one extra output field, `relation_family`, on
the edge pair; drop that field if the existing schema is kept.

---

```text
You connect fact sentences into a knowledge graph. Each fact is one atomic, self-contained sentence. An edge between
two facts tells a later reader: "read these two together: combined, they tell you something neither tells you
alone." Your job is to find exactly those pairs, and to name what the combination reveals.

=====================================================================================================
1. THE CORE TEST: DOES COMBINING THE TWO FACTS CREATE NEW UNDERSTANDING?
=====================================================================================================
Before you emit any edge, do this test in your head:
  (a) Write the one sentence that merges fact A and fact B.
  (b) Ask: does the merged sentence tell me something that I could not know from A alone or from B alone?
      For example: WHY something happened, WHAT it led to, WHAT made it possible or stopped it, WHEN it happened
      relative to something else, WHAT larger event it belongs to, WHICH case of a general statement it is,
      WHO an unnamed person turns out to be, or THAT two reports cannot both be true.
  (c) If the merged sentence is just "A, and also B" (two facts side by side, nothing gained), do NOT link them.

Things that are NOT enough for an edge:
  - The two facts mention the same person, place, object or topic. Shared entities are already linked elsewhere in
    the graph.
  - The two facts happen to have dates, so one is "earlier", but neither affects or frames the other.
  - The two facts belong to the same broad category ("X is a city", "Y is a city").
  - One fact repeats the other in different words without adding anything.
  - Two facts use the same name but come from different stories or documents with no sign that they describe the
    same person or event.

Fewer, meaningful edges are far better than many vague ones. A missing weak edge costs little; a wrong or empty
edge misleads the reader.

=====================================================================================================
2. RELATION FAMILIES
=====================================================================================================
Pick the ONE family and relation that best describes what the combination reveals. If two relations hold (for
example a cause, which always also implies an order in time), choose the more informative one: cause beats
precondition, precondition beats plain time order, and time order is used only when nothing stronger holds.

-----------------------------------------------------------------------------------------------------
2.1 CONTINGENCY: one fact brings about, makes possible, or blocks the other
-----------------------------------------------------------------------------------------------------
  CAUSE: given A, B was inevitable or directly brought about, according to the text.
         Test: "Because A, B happened."
  PRECONDITION / ENABLE: had A not happened, B would not have happened, but A alone did not force B.
         Test: "Without A, B could not have happened." This includes implicit conditions with no connective word
         in the text (a skill, permission, resource or prior event that B depends on). These are the edges
         annotators miss most often, so look for them actively.
  PREVENT: A stops B from happening, or is why B did not happen.
         Test: "Because of A, B did not happen."
  MOTIVE: A (a feeling, belief, goal, fear, order, promise) is why a person does B. Name the feeling or goal in
         the label: anger, fear, jealousy, greed, duty, love, revenge, ambition.
  PURPOSE: B is the goal that the action in A serves.
         Test: "A was done in order to B."
Not a precondition: when B merely PRESUPPOSES A by definition. A "second marriage" presupposes a first one, and a
"re-election" presupposes an election. Do not call that a cause or precondition. Use a temporal or recurrence
relation, or no edge.

-----------------------------------------------------------------------------------------------------
2.2 INHERITANCE: the two facts describe the same event, a part of it, or a repeat of it
-----------------------------------------------------------------------------------------------------
  SAME EVENT (coreference): both facts describe the same single occurrence, and each adds a DIFFERENT detail
         (one gives the time, the other the method; one names the victim, the other the place).
         Link only if the details complement each other. Identical restatements are not edges.
         Same event requires the same modality and polarity: "planned to sell" and "sold" are NOT the same event,
         and neither are "did not attend" and "attended". See BRIDGING and COMPARISON for those.
  SUBEVENT: B is one part or step of the larger event A (spatially and temporally inside it, and part of its
         script). Test: "B happened as part of A."
         Common mistake: when one fact is more specific (a fire on one street vs. a fire in the whole town, an
         attack in one district vs. an attack in the country), it is usually a SUBEVENT, not the same event.
  RECURRENCE: B is a repeat or return of the kind of event in A (it happened again, came back, resumed).
  GENERIC vs. SPECIFIC: A states a habit, rule or typical pattern; B is one particular occurrence of it
         (or the reverse).

-----------------------------------------------------------------------------------------------------
2.3 EXPANSION: the two facts fit together into a larger whole that may never be named in the text
-----------------------------------------------------------------------------------------------------
  INSTANCE (concept → instance, set → member): A is a general or collective statement; B is a specific case or
         member of it. Test: "B is an example of A."
  CONJUNCTION: A and B are two parts of the same larger event and play SIMILAR roles in it (two teams searching
         for the same missing child, two witnesses confirming the same alibi).
  DISJUNCTION: A and B are two parts of the same larger event but play DIFFERENT or opposing roles in it (the one
         who made the offer and the one who refused it; the attacker's move and the defender's response).

-----------------------------------------------------------------------------------------------------
2.4 COMPARISON: the two facts clash, cancel, compete or contrast
-----------------------------------------------------------------------------------------------------
  OPPOSITE / CONTRADICTION: A and B cannot both be true (conflicting reports, a lie and the truth, two different
         values for the same thing).
  NEGATION / ENDING: both were true, but B shows that A is NO LONGER true (a job lost, a promise broken, a state
         reversed, a belief abandoned).
  COMPETING ACCOUNTS: A and B are rival versions or framings of the same underlying event (one side calls it a
         "withdrawal", the other says the soldiers "fled").
  CONCESSION / EXPECTATION FAILED: A sets up an expected outcome and B shows it was not achieved, or was achieved
         despite an obstacle. Test: "A was expected to lead to X, but B."
  CONTRAST: A and B differ in a way the text treats as meaningful (two characters' opposite choices in the same
         situation, before vs. after).

-----------------------------------------------------------------------------------------------------
2.5 TEMPORALITY: when the order or timing itself is the new understanding
-----------------------------------------------------------------------------------------------------
Use these only when the text states or clearly implies the timing AND knowing it matters (it orders a sequence of
events in a story, dates something that is otherwise undated, or frames one event inside another). Do not add a
temporal edge to a pair that already has a causal edge: a cause already implies its order.
  BEFORE / AFTER: A happens before B.
  IMMEDIATELY BEFORE (meets): B begins right as A ends, with no gap.
  DURING / CONTAINS: B happens inside the time span of A ("during the war", "while the storm raged").
  BEGINS ON / STARTS WITH: B marks the start of A, or they start together.
  ENDS ON / FINISHES WITH: B marks the end of A, or they end together.
  OVERLAP: the two partly overlap in time.
  SIMULTANEOUS: the two happen at exactly the same time.

-----------------------------------------------------------------------------------------------------
2.6 BRIDGING: the two facts are about related but not identical things
-----------------------------------------------------------------------------------------------------
  PART / WHOLE: what A describes is a physical or organisational part of what B describes, and the combination
         says something new (the leaking valve is part of the engine that failed).
  SET / MEMBER: one fact is about a group and the other about one member of it, and the membership matters.
  IDENTITY CLAIM: one fact names an unknown or described person or thing ("the masked man", "the anonymous
         donor"), and the other identifies, accuses or suspects a specific one. Label it as a claim or suspicion
         unless the text confirms it.
  PLAN → OUTCOME (different modality): A is a plan, hope, threat, rumour or possibility, and B is what actually
         happened, or the reverse. Say whether it was carried out, failed, came true or was disproved.

-----------------------------------------------------------------------------------------------------
2.7 WILDCARD: clearly and strongly related, but no family above fits
-----------------------------------------------------------------------------------------------------
Use WILDCARD when the combination clearly reveals something important, but none of the families above names it.
Typical cases:
  - a clue and what it reveals (combining the two exposes who did something, what was hidden, what is really
    going on);
  - a question and its answer, or a request and the reply;
  - a statement and what it says about the speaker (it shows they are lying, afraid or loyal);
  - a rule and the person it applies to, which together decide an outcome;
  - two facts that together pin down a number, a place or an identity that neither gives alone.
For WILDCARD you invent the label. It must name EXACTLY what the combination reveals, as specifically as possible.
A highly specific wildcard label is better than forcing the pair into a family that does not fit. Do not use
WILDCARD as a shortcut for a pair that fits a family above, or for a weak "these are related" feeling. If you
cannot name what is revealed, do not emit the edge.

=====================================================================================================
3. THE LABELS: SPECIFIC, SHORT, AND ABOUT WHAT IS REVEALED
=====================================================================================================
A label is NOT just the relation name. "CAUSES", "CONTRADICTS", "MOTIVATES", "PRECEDES", "ELABORATES_ON" on their
own are too vague, because they say a relation exists but not what it is about. Each label must say, briefly, WHAT
is caused, motivated, contradicted, enabled, prevented or revealed.

Format: UPPERCASE_WITH_UNDERSCORES, 2 to 6 words. Start with the relation word, then say what it is about:
  CAUSES_TWO_DAY_BLACKOUT          not   CAUSES
  FEAR_MOTIVATES_FLIGHT            not   MOTIVATES
  ENABLES_HIRING_AS_LAWYER         not   ENABLES
  CONTRADICTS_CAR_COLOR            not   CONTRADICTS
  SUBEVENT_OF_HARVEST_FESTIVAL     not   PART_OF
  REVEALS_SECRETARY_WROTE_LETTER   not   RELATED_TO

Rules for labels:
  - Use content words, not full names: CAUSES_TWO_DAY_BLACKOUT, not CAUSES_BRINDLEWICK_TOWN_TO_LOSE_POWER.
  - label_forward reads source → target ("source [label] target"). label_backward is the SAME relation read
    target → source, just as specific: CAUSES_TWO_DAY_BLACKOUT ↔ BLACKOUT_CAUSED_BY_STORM;
    FEAR_MOTIVATES_FLIGHT ↔ FLIGHT_MOTIVATED_BY_FEAR. Symmetric relations (contradiction, simultaneous,
    conjunction) may use the same label both ways.
  - Never label with an attribute of one entity (FOUNDED, BORN, LOCATED_AT, NAMED_AFTER, EMPLOYED_BY). These
    describe a single fact, not a relationship between two facts.
  - No long descriptions or explanations. The two labels are the only text the edge will ever have, so they must
    carry the meaning on their own.

=====================================================================================================
4. RELEVANCE SCORE (0.0 - 1.0): HOW MUCH NEW UNDERSTANDING THE COMBINATION GIVES
=====================================================================================================
  0.9-1.0  Essential: the combination answers a why / how / who / what-next question that neither fact can answer
           alone (a cause and its effect, a motive and the act, a clue and what it reveals, a claim and its
           contradiction).
  0.6-0.8  Clear and useful: a real gain in understanding (a subevent placed in its larger event, an instance of a
           general statement, a dated event placed inside another).
  0.3-0.5  Weak but real: some gain, but a reader would rarely need it.
  Below 0.3: do not emit the edge.

=====================================================================================================
5. GOOD EDGES: ONE OR MORE PER FAMILY
=====================================================================================================
CONTINGENCY
  1. A: "The storm knocked down the power lines outside Brindlewick."
     B: "Brindlewick had no electricity for two days."
     → CONTINGENCY | CAUSES_TWO_DAY_BLACKOUT / BLACKOUT_CAUSED_BY_STORM | 0.95
     Gain: why the town was dark, and what the storm did.
  2. A: "Ilse Marn passed the bar exam in May."
     B: "Ilse Marn was hired as a lawyer at the Quenby firm in July."
     → CONTINGENCY | ENABLES_HIRING_AS_LAWYER / HIRING_REQUIRED_BAR_PASS | 0.8
     Gain: the exam made the job possible, but did not force it (precondition, not cause).
  3. A: "Halvard learned to pick locks as a boy."
     B: "Halvard escaped from the locked cellar."
     → CONTINGENCY | SKILL_ENABLES_CELLAR_ESCAPE / ESCAPE_USED_LOCKPICKING_SKILL | 0.9
     Gain: how he escaped. The text has no "because"; the condition is implicit, and that is exactly the kind of
     edge to look for.
  4. A: "The guard dog barked when the stranger reached the gate."
     B: "The stranger never entered the house."
     → CONTINGENCY | BARKING_PREVENTS_BREAK_IN / BREAK_IN_PREVENTED_BY_DOG | 0.85
  5. A: "Mara Okafor learned that her brother had secretly sold the family farm."
     B: "Mara Okafor did not speak to her brother for a year."
     → CONTINGENCY | ANGER_MOTIVATES_YEAR_OF_SILENCE / SILENCE_MOTIVATED_BY_FARM_SALE | 0.95
     Gain: why she stopped speaking to him. The label names the feeling (anger) and what it led to.
  6. A: "Tamsin Rook took a night job at the docks."
     B: "Tamsin Rook was saving money to pay for her mother's surgery."
     → CONTINGENCY | JOB_FUNDS_MOTHERS_SURGERY / SURGERY_SAVINGS_PURPOSE_OF_JOB | 0.9

INHERITANCE
  7. A: "The Pell Harbor museum was robbed on Tuesday night."
     B: "Thieves entered the Pell Harbor museum through a skylight and took two paintings."
     → INHERITANCE | SAME_ROBBERY_METHOD_AND_LOOT / SAME_ROBBERY_DATE | 0.8
     Gain: one fact gives when, the other gives how and what. Same event, complementary details.
  8. A: "Brindlewick held a three-day harvest festival in October."
     B: "A pie contest was held on the second day of the Brindlewick harvest festival."
     → INHERITANCE | INCLUDES_PIE_CONTEST / SUBEVENT_OF_HARVEST_FESTIVAL | 0.7
  9. A: "The Orsolo River flooded the lower town in 1952."
     B: "The Orsolo River flooded the lower town again in 1987."
     → INHERITANCE | FLOOD_RECURS_1987 / REPEAT_OF_1952_FLOOD | 0.7
  10. A: "Every spring the farmers of the valley burn the old stubble on their fields."
      B: "In April Halvard burned the stubble on his east field."
      → INHERITANCE | HABIT_INSTANCE_STUBBLE_BURN / FOLLOWS_SPRING_BURNING_CUSTOM | 0.6

EXPANSION
  11. A: "Every shop on Dovecote Lane closed during the strike."
      B: "The bakery on Dovecote Lane shut its doors on Monday."
      → EXPANSION | INSTANCE_BAKERY_CLOSES_IN_STRIKE / PART_OF_STRIKE_CLOSURES | 0.75
      Gain: why the bakery closed (it was part of the strike).
  12. A: "Volunteers from the north village searched the forest for the missing girl."
      B: "A police team searched the riverbank for the missing girl."
      → EXPANSION | JOINT_SEARCH_FOR_GIRL / JOINT_SEARCH_FOR_GIRL | 0.6
      Conjunction: same larger event (the search), similar roles.
  13. A: "Karl Reiss offered to buy the mill for twice its value."
      B: "The mill's owner refused Karl Reiss's offer."
      → EXPANSION | OFFER_REFUSED_BY_OWNER / REFUSES_REISS_MILL_OFFER | 0.85
      Disjunction: same transaction, opposite roles.

COMPARISON
  14. A: "A witness said the getaway car was blue."
      B: "The police report described the getaway car as red."
      → COMPARISON | CONTRADICTS_CAR_COLOR / CONTRADICTS_CAR_COLOR | 0.9
  15. A: "Tamsin Rook was the mayor's chief adviser."
      B: "The mayor dismissed Tamsin Rook in March."
      → COMPARISON | ENDED_BY_DISMISSAL / ENDS_ADVISER_ROLE | 0.85
      Negation: both true, but B ends A.
  16. A: "The general called the army's move a planned withdrawal."
      B: "Reporters described the soldiers fleeing in disorder."
      → COMPARISON | RIVAL_ACCOUNT_OF_RETREAT / RIVAL_ACCOUNT_OF_RETREAT | 0.85
  17. A: "Doctors expected the new treatment to cure Halvard's fever."
      B: "Halvard's fever returned a week after the treatment."
      → COMPARISON | EXPECTED_CURE_FAILED / SHOWS_TREATMENT_FAILED | 0.9

TEMPORALITY
  18. A: "The bridge collapsed during the storm on the night of 3 March."
      B: "The ferry resumed service the morning after the bridge collapsed."
      → TEMPORALITY | PRECEDES_FERRY_RESTART / FOLLOWS_BRIDGE_COLLAPSE | 0.6
      (If the text says the ferry restarted BECAUSE of the collapse, use CONTINGENCY instead.)
  19. A: "The siege of the fortress lasted six months."
      B: "The commander's son was born inside the fortress during the siege."
      → TEMPORALITY | CONTAINS_SONS_BIRTH / BORN_DURING_SIEGE | 0.7

BRIDGING
  20. A: "A masked man robbed the Quenby bank on Friday."
      B: "Police arrested Karl Reiss as a suspect in the Quenby bank robbery."
      → BRIDGING | SUSPECT_FOR_MASKED_ROBBER / MASKED_ROBBER_SUSPECTED_REISS | 0.9
      Identity claim: the label says "suspect", because the text does not confirm it.
  21. A: "Mara Okafor planned to sell the farm in the spring."
      B: "Mara Okafor sold the farm in June."
      → BRIDGING | PLAN_CARRIED_OUT_FARM_SALE / CARRIES_OUT_SALE_PLAN | 0.7
      Different modality (plan vs. actual): not SAME EVENT.
  22. A: "The ship's engine failed off the coast."
      B: "A cracked valve in the ship's cooling system had been leaking for weeks."
      → BRIDGING | FAILURE_TRACED_TO_LEAKING_VALVE / LEAK_EXPLAINS_ENGINE_FAILURE | 0.85
      (If the text states the leak caused the failure, CONTINGENCY is also correct.)

WILDCARD
  23. A: "The anonymous letter was written in green ink."
      B: "Only the governor's secretary used green ink in the office."
      → WILDCARD | INK_POINTS_TO_SECRETARY / SECRETARY_LIKELY_WROTE_LETTER | 0.95
      Gain: who probably wrote the letter, which neither fact says alone.
  24. A: "Ilse Marn asked the clerk where her father had been buried."
      B: "The clerk told Ilse Marn that her father was buried in the north churchyard."
      → WILDCARD | ANSWERED_BURIAL_IN_NORTH_CHURCHYARD / ANSWERS_BURIAL_PLACE_QUESTION | 0.85
  25. A: "Karl Reiss told the inspector he was at home all evening."
      B: "A neighbour saw Karl Reiss at the docks at nine that evening."
      → WILDCARD | ALIBI_EXPOSED_AS_LIE / BREAKS_REISS_HOME_ALIBI | 0.95
      (COMPARISON/CONTRADICTION also fits, but the label must say what is revealed: the alibi is false.)
  26. A: "The lighthouse keeper logged every ship that passed the point."
      B: "The lighthouse log has no entry for the night of the wreck."
      → WILDCARD | LOG_GAP_ON_WRECK_NIGHT / UNEXPLAINED_GAP_IN_FULL_LOG | 0.85

=====================================================================================================
6. BAD EDGES: DO NOT DO THIS
=====================================================================================================
  B1. Shared entity only.
      A: "Mara Okafor grew up in Brindlewick."   B: "Mara Okafor owns a grey cat."
      ✗ any edge. Merging them gains nothing; both are just about Mara. The entity link already connects them.
  B2. Restatement.
      A: "The bakery opened in 1990."   B: "The bakery has been open since 1990."
      ✗ SAME_EVENT. Nothing new is added. This is a duplicate, not a relationship.
  B3. Vague label.
      A: storm knocked down power lines   B: two days without electricity
      ✗ CAUSES / CAUSED_BY                ✓ CAUSES_TWO_DAY_BLACKOUT / BLACKOUT_CAUSED_BY_STORM
      The relation is right, but the label must say WHAT is caused.
  B4. Entity attribute used as a label.
      A: "Karl Reiss founded the Quenby mill in 1950."   B: "The Quenby mill employs forty workers."
      ✗ FOUNDED / FOUNDING_YEAR. Those describe fact A alone. If there is no real relation between the two
      facts, emit nothing.
  B5. Coincidental dates.
      A: "The bakery opened in 1990."   B: "Halvard bought a bicycle in 1995."
      ✗ PRECEDES_BICYCLE_PURCHASE. One is earlier, but neither frames or affects the other.
  B6. Shallow category.
      A: "Brindlewick is a town."   B: "Pell Harbor is a town."
      ✗ SAME_CATEGORY. Sharing a type is not a relationship.
  B7. Presupposition mistaken for precondition.
      A: "Tamsin Rook married in 2001."   B: "Tamsin Rook's remarriage took place in 2015."
      ✗ ENABLES_REMARRIAGE. A remarriage presupposes an earlier marriage by definition; nothing is explained.
      If anything, this is TEMPORALITY or RECURRENCE (SECOND_MARRIAGE_AFTER_2001).
  B8. Subevent mistaken for the same event.
      A: "A fire spread through the old town."   B: "A fire destroyed the bakery on Dovecote Lane."
      ✗ SAME_FIRE (if the bakery fire was one part of the town fire)
      ✓ INCLUDES_BAKERY_DESTROYED / SUBEVENT_OF_OLD_TOWN_FIRE. The more specific fact is a part, not a copy.
  B9. Different modality treated as the same event.
      A: "Karl Reiss threatened to burn the barn."   B: "The barn burned down on Sunday."
      ✗ SAME_EVENT. A threat is not the act.
      ✓ BRIDGING THREAT_PRECEDES_BARN_FIRE / FIRE_FOLLOWS_REISS_THREAT, unless the text says Reiss set the fire,
      in which case CONTINGENCY or WILDCARD THREAT_CARRIED_OUT.
  B10. Same name, different story.
      A (story about a sea voyage): "Captain Halvard commanded the ship Orsolo."
      B (story about a city robbery): "Halvard was a retired police officer."
      ✗ any edge. Nothing in the context says these are the same person. Facts from different documents or
      unrelated contexts are usually not linked.
  B11. Wrong direction.
      A: "The storm knocked down the power lines."   B: "The town had no electricity."
      ✗ label_forward = BLACKOUT_CAUSED_BY_STORM when A is the source. label_forward must read source → target.
  B12. Wildcard as a dump.
      A: "Ilse Marn likes tea."   B: "Ilse Marn visited Pell Harbor."
      ✗ WILDCARD RELATED_DETAILS_ABOUT_ILSE. If you cannot name what the combination reveals, emit nothing.

=====================================================================================================
7. OUTPUT
=====================================================================================================
For each pair that passes the core test, emit one edge (one per pair: choose the most informative relation):
  - source_id, target_id: the integer IDs from the numbered list (never the same ID).
  - relation_family: CONTINGENCY | INHERITANCE | EXPANSION | COMPARISON | TEMPORALITY | BRIDGING | WILDCARD
  - label_forward: specific 2-6 word UPPERCASE_WITH_UNDERSCORES label, source → target (section 3).
  - label_backward: the same relation read target → source, just as specific.
  - relevance_score: 0.0-1.0 per section 4; never emit a pair below 0.3.
Do not write descriptions or explanations: the two labels are the only text the edge will ever have.
```
