# Insurance contracts: smoke_qa set

Scores in percent. QA: mean judge score (1.0 / 0.5 / 0.0) per question; Pooled = over every question of the set, Macro = mean of the per-contract scores. Fact recall: share of facts the judge finds supported by the retrieved context. Ctx tokens: mean retrieved-context length (cl100k).

## QA

| System | CAT1 | CAT2 | CAT3 | CAT4 | CAT5 | All | All excl. CAT5 | Macro | Ctx tokens | n |
|---|---|---|---|---|---|---|---|---|---|---|
| hipporag2 (partial) | 100.0 | 100.0 | 100.0 | 50.0 | 100.0 | 90.0 | 87.5 | 90.0 | 5723.6 | 10 |
| logicrag (partial) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 109.7 | 10 |
| memgraphrag (partial) | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 | 90.0 | 87.5 | 90.0 | 1196.5 | 10 |
| hypergraphrag (partial) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 18831.6 | 10 |
| ms_graphrag (partial) | 50.0 | 100.0 | 100.0 | 100.0 | 100.0 | 90.0 | 87.5 | 90.0 | 9110.0 | 10 |
| hygrag (partial) | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 | 90.0 | 87.5 | 90.0 | 5284.1 | 10 |

## QA per contract

### 2025_pdp_contract

| System | CAT1 | CAT2 | CAT3 | CAT4 | CAT5 | All | All excl. CAT5 | Fact recall | Ctx tokens | n |
|---|---|---|---|---|---|---|---|---|---|---|
| hipporag2 | 100.0 | 100.0 | 100.0 | 50.0 | 100.0 | 90.0 | 87.5 | - | 5723.6 | 10 |
| logicrag | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | - | 109.7 | 10 |
| memgraphrag | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 | 90.0 | 87.5 | - | 1196.5 | 10 |
| hypergraphrag | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | - | 18831.6 | 10 |
| ms_graphrag | 50.0 | 100.0 | 100.0 | 100.0 | 100.0 | 90.0 | 87.5 | - | 9110.0 | 10 |
| hygrag | 100.0 | 50.0 | 100.0 | 100.0 | 100.0 | 90.0 | 87.5 | - | 5284.1 | 10 |

## Diagnostics

| System | Empty contexts | Abstained on CAT1-4 | Judge unparsed | Reader hit max_tokens | Reader without 'Answer:' |
|---|---|---|---|---|---|
| hipporag2 | 0 | 1 | 0 | 0 | 0 |
| logicrag | 0 | 0 | 0 | 0 | 0 |
| memgraphrag | 0 | 1 | 0 | 0 | 0 |
| hypergraphrag | 0 | 0 | 0 | 0 | 0 |
| ms_graphrag | 0 | 1 | 0 | 0 | 0 |
| hygrag | 0 | 1 | 0 | 0 | 0 |
