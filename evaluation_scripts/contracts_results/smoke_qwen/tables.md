# Insurance contracts: smoke set

Scores in percent. QA: mean judge score (1.0 / 0.5 / 0.0) per question; Pooled = over every question of the set, Macro = mean of the per-contract scores. Fact recall: share of facts the judge finds supported by the retrieved context. Ctx tokens: mean retrieved-context length (cl100k).

## QA

| System | CAT1 | CAT2 | CAT3 | CAT4 | CAT5 | All | All excl. CAT5 | Macro | Ctx tokens | n |
|---|---|---|---|---|---|---|---|---|---|---|
| hipporag2 | 100.0 | 87.5 | 100.0 | 100.0 | 100.0 | 97.5 | 96.9 | 97.5 | 3561.3 | 20 |
| logicrag | 100.0 | 87.5 | 100.0 | 100.0 | 50.0 | 87.5 | 96.9 | 87.5 | 137.3 | 20 |
| memgraphrag | 100.0 | 87.5 | 87.5 | 75.0 | 100.0 | 90.0 | 87.5 | 90.0 | 1208.8 | 20 |
| hypergraphrag | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 13977.0 | 20 |
| ms_graphrag | 100.0 | 100.0 | 75.0 | 100.0 | 100.0 | 95.0 | 93.8 | 95.0 | 7483.1 | 20 |
| hygrag | 100.0 | 62.5 | 100.0 | 100.0 | 100.0 | 92.5 | 90.6 | 92.5 | 3597.0 | 20 |

## Fact recall

| System | Pooled | Macro | contract_1_term_life | 2025_pdp_contract | Ctx tokens | n |
|---|---|---|---|---|---|---|
| hipporag2 | 100.0 | 100.0 | 100.0 | 100.0 | 3526.2 | 20 |
| logicrag | 100.0 | 100.0 | 100.0 | 100.0 | 223.4 | 20 |
| memgraphrag | 90.0 | 90.0 | 100.0 | 80.0 | 1200.0 | 20 |
| hypergraphrag | 100.0 | 100.0 | 100.0 | 100.0 | 14404.1 | 20 |
| ms_graphrag | 100.0 | 100.0 | 100.0 | 100.0 | 7543.1 | 20 |
| hygrag | 100.0 | 100.0 | 100.0 | 100.0 | 3472.4 | 20 |

## QA per contract

### contract_1_term_life

| System | CAT1 | CAT2 | CAT3 | CAT4 | CAT5 | All | All excl. CAT5 | Fact recall | Ctx tokens | n |
|---|---|---|---|---|---|---|---|---|---|---|
| hipporag2 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 1399.0 | 10 |
| logicrag | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 80.0 | 100.0 | 100.0 | 119.3 | 10 |
| memgraphrag | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 1222.2 | 10 |
| hypergraphrag | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 8205.0 | 10 |
| ms_graphrag | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 6136.1 | 10 |
| hygrag | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 1932.4 | 10 |

### 2025_pdp_contract

| System | CAT1 | CAT2 | CAT3 | CAT4 | CAT5 | All | All excl. CAT5 | Fact recall | Ctx tokens | n |
|---|---|---|---|---|---|---|---|---|---|---|
| hipporag2 | 100.0 | 75.0 | 100.0 | 100.0 | 100.0 | 95.0 | 93.8 | 100.0 | 5723.6 | 10 |
| logicrag | 100.0 | 75.0 | 100.0 | 100.0 | 100.0 | 95.0 | 93.8 | 100.0 | 155.4 | 10 |
| memgraphrag | 100.0 | 75.0 | 75.0 | 50.0 | 100.0 | 80.0 | 75.0 | 80.0 | 1195.3 | 10 |
| hypergraphrag | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 19748.9 | 10 |
| ms_graphrag | 100.0 | 100.0 | 50.0 | 100.0 | 100.0 | 90.0 | 87.5 | 100.0 | 8830.1 | 10 |
| hygrag | 100.0 | 25.0 | 100.0 | 100.0 | 100.0 | 85.0 | 81.2 | 100.0 | 5261.6 | 10 |

## Diagnostics

| System | Empty contexts | Abstained on CAT1-4 | Judge unparsed | Reader hit max_tokens | Reader without 'Answer:' |
|---|---|---|---|---|---|
| hipporag2 | 0 | 1 | 0 | 0 | 0 |
| logicrag | 0 | 0 | 0 | 0 | 0 |
| memgraphrag | 0 | 2 | 0 | 0 | 0 |
| hypergraphrag | 0 | 0 | 0 | 0 | 0 |
| ms_graphrag | 0 | 0 | 0 | 0 | 0 |
| hygrag | 0 | 1 | 0 | 0 | 0 |
