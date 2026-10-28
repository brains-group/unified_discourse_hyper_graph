| system | n | EM | F1 | Contain-Acc | LLM-Acc | context tokens mean / max |
|---|---|---|---|---|---|---|
| HyperGraphRAG | 250 | 59.6 | 74.8 | 74.0 | 85.6 | 22691 / 31665 |
| UNH PPR (t2000_f40_dmp06) | 250 | 60.4 | 74.1 | 71.6 | 86.4 | 1427 / 1999 |
| HippoRAG 2 | 250 | 59.6 | 71.9 | 70.8 | 80.4 | 662 / 1762 |
| LogicRAG | 250 | 56.4 | 71.3 | 68.8 | 81.6 | 70 / 270 |
| HyGRAG | 250 | 56.4 | 69.3 | 66.4 | 79.2 | 1056 / 2007 |
| UNH old retriever (full view) | 250 | 54.0 | 65.8 | 64.4 | 76.0 | 964 / 1866 |
| MS GraphRAG | 250 | 51.6 | 65.4 | 63.6 | 75.6 | 9737 / 11626 |
| MemGraphRAG | 250 | 50.4 | 61.4 | 61.6 | 68.4 | 1307 / 1309 |
| UNH old retriever (hypergraph view) | 250 | 46.8 | 56.6 | 56.0 | 64.8 | 1073 / 1623 |
| UNH old retriever (discourse view) | 250 | 45.2 | 56.6 | 56.8 | 64.8 | 734 / 1713 |

UNH PPR path edges: {'pct_edges_discourse': 41.5, 'pct_edges_entity': 58.5, 'pct_hops_discourse': 57.0, 'pct_hops_entity': 43.0, 'mean_edges_per_question': 41.5, 'mean_distinct_facts': 50.2, 'answer_in_context_pct': 86.0, 'mean_seconds': 2.413}
