| system | n | EM | F1 | Contain-Acc | LLM-Acc | context tokens mean / max |
|---|---|---|---|---|---|---|
| UNH PPR (UNH-H, hypergraph view) | 1000 | 62.2 | 75.2 | 71.9 | 85.0 | 1362 / 1999 |
| HyperGraphRAG | 1000 | 61.4 | 74.9 | 74.5 | 86.4 | 22443 / 31665 |
| UNH PPR (full view) | 1000 | 61.3 | 74.8 | 72.2 | 85.4 | 1435 / 2000 |
| HippoRAG 2 | 1000 | 61.3 | 73.5 | 70.6 | 82.8 | 661 / 2316 |
| LogicRAG | 1000 | 58.2 | 72.8 | 70.2 | 82.9 | 72 / 279 |
| UNH PPR (UNH-D, discourse view) | 1000 | 59.2 | 72.3 | 69.4 | 82.8 | 1577 / 2000 |
| MS GraphRAG | 1000 | 56.4 | 69.2 | 67.1 | 79.6 | 9718 / 11700 |
| UNH old retriever (full view) | 1000 | 54.5 | 67.5 | 66.2 | 77.8 | 950 / 1866 |
| HyGRAG | 1000 | 54.1 | 67.4 | 64.8 | 78.6 | 1044 / 2136 |
| MemGraphRAG | 1000 | 51.2 | 62.6 | 60.9 | 70.6 | 1306 / 1309 |
| UNH old retriever (hypergraph view) | 1000 | 49.1 | 60.5 | 59.6 | 70.2 | 1068 / 1963 |
| UNH old retriever (discourse view) | 1000 | 47.9 | 59.6 | 57.1 | 67.4 | 717 / 1713 |

UNH PPR full path stats: {'pct_edges_discourse': 41.3, 'pct_edges_entity': 58.7, 'pct_hops_discourse': 57.0, 'pct_hops_entity': 43.0, 'mean_edges_per_question': 42.0, 'mean_distinct_facts': 50.2, 'answer_in_context_pct': 85.2, 'mean_tokens': 1435, 'mean_seconds': 11.302}
UNH PPR hypergraph path stats: {'pct_edges_discourse': 0.0, 'pct_edges_entity': 100.0, 'pct_hops_discourse': 0.0, 'pct_hops_entity': 100.0, 'mean_edges_per_question': 42.0, 'mean_distinct_facts': 47.0, 'answer_in_context_pct': 85.0, 'mean_tokens': 1362, 'mean_seconds': 16.39}
UNH PPR discourse path stats: {'pct_edges_discourse': 100.0, 'pct_edges_entity': 0.0, 'pct_hops_discourse': 100.0, 'pct_hops_entity': 0.0, 'mean_edges_per_question': 40.6, 'mean_distinct_facts': 56.2, 'answer_in_context_pct': 80.1, 'mean_tokens': 1577, 'mean_seconds': 16.969}
