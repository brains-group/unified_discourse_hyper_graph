| system | n | EM | F1 | Contain-Acc | LLM-Acc | context tokens mean / max |
|---|---|---|---|---|---|---|
| HyperGraphRAG | 1000 | 44.5 | 56.4 | 51.7 | 62.1 | 22597 / 30179 |
| UNH PPR (full view) | 1000 | 45.2 | 54.6 | 49.3 | 59.0 | 2316 / 3752 |
| HippoRAG 2 | 1000 | 39.8 | 50.6 | 43.9 | 53.9 | 719 / 1688 |
| MS GraphRAG | 1000 | 38.2 | 49.1 | 42.9 | 54.2 | 9598 / 11779 |
| LogicRAG | 1000 | 35.8 | 48.5 | 42.3 | 52.8 | 119 / 295 |
| UNH old retriever (full view) | 1000 | 36.4 | 47.2 | 40.7 | 50.1 | 1039 / 2513 |
| UNH old retriever (hypergraph view) | 1000 | 34.4 | 45.1 | 37.7 | 47.5 | 1120 / 2049 |
| HyGRAG | 1000 | 34.1 | 44.5 | 37.1 | 47.7 | 1101 / 1825 |
| MemGraphRAG | 1000 | 30.9 | 40.5 | 34.4 | 42.9 | 1307 / 1309 |
| UNH old retriever (discourse view) | 1000 | 23.6 | 33.3 | 27.6 | 35.3 | 582 / 1591 |

UNH PPR full path stats: {'pct_edges_discourse': 38.7, 'pct_edges_entity': 61.3, 'pct_hops_discourse': 54.5, 'pct_hops_entity': 45.5, 'mean_edges_per_question': 46.8, 'mean_distinct_facts': 52.7, 'answer_in_context_pct': 72.9, 'mean_tokens': 2316, 'mean_seconds': 37.319}
