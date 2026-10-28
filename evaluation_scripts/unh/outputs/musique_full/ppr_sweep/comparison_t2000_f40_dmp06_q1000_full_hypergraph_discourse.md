| system | n | EM | F1 | Contain-Acc | LLM-Acc | context tokens mean / max |
|---|---|---|---|---|---|---|
| HyperGraphRAG | 1000 | 44.5 | 56.4 | 51.7 | 62.1 | 22597 / 30179 |
| UNH PPR (UNH-H, hypergraph view) | 1000 | 43.5 | 55.0 | 46.7 | 58.6 | 1538 / 2000 |
| UNH PPR (full view) | 1000 | 42.8 | 54.3 | 47.5 | 60.3 | 1589 / 2000 |
| HippoRAG 2 | 1000 | 39.8 | 50.6 | 43.9 | 53.9 | 719 / 1688 |
| MS GraphRAG | 1000 | 38.2 | 49.1 | 42.9 | 54.2 | 9598 / 11779 |
| LogicRAG | 1000 | 35.8 | 48.5 | 42.3 | 52.8 | 119 / 295 |
| UNH old retriever (full view) | 1000 | 36.4 | 47.2 | 40.7 | 50.1 | 1039 / 2513 |
| UNH PPR (UNH-D, discourse view) | 1000 | 34.6 | 45.9 | 39.1 | 51.0 | 1577 / 2000 |
| UNH old retriever (hypergraph view) | 1000 | 34.4 | 45.1 | 37.7 | 47.5 | 1120 / 2049 |
| HyGRAG | 1000 | 34.1 | 44.5 | 37.1 | 47.7 | 1101 / 1825 |
| MemGraphRAG | 1000 | 30.9 | 40.5 | 34.4 | 42.9 | 1307 / 1309 |
| UNH old retriever (discourse view) | 1000 | 23.6 | 33.3 | 27.6 | 35.3 | 582 / 1591 |

UNH PPR full path stats: {'pct_edges_discourse': 38.7, 'pct_edges_entity': 61.3, 'pct_hops_discourse': 54.5, 'pct_hops_entity': 45.5, 'mean_edges_per_question': 46.8, 'mean_distinct_facts': 52.7, 'answer_in_context_pct': 69.0, 'mean_tokens': 1589, 'mean_seconds': 17.27}
UNH PPR hypergraph path stats: {'pct_edges_discourse': 0.0, 'pct_edges_entity': 100.0, 'pct_hops_discourse': 0.0, 'pct_hops_entity': 100.0, 'mean_edges_per_question': 47.2, 'mean_distinct_facts': 50.2, 'answer_in_context_pct': 68.8, 'mean_tokens': 1538, 'mean_seconds': 17.315}
UNH PPR discourse path stats: {'pct_edges_discourse': 100.0, 'pct_edges_entity': 0.0, 'pct_hops_discourse': 100.0, 'pct_hops_entity': 0.0, 'mean_edges_per_question': 38.3, 'mean_distinct_facts': 54.4, 'answer_in_context_pct': 58.1, 'mean_tokens': 1577, 'mean_seconds': 17.391}
