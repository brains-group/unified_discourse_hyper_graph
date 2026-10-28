| variant | view | n | Accuracy | Acc-HARD | Acc-EASY | context tokens mean / max | overrides |
|---|---|---|---|---|---|---|---|
| base | full | 181 | 69.6 | 64.2 | 77.8 | 1450 / 1999 | {} |
| f60 | full | 181 | 73.5 | 67.0 | 83.3 | 1832 / 2000 | {"force_top_facts": 60} |
| f80 | full | 181 | 72.4 | 65.1 | 83.3 | 1972 / 2000 | {"force_top_facts": 80, "final_top_k": 120} |
| s50 | full | 181 | 69.6 | 62.4 | 80.6 | 1778 / 2000 | {"top_k_seeds": 50} |
| s50_f60 | full | 181 | 68.5 | 58.7 | 83.3 | 1896 / 2000 | {"top_k_seeds": 50, "force_top_facts": 60} |
| seed10 | full | 181 | 69.1 | 63.3 | 77.8 | 1458 / 2000 | {"seed_facts": 10} |
| seed20 | full | 181 | 66.8 | 56.9 | 81.9 | 1472 / 1998 | {"seed_facts": 20, "fact_candidates": 100} |
| rel300_f60 | full | 181 | 71.8 | 63.3 | 84.7 | 1834 / 2000 | {"relevant_facts": 300, "ce_fact_pool": 200, "force_top_facts": 60} |
| dmp05 | full | 181 | 69.1 | 61.5 | 80.6 | 1447 / 2000 | {"damping": 0.5} |
| dmp07 | full | 181 | 69.1 | 62.4 | 79.2 | 1454 / 2000 | {"damping": 0.7} |
| r2 | full | 181 | 72.9 | 66.1 | 83.3 | 1445 / 2000 | {"ppr_rounds": 2} |
| d3 | full | 181 | 71.8 | 67.0 | 79.2 | 1449 / 1998 | {"max_depth": 3} |
| d5_b8 | full | 181 | 70.7 | 63.3 | 81.9 | 1489 / 1998 | {"max_depth": 5, "beam_width": 8} |
| wide_t3000 | full | 181 | 70.7 | 61.5 | 84.7 | 2570 / 3000 | {"top_k_seeds": 50, "seed_facts": 10, "relevant_facts": 300, "ce_fact_pool": 200, "force_top_facts": 80, "final_top_k": 150, "context_tokens": 3000} |
