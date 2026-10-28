"""[unh_v1] Minimal vendored subset of SAG-Benchmark's pipeline.evaluation.utils.

Only load_utils.py, sample_identity.py and reproduce_dataset.py are copied (verbatim). The upstream
package __init__ also imports mlflow/token-tracking helpers that the MS GraphRAG steps never use.
"""
