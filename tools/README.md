# Tools

Utility scripts kept outside the main training entrypoints.

- `evaluate_assemble_score_fusion.py`
  - mainline assemble-score evaluation entrypoint
  - computes `score_final = (1-alpha) * norm(score_base) + alpha * norm(score_aux)`
  - wraps `evaluate_late_score_fusion.py` for backward compatibility
- `evaluate_late_score_fusion.py`
  - legacy name for assemble-score evaluation used by earlier experiment scripts
- `evaluate_imputation_metrics.py`
  - offline imputation-quality evaluation
- `diagnose_imputation_vs_zero.py`
  - diagnostic comparison between zero-filled and imputed pathways
