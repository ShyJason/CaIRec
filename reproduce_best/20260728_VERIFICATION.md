# Stage 2 verification — 2026-07-28

This verification used commit `3f40fa8`, Python 3.10, PyTorch
`1.13.0+cu117`, and three NVIDIA A40 GPUs. Each dataset used its fixed
`reproduce_best/20260719/*.sh` command, retained Stage 1.2 checkpoint, and
strict validation-based checkpoint selection.

| Dataset | Recorded Recall@20 | Verified Recall@20 | Delta | Recorded NDCG@20 | Verified NDCG@20 | Delta | Recorded / verified best epoch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Clothing | 0.08141 | 0.07947 | -0.00194 | 0.03612 | 0.03566 | -0.00046 | 280 / 303 |
| Beauty | 0.08418 | 0.08533 | +0.00115 | 0.03386 | 0.03439 | +0.00053 | 177 / 176 |
| Sports | 0.10579 | 0.10620 | +0.00041 | 0.04735 | 0.04761 | +0.00026 | 239 / 229 |

Beauty and Sports reproduce the recorded performance almost exactly and are
slightly higher in this run. Clothing remains in the same performance range
but is lower by `0.00194` Recall@20 (about 2.4% relative) and `0.00046`
NDCG@20 (about 1.3% relative). Therefore, the release reproduces the overall
reported performance, although Clothing is not bitwise or numerically exact.

Run tags:

- `verify_3f40fa8_clothing_20260728_132300`
- `verify_3f40fa8_beauty_20260728_132300`
- `verify_3f40fa8_sports_20260728_132300`

When the external `exp_report` asset directory is mounted at the repository
root, each complete log is located at:

```text
exp_report/<dataset>/reproduce_best_20260719/<run-tag>/<run-tag>.launch.log
```

During the first attempted run, full forward execution exposed a missing
`get_recommender_modal_features` method left by branch cleanup. Commit
`3f40fa8` restored the minimal recommendation feature bridge and added a
regression test before these three successful runs were started.
