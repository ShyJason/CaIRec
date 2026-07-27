# Decoupled Latent Adapter Variant

This folder packages the adapter-based MMRec variant without changing the
upstream checkout by default. It is intended for moving the current adapter
line to another machine through GitHub.

## What This Variant Contains

- Decoupled completion and recommendation projection spaces.
- Frozen completion-side projection/imputer in Stage 2.
- Recommendation-side adapter for completed missing modality representations.
- Optional `completion_adapter_mode` values:
  - `linear_ln`
  - `identity`
  - `residual_mlp`
- Adapter alignment loss controlled by `gamma_align`.
- Clothing adapter mainline documentation and experiment references.

## Apply On Another Machine

From a clean clone of the repository:

```bash
git checkout -b adapter-decoupled-latent
bash variants/adapter_decoupled_latent/apply.sh
```

The script applies `adapter_decoupled_latent.patch` to the current branch. Use a
new branch so the upstream main branch remains untouched.

## Current Best Adapter Result

Current best adapter-line Clothing result:

```text
train missing_rate = 0.3
eval_missing_rate = 0.5
completion_adapter_mode = linear_ln
gamma_align = 0.00125
Recall@20 = 0.07886
NDCG@20 = 0.03524
best_epoch = 185
```

Evidence:

```text
exp_report/clothing/stage2_hparam_mr0p3_gamma000125_20260702_142715/log/run_20260702_142715.log
```

## Notes

The stronger `residual_mlp` adapter is included as an option, but the current
search did not beat `linear_ln`. The best residual MLP strict result was:

```text
hidden = 256
dropout = 0.0
Recall@20 = 0.07801
NDCG@20 = 0.03473
```

