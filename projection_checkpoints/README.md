# Projection checkpoints

Stage 1 loads one of these fixed projection-only checkpoints by dataset:

| Dataset | File | Source experiment |
| --- | --- | --- |
| Clothing | `clothing.pth` | `clothing_unified_mr0p5_fixed_stage12_seed2023_20260718` |
| Beauty | `beauty.pth` | `beauty_unified_mr0p5_seed2023_hsearch_20260717` |
| Sports | `sports.pth` | `amazon_single_modality_2gpu_20260715_sports` |

Each checkpoint contains four tensors:

- `comp_proj_v.weight`
- `comp_proj_v.bias`
- `comp_proj_t.weight`
- `comp_proj_t.bias`

The hashes in `SHA256SUMS` identify the exact files used by the retained
Stage 1 result lineage.
