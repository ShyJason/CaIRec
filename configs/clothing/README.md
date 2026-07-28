# Clothing configurations

`paper_stage1_1.yaml`, `paper_stage1_2.yaml`, and `paper_stage2.yaml` are the
complete Clothing pipeline. The recorded paper result uses `unified_static`,
train/eval missing rate `0.5`, payload seed `2023`, model seed `2023`,
mean modality fusion, no Rec Neighbor contrastive loss, and early stopping
patience `50`.

Recall@20 `0.08141` and NDCG@20 `0.03612` at epoch `280` were produced by the
older posterior-reliability setting and are retained only as historical
reference. The current mean-fusion setting must be re-evaluated before a
replacement result is reported.

Use `reproduce_best/20260719/clothing.sh` from the repository root.
