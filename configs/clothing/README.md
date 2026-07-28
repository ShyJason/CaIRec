# Clothing configurations

`paper_stage1_1.yaml`, `paper_stage1_2.yaml`, and `paper_stage2.yaml` are the
complete Clothing pipeline. The recorded paper result uses `unified_static`,
train/eval missing rate `0.5`, payload seed `2023`, model seed `2023`,
posterior reliability in fusion with scale `50`, and early stopping patience
`50`.

The recorded strict-test result is Recall@20 `0.08141` and NDCG@20 `0.03612`
at best epoch `280`.

Use `reproduce_best/20260719/clothing.sh` from the repository root.
