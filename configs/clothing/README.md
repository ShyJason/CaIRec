# Clothing configurations

`paper_stage1_1.yaml`, `paper_stage1_2.yaml`, and `paper_stage2.yaml` are the
complete Clothing pipeline. The recorded paper result uses `unified_static`,
train/eval missing rate `0.5`, payload seed `2023`, model seed `2023`,
mean modality fusion, and early stopping patience `50`.

Run `DATASET=clothing bash run_mmrec_mainline.sh` from the repository root to
execute Stage 1.1, Stage 1.2, and Stage 2 in order.
