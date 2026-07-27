#!/usr/bin/env bash
set -euo pipefail

echo "START $(date -Is)"
RUN_GPU="${RUN_GPU:-3}"
CUDA_VISIBLE_DEVICES="${RUN_GPU}" PYTHONUNBUFFERED=1 .venv/bin/python main.py \
  --config configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml \
  --suffix itemgraph_singlecore_gcnadapter_completed_cf03_img035_txt035_modala025_nocl_seed2023_20260615_gpu3 \
  --dataset clothing \
  --exp_mode mm \
  --device_id 0 \
  --seed 2023 \
  --dataset_seed 0 \
  --train_stage recommender \
  --freeze_imputer 1 \
  --freeze_decoder 1 \
  --disable_imputation 0 \
  --feature_bridge_mode raw_decoder \
  --gcn_frontend_mode original_linear \
  --imputer_ckpt exp_report/clothing/stage1_2_clothing_obsneg_debiased002_cl015_t01_r03_from_stage11e4_20260605/ckpt/stage1_2_clothing_obsneg_debiased002_cl015_t01_r03_from_stage11e4_20260605_imputer_backprop_50_epoch49.pth \
  --epoch 200 \
  --early_stop 10000 \
  --eva_interval 1 \
  --batch_size 2048 \
  --lr 0.01 \
  --lr_rec 0.01 \
  --lr_imp 0.0002 \
  --lr_decoder 0.00005 \
  --reg_coeff 0.01 \
  --penalty_coeff 1 \
  --max_info_coeff 0.01 \
  --min_info_coeff 0.000001 \
  --modality_bpr_coeff 1.0 \
  --evaluation_protocol strict \
  --selection_mode val \
  --strict_probe_test_interval 0 \
  --save 1 \
  --topk "[10, 20, 30, 40, 50]" \
  --rec_neighbor_cl_weight 0 \
  --rec_neighbor_cl_temp 0.2 \
  --rec_neighbor_cl_bank_size 256 \
  --item_graph_kind fused_completed \
  --item_graph_topk 20 \
  --item_graph_norm rw \
  --item_graph_cf_weight 0.3 \
  --item_graph_image_weight 0.35 \
  --item_graph_text_weight 0.35 \
  --item_graph_audio_weight 0 \
  --item_graph_modal_alpha 0.25 \
  --item_graph_modal_layers 1 \
  --item_graph_modal_target all
echo "EXIT $? $(date -Is)"
