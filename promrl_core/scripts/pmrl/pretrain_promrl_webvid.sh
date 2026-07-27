
# 
CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-webvid.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_webvid \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 16 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_webvid \
--feature_inference false





# 10/31/2025 14:16:40 - INFO - __main__ -   {'video_r1': 52.1, 'video_recall': '52.1/71.3/78.2', 'video_ravg': 67.2, 'txt_r1': 51.8, 'txt_recall': '51.8/73.4/79.8', 'txt_ravg': 68.3}