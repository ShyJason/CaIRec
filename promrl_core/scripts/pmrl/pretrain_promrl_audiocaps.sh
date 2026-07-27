
# # ./outputs/pretrain_gram_audiocaps/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \

CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps2.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_audiocaps_3modalities_0.05_itm \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.05 \
--log_name pretrain_promrl_audiocaps_3modalities_0.05_itm \
--feature_inference false


CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-msrvtt2.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_msrvtt_3modalities_0_itm \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_msrvtt_3modalities_0_itm \
--feature_inference false



# CUDA_VISIBLE_DEVICES=3,4 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 2 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 1e-5 \
# --checkpointing true \
# --first_eval true \
# --save_best true \
# --config ./config/pmrl/finetune_cfg/retrieval-audiocaps2.json \
# --pretrain_dir ./pretrain_vast \
# --checkpoint ./outputs/pretrain_promrl_audiocaps/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
# --model_type promrl \
# --tau1 0.05 \
# --tau2 0.1 \
# --valid_freq 8 \
# --lambda_itm 0.1 \
# --output_dir outputs/pretrain_promrl_audiocaps2 \
# --log_name pretrain_promrl_audiocaps2 \
# --feature_inference false


# CUDA_VISIBLE_DEVICES=3,4 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 2 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 1e-5 \
# --checkpointing true \
# --first_eval true \
# --save_best true \
# --config ./config/pmrl/finetune_cfg/retrieval-msrvtt2.json \
# --pretrain_dir ./pretrain_vast \
# --output_dir outputs/pretrain_promrl_audiocaps_msrvtt2 \
# --checkpoint ./outputs/pretrain_promrl_audiocaps2/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
# --model_type promrl \
# --tau1 0.05 \
# --tau2 0.1 \
# --valid_freq 8 \
# --lambda_itm 0.1 \
# --log_name pretrain_promrl_audiocaps_msrvtt2 \
# --feature_inference false

