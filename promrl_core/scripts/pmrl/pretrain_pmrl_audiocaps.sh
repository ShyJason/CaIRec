
# 
CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps2.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_pmrl_audiocaps \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_pmrl_audiocaps \
--feature_inference false



CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9839 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_pmrl_msrvtt_full \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_pmrl_msrvtt_full \
--feature_inference false


CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9839 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_pmrl_msrvtt_full2 \
--checkpoint ./outputs/pretrain_pmrl_msrvtt_full/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_pmrl_msrvtt_full2 \
--feature_inference false




# outputs/pretrain_pmrl_audiocaps