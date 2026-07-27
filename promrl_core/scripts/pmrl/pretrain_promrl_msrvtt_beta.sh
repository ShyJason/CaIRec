
# 
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
--output_dir outputs/pretrain_promrl_msrvtt_beta0.1 \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_msrvtt_beta0.1 \
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
--output_dir outputs/pretrain_promrl_msrvtt_beta0.2 \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_msrvtt_beta0.2 \
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
--output_dir outputs/pretrain_promrl_msrvtt_beta1 \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_msrvtt_beta1 \
--feature_inference false


# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 4 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 1e-5 \
# --checkpointing true \
# --first_eval true \
# --save_best true \
# --config ./config/pmrl/finetune_cfg/retrieval-msrvtt2.json \
# --pretrain_dir ./pretrain_vast \
# --output_dir outputs/pretrain_pmrl_msrvtt \
# --checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
# --model_type pmrl \
# --tau1 0.05 \
# --tau2 0.1 \
# --valid_freq 8 \
# --lambda_itm 0.1 \
# --log_name pretrain_pmrl_msrvtt \
# --feature_inference false


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
--output_dir outputs/pretrain_triangle_msrvtt \
--checkpoint ./triangle_pretraining/ckpt/model_step_200.pt \
--model_type triangle \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_triangle_msrvtt \
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
--output_dir outputs/pretrain_gram_msrvtt \
--checkpoint ./SVD_MM/GRAM/GRAM_pretrained_4modalities/ckpt/base.pt \
--model_type gram \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_gram_msrvtt \
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
--config ./config/pmrl/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_pmrl_msrvtt_ori \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_pmrl_msrvtt_ori \
--feature_inference false

# outputs/pretrain_promrl_audiocaps