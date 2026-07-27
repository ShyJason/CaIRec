
# 

CUDA_SET="${CUDA_VISIBLE_DEVICES:-3,4,5,6}"
NPROC="${NPROC_PER_NODE:-4}"
BASE_PMCL_CKPT="./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt"

CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node "${NPROC}" \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps_msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_triangle_audiocaps_msrvtt \
--checkpoint ./triangle_pretraining/ckpt/model_step_200.pt \
--model_type triangle \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_triangle_audiocaps_msrvtt \
--feature_inference false


CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node "${NPROC}" \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps_msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_gram_audiocaps_msrvtt \
--checkpoint ./SVD_MM/GRAM/GRAM_pretrained_4modalities/ckpt/base.pt \
--model_type gram \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_gram_audiocaps_msrvtt \
--feature_inference false

CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node "${NPROC}" \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps_msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_audiocaps_msrvtt_3modalities \
--checkpoint "${BASE_PMCL_CKPT}" \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_audiocaps_msrvtt_3modalities \
--feature_inference false



CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node "${NPROC}" \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps_msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_vast_audiocaps_msrvtt \
--model_type vast \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_vast_audiocaps_msrvtt \
--feature_inference false

CUDA_VISIBLE_DEVICES="${CUDA_SET}" torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node "${NPROC}" \
--master_port 9836 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps_msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_pmrl_audiocaps_msrvtt_2 \
--checkpoint "${BASE_PMCL_CKPT}" \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_pmrl_audiocaps_msrvtt_2 \
--feature_inference false

# outputs/pretrain_promrl_audiocaps
