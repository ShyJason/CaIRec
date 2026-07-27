
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
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps2.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_triangle_audiocaps \
--checkpoint ./triangle_pretraining/ckpt/model_step_200.pt \
--model_type triangle \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_triangle_audiocaps \
--feature_inference false


# outputs/pretrain_gram_audiocaps
