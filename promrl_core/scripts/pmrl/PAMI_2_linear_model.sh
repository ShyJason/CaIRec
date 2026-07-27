# CUDA_VISIBLE_DEVICES=1,1 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 4 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 1e-5 \
# --checkpointing true \
# --first_eval false \
# --save_best true \
# --config ./config/pmrl/pretrain_cfg/pretrain_pmrl.json \
# --pretrain_dir ./pretrain_vast \
# --output_dir outputs/PAMI_1_train_prml \
# --model_type pmrl_PAMI \
# --tau1 1 \ 
# --tau2 0.1 \
# --lambda_itm 0.1 \
# --log_name PAMI_1_train_prml 


# # 1. set the larger temperature for the contrastive loss, so that enhance the effect of the first singular value.
