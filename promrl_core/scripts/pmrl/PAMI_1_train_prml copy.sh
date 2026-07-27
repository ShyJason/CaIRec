CUDA_VISIBLE_DEVICES=1,2 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run_efficiency.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/pretrain_cfg/pretrain_pmrl.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/PAMI_1_train_prml_efficiency \
--model_type pmrl \
--lambda_itm 0.1 \
--log_name PAMI_1_train_prml_efficiency 


CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run_efficiency.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/pretrain_cfg/pretrain_pmrl.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/PAMI_1_train_vast_efficiency \
--model_type vast \
--lambda_itm 0.1 \
--log_name PAMI_1_train_vast_efficiency 


CUDA_VISIBLE_DEVICES=1,2 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run_efficiency.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/pretrain_cfg/pretrain_pmrl.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/PAMI_1_train_gram_efficiency \
--model_type gram \
--lambda_itm 0.1 \
--log_name PAMI_1_train_gram_efficiency 


# 1. set the larger temperature for the contrastive loss, so that enhance the effect of the first singular value.
