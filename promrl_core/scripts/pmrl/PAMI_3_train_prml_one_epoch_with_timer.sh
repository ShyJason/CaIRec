CUDA_VISIBLE_DEVICES=1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/pretrain_cfg/pretrain_pmrl.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/PAMI_3_train_prml_one_epoch_with_timer \
--model_type pmrl_PAMI \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name PAMI_3_train_prml_one_epoch_with_timer \
--timer true 

CUDA_VISIBLE_DEVICES=0,2 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/pretrain_cfg/pretrain_pmrl.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/PAMI_3_train_prml_one_epoch_with_timer \
--model_type GRAM_PAMI \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name PAMI_3_train_prml_one_epoch_with_timer 


# Also GRAM: forward time, backward time, DET time.
# set timer: forward time, backward time, w/wo SVD time.
