CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/pretrain_cfg/pretrain_pmrl.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/PAMI_1_train_prml \
--model_type pmrl_PAMI \
--tau1 0.05 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name PAMI_1_train_prml 


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./outputs/PAMI_1_train_prml/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type pmrl_PAMI \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./output/retrieval_pami_all_zeroshot \
--log_name test_pmrl_pami_all_zeroshot \
--feature_inference false


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot_ori.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./outputs/PAMI_1_train_prml/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type pmrl_PAMI \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./output/retrieval_pami_all_zeroshot_ori \
--log_name test_pmrl_pami_all_zeroshot \
--feature_inference false

# 1. set the larger temperature for the contrastive loss, so that enhance the effect of the first singular value.
