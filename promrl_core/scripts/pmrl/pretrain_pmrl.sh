CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun \
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
--output_dir outputs/pretrain_pmrl_3_modalities \
--model_type pmrl \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name pretrain_pmrl_3_modalities 


CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun \
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
--checkpoint outputs/pretrain_pmrl_3_modalities/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/pmrl/retrieval-all-zeroshot_ori_3_modalities \
--log_name test_pmrl_all_zeroshot