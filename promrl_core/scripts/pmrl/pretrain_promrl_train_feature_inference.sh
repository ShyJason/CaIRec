
# 
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
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_train_feature_inference \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--mode 'testing' \
--tau1 0.0 \
--tau2 0.1 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_train_feature_inference \
--feature_inference true


