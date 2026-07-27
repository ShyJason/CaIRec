# # CUDA_VISIBLE_DEVICES=0 torchrun \
# # --nnodes 1 \
# # --node_rank 0 \
# # --nproc_per_node 1 \
# # --master_port 9838 \
# # ./run.py \
# # --learning_rate 5e-5 \
# # --checkpointing true \
# # --first_eval false \
# # --config ./config/pmrl/pretrain_cfg/pretrain_promrl_test_feature_inference.json \
# # --pretrain_dir ./pretrain_vast \
# # --checkpoint outputs/pretrain_vast_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
# # --mode 'testing' \
# # --model_type promrl \
# # --save_best true \
# # --tau1 0.01 \
# # --tau2 0.1 \
# # --lambda_itm 0.1 \
# # --output_dir ./outputs/features_vast_mini \
# # --log_name features_vast_mini \
# # --feature_inference true 



# # CUDA_VISIBLE_DEVICES=0 torchrun \
# # --nnodes 1 \
# # --node_rank 0 \
# # --nproc_per_node 1 \
# # --master_port 9838 \
# # ./run.py \
# # --learning_rate 5e-5 \
# # --checkpointing true \
# # --first_eval false \
# # --config ./config/pmrl/pretrain_cfg/pretrain_promrl_test_feature_inference.json \
# # --pretrain_dir ./pretrain_vast \
# # --checkpoint outputs/pretrain_gram_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
# # --mode 'testing' \
# # --model_type promrl \
# # --save_best true \
# # --tau1 0.01 \
# # --tau2 0.1 \
# # --lambda_itm 0.1 \
# # --output_dir ./outputs/features_gram_mini \
# # --log_name features_gram_mini \
# # --feature_inference true 

# # ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt

# # outputs/pretrain_pmrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt 

# # ./outputs/pretrain_pmrl/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt 

# CUDA_VISIBLE_DEVICES=0 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 1 \
# --master_port 9838 \
# ./run.py \
# --learning_rate 5e-5 \
# --checkpointing true \
# --first_eval false \
# --config ./config/pmrl/pretrain_cfg/pretrain_promrl_test_feature_inference.json \
# --pretrain_dir ./pretrain_vast \
# --checkpoint outputs/pretrain_pmrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt  \
# --mode 'testing' \
# --model_type promrl \
# --save_best true \
# --tau1 0.01 \
# --tau2 0.1 \
# --lambda_itm 0.1 \
# --output_dir ./outputs/features_pmrl2_mini \
# --log_name features_pmrl2_mini \
# --feature_inference true 

# CUDA_VISIBLE_DEVICES=0 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 1 \
# --master_port 9838 \
# ./run.py \
# --learning_rate 5e-5 \
# --checkpointing true \
# --first_eval false \
# --config ./config/pmrl/pretrain_cfg/pretrain_promrl_test_feature_inference.json \
# --pretrain_dir ./pretrain_vast \
# --checkpoint ./outputs/pretrain_pmrl/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt  \
# --mode 'testing' \
# --model_type promrl \
# --save_best true \
# --tau1 0.01 \
# --tau2 0.1 \
# --lambda_itm 0.1 \
# --output_dir ./outputs/features_pmrl3_mini \
# --log_name features_pmrl3_mini \
# --feature_inference true 





CUDA_VISIBLE_DEVICES=1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_test_feature_classification.json \
--pretrain_dir ./pretrain_vast \
--checkpoint outputs/pretrain_vast_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_vast \
--log_name features_vast \
--feature_inference true 



CUDA_VISIBLE_DEVICES=1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_test_feature_classification.json \
--pretrain_dir ./pretrain_vast \
--checkpoint outputs/pretrain_gram_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_gram \
--log_name features_gram \
--feature_inference true 

# ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt

# outputs/pretrain_pmrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt 

# ./outputs/pretrain_pmrl/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt 

CUDA_VISIBLE_DEVICES=1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_test_feature_classification.json \
--pretrain_dir ./pretrain_vast \
--checkpoint outputs/pretrain_pmrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt  \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_pmrl2 \
--log_name features_pmrl2 \
--feature_inference true 



