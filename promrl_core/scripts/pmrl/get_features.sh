CUDA_VISIBLE_DEVICES=0 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./triangle_pretraining/ckpt/model_step_200.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_triangle_mini \
--log_name features_triangle_mini \
--feature_inference true 




CUDA_VISIBLE_DEVICES=6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./outputs/pretrain_promrl_audiocaps_msrvtt_3modalities/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas_ori.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_promrl_am_mini \
--log_name features_promrl_am_mini \
--feature_inference true 


CUDA_VISIBLE_DEVICES=6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_pmrl_mini \
--log_name features_pmrl_mini \
--feature_inference true 



CUDA_VISIBLE_DEVICES=6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint outputs/pretrain_pmrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_pmrl_am_mini \
--log_name features_pmrl_am_mini \
--feature_inference true 


CUDA_VISIBLE_DEVICES=6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint outputs/pretrain_vast_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_vast_am_mini \
--log_name features_vast_am_mini \
--feature_inference true 

CUDA_VISIBLE_DEVICES=6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint outputs/pretrain_gram_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_gram_am_mini \
--log_name features_gram_am_mini \
--feature_inference true 


CUDA_VISIBLE_DEVICES=6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint outputs/pretrain_triangle_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_triangle_am_mini \
--log_name features_triangle_am_mini \
--feature_inference true 



CUDA_VISIBLE_DEVICES=0 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./outputs/pretrain_pmrl/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_pmrl \
--log_name features_pmrl \
--feature_inference true 






CUDA_VISIBLE_DEVICES=6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 1 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/pretrain_cfg/pretrain_promrl_train_feature_inference.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/features_promrl_am_mini2 \
--log_name features_promrl_am_mini2 \
--feature_inference true 