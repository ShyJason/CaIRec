
# 
CUDA_VISIBLE_DEVICES=0,1,5,6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9836 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-msrvtt2.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_msrvtt_3modalities_detach \
--checkpoint ./outputs/pretrain_promrl_msrvtt_3modalities/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 32 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_msrvtt_3modalities_detach \
--feature_inference false


# --checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
# --checkpoint ./outputs/pretrain_promrl_msrvtt_detach3/ckpt/best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt \

CUDA_VISIBLE_DEVICES=0,1,5,6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps2.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./outputs/pretrain_promrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 32 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_audiocaps_msrvtt_3modalities_audiocaps \
--feature_inference false



CUDA_VISIBLE_DEVICES=0,1,5,6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt \
--checkpoint ./outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--mode 'testing' \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 32 \
--lambda_itm 0.1 \
--log_name pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt \
--feature_inference false



CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9836 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-audiocaps2.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_audiocaps \
--checkpoint ./outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type promrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 32 \
--lambda_itm 0 \
--log_name pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_msrvtt_audiocaps \
--feature_inference false


CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9836 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/retrieval-all-zeroshot_promrl_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_audiocaps \
--checkpoint ./outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_audiocaps/ckpt/best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt \
--model_type promrl \
--mode 'testing' \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 32 \
--lambda_itm 0 \
--log_name retrieval-all-zeroshot_promrl_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_audiocaps \
--feature_inference false



CUDA_VISIBLE_DEVICES=0,1,5,6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/retrieval-all-zeroshot_gram_msrvtt3 \
--checkpoint ./outputs/pretrain_gram_all_zeroshot_audiocaps_msrvtt3/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type gram \
--mode 'testing' \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name retrieval-all-zeroshot_gram_msrvtt3 \
--feature_inference false



CUDA_VISIBLE_DEVICES=0,1,5,6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_audiocaps \
--checkpoint ./outputs/pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_audiocaps/ckpt/best_ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva.pt \
--model_type promrl \
--mode 'testing' \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 32 \
--lambda_itm 0 \
--log_name pretrain_promrl_all_zeroshot_audiocaps_msrvtt_3modalities_audiocaps_msrvtt_audiocaps \
--feature_inference false




# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 4 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 1e-5 \
# --checkpointing true \
# --first_eval true \
# --save_best true \
# --config ./config/pmrl/finetune_cfg/retrieval-msrvtt2.json \
# --pretrain_dir ./pretrain_vast \
# --output_dir outputs/pretrain_pmrl_msrvtt \
# --checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
# --model_type pmrl \
# --tau1 0.05 \
# --tau2 0.1 \
# --valid_freq 8 \
# --lambda_itm 0.1 \
# --log_name pretrain_pmrl_msrvtt \
# --feature_inference false


CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-msrvtt2.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_triangle_msrvtt \
--checkpoint ./triangle_pretraining/ckpt/model_step_200.pt \
--model_type triangle \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_triangle_msrvtt \
--feature_inference false


CUDA_VISIBLE_DEVICES=2,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_gram_all_zeroshot_audiocaps_msrvtt2 \
--checkpoint ./outputs/pretrain_gram_msrvtt2/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type gram \
--mode 'testing' \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_gram_all_zeroshot_audiocaps_msrvtt2 \
--feature_inference false




CUDA_VISIBLE_DEVICES=0,1,5,6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9838 \
./run.py \
--learning_rate 2e-5 \
--checkpointing true \
--first_eval false \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-msrvtt2.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_gram_msrvtt2 \
--checkpoint ./outputs/pretrain_gram_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type gram \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 32 \
--lambda_itm 0.1 \
--itm_ratio 0.1 \
--log_name pretrain_gram_msrvtt2 \
--feature_inference false



CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 1e-5 \
--checkpointing true \
--first_eval true \
--save_best true \
--config ./config/pmrl/finetune_cfg/retrieval-msrvtt.json \
--pretrain_dir ./pretrain_vast \
--output_dir outputs/pretrain_pmrl_msrvtt_ori \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--model_type pmrl \
--tau1 0.05 \
--tau2 0.1 \
--valid_freq 8 \
--lambda_itm 0.1 \
--log_name pretrain_pmrl_msrvtt_ori \
--feature_inference false

# outputs/pretrain_promrl_audiocaps