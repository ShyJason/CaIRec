CUDA_VISIBLE_DEVICES=2,7 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_gram_audiocaps/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type gram \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_gram_audiocaps \
--log_name test_gram_all_zeroshot_audiocaps \
--feature_inference false 


CUDA_VISIBLE_DEVICES=2,7 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_pmrl_audiocaps/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_pmrl_audiocaps \
--log_name test_pmrl_all_zeroshot_audiocaps \
--feature_inference false 


CUDA_VISIBLE_DEVICES=2,3 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-new.json \
--checkpoint ./SVD_MM/GRAM/GRAM_pretrained_4modalities/ckpt/base.pt  \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type gram \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_gram_new \
--log_name test_gram_all_zeroshot_new \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_gram_new/ --log-file ./outputs/retrieval-all-zeroshot_gram_new/log/log.txt --prefix gram_new_


CUDA_VISIBLE_DEVICES=2,3 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-new.json \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_pmrl_new \
--log_name test_pmrl_all_zeroshot_new \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_pmrl_new/ --log-file ./outputs/retrieval-all-zeroshot_pmrl_new/log/log.txt --prefix pmrl_new_

CUDA_VISIBLE_DEVICES=2,3 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9839 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-new.json \
--pretrain_dir ./pretrain_vast \
--checkpoint ./triangle_pretraining/ckpt/model_step_200.pt \
--mode 'testing' \
--model_type triangle \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_triangle_new \
--log_name test_triangle_all_zeroshot_new \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_triangle_new/ --log-file ./outputs/retrieval-all-zeroshot_triangle_new/log/log.txt --prefix triangle_new_


# base.pt  model_step_459.pt

