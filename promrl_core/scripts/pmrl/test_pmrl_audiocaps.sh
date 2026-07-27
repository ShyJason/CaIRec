
CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_pmrl_msrvtt_full/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_prmrl_msrvtt_full \
--log_name test_pmrl_msrvtt_full \
--feature_inference false 


python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_prmrl_audiocaps/ --log-file ./outputs/retrieval-all-zeroshot_prmrl_audiocaps/log/log.txt --prefix pmrl_audiocaps_



CUDA_VISIBLE_DEVICES=0,1 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9838 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/finetune_intra_0.02_inter_0.2_itm_0.1_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_pmrl_msrvtt_full2 \
--log_name test_pmrl_msrvtt_all_zeroshot_full2 \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_prmrl_msrvtt_full/ --log-file ./outputs/retrieval-all-zeroshot_prmrl_msrvtt_full/log/log.txt --prefix pmrl_msrvtt_full_