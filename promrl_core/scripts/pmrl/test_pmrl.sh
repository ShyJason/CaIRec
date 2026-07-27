
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
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
--checkpoint ./outputs/pretrain_pmrl/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./output/retrieval-all-zeroshot \
--log_name test_pmrl_all_zeroshot

python extract_results.py --outputs-dir ./outputs/pmrl/retrieval-all-zeroshot_ori/ --log-file ./outputs/pmrl/retrieval-all-zeroshot_ori/log/log.txt --prefix pmrl_



# CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 4 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 5e-5 \
# --checkpointing true \
# --first_eval false \
# --config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
# --pretrain_dir ./pretrain_vast \
# --checkpoint ./SVD_MM/PMCL/output/gram/pretrain_pmcl/downstream/pretrain_intra_0.05_inter_0.1_itm_0.1/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
# --mode 'testing' \
# --model_type pmrl \
# --save_best true \
# --tau1 0.01 \
# --tau2 0.1 \
# --lambda_itm 0.1 \
# --output_dir ./outputs/pmrl/retrieval-all-zeroshot_ori \
# --log_name test_pmrl_all_zeroshot


# python extract_results.py --outputs-dir ./outputs/pmrl/retrieval-all-zeroshot_ori/ --log-file ./outputs/pmrl/retrieval-all-zeroshot_ori/log/log.txt --prefix pmrl_



# CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun \
# --nnodes 1 \
# --node_rank 0 \
# --nproc_per_node 4 \
# --master_port 9834 \
# ./run.py \
# --learning_rate 5e-5 \
# --checkpointing true \
# --first_eval false \
# --config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
# --pretrain_dir ./pretrain_vast \
# --mode 'testing' \
# --model_type pmrl \
# --save_best true \
# --tau1 0.01 \
# --tau2 0.1 \
# --lambda_itm 0.1 \
# --output_dir ./outputs/retrieval-all-zeroshot_vast \
# --log_name test_vast_all_zeroshot




