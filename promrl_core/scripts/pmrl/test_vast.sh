CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 4 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-new.json \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_vast_new \
--log_name test_vast_all_zeroshot_new \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_vast_new/ --log-file ./outputs/retrieval-all-zeroshot_vast_new/log/log.txt --prefix vast_new_



