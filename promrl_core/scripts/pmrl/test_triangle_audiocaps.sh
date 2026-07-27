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
--checkpoint ./outputs/pretrain_triangle_audiocaps/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type triangle \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_triangle_audiocaps \
--log_name test_triangle_audiocaps_all_zeroshot \
--feature_inference false 


python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_triangle_audiocaps/ --log-file ./outputs/retrieval-all-zeroshot_triangle_audiocaps/log/log.txt --prefix triangle_audiocaps_


