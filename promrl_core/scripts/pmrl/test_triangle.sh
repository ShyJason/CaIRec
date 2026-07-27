
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
--pretrain_dir ./pretrain_vast \
--checkpoint ./triangle_pretraining/ckpt/model_step_200.pt \
--mode 'testing' \
--model_type triangle \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./output/retrieval-all-zeroshot_ori \
--log_name test_triangle_all_zeroshot_ori \
--feature_inference false 


python extract_results.py --outputs-dir ./output/retrieval-all-zeroshot_ori/ --log-file ./output/retrieval-all-zeroshot_ori/log/log.txt --prefix triangle_ori_