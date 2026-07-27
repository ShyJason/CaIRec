CUDA_VISIBLE_DEVICES=5,6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9836 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_pmrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_pmrl_audiocaps_msrvtt \
--log_name test_pmrl_audiocaps_msrvtt_all_zeroshot \
--feature_inference false 


python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_pmrl_audiocaps_msrvtt/ --log-file ./outputs/retrieval-all-zeroshot_pmrl_audiocaps_msrvtt/log/log.txt --prefix pmrl_audiocaps_msrvtt_


CUDA_VISIBLE_DEVICES=5,6 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9836 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_gram_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type gram \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_gram_audiocaps_msrvtt \
--log_name test_gram_audiocaps_msrvtt_all_zeroshot \
--feature_inference false 


python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_gram_audiocaps_msrvtt/ --log-file ./outputs/retrieval-all-zeroshot_gram_audiocaps_msrvtt/log/log.txt --prefix gram_audiocaps_msrvtt_



CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9836 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_triangle_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type triangle \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_triangle_audiocaps_msrvtt \
--log_name test_triangle_audiocaps_msrvtt_all_zeroshot \
--feature_inference false 


python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_triangle_audiocaps_msrvtt/ --log-file ./outputs/retrieval-all-zeroshot_triangle_audiocaps_msrvtt/log/log.txt --prefix triangle_audiocaps_msrvtt_


CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9836 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_promrl_audiocaps_msrvtt_3modalities/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--itm_rerank_num 10 \
--output_dir ./outputs/retrieval-all-zeroshot_promrl_audiocaps_msrvtt_3modalities_10rerank \
--log_name test_promrl_audiocaps_msrvtt_3modalities_10rerank_all_zeroshot \
--feature_inference false 


python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_promrl_audiocaps_msrvtt_3modalities_10rerank/ --log-file ./outputs/retrieval-all-zeroshot_promrl_audiocaps_msrvtt_3modalities_10rerank/log/log.txt --prefix promrl_audiocaps_msrvtt_3modalities_10rerank_


CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9836 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_vast_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type vast \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_vast_audiocaps_msrvtt \
--log_name test_vast_audiocaps_msrvtt_all_zeroshot \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_vast_audiocaps_msrvtt/ --log-file ./outputs/retrieval-all-zeroshot_vast_audiocaps_msrvtt/log/log.txt --prefix vast_audiocaps_msrvtt_