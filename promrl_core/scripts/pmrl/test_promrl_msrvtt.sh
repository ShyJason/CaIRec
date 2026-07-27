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
--checkpoint ./outputs/pretrain_promrl_audiocaps_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type promrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval_promrl_audiocaps_msrvtt_3modalities \
--log_name test_promrl_audiocaps_msrvtt_3modalities \
--feature_inference false 


python extract_results.py --log-file ./outputs/retrieval_promrl_msrvtt_3modalities_detach/log/log.txt --prefix promrl_msrvtt_3modalities_detach



CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_pmrl_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_pmrl_msrvtt \
--log_name test_pmrl_msrvtt_all_zeroshot \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_pmrl_msrvtt/ --log-file ./outputs/retrieval-all-zeroshot_pmrl_msrvtt/log/log.txt --prefix pmrl_msrvtt_


CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_triangle_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type triangle \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_triangle_msrvtt \
--log_name test_triangle_msrvtt_all_zeroshot \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_triangle_msrvtt/ --log-file ./outputs/retrieval-all-zeroshot_triangle_msrvtt/log/log.txt --prefix triangle_msrvtt_


CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_gram_msrvtt/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type gram \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_gram_msrvtt \
--log_name test_gram_msrvtt_all_zeroshot \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_gram_msrvtt/ --log-file ./outputs/retrieval-all-zeroshot_gram_msrvtt/log/log.txt --prefix gram_msrvtt_


CUDA_VISIBLE_DEVICES=3,4 torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
--master_port 9834 \
./run.py \
--learning_rate 5e-5 \
--checkpointing true \
--first_eval false \
--config ./config/pmrl/finetune_cfg/retrieval-all-zeroshot.json \
--checkpoint ./outputs/pretrain_pmrl_msrvtt_ori/ckpt/best_ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas.pt \
--pretrain_dir ./pretrain_vast \
--mode 'testing' \
--model_type pmrl \
--save_best true \
--tau1 0.01 \
--tau2 0.1 \
--lambda_itm 0.1 \
--output_dir ./outputs/retrieval-all-zeroshot_pmrl_msrvtt_ori \
--log_name test_pmrl_msrvtt_all_zeroshot_ori \
--feature_inference false 

python extract_results.py --outputs-dir ./outputs/retrieval-all-zeroshot_pmrl_msrvtt_ori/ --log-file ./outputs/retrieval-all-zeroshot_pmrl_msrvtt_ori/log/log.txt --prefix pmrl_msrvtt_ori_


"""

11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 40.6, 'video_recall': '40.6/67.4/77.7', 'video_ravg': 61.9} !!! Better
11/09/2025 03:42:15 - INFO - __main__ -   ==== evaluation--ret%tvas--msrvtt_ret_msrvtt_ret_ret_itm_tvas========

11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 60.9, 'video_recall': '60.9/83.0/87.3', 'video_ravg': 77.1, 'txt_r1': 60.7, 'txt_recall': '60.7/80.5/87.1', 'txt_ravg': 76.1}!!! Comparable


11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 34.1, 'video_recall': '34.1/62.5/74.2', 'video_ravg': 56.9}  # Better 
11/09/2025 03:42:15 - INFO - __main__ -   ==== evaluation--ret%tva--activitynet_ret_activitynet_ret_ret_itm_tva========

11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 56.1, 'video_recall': '56.1/80.7/87.9', 'video_ravg': 74.9, 'txt_r1': 54.1, 'txt_recall': '54.1/81.6/89.5', 'txt_ravg': 75.1}



11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 31.6, 'video_recall': '31.6/59.0/67.2', 'video_ravg': 52.6}
11/09/2025 03:42:15 - INFO - __main__ -   ==== evaluation--ret%tva--didemo_ret_didemo_ret_ret_itm_tva========

11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 55.3, 'video_recall': '55.3/75.6/80.5', 'video_ravg': 70.5, 'txt_r1': 52.3, 'txt_recall': '52.3/76.2/82.0', 'txt_ravg': 70.2}


11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 57.3, 'video_recall': '57.3/88.3/94.6', 'video_ravg': 80.1}
11/09/2025 03:42:15 - INFO - __main__ -   ==== evaluation--ret%tvas--vatex_ret_vatex_ret_ret_itm_tvas========

11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 82.1, 'video_recall': '82.1/96.7/98.1', 'video_ravg': 92.3, 'txt_r1': 78.8, 'txt_recall': '78.8/96.7/98.0', 'txt_ravg': 91.1}



11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 43.5, 'video_recall': '43.5/76.4/87.4', 'video_ravg': 69.1}
11/09/2025 03:42:15 - INFO - __main__ -   ==== evaluation--ret%tva--audiocaps_ret_audiocaps_ret_ret_itm_tva========

11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 47.6, 'video_recall': '47.6/80.8/88.2', 'video_ravg': 72.2, 'txt_r1': 50.7, 'txt_recall': '50.7/82.4/91.3', 'txt_ravg': 74.8}



11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 17.7, 'video_recall': '17.7/42.2/54.2', 'video_ravg': 38.0}
11/09/2025 03:42:15 - INFO - __main__ -   ==== evaluation--ret%ta--clothov2_ret_clothov2_ret_ret_itm_ta========

11/09/2025 03:42:15 - INFO - __main__ -   {'video_r1': 22.3, 'video_recall': '22.3/47.0/58.5', 'video_ravg': 42.6, 'txt_r1': 21.2, 'txt_recall': '21.2/46.0/59.1', 'txt_ravg': 42.1}


Basically, V-T consine improcves, while A-T decreases.

"""