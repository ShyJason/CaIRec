# MMRec Mainline

This is the single source of truth for the current MMRec mainline. Older
progress notes and experiment reports are historical unless they point back to
this file.

## Mainline Pipeline

```text
PROMRL completion
  -> raw decoder v2
  -> original I3 linear MLP
  -> modality GCNs
  -> multi-source completed-feature item-item graph residual
  -> true-missing GCN InfoNCE
  -> fusion
  -> BPR
```

The active stage layout is:

- Stage 1.1: `configs/<dataset>/stage1_1_imputer_param.yaml`
- Stage 1.2: `configs/<dataset>/stage1_2_decoder_v2.yaml`
- Stage 2: dataset canonical mainline config, currently
  `configs/clothing/mainline_mr0p1.yaml` for Clothing 10% missing-rate runs

Stage 1.2 only supports `stage1_2_mode=observed` for new runs. Both the rec/NLL
target and decoder target use genuinely observed modalities; the previous
pseudo-missing target branch has been removed. All other Stage 1.2 settings stay
on the mainline setup, including `decode_loss_grad_mode=detached` and
`generative_update_mode=fixed`.

Some historical Clothing best-style checkpoints log
`stage1_rec_loss_mode=calmrl_pseudo`. Treat this as the historical name for the
old pseudo-missing Stage 1.2 branch when interpreting those checkpoints. New
Stage 1.2 runs should use `stage1_2_mode=observed`.

## Protocol Names

- Missing-rate protocol: train uses `--missing_rate <rate>` and validation/test
  use `--eval_missing_rate 0.5`, matching the historical fixed 50% eval mask.
- True full modality: train, validation, and test all use complete modalities:
  `--missing_rate 0 --eval_missing_rate 0`.
- Results from these protocols must not be compared as the same setting unless
  both train and eval missing rates match.

## Current Best Results

| Dataset | Protocol | Config | Result | Evidence |
| --- | --- | --- | --- | --- |
| `clothing` | `mm`, train missing `0.1`, eval/test missing `0.5`, seed `2023` | `configs/clothing/mainline_mr0p1.yaml` | Recall@20 `0.08054`, NDCG@20 `0.03550`, best epoch `127` | `exp_report/clothing/topk10_local_hparam_search/clothing_topk10_local_mr0p1_4x_gpu56_20260625_105012/summary.tsv` |
| `clothing` | `mm`, train missing `0.3`, eval/test missing `0.5`, seed `2023` | `configs/clothing/mainline_mr0p1.yaml` plus compact-search overrides and the 30% Stage 1.2 checkpoint | Recall@20 `0.07918`, NDCG@20 `0.03599`, best epoch `198` | `exp_report/clothing/mr0p3_compact_hparam_search/clothing_mr0p3_compact_smoke_gpu07_20260627_145004/phase4_confirm/summary.tsv` |
| `clothing` | `mm`, train missing `0.3`, eval/test missing `0.5`, seed `2023`, SMORE native-missing pretrained modal features | SMORE `image_item_embeds/text_item_embeds` + observed Stage 1.1 epoch4 imputer + identity/shared bridge + fixed I3 item graph | Recall@20 `0.08480`, NDCG@20 `0.03788`, best epoch `68` | `exp_report/clothing/smore_native_missing_item_embeds_iigraph_imputed_search/smore_native_imputed_stage2_search_20260702_gpu0/logs/reccl0075_topk10.log`; details in `docs/SMORE_NATIVE_MISSING_IMPUTED_SEARCH_CN.md` |
| `clothing` | true full modality, train/eval/test missing `0`, seed `2023` | historical command used `configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml` plus CLI overrides | Recall@20 `0.08304`, NDCG@20 `0.03699`, best epoch `79` | `exp_report/clothing/full_modality_true_doc_mainline_topk10_gpu7/run.log` |
| `baby` | `mm`, train missing `0.3`, seed `2023` | historical raw-decoder baseline | Recall@20 `0.08435`, NDCG@20 `0.03751`, best epoch `109` | `README.md` |
| `tiktok` | TBD | TBD | TBD | TBD |
| `microlens` | TBD | TBD | TBD | TBD |
| `microlens100k` | TBD | TBD | TBD | TBD |

## Clothing Canonical Mainline

Current verified Clothing 10% missing-rate mainline:

```text
dataset = clothing
exp_mode = mm
train missing_rate = 0.1
eval_missing_rate = 0.5
seed = 2023
dataset_seed = 0
stage1.2 style = best-style no-CL
stage1.2 checkpoint epoch = 47

config = configs/clothing/mainline_mr0p1.yaml
train_stage = recommender
epoch = 200
early_stop = 20
batch_size = 2048
selection_mode = val
recommendation_selection_metric = recall
recommendation_selection_topk = 20
evaluation_protocol = strict

feature_bridge_mode = raw_decoder
gcn_frontend_mode = original_linear
freeze_imputer = 1
freeze_decoder = 1
recommender_allow_modal_grad = 0
disable_imputation = 0

lr = 0.01
lr_rec = 0.01
lr_imp = 0.0002
lr_decoder = 0.00005
reg_coeff = 0.01
modality_bpr_coeff = 1.0

item_graph_kind = fused_completed
item_graph_topk = 10
item_graph_norm = rw
item_graph_cf_weight = 0.2
item_graph_image_weight = 0.4
item_graph_text_weight = 0.4
item_graph_audio_weight = 0.0
item_graph_modal_alpha = 0.25
item_graph_modal_layers = 1
item_graph_modal_target = all

rec_neighbor_cl_weight = 0.005
rec_neighbor_cl_temp = 0.2
rec_neighbor_cl_bank_size = 256
```

Reproduction command:

```bash
python -u main.py \
  --config configs/clothing/mainline_mr0p1.yaml \
  --device_id 0 \
  --imputer_ckpt exp_report/clothing/stage1_2_clothing_mm_mr0p1_beststyle_nocl_clothing_mr0p1_latest_20260624_163902/ckpt/stage1_2_clothing_mm_mr0p1_beststyle_nocl_clothing_mr0p1_latest_20260624_163902_imputer_backprop_50_epoch47.pth \
  --suffix stage2_clothing_mr0p1_mainline_repro
```

The evidence run used the same settings through
`configs/clothing/stage2_decoder_mm_itemgraph_completed.yaml` plus CLI
overrides. Its final strict test metrics were:

```text
Recall@20 = 0.08054
NDCG@20 = 0.03550
best_epoch = 127
```

Evidence:

```text
exp_report/clothing/topk10_local_hparam_search/clothing_topk10_local_mr0p1_4x_gpu56_20260625_105012/summary.tsv
exp_report/clothing/topk10_local_hparam_search/clothing_topk10_local_mr0p1_4x_gpu56_20260625_105012/logs/stage2_clothing_mr0p1_topk10_graph_modal040_clothing_topk10_local_mr0p1_4x_gpu56_20260625_105012.log
```

## Clothing 30% Missing-Rate Best So Far

Current verified Clothing 30% missing-rate best-so-far uses the 30% Stage 1.2
checkpoint and the compact-search overrides selected by validation Recall@20.
The same setting was confirmed with seed `2023` and checked again with seed
`1`.

```text
dataset = clothing
exp_mode = mm
train missing_rate = 0.3
eval_missing_rate = 0.5
seed = 2023
dataset_seed = 0
stage1.2 style = best-style no-CL
stage1.2 checkpoint epoch = 49

config = configs/clothing/mainline_mr0p1.yaml
train_stage = recommender
epoch = 200
early_stop = 20
batch_size = 2048
selection_mode = val
recommendation_selection_metric = recall
recommendation_selection_topk = 20
evaluation_protocol = strict

item_graph_kind = fused_completed
item_graph_topk = 8
item_graph_norm = rw
item_graph_cf_weight = 0.25
item_graph_image_weight = 0.375
item_graph_text_weight = 0.375
item_graph_audio_weight = 0.0
item_graph_modal_alpha = 0.25
item_graph_modal_layers = 1
item_graph_modal_target = all

rec_neighbor_cl_weight = 0.01
rec_neighbor_cl_temp = 0.2
rec_neighbor_cl_bank_size = 256
```

Reproduction command:

```bash
python -u main.py \
  --config configs/clothing/mainline_mr0p1.yaml \
  --device_id 0 \
  --dataset clothing \
  --exp_mode mm \
  --train_stage recommender \
  --missing_rate 0.3 \
  --eval_missing_rate 0.5 \
  --seed 2023 \
  --dataset_seed 0 \
  --imputer_ckpt exp_report/clothing/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840/ckpt/stage1_2_clothing_mm_mr0p3_beststyle_nocl_clothing_mr0p3_latest_20260624_174840_imputer_backprop_50_epoch49.pth \
  --suffix stage2_clothing_mr0p3_compact_confirm_repro \
  --epoch 200 \
  --early_stop 20 \
  --eva_interval 1 \
  --batch_size 2048 \
  --lr 0.01 \
  --lr_rec 0.01 \
  --lr_imp 0.0002 \
  --lr_decoder 0.00005 \
  --freeze_imputer 1 \
  --freeze_decoder 1 \
  --recommender_allow_modal_grad 0 \
  --feature_bridge_mode raw_decoder \
  --gcn_frontend_mode original_linear \
  --disable_imputation 0 \
  --modality_bpr_coeff 1.0 \
  --reg_coeff 0.01 \
  --evaluation_protocol strict \
  --selection_mode val \
  --strict_probe_test_interval 0 \
  --recommendation_selection_metric recall \
  --recommendation_selection_topk 20 \
  --rec_neighbor_cl_weight 0.01 \
  --rec_neighbor_cl_temp 0.2 \
  --rec_neighbor_cl_bank_size 256 \
  --item_graph_kind fused_completed \
  --item_graph_topk 8 \
  --item_graph_norm rw \
  --item_graph_cf_weight 0.25 \
  --item_graph_image_weight 0.375 \
  --item_graph_text_weight 0.375 \
  --item_graph_audio_weight 0.0 \
  --item_graph_modal_alpha 0.25 \
  --item_graph_modal_layers 1 \
  --item_graph_modal_target all \
  --tensorboard 0 \
  --save 1 \
  --topk [10, 20, 30, 40, 50]
```

Final strict test metrics:

```text
Recall@10 = 0.05243
NDCG@10 = 0.02918
Recall@20 = 0.07918
NDCG@20 = 0.03599
Recall@30 = 0.09827
NDCG@30 = 0.04009
Recall@40 = 0.11269
NDCG@40 = 0.04290
Recall@50 = 0.12658
NDCG@50 = 0.04544
best_epoch = 198
```

Seed `1` stability check for the same setting:

```text
Recall@10 = 0.05320
NDCG@10 = 0.02899
Recall@20 = 0.07920
NDCG@20 = 0.03556
Recall@30 = 0.09740
NDCG@30 = 0.03947
Recall@40 = 0.11259
NDCG@40 = 0.04243
Recall@50 = 0.12527
NDCG@50 = 0.04474
best_epoch = 166
```

Evidence:

```text
exp_report/clothing/mr0p3_compact_hparam_search/clothing_mr0p3_compact_smoke_gpu07_20260627_145004/phase4_confirm/summary.tsv
exp_report/clothing/mr0p3_compact_hparam_search/clothing_mr0p3_compact_smoke_gpu07_20260627_145004/phase4_confirm/summary_topk.tsv
exp_report/clothing/mr0p3_compact_hparam_search/clothing_mr0p3_compact_smoke_gpu07_20260627_145004/phase4_confirm/logs/stage2_clothing_mr0p3_p4_top2_confirm_seed2023_clothing_mr0p3_compact_smoke_gpu07_20260627_145004.log
exp_report/clothing/mr0p3_compact_hparam_search/clothing_mr0p3_compact_smoke_gpu07_20260627_145004/phase4_confirm/logs/stage2_clothing_mr0p3_p4_top2_confirm_seed1_clothing_mr0p3_compact_smoke_gpu07_20260627_145004.log
```

The `batch_size=512` follow-up used the same settings except for batch size and
finished lower: Recall@20 `0.07607`, NDCG@20 `0.03392`, best epoch `73`.

## Clothing 30% Missing-Rate SMORE-Pretrained Best

This is the current best Clothing 30% experiment that replaces MMRec's original
modal inputs with SMORE native-missing pretrained item embeddings. It is not the
canonical raw-feature MMRec mainline above; keep the protocol label explicit
when comparing results.

```text
dataset = clothing
exp_mode = mm
train missing_rate = 0.3
eval_missing_rate = 0.5
seed = 2023
dataset_seed = 0

pretrained modal features = SMORE native-missing raw image_item_embeds/text_item_embeds
feature dimension = 64
modal_feature_override_dir = /home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_native_missing_raw_input_compact_clothing_mr0p3_20260702/native_missing_raw_mr0p3
modal_feature_image_file = image_item_embeds.npy
modal_feature_text_file = text_item_embeds.npy

Stage 1.1 = observed imputer_param
Stage 1.1 epochs = 5
Stage 1.1 checkpoint = epoch4
Stage 1.2 = disabled / not used

promrl_projection_mode = identity
feature_bridge_mode = shared_identity
gcn_frontend_mode = identity
disable_imputation = 0
freeze_imputer = 1
freeze_decoder = 1
recommender_allow_modal_grad = 0

config = configs/clothing/mainline_mr0p1.yaml
train_stage = recommender
epoch = 200
early_stop = 20
batch_size = 2048
selection_mode = val
recommendation_selection_metric = recall
recommendation_selection_topk = 20
evaluation_protocol = strict

lr = 0.01
lr_rec = 0.01
lr_imp = 0.0002
lr_decoder = 0.00005
reg_coeff = 0.01
modality_bpr_coeff = 1.0

item_graph_kind = fused_completed
item_graph_feature_space = shared
item_graph_topk = 10
item_graph_norm = rw
item_graph_cf_weight = 0.2
item_graph_image_weight = 0.4
item_graph_text_weight = 0.4
item_graph_audio_weight = 0.0
item_graph_modal_alpha = 0.25
item_graph_modal_layers = 1
item_graph_modal_target = all

rec_neighbor_cl_weight = 0.0075
rec_neighbor_cl_temp = 0.2
rec_neighbor_cl_bank_size = 256
```

Stage 1.1 imputer checkpoint:

```text
exp_report/clothing/clothing_mr0p3_smore_native_missing_raw_identity_stage11_observed_5e_20260702_gpu0/ckpt/clothing_mr0p3_smore_native_missing_raw_identity_stage11_observed_5e_20260702_gpu0_imputer_param_50_epoch4.pth
```

Stage 2 reproduction command:

```bash
python -u main.py \
  --config configs/clothing/mainline_mr0p1.yaml \
  --device_id 0 \
  --dataset clothing \
  --exp_mode mm \
  --train_stage recommender \
  --suffix stage2_clothing_mr0p3_smore_native_imputed_reccl0075_topk10_smore_native_imputed_stage2_search_20260702_gpu0 \
  --missing_rate 0.3 \
  --eval_missing_rate 0.5 \
  --seed 2023 \
  --dataset_seed 0 \
  --batch_size 2048 \
  --lr 0.01 \
  --lr_rec 0.01 \
  --lr_imp 0.0002 \
  --lr_decoder 0.00005 \
  --contra_dim 64 \
  --d_beta 32 \
  --imputer_ckpt exp_report/clothing/clothing_mr0p3_smore_native_missing_raw_identity_stage11_observed_5e_20260702_gpu0/ckpt/clothing_mr0p3_smore_native_missing_raw_identity_stage11_observed_5e_20260702_gpu0_imputer_param_50_epoch4.pth \
  --disable_imputation 0 \
  --freeze_imputer 1 \
  --freeze_decoder 1 \
  --recommender_allow_modal_grad 0 \
  --promrl_projection_mode identity \
  --feature_bridge_mode shared_identity \
  --gcn_frontend_mode identity \
  --modal_feature_override_dir /home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_native_missing_raw_input_compact_clothing_mr0p3_20260702/native_missing_raw_mr0p3 \
  --modal_feature_image_file image_item_embeds.npy \
  --modal_feature_text_file text_item_embeds.npy \
  --item_graph_kind fused_completed \
  --item_graph_feature_space shared \
  --item_graph_topk 10 \
  --item_graph_cf_weight 0.2 \
  --item_graph_image_weight 0.4 \
  --item_graph_text_weight 0.4 \
  --item_graph_modal_alpha 0.25 \
  --rec_neighbor_cl_weight 0.0075 \
  --selection_mode val \
  --recommendation_selection_metric recall \
  --recommendation_selection_topk 20 \
  --evaluation_protocol strict \
  --early_stop 20 \
  --tensorboard 0 \
  --save 1
```

Final strict test metrics:

```text
Recall@10 = 0.05729
NDCG@10 = 0.03088
Recall@20 = 0.08480
NDCG@20 = 0.03788
Recall@30 = 0.10369
NDCG@30 = 0.04193
Recall@40 = 0.12011
NDCG@40 = 0.04513
Recall@50 = 0.13426
NDCG@50 = 0.04771
best_epoch = 68
best_val_recall@20 = 0.08544
```

Evidence:

```text
exp_report/clothing/smore_native_missing_item_embeds_iigraph_imputed_search/smore_native_imputed_stage2_search_20260702_gpu0/logs/reccl0075_topk10.log
exp_report/clothing/stage2_clothing_mr0p3_smore_native_imputed_reccl0075_topk10_smore_native_imputed_stage2_search_20260702_gpu0/ckpt/stage2_clothing_mr0p3_smore_native_imputed_reccl0075_topk10_smore_native_imputed_stage2_search_20260702_gpu0_recommender_1.0_epoch68.pth
docs/SMORE_NATIVE_MISSING_IMPUTED_SEARCH_CN.md
```

## Ablation Boundaries

Single-source item-item graph runs are ablations, not the default mainline. They
should still use `item_graph_kind=fused_completed`, but set only one source
weight positive.

The following branches are not the current mainline:

- score-level assemble fusion with an auxiliary all-gate recommender
- `mlp_impute`
- `semantic_bridge`
- `shared_gcn`
- `dual_head_decoder`
- `shared_direct`
- observed Stage 1.2 target mode as the default imputer branch
- stage-1 CL for Clothing default runs
- single-source item-item graph as a default setting
