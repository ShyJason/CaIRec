# Sports Search Records

This file summarizes the local Sports hyperparameter search records that are
useful for continuing the search. All records below use the Sports MM setting
with train missing rate 0.3 unless noted otherwise.

## Common Protocol

Common Stage 2 protocol used by the meaningful searches:

```text
dataset = sports
exp_mode = mm
train_stage = recommender
missing_rate = 0.3
seed = 1
dataset_seed = 0
missing_mask_protocol = i3
evaluation_protocol = strict
selection_mode = val
recommendation_selection_metric = recall
recommendation_selection_topk = 20
feature_bridge_mode = raw_decoder
gcn_frontend_mode = original_linear
freeze_imputer = 1
freeze_decoder = 1
recommender_allow_modal_grad = 0
item_graph_kind = fused_completed
item_graph_norm = rw
item_graph_modal_layers = 1
item_graph_modal_target = all
```

Baseline Stage 2 config:

```text
configs/sports/stage2_decoder_mm.yaml
```

The best current run is not the raw YAML default; it depends on CLI overrides.

## Stage 1 Inputs

### Stage 1.1

Historical Stage 1.1 checkpoint used by the best Stage 1.2 run:

```text
exp_report/sports/stage1_1_sports_imputer_param_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/ckpt/stage1_1_sports_imputer_param_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817_imputer_param_50_epoch2.pth
```

Evidence log:

```text
exp_report/sports/stage1_1_sports_imputer_param_mmrec_sports_mm_mr0.3_seed1_mmrec_sports_mm_fixedmissing_20260524_165817/log/run_20260524_165817.log
```

Key settings:

```text
config = configs/sports/stage1_1_imputer_param.yaml
seed = 1
dataset_seed = 0
epoch = 5
batch_size = 256
lr = 0.001
generative_update_mode = em
best_epoch = 2
```

### Stage 1.2

Stage 1.2 checkpoint used by the current best Stage 2 result:

```text
exp_report/sports/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233/ckpt/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233_imputer_backprop_50_epoch49.pth
```

Evidence log:

```text
exp_report/sports/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233/run_full.log
```

Key settings:

```text
config = configs/sports/stage1_2_decoder_v2.yaml
seed = 1
dataset_seed = 0
epoch = 50
batch_size = 256
lr = 0.0005
lr_imp = 0.0005
lr_decoder = 0.0002
stage1_profile = v2
stage1_v2_loss_preset = balanced
generative_update_mode = fixed
stage1_rec_loss_mode = observed
decode_loss_target_mode = observed
decode_loss_grad_mode = detached
save_all_epochs = 1
```

Important: the Stage 1.2 log reports best epoch 46, but the current best
Stage 2 search used the epoch 49 checkpoint.

### Reproduction Status, 2026-07-01

The 0.10489 Recall@20 result is tied to the historical Stage 1.2 checkpoint
listed above. A same-config Stage 1.1 -> Stage 1.2 rebuild did not reproduce the
historical Stage 2 result.

Rebuild attempt:

```text
Stage 1.1:
config = configs/sports/stage1_1_imputer_param.yaml
seed = 1
dataset_seed = 0
epoch = 5
batch_size = 256
lr = 0.001
generative_update_mode = em

Stage 1.2:
input = rebuilt Stage 1.1 epoch2
config = configs/sports/stage1_2_decoder_v2.yaml
used checkpoint = rebuilt epoch49

Stage 2:
candidate = reg1em03
batch_size = 2048
lr = lr_rec = 0.002
reg_coeff = 0.001
item_graph_topk = 8
evaluation_protocol = strict
```

Observed rebuild result:

```text
Recall@20 = 0.10200
NDCG@20 = 0.04578
best_epoch = 313
```

Expected historical result:

```text
Recall@20 = 0.10489
NDCG@20 = 0.04742
best_epoch = 358
```

Rebuild log reported by the 2026-07-01 attempt:

```text
exp_report/sports/repro_best/sports_repro_reg1em03_seed1_20260701_001718/logs/stage2_sports_mr0p3_bs2048_reg1em03_sports_repro_reg1em03_seed1_20260701_001718.log
```

Do not treat the Stage 1 rebuild as equivalent to the historical checkpoint
unless another run closes this gap. For follow-up search, either recover/use the
historical Stage 1.2 checkpoint exactly, or explicitly reset the baseline to the
rebuilt-chain result above.

## Search 1: Sequential Sports Hparam Search, 2026-06-13

Script:

```text
scripts/run_sports_hparam_search.py
```

Run directory:

```text
exp_report/sports/hparam_search/sports_hparam_20260613_094108
```

Status: partial. Only `01_batch_size` and `02_lr_rec` completed. The planned
later steps did not produce completed summaries in this run.

Planned search ranges from the script:

```text
batch_size: [128, 256, 512, 1024]
lr_rec: [0.0005, 0.001, 0.002, 0.005]
modality_bpr_coeff: [0.0, 0.1, 0.2, 0.5, 1.0]
reg_coeff: [1e-5, 1e-4, 3e-4, 1e-3]
rec_neighbor_cl_weight: [0.0, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]
```

Completed batch-size step:

| candidate | best_epoch | val_R20 | val_N20 | test_R20 | test_N20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bs128 | 90 | 0.09296 | 0.04088 | 0.09445 | 0.04225 |
| bs256 | 144 | 0.09458 | 0.04205 | 0.09541 | 0.04264 |
| bs512 | 198 | 0.09352 | 0.04142 | 0.09585 | 0.04277 |
| bs1024 | 199 | 0.09268 | 0.04079 | 0.09268 | 0.04079 |

Completed learning-rate step:

| candidate | best_epoch | val_R20 | val_N20 | test_R20 | test_N20 |
| --- | ---: | ---: | ---: | ---: | ---: |
| lr0p0005 | 197 | 0.09209 | 0.04102 | 0.09371 | 0.04133 |
| lr0p001 | 144 | 0.09458 | 0.04205 | 0.09541 | 0.04264 |
| lr0p002 | 86 | 0.09264 | 0.04101 | 0.09489 | 0.04284 |
| lr0p005 | 41 | 0.08900 | 0.03983 | 0.09262 | 0.04140 |

Takeaway: this search suggested `batch_size=256` and `lr_rec=0.001` by val
Recall@20, but it was not a complete search.

## Search 2: Mainline After Clothing Transfer, 2026-06-25

Script:

```text
scripts/run_sports_mainline_hparam_search.sh
```

Run directory:

```text
exp_report/sports/mainline_hparam_search/sports_mainline_after_clothing_20260625_112854
```

Summary:

```text
exp_report/sports/mainline_hparam_search/sports_mainline_after_clothing_20260625_112854/summary.tsv
```

Results:

| rank | candidate | Recall@20 | NDCG@20 | best_epoch |
| ---: | --- | ---: | ---: | ---: |
| 1 | clothing_topk10_transfer | 0.10030 | 0.04534 | 156 |
| 2 | sports_prev_center | 0.09770 | 0.04371 | 196 |
| 3 | ckpt_e19 | 0.09770 | 0.04371 | 196 |
| 4 | ckpt_e18 | 0.09728 | 0.04348 | 195 |
| 5 | ckpt_e15 | 0.09549 | 0.04296 | 195 |
| 6 | sports_current_mainline | 0.09531 | 0.04243 | 195 |

Best candidate parameters:

```text
tag = clothing_topk10_transfer
stage1.2_ckpt_epoch = 19
batch_size = 256
modality_bpr_coeff = 1.0
lr_rec = 0.001
reg_coeff = 0.0001
item_graph_modal_alpha = 0.25
rec_neighbor_cl_weight = 0.005
rec_neighbor_cl_temp = 0.2
rec_neighbor_cl_bank_size = 256
item_graph_cf_weight = 0.3
item_graph_image_weight = 0.35
item_graph_text_weight = 0.35
item_graph_topk = 10
```

Takeaway: transferring Clothing topk10 style to Sports was better than the
previous Sports center.

## Search 3: Multigraph Mainline Search, 2026-06-26

Script:

```text
scripts/run_sports_multigraph_mainline_search.sh
```

Run directory:

```text
exp_report/sports/multigraph_mainline_search/sports_multigraph_mainline_mr0p3_seed1_gpu56_20260626_080855
```

Summary:

```text
exp_report/sports/multigraph_mainline_search/sports_multigraph_mainline_mr0p3_seed1_gpu56_20260626_080855/summary.tsv
```

Top completed results:

| rank | candidate | Recall@20 | NDCG@20 | best_epoch |
| ---: | --- | ---: | ---: | ---: |
| 1 | topk8 | 0.10168 | 0.04617 | 198 |
| 2 | center_ckpt15 | 0.10101 | 0.04612 | 196 |
| 3 | topk12 | 0.10096 | 0.04559 | 197 |
| 4 | topk5 | 0.10081 | 0.04591 | 187 |
| 5 | center_transfer_topk10 | 0.10030 | 0.04534 | 156 |
| 6 | center_ckpt18 | 0.10002 | 0.04565 | 166 |

Best candidate parameters:

```text
tag = topk8
stage1.2_ckpt_epoch = 19
batch_size = 256
modality_bpr_coeff = 1.0
lr_rec = 0.001
reg_coeff = 0.0001
item_graph_modal_alpha = 0.25
rec_neighbor_cl_weight = 0.005
rec_neighbor_cl_temp = 0.2
rec_neighbor_cl_bank_size = 256
item_graph_cf_weight = 0.3
item_graph_image_weight = 0.35
item_graph_text_weight = 0.35
item_graph_topk = 8
```

Takeaway: topk8 improved over topk10 under the same transferred center.

## Search 4: Topk8 + Clothing-Style Stage1 + Batch 2048, 2026-06-27

Script:

```text
scripts/run_sports_topk8_clothingstage1_bs2048_search.sh
```

Run directory:

```text
exp_report/sports/topk8_clothingstage1_bs2048_search/sports_topk8_clothingstage1_bs2048_2gpu_noepochcap_20260627_233347
```

Summary:

```text
exp_report/sports/topk8_clothingstage1_bs2048_search/sports_topk8_clothingstage1_bs2048_2gpu_noepochcap_20260627_233347/summary.tsv
```

Command files for every completed candidate live under:

```text
exp_report/sports/topk8_clothingstage1_bs2048_search/sports_topk8_clothingstage1_bs2048_2gpu_noepochcap_20260627_233347/logs/*.log.cmd
```

Top completed results:

| rank | candidate | Recall@20 | NDCG@20 | best_epoch |
| ---: | --- | ---: | ---: | ---: |
| 1 | reg1em03 | 0.10489 | 0.04742 | 358 |
| 2 | reg3em04 | 0.10353 | 0.04696 | 353 |
| 3 | reg2em04 | 0.10342 | 0.04658 | 330 |
| 4 | mbpr1p5 | 0.10305 | 0.04672 | 302 |
| 5 | mbpr0p5 | 0.10294 | 0.04596 | 353 |
| 6 | reccl0p015 | 0.10287 | 0.04627 | 294 |
| 7 | reccl_bank128 | 0.10247 | 0.04616 | 300 |
| 8 | reccl0p003 | 0.10237 | 0.04627 | 300 |
| 9 | reccl_bank1024 | 0.10235 | 0.04636 | 311 |
| 10 | reccl_temp0p25 | 0.10231 | 0.04629 | 302 |
| 11 | lr0p002 | 0.10231 | 0.04628 | 311 |
| 12 | reccl_bank512 | 0.10218 | 0.04620 | 296 |
| 13 | reg5em05 | 0.10212 | 0.04619 | 311 |
| 14 | reccl_temp0p15 | 0.10209 | 0.04621 | 312 |
| 15 | lr0p004 | 0.10201 | 0.04620 | 188 |
| 16 | mbpr2p0 | 0.10198 | 0.04655 | 310 |
| 17 | lr0p003 | 0.10189 | 0.04621 | 234 |
| 18 | topk12 | 0.10185 | 0.04580 | 302 |
| 19 | lr0p0015 | 0.10183 | 0.04540 | 342 |
| 20 | reccl0p010 | 0.10177 | 0.04606 | 293 |
| 21 | graph_modal040 | 0.10128 | 0.04562 | 309 |
| 22 | topk10 | 0.10104 | 0.04584 | 292 |
| 23 | center_lr0p001 | 0.10104 | 0.04517 | 480 |
| 24 | lr0p006 | 0.10086 | 0.04536 | 129 |
| 25 | topk5 | 0.10043 | 0.04512 | 229 |
| 26 | topk20 | 0.10041 | 0.04519 | 329 |
| 27 | reccl0 | 0.09988 | 0.04453 | 211 |
| 28 | graph_cf040 | 0.09972 | 0.04445 | 207 |
| 29 | topk15 | 0.09965 | 0.04449 | 239 |

Center for this search:

```text
stage1.2_ckpt = exp_report/sports/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233/ckpt/stage1_2_sports_clothing_stage1style_mr0p3_seed1_20260627_004233_imputer_backprop_50_epoch49.pth
batch_size = 2048
modality_bpr_coeff = 1.0
lr_rec = 0.002
reg_coeff = 0.0001
item_graph_modal_alpha = 0.25
rec_neighbor_cl_weight = 0.005
rec_neighbor_cl_temp = 0.2
rec_neighbor_cl_bank_size = 256
item_graph_cf_weight = 0.3
item_graph_image_weight = 0.35
item_graph_text_weight = 0.35
item_graph_topk = 8
```

Current best candidate parameters:

```text
tag = reg1em03
batch_size = 2048
lr = 0.002
lr_rec = 0.002
reg_coeff = 0.001
modality_bpr_coeff = 1.0
rec_neighbor_cl_weight = 0.005
rec_neighbor_cl_temp = 0.2
rec_neighbor_cl_bank_size = 256
item_graph_topk = 8
item_graph_cf_weight = 0.3
item_graph_image_weight = 0.35
item_graph_text_weight = 0.35
item_graph_modal_alpha = 0.25
```

Best command file:

```text
exp_report/sports/topk8_clothingstage1_bs2048_search/sports_topk8_clothingstage1_bs2048_2gpu_noepochcap_20260627_233347/logs/stage2_sports_mr0p3_bs2048_reg1em03_sports_topk8_clothingstage1_bs2048_2gpu_noepochcap_20260627_233347.log.cmd
```

Best final strict test metrics:

```text
Recall@10 = 0.07046
NDCG@10 = 0.03855
Recall@20 = 0.10489
NDCG@20 = 0.04742
Recall@30 = 0.13055
NDCG@30 = 0.05302
Recall@40 = 0.15192
NDCG@40 = 0.05727
Recall@50 = 0.16855
NDCG@50 = 0.06035
best_epoch = 358
```

Takeaway: the strongest improvements came from the new Stage 1.2 checkpoint,
larger batch size, topk8 graph, and stronger regularization. Removing
rec-neighbor CL was bad in this search (`reccl0` dropped to Recall@20 0.09988).

## Suggested Next Search Directions

If the historical Stage 1.2 checkpoint is available, start from the current
historical best:

```text
batch_size = 2048
lr_rec = 0.002
reg_coeff = 0.001
modality_bpr_coeff = 1.0
rec_neighbor_cl_weight = 0.005
rec_neighbor_cl_temp = 0.2
rec_neighbor_cl_bank_size = 256
item_graph_topk = 8
item_graph_cf/image/text = 0.3/0.35/0.35
item_graph_modal_alpha = 0.25
```

Reasonable next local ranges:

```text
reg_coeff: [0.0006, 0.0008, 0.001, 0.0012, 0.0015, 0.002]
modality_bpr_coeff: [0.8, 1.0, 1.2, 1.5]
lr_rec: [0.0015, 0.002, 0.0025, 0.003]
rec_neighbor_cl_weight: [0.003, 0.005, 0.0075, 0.01, 0.015]
item_graph_topk: [6, 8, 10, 12]
item_graph_modal_alpha: [0.15, 0.2, 0.25, 0.3, 0.35]
item_graph_cf/image/text: [0.25/0.375/0.375, 0.3/0.35/0.35, 0.2/0.4/0.4]
```

High-priority full verification:

```text
Run the best historical config across seeds [1, 12, 123, 1234, 12345].
```

Risk note: the current best is seed 1 only. Do not promote it as a stable
multi-seed mainline until the multi-seed check is complete. Also do not mix
historical-checkpoint results with rebuilt-Stage1 results in the same search
table.

If the historical checkpoint cannot be recovered, use the rebuilt-chain result
as the new baseline:

```text
baseline = rebuilt Stage1 chain, reg1em03
Recall@20 = 0.10200
NDCG@20 = 0.04578
best_epoch = 313
```

In that case, rerun a compact local search around the same center before
launching a wide search:

```text
reg_coeff: [0.0006, 0.0008, 0.001, 0.0012, 0.0015]
modality_bpr_coeff: [0.8, 1.0, 1.2, 1.5]
lr_rec: [0.0015, 0.002, 0.0025]
rec_neighbor_cl_weight: [0.003, 0.005, 0.0075, 0.01]
item_graph_topk: [6, 8, 10]
```
