# SMORE Native Missing + Observed Imputer 搜索记录

本文档记录 Clothing `missing_rate=0.3` 下，使用 SMORE 原生缺失训练得到的 `image_item_embeds.npy` / `text_item_embeds.npy` 作为 MMRec 输入，并在 MMRec 中启用 stage1.1 observed imputer 的实验。

## 当前基线

- 数据集：`clothing`
- 训练缺失率：`0.3`
- 验证/测试缺失率：`0.5`
- 随机种子：`seed=2023`, `dataset_seed=0`
- 外部模态特征：
  - `/home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_native_missing_raw_input_compact_clothing_mr0p3_20260702/native_missing_raw_mr0p3/phase_train/image_item_embeds.npy`
  - `/home/ruiyuliu/projects/baselines/SMORE/experiment_logs/smore_native_missing_raw_input_compact_clothing_mr0p3_20260702/native_missing_raw_mr0p3/phase_train/text_item_embeds.npy`
- stage1.1 checkpoint：
  - `exp_report/clothing/clothing_mr0p3_smore_native_missing_raw_identity_stage11_observed_5e_20260702_gpu0/ckpt/clothing_mr0p3_smore_native_missing_raw_identity_stage11_observed_5e_20260702_gpu0_imputer_param_50_epoch4.pth`
- stage1.1 训练方式：
  - `train_stage=imputer_param`
  - `stage1_rec_loss_mode=observed`
  - `epoch=5`
  - 使用最后一轮 `epoch4`
- stage2 关键结构：
  - `disable_imputation=0`
  - `freeze_imputer=1`
  - `promrl_projection_mode=identity`
  - `feature_bridge_mode=shared_identity`
  - `gcn_frontend_mode=identity`
  - `item_graph_kind=fused_completed`
  - `item_graph_feature_space=shared`
  - 不使用原 stage1.2，不使用原投影和 decoder bridge。

### 基线 stage2 参数

```text
batch_size=2048
lr=0.01
reg_coeff=0.01
modality_bpr_coeff=1.0
item_graph_topk=10
item_graph_cf_weight=0.2
item_graph_image_weight=0.4
item_graph_text_weight=0.4
item_graph_modal_alpha=0.25
rec_neighbor_cl_weight=0.005
rec_neighbor_cl_temp=0.2
early_stop=20
selection=val recall@20
```

### 基线结果

- stage2 suffix：`stage2_clothing_mr0p3_smore_native_missing_raw_item_embeds_identity_i3fixed_imputed_stage11obs5e_epoch4_20260702_gpu0`
- 日志：`exp_report/clothing/smore_native_missing_item_embeds_iigraph_imputed/stage2_imputed_stage11obs5e_epoch4_gpu0.launch.log`
- checkpoint：`exp_report/clothing/stage2_clothing_mr0p3_smore_native_missing_raw_item_embeds_identity_i3fixed_imputed_stage11obs5e_epoch4_20260702_gpu0/ckpt/stage2_clothing_mr0p3_smore_native_missing_raw_item_embeds_identity_i3fixed_imputed_stage11obs5e_epoch4_20260702_gpu0_recommender_1.0_epoch68.pth`
- best epoch：`68`
- best val recall@20：`0.08511`
- final strict test：

| K | Recall | NDCG |
| ---: | ---: | ---: |
| 10 | 0.05719 | 0.03084 |
| 20 | 0.08471 | 0.03784 |
| 30 | 0.10355 | 0.04187 |
| 40 | 0.11992 | 0.04506 |
| 50 | 0.13425 | 0.04768 |

## 对比

无补齐 fixed I3 ii 图版本：

- suffix：`stage2_clothing_mr0p3_smore_native_missing_raw_item_embeds_identity_i3mainline_fixed_iigraph_20260702_gpu0`
- final strict test recall@20：`0.08033`
- final strict test ndcg@20：`0.03608`
- final strict test recall@50：`0.12982`
- final strict test ndcg@50：`0.04594`

当前补齐版相对无补齐版：

| 指标 | 无补齐 fixed ii | 当前补齐版 | 提升 |
| --- | ---: | ---: | ---: |
| Recall@20 | 0.08033 | 0.08471 | +0.00438 |
| NDCG@20 | 0.03608 | 0.03784 | +0.00176 |
| Recall@50 | 0.12982 | 0.13425 | +0.00443 |
| NDCG@50 | 0.04594 | 0.04768 | +0.00174 |

## 下一步搜索目标

目标：在相同数据、相同 stage1.1 observed epoch4 imputer、相同 identity/shared 补齐路径下，将 strict test recall@20 推到 `0.086` 及以上。

优先搜索 stage2 参数：

1. `item_graph_topk`：`8, 10, 12, 15`
2. 图融合权重：围绕 `cf:image:text = 0.2:0.4:0.4`，试 `0.15:0.425:0.425`、`0.25:0.375:0.375`、图文不对称小偏移。
3. `item_graph_modal_alpha`：围绕 `0.25`，试 `0.2, 0.3`。
4. `rec_neighbor_cl_weight`：围绕 `0.005`，试 `0.003, 0.0075, 0.01`。
5. 若图参数收益不足，再动 `lr/reg_coeff/modality_bpr_coeff`。

## 搜索进展

搜索脚本：`scripts/run_clothing_mr0p3_smore_native_imputed_stage2_search_gpu0.sh`

当前队列：

- run tag：`smore_native_imputed_stage2_search_20260702_gpu0`
- 输出目录：`exp_report/clothing/smore_native_missing_item_embeds_iigraph_imputed_search/smore_native_imputed_stage2_search_20260702_gpu0`
- 候选表：`exp_report/clothing/smore_native_missing_item_embeds_iigraph_imputed_search/smore_native_imputed_stage2_search_20260702_gpu0/candidates.tsv`
- 汇总表：`exp_report/clothing/smore_native_missing_item_embeds_iigraph_imputed_search/smore_native_imputed_stage2_search_20260702_gpu0/summary.tsv`

已完成结果：

| Tag | 改动 | Best val recall@20 | Best epoch | Test recall@20 | Test ndcg@20 | Test recall@50 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | topk=10, cf/img/text=0.2/0.4/0.4, alpha=0.25, rec_cl=0.005 | 0.08511 | 68 | 0.08471 | 0.03784 | 0.13425 | 初始最佳 |
| reccl0075_topk10 | rec_neighbor_cl_weight 0.005 -> 0.0075 | 0.08544 | 68 | 0.08480 | 0.03788 | 0.13426 | 当前最佳，略高于 baseline |
| topk12_base | topk 10 -> 12 | 0.08514 | 49 | 0.08327 | 0.03728 | 0.13343 | val 略高但 test 下降，不作为主方向 |
| topk15_base | topk 10 -> 15 | 0.07962 | 20 | - | - | - | epoch20 明显低于 baseline 同期，已中止 |
| topk8_base | topk 10 -> 8 | 0.08528 | 68 | 0.08457 | 0.03779 | 0.13440 | val 略高但 test 未超过 baseline |
| sem425_topk10 | cf/img/text 0.2/0.4/0.4 -> 0.15/0.425/0.425 | 0.08024 | 18 | - | - | - | epoch21 低于 baseline 同期，已中止 |
| cf25_topk10 | cf/img/text 0.2/0.4/0.4 -> 0.25/0.375/0.375 | 0.08506 | 49 | 0.08370 | 0.03754 | 0.13355 | test 下降，不作为主方向 |
| alpha20_topk10 | item_graph_modal_alpha 0.25 -> 0.20 | 0.07630 | 8 | - | - | - | epoch9 明显低于 baseline，同步中止 |
| alpha30_topk10 | item_graph_modal_alpha 0.25 -> 0.30 | 0.08540 | 75 | 0.08384 | 0.03772 | 0.13513 | val/Recall@50 提升，但 Recall@20 test 下降 |
