# Stage 1 实验记录

## Summary

这份文档记录当前单数据集三步方案中，`baby` 数据集的第一阶段训练结果：

1. `stage 1.1`: `imputer_param`
2. `stage 1.2`: `imputer_backprop`

当前结论：

- `stage 1.1` 已稳定补齐参数 `W / mu / log_sigma`
- `stage 1.2` 已在此基础上进一步训练 `Contra_head + ITM/MLP`
- 第一阶段整体训练稳定，可进入 `stage 2`

## Experiment Setup

- 数据集：`baby`
- 模态设置：`ff`
- 设备：`cuda:4`
- 缺失率参数：`0.3`
- 当前目标：只完成第一阶段，不训练推荐主干

### Stage 1.1

- 阶段名：`imputer_param`
- 脚本：[run_stage1_1_baby_imputer_param.sh](../../run_stage1_1_baby_imputer_param.sh:1)
- 默认轮数：`5`
- 默认 loss：

```text
L_1.1 = 1.0 * rec_loss
```

- checkpoint：
  - [stage1_1_baby_imputer_param_imputer_param_50_epoch4.pth](../../exp_report/baby/stage1_1_baby_imputer_param/ckpt/stage1_1_baby_imputer_param_imputer_param_50_epoch4.pth)

### Stage 1.2

- 阶段名：`imputer_backprop`
- 脚本：[run_stage1_2_baby_imputer_backprop_decoder_v2.sh](../../run_stage1_2_baby_imputer_backprop_decoder_v2.sh:1)
- 默认轮数：`5`
- 默认 loss：

```text
L_1.2 = loss_intra + loss_inter + loss_itm + 0.1 * rec_loss
```

- 加载的 checkpoint：
  - [stage1_1_baby_imputer_param_imputer_param_50_epoch4.pth](../../exp_report/baby/stage1_1_baby_imputer_param/ckpt/stage1_1_baby_imputer_param_imputer_param_50_epoch4.pth)
- 输出 checkpoint：
  - [stage1_2_baby_imputer_backprop_imputer_backprop_50_epoch4.pth](../../exp_report/baby/stage1_2_baby_imputer_backprop/ckpt/stage1_2_baby_imputer_backprop_imputer_backprop_50_epoch4.pth)

## Results

### Stage 1.1: `imputer_param`

每轮结果如下：

| epoch | loss_s1 | promrl_rec | train_time(s) |
|---|---:|---:|---:|
| 0 | -2.30739 | -2.30739 | 2.91 |
| 1 | -2.31614 | -2.31614 | 2.57 |
| 2 | -2.31735 | -2.31735 | 2.53 |
| 3 | -2.31778 | -2.31778 | 2.41 |
| 4 | -2.31893 | -2.31893 | 2.51 |

观察：

- `promrl_rec` 单调变得更稳定，说明补齐参数已经脱离随机初始化
- 该阶段只训练补齐参数，因此：
  - `promrl_intra = 0`
  - `promrl_inter = 0`
  - `promrl_itm = 0`
  - 推荐相关 loss 全为 `0`
- 总训练时间约 `13.03s`

阶段结论：

- `stage 1.1` 运行正常
- `rec_loss` 数值稳定，没有发散或 NaN
- 可以进入 `stage 1.2`

### Stage 1.2: `imputer_backprop`

每轮结果如下：

| epoch | loss_s1 | promrl_intra | promrl_inter | promrl_itm | promrl_rec | train_time(s) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | -0.03585 | 0.01723 | 0.07073 | 0.06371 | -1.87528 | 54.75 |
| 1 | -0.04426 | 0.00360 | 0.05653 | 0.06367 | -1.68053 | 54.08 |
| 2 | -0.04564 | 0.00297 | 0.05469 | 0.06366 | -1.66961 | 54.03 |
| 3 | -0.04602 | 0.00256 | 0.05384 | 0.06365 | -1.66074 | 53.90 |
| 4 | -0.04628 | 0.00237 | 0.05331 | 0.06365 | -1.65611 | 53.97 |

观察：

- `promrl_intra` 明显下降：
  - `0.01723 -> 0.00237`
- `promrl_inter` 稳定下降：
  - `0.07073 -> 0.05331`
- `promrl_itm` 基本稳定：
  - `0.06371 -> 0.06365`
- `promrl_rec` 从 `-1.87528` 回到 `-1.65611`
  - 没有数值发散
  - 说明补齐参数和判别空间在趋于稳定
- 总训练时间约 `271.14s`

阶段结论：

- `stage 1.2` 训练稳定
- `Contra_head + ITM/MLP` 已在 `stage 1.1` 的参数基础上完成对齐
- 当前 checkpoint 可以作为正式 `imputer_ckpt` 输入到 `stage 2`

## Interpretation

当前第一阶段的整体结论如下：

1. `stage 1.1` 已完成“只训练补齐参数”的任务
2. `stage 1.2` 已完成“补齐参数 + 反向传播训练表示层与 ITM/MLP”的任务
3. 第一阶段没有异常：
   - 无 NaN
   - 无 loss 爆炸
   - 无明显数值震荡
4. 当前结果支持进入第二阶段推荐训练

需要注意：

- 第一阶段的 `HR/Recall/NDCG` 为 `0` 是正常现象
- 因为第一阶段不训练推荐主干，这些指标不用于判断第一阶段是否成功

## Next Step

下一步应固定使用 `stage 1.2` 的 checkpoint，分别运行：

1. `stage2 + fm + with imputation`
2. `stage2 + fm + without imputation`
3. `stage2 + mm + with imputation`
4. `stage2 + mm + without imputation`

推荐入口：

- [run_stage2_baby_recommender_decoder.sh](../../run_stage2_baby_recommender_decoder.sh:1)
- [run_stage2_baby_recommender_decoder_no_impute.sh](../../run_stage2_baby_recommender_decoder_no_impute.sh:1)

## Stage 2 Progress

### Stage 2: `baby + fm + with imputation`

- 阶段名：`recommender`
- 模式：`fm`
- 脚本：[run_stage2_baby_recommender_decoder.sh](../../run_stage2_baby_recommender_decoder.sh:1)
- 使用 checkpoint：
  - [stage1_2_baby_imputer_backprop_imputer_backprop_50_epoch4.pth](../../exp_report/baby/stage1_2_baby_imputer_backprop/ckpt/stage1_2_baby_imputer_backprop_imputer_backprop_50_epoch4.pth)

最优结果：

| best epoch | HR@10 | NDCG@10 | HR@20 | NDCG@20 | HR@50 | NDCG@50 |
|---|---:|---:|---:|---:|---:|---:|
| 9 | 0.04328 | 0.02370 | 0.06784 | 0.03003 | 0.12332 | 0.04124 |

训练趋势：

- `main_bpr_loss`: `0.67615 -> 0.32622`
- `modality_bpr_loss`: `1.17195 -> 0.70064`
- `val ndcg@10`: `0.00576 -> 0.02214`
- `test ndcg@10`: `0.00575 -> 0.02370`

当前判断：

- `stage2 + fm + with imputation` 训练稳定
- 推荐指标持续上升，到第 `9` 轮仍在提升，说明训练有效
- 这组结果已经可以作为 `fm` 主实验结果之一
- 但是否证明补齐有效，仍需等待：
  - `stage2 + fm + without imputation`
  - `stage2 + mm + with/without imputation`
