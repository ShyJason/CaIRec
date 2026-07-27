# MMRec 当前实验进度

本文档只保留当前仍在仓库中实际保留的主线结果，避免继续引用已经清理掉的历史实验目录。

## 1. 当前结论

- `legacy` 是当前默认主线，也是后续复现实验的唯一默认入口。
- `v2` 仅作为 `stage1_2` 的可选支线保留，不覆盖旧脚本、不替换旧配置、不改旧结果口径。
- 当前主线关注 `baby + mm + seed2023`，其余旧扫参、smoke、长周期和跨数据集试跑目录已清理。

## 2. 当前保留结果

### 2.1 Stage 1.1 主线基线

- 目录：
  - [stage1_1_baby_seed2023_completion_mm_joint](../exp_report/baby/stage1_1_baby_seed2023_completion_mm_joint)

### 2.2 Stage 1.2 `legacy` 主线

- 历史基线：
  - [stage1_2_baby_seed2023_mm_epoch20_diag](../exp_report/baby/stage1_2_baby_seed2023_mm_epoch20_diag)
- Phase 1 回退安全复跑：
  - [stage1_2_baby_seed2023_mm_phase1_legacy_rerun](../exp_report/baby/stage1_2_baby_seed2023_mm_phase1_legacy_rerun)

Phase 1 结论：

- `best_epoch = 0` 与历史基线一致
- `train/test mse` 与 `cosine` 一致
- `stage1_profile=legacy` 下，当前代码没有破坏原有可复现实验链路

### 2.3 Stage 1.2 `v2` 支线

- 目录：
  - [stage1_2_baby_seed2023_mm_phase2_v2](../exp_report/baby/stage1_2_baby_seed2023_mm_phase2_v2)

Phase 2 结论：

- `best_epoch = 4`，不再固定在 `0`
- 被选中的 held-out 指标：
  - `val_shared_cosine_gap = 0.238781`
  - `val_missing_decode_cosine = 0.495757`
  - `val_shared_mse = 0.021679`
- 相比 `legacy` epoch 0：
  - `test mse`: `0.021918 -> 0.021541`
  - `test cosine`: `0.298631 -> 0.310683`

解释：

- `v2` 在 `stage1_2` 上有正向信号
- 但仍然只是支线观察结果，不应替代主线默认配置

### 2.4 Stage 2 对照

- `legacy`:
  - [stage2_baby_seed2023_mm_phase1_legacy](../exp_report/baby/stage2_baby_seed2023_mm_phase1_legacy)
- `v2`:
  - [stage2_baby_seed2023_mm_phase2_v2](../exp_report/baby/stage2_baby_seed2023_mm_phase2_v2)

两条路径的 `best_epoch` 都是 `29`。`v2` 在 `NDCG` 上是小幅正向，在 `Recall/HR` 上不是单边优势。

| 路径 | Recall@20 | NDCG@20 | Recall@50 | NDCG@50 |
|---|---:|---:|---:|---:|
| `legacy` | 0.07711 | 0.03310 | 0.13669 | 0.04513 |
| `v2` | 0.07653 | 0.03322 | 0.13547 | 0.04517 |

当前判断：

- `v2` 可以继续保留用于后续验证
- 默认主线仍维持 `legacy`

## 3. 当前仓库约定

- 主线脚本：
  - `run_stage1_2_baby_imputer_backprop_decoder_v2.sh`
- 主线配置：
  - `configs/baby/stage1_2_decoder_v2.yaml`
- 支线配置：
  - `configs/baby/stage1_2_decoder_v2.yaml`

约束：

- `legacy` 默认行为不可变
- `v2` 只能显式开启
- `legacy` 与 `v2` 的 checkpoint/output 目录必须隔离

## 4. 已完成清理

已删除：

- 旧的 `baby/clothing/tiktok` smoke、grid、longcycle、demo、tensorboard 和历史扫参目录
- `__pycache__`
- `.DS_Store`
- 零散 launcher/background 日志

当前仅保留上文列出的最小结果集，用于后续主线复现和 `v2` 对照验证。
