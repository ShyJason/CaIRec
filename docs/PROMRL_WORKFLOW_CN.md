# MMRec: Missing-Modality Recommendation 当前 Workflow

## 1. 当前目标边界

当前代码路径明确遵守以下边界：

- 保留原始 I3 的推荐骨干：
  - `original linear MLP`
  - `GCN`
  - `fusion`
  - `BPR`
- 保留模态级推荐监督，但只保留到 **env-wise modality BPR**
- 不恢复以下原始 I3 组件：
  - `penalty_loss`
  - `InfoNCE`
  - `CLUB`
  - `mutual_info`
  - `IRM / invariant learning`

因此当前系统不是原始 I3 的完整复现，而是：

`PROMRL completion -> raw_decoder -> original I3 linear MLP -> GCN -> fusion -> BPR`

并辅以：

- `env-wise modality BPR`
- `regularization`

补充约束：

- `legacy` 是默认主线
- `v2` 仅作为 `stage1_2` 的显式 opt-in 支线
- 主线脚本和主线配置保持原名，不因支线实验而改变默认语义

## 2. 当前数据流

### 2.1 缺失协议

当前仍使用“老协议”：

1. 先对训练集和测试集做**第一次全局人工缺失**
2. 训练和测试阶段直接使用这批第一次缺失后的特征
3. 不再使用后续那版“两层缺失 / 第二次 mask only for eligible items”的协议

对应代码：

- [dataset_loader.py](../dataset_loader.py)
- [model.py](../model.py) 中的 `init_missing_modality_set()` 与 `set_missing_modality_via_env()`

### 2.2 `ff / fm / mf / mm` 的含义

- `ff`
  - 训练完整模态
  - 测试完整模态
- `fm`
  - 训练完整模态
  - 测试缺失模态
- `mf`
  - 训练缺失模态
  - 测试完整模态
- `mm`
  - 训练缺失模态
  - 测试缺失模态

这一定义与原始 I3 的协议保持一致。

## 3. 当前主线结构

### 3.1 有补齐时

当前默认带补齐主线为：

`missing/full raw modality features -> PROMRL -> raw_decoder -> original linear MLP -> modal GCN -> fusion -> main BPR`

说明：

- `PROMRL`
  - 负责投影、后验推断、缺失补齐
- `raw_decoder`
  - 把补齐后的共享表示桥接回推荐器可用的原始模态空间
- `original linear MLP`
  - 即 `MGCN` 前端单层 `nn.Linear`
- `modal GCN`
  - 每个模态分支各自做图传播
- `fusion`
  - 最终融合 user/item embedding

### 3.2 无补齐时

无补齐对照组主线为：

`missing raw modality features (missing part = 0) -> original linear MLP -> modal GCN -> fusion -> main BPR`

注意：

- 缺失模态在输入层被置 0
- 但进入 `Linear` 之后会因为 bias 和后续 GCN 传播变成非零表示
- 所以“无补齐”不等于“完全没有该模态信息”

## 4. 当前 Stage 1 Workflow

### 4.1 Stage 1.1: `imputer_param`

作用：

- 只训练生成参数相关部分
- 作为补齐器的初始化步骤

当前输入：

- 使用第一次缺失后的训练特征

当前 loss：

- `alpha_rec * promrl_rec`

### 4.2 Stage 1.2: `imputer_backprop`

当前已按“更接近 PROMRL”的方向做了两项调整：

1. `stage1` 改成 **item-batch**
   - 不再通过 `PairSample` 的 `pos/neg` 唯一 item 集合间接训练
   - 直接从训练 item 集合做补齐训练
2. `rec_loss` 改成按 **observed-modality pattern** 计算
   - 不再只在“所有模态都观测到”的样本上算

此外：

- `itm_loss` 也已经改成基于 **completed features** 计算
- 第一次已缺 item 可以更像 PROMRL 那样直接参与 completion training

当前 `stage1.2` 组成项：

- `promrl_rec`
- `promrl_decode`
- `loss_intra`
- `loss_inter`
- `loss_itm`

当前选模：

- 仍支持按显式补齐指标选 best

相关文件：

- [main.py](../main.py)
- [model.py](../model.py)
- [session.py](../session.py)

## 5. 当前 Stage 2 Workflow

### 5.1 基础推荐损失

当前 recommender 阶段总损失为：

`L_stage2 = main_bpr_loss + modality_bpr_loss + reg_loss`

其中：

- `main_bpr_loss`
  - 最终融合后的 user/item embedding 上的 BPR
- `modality_bpr_loss`
  - 当前为 **env-wise modality BPR**
- `reg_loss`
  - L2 regularization

### 5.2 当前保留的模态级推荐监督

当前不再使用最初那版“branch-wise modality BPR”，已经改成更接近原始 I3 的：

- **env-wise modality BPR**

实现逻辑：

1. 构造 `mix_ration`
   - 两模态时为：
     - `[1, 0]`
     - `[0, 1]`
2. 调用当前模型中的 `get_env_emb(...)`
3. 对每个 environment 单独算一个 BPR
4. 求和得到 `modality_bpr_loss`

当前**明确不做**：

- `penalty_loss = var(env_penalty)`
- `invariant_learning_emb(...)`
- `InfoNCE`
- `CLUB`
- `mutual_info`

这保证了：

- 保留“模态级推荐监督”
- 不重新引入“不变学习 + 信息瓶颈”

## 6. 当前 Stage 3 Workflow

当前 stage3 仍保留为可选方向，不是当前主线重点。

已保留入口：

- [scripts/legacy/run_stage3_baby_task_aware_imputer.sh](../scripts/legacy/run_stage3_baby_task_aware_imputer.sh)

当前结论：

- `full joint` 不适合作为主线
- `task-aware imputer` 是相对更稳的推荐导向补齐微调方式
- 但当前主要工作重心已经转到：
  - 先让 backbone 更利用模态
  - 再观察补齐收益是否放大

## 7. 当前关键修改记录

### 7.1 已完成的结构/训练更新

1. 删除了原始 I3 的：
   - `IRM`
   - `MI / CLUB`
2. 保留并沿用：
   - `original linear MLP`
   - `GCN`
   - `fusion`
   - `BPR`
3. 将 `PROMRL` 作为前置补齐模块接入
4. `stage1` 改为更接近 PROMRL 的 item-batch + pattern-wise completion training
5. recommender 阶段恢复了 `env-wise modality BPR`

### 7.2 已废弃或不再使用的方向

1. 两层缺失 / 第二次 mask 的“新协议”
   - 已回退
2. 只靠 `main_bpr_loss + reg_loss` 的 recommender
   - 已不再作为当前主线
3. 旧 feature-bridge 支线
   - 已清理，当前只保留 `raw_decoder`

## 8. 当前推荐运行入口

### 8.1 无补齐

- [run_stage2_baby_recommender_control.sh](../run_stage2_baby_recommender_control.sh)

### 8.2 带补齐

- [run_stage2_baby_recommender_decoder.sh](../run_stage2_baby_recommender_decoder.sh)

当前 recommender 默认实验口径与原始 I3 保持一致：

- `selection_mode=val`
- `recommendation_selection_metric=recall`
- `recommendation_selection_topk=20`
- `early_stop=20`
- `epoch=200` 作为训练上限

也就是后续 stage2 实验统一按 `val Recall@20` 选 best ckpt，并在连续 `20`
个 epoch 不刷新时提前停止。

### 8.3 Stage 1

- [run_stage1_1_baby_imputer_param.sh](../run_stage1_1_baby_imputer_param.sh)
- [run_stage1_2_baby_imputer_backprop_decoder_v2.sh](../run_stage1_2_baby_imputer_backprop_decoder_v2.sh)

### 8.4 归档脚本与诊断工具

以下内容不再属于当前主线，但仍保留在仓库中：

- 历史实验脚本：
  - [scripts/legacy](../scripts/legacy)
- 离线诊断工具：
  - [tools](../tools)

其中包括：

- 旧 joint / stage3 探索脚本
- 长周期报告脚本
- stage1 参数网格搜索脚本
- 补齐质量离线评测
- zero-fill vs imputation 诊断

## 9. 评估协议分支

当前代码显式支持两条评估支路，通过：

- `--evaluation_protocol legacy|strict`

控制。

### 9.1 `legacy`

保留当前历史行为：

- `stage2/stage3`
  - 每当 `val` 变好，就立刻跑一次 `test`
- `stage1.2`
  - 每个 epoch 都会打印 `train/test` 的补齐指标

特点：

- 方便调试和快速观察趋势
- 但存在 `test peeking`

### 9.2 `strict`

严格评估流程：

- `stage2/stage3`
  - 训练过程中只看 `val`
  - 只按 `val` 选 best
  - 训练结束后，用 best ckpt 只跑一次最终 `test`
- `stage1.2`
  - 训练过程中只看 `train` 的补齐指标
  - 训练结束后，用 best ckpt 只补一次最终 `test` 补齐指标

特点：

- 不把 `test` 暴露给训练/调参过程
- 更适合作为正式结果口径

### 9.3 当前建议

- 日常调试：
  - 用 `legacy`
- 正式汇报/论文结果：
  - 用 `strict`

## 10. 当前 workflow 的一句话总结

当前 workflow 可以概括为：

**在不恢复原始 I3 不变学习与信息瓶颈的前提下，保留其推荐骨干和 environment-wise 模态级推荐监督，并在其前端接入 PROMRL 补齐模块；当前已经验证，只要 backbone 具备足够的模态敏感性，补齐模块就能在 `fm/mm` 缺模态场景下带来稳定增益。**
