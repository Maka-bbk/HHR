# HHR J0 单阶段动作元轨迹框架

本文档描述当前实现，而不是尚未实现的设想。J0 的目标是用一个随机初始化模型、一个优化器和一条连续检查点（checkpoint，模型检查点）谱系，同时学习局部内容、边界、向量量化（vector quantization，VQ）码本、物理状态和完整试次轨迹。它不加载 HAPPY 或 A2 检查点。

## 先回答损失继承与阶段数

结论如下：

- 不建议把 HAPPY 的损失整包继承到 J0。当前 J0 明确不使用窗口分类交叉熵、InfoNCE（信息噪声对比估计）、SupCon（监督式对比学习）、DINO 聚类损失、MeMax（平均熵最大化）或 HAPPY 检查点蒸馏。
- 用户给出的 `train_offline` 日志来自 HAPPY 离线窗口预训练，不是 A2 编码器训练，也不是 J0。该次运行的有效权重是 `cls=1.0, cluster=0.0, contrast=0.65, supcon=0.35`，所以 `sup_con_loss` 和 `contrastive_loss` 都进入总损失，`cluster_loss` 虽被计算和打印但权重为零：

  ```text
  loss = 1.0 * cls_loss
       + 0.0 * cluster_loss
       + 0.65 * contrastive_loss
       + 0.35 * sup_con_loss
  ```

  用日志中四位小数近似计算为 `0.3136 + 0 + 0.65×4.0869 + 0.35×3.5613 ≈ 4.21654`，与日志的 `4.21657` 一致；微小差异来自打印舍入。
- A2 是独立的两阶段基线编码器：它自己的训练目标包括变点监督、内容—边界对齐、时序预测、试次辅助分类和防坍缩正则，并明确关闭 InfoNCE。不能把“A2 新损失”和上面这条 HAPPY 日志理解为在同一次反向传播中共存。
- J0 已把“表示学习→边界→码本→轨迹编码”缩成一次连续训练；训练中没有先训编码器再冻结做码本的第二阶段。训练前的原始运动学校准是确定性变换，不是预训练。
- 但当前最终 CGCD（广义类别发现）评价仍会在选定检查点后拟合一次半监督 KMeans（K 均值）。因此它是“单阶段表示训练”，不是已经无需聚类、无需类别数先验的端到端部署分类器。

这项隔离是必要的：若同时加入 HAPPY、A2 和轨迹损失，即使指标提高，也无法判断收益来自窗口判别、对比学习、边界学习还是轨迹顺序。首轮应先比较 J0-U 与 J0-T；之后若证据支持，再逐项加入候选损失并做匹配消融。

## 研究问题与不可变协议

研究问题是：USC-HAD 完整活动试次是否更适合表示成有序动作元轨迹，而不是单个池化特征。

- 分类单位始终是完整变长试次；局部窗口不接收活动标签。
- 每个成员运行只有一个随机初始化模型、一个 AdamW（带权重衰减的 Adam）优化器和一条连续训练谱系。
- 训练、验证和外层测试受试者互斥。
- 归一化、原始运动学边界校准和训练期码本学习只使用训练受试者的旧六类数据。
- 外层测试传感器在训练和检查点选择完成前保持锁定；选中 `checkpoint_best.pt` 并计算其 SHA256 后才解锁。
- 主实验 `primitive_only` 的轨迹分类器只接收离散动作元编号、顺序、边界/转移和持续时间；不接收连续局部内容、量化码向量、物理状态向量或 mean/max pooling（均值/最大值池化）。`state_only` 与 `primitive_plus_state` 只能作为显式归因消融，不能与主结果混报。
- 活动标签只能进入 J0-T 的完整试次轨迹交叉熵；标签到边界分支的梯度在轨迹入口处截断。

## 数据与 80% 旧类轨迹标签协议

默认使用 USC-HAD 原始 MAT 完整试次：14 名受试者、12 类活动、每类 5 条试次，共 840 条；采样率 100 Hz、六个惯性通道。旧类为活动 1–6，新类为活动 7–12。

七折外测受试者对固定为：

```text
fold 1..7 test: (11,10), (2,13), (3,9), (7,1), (12,8), (5,4), (14,6)
fold 1..7 val : (2,13), (3,9), (7,1), (12,8), (5,4), (14,6), (11,10)
```

每折余下 10 名受试者用于训练。默认 `anomaly-policy=report` 时，每折有 300 条 train-old、60 条 validation-old 和 120 条 outer-test-all。

80% 标签不是从 300 条训练试次全局随机抽取，而是按每个 `(训练受试者, 旧活动)` 独立分层：五条试次按 `seed` 确定性洗牌，四条标为 labelled（有标签），一条标为 unlabelled（无标签）。因此默认是 240 条有标签、60 条无标签。若显式排除 `Subject14/a3t2`，对应四条试次的层采用 3/1。

- J0-U：训练和验证的 `trajectory_label_mask` 全为假；损失接口甚至拒绝传入全是 sentinel（哨兵值）的活动标签。
- J0-T：只有上述 80% train-old 标签进入训练轨迹交叉熵；其余训练行目标为 `-100`。validation-old 的标签 100% 可见，但只用于验证损失和检查点选择。
- outer-test：两种 profile（配置档）都没有模型可见标签。

必须注意一个容易误读的区别：80/20 划分只控制训练期轨迹交叉熵。最终半监督 KMeans 使用全部 train-old 的真实标签作为旧类锚点，包括 J0-T 中训练期隐藏的 20%，J0-U 评价也同样使用全部 train-old 标签。因此 J0-U 是“表征训练无标签”，不是“从训练到评价都无标签”。

`Subject14/a3t2` 默认按文件名保留为活动 3，同时报告 MAT 内部活动 2 的冲突；`anomaly-policy=exclude` 可删除它，但任何模式都不会静默改标。

## 当前模型路径

```text
完整试次 [B,C,T]
  -> 128-sample frame / 64-sample stride（含必要的右侧 padding 尾帧）
  -> 局部内容编码器 + 独立边界编码器 + 物理状态编码器
  -> train-old 原始运动学 q50/q90 边界锚点
  -> 可微 K=32 VQ 动作元码本
  -> straight-through hard assignment（直通估计硬分配）
  -> 动作元编号 + 边界/转移 + 持续时间
  -> masked trajectory encoder（掩码轨迹编码器）
  -> 轨迹嵌入 + old6 轨迹 logits（分类分数）
```

训练时 Gumbel-Softmax（耿贝尔软最大）采用直通硬分配：前向送入轨迹编码器的每个有效位置严格是 one-hot（独热）动作元，反向使用软梯度，因此既可联合优化，也不能把连续内容偷偷编码进完整 32 维概率分布。评价同样使用硬动作元。按连续相同标记或预测边界形成的变长动作元区间用于评价、可视化和导出。原始边界锚点只覆盖完整原始帧；额外右侧 padding 尾帧对应的边界保持不确定，不进入边界监督。

模型仍计算无标签物理状态描述符，用于码本状态重建等表示约束；默认分类入口不会读取该连续向量。这样 Sit/Stand（坐/站）所需的姿态信息只能先被离散码本吸收，再通过动作元轨迹影响分类。若 `state_only` 明显更好，只能说明现有码本丢失了必要物理信息，不能将其直接替换为主方法并声称动作元有效。

## J0-U/J0-T 的 12 项完整损失闭环

每个 batch（批次）只形成一个标量总损失，并执行一次反向传播：

```text
L_total = Σ_i λ_i(e) L_i
```

当前实现的 12 项如下；表中权重是完整基础权重。

| # | 实现名 | 基础权重 | 监督范围与梯度语义 |
|---:|---|---:|---|
| 1 | `raw_boundary` | 1.00 | 原始运动学 q50 稳定锚点与 q90 变化锚点上的分组平衡 BCE（二元交叉熵）；中间不确定区与 padding 不参与。 |
| 2 | `vq_commitment` | 0.25 | 编码状态逼近已停止梯度的量化状态，只更新编码器侧。 |
| 3 | `vq_codebook` | 1.00 | 码字逼近已停止梯度的编码状态，只更新码本侧。 |
| 4 | `content_reconstruction` | 0.50 | 有效窗口上的内容重建 Huber loss（胡贝尔损失），目标停止梯度。 |
| 5 | `stable_next_content` | 0.25 | 仅跨原始 q50 稳定边界预测下一内容；目标停止梯度，不能跨 q90 变化锚点平滑。 |
| 6 | `utilization_floor` | 0.05 | 平均软分配的归一化熵低于 0.35 时才处罚的单侧 hinge（铰链）平方；达到下限后梯度为零，不强迫均匀占用。 |
| 7 | `boundary_sparsity` | 0.02 | 排除 q90 锚点后，预测边界率高于 0.35 才处罚的单侧预算。 |
| 8 | `minimum_duration` | 0.05 | 软性抑制边缘处和相邻过密边界；默认最短 2 个模型帧。 |
| 9 | `state_reconstruction` | 0.50 | 从码字重建无标签物理状态描述符；目标停止梯度。 |
| 10 | `masked_token_prediction` | 1.00 | 在掩码有效轨迹位置预测停止梯度的整数 VQ 伪标记；默认掩码比例 0.15。 |
| 11 | `masked_state_reconstruction` | 0.50 | 在相同掩码位置重建停止梯度的物理状态目标。 |
| 12 | `trajectory_ce` | J0-U: 0；J0-T: 1.00 | 只对显式 `trajectory_label_mask=true` 的旧类完整试次计算交叉熵。 |

J0-U 不只是把第 12 项乘零：它从接口层禁止 `trajectory_logits`、活动标签和有标签 mask 进入损失。J0-T 则强制第 12 项权重大于零，并强制未标注行只能携带 `-100`。两种配置均不包含 HAPPY/A2 的隐藏附加项；`run_manifest.json` 中的 `loss_config` 会写出完整审计字段。

### 连续课程与验证权重

默认 `warmup_epochs=10`。第 `e` 个 epoch（训练轮次）使用

```text
r(e) = max(0.05, min(1, e / warmup_epochs))
```

`stable_next_content`、`masked_token_prediction`、`masked_state_reconstruction`，以及 J0-T 的 `trajectory_ce` 乘以 `r(e)`；其余项保持基础权重。同一模型和优化器不重置，所以这仍是一阶段连续课程（curriculum，课程式训练），不是两个训练阶段。码本温度默认从 2.0 几何退火到 0.25。

每轮验证始终使用表中的完整基础权重和硬码本，不用当轮的 warmup 权重。默认 `checkpoint-selection=common_unsupervised`：J0-U 与 J0-T 都先按较低的无标签验证目标选点，再以较高有效码字数打破平局；因此两臂不会因为 J0-T 使用验证活动标签选点而产生额外优势。`profile_default` 只保留为诊断消融，其中 J0-T 才按 validation trajectory macro-F1（验证轨迹宏平均 F1）选点。

## 最终评价：传导式约束半监督 KMeans

选定检查点之后，代码执行 transductive constrained semi-supervised KMeans（传导式约束半监督 K 均值）：

1. 全部 train-old 轨迹嵌入和旧类标签作为锚点；每个旧类必须至少有一个锚点。
2. 全部 outer-test 轨迹嵌入共同作为无标签样本参加聚类拟合；其真值不传入拟合函数。
3. 行向量先做 L2 normalization（L2 归一化）。旧六类中心由对应锚点均值初始化；另外六个中心从 outer-test 特征用 KMeans++ 初始化。
4. 锚点的旧类簇身份固定，但旧类中心坐标并非冻结：每轮会用该类锚点和当前分到该簇的 outer-test 样本共同更新。每个 outer-test 样本可选择任一旧类或新类中心。
5. 默认运行 10 个 restart（重启），选择包含锚点和 outer-test 距离的总 inertia（簇内平方距离和）最小者。
6. 拟合完成后才读取 outer-test 真值；旧类簇 0–5 保持原身份，新簇 6–11用 Hungarian matching（匈牙利匹配）命名，然后计算 All/Old/New accuracy、H-score、macro-F1、逐类召回率和混淆矩阵。

这里有两个不能省略的限制：

- `num_classes=12` 是 oracle total class count（先验真值给定的总类别数），`num_old_classes=6` 也已知；该实验没有自动发现类别数。
- “传导式”表示一个 outer-test 样本的结果依赖整批 outer-test 特征。测试标签没有进入聚类拟合，但这仍是离线表示/CGCD 筛查，不是 inductive（归纳式）单样本分类器、online CGCD（在线广义类别发现）或部署式无标签分类器。

J0-T 的 direct old trajectory classifier（直接旧类轨迹分类器）只能输出旧六类，不能预测新类。J0-U 的该分类头没有分类损失训练，其输出只应视为随机头诊断，不能当作 J0-U 分类能力。

顺序消融会打乱预测动作元 run block（连续动作元区块），保留区块内部帧顺序，并在 identity（恒等）与 shuffle（打乱）两侧都重算转移/持续时间、清零 learned-boundary（学习边界）入口。主实验应看 `identity_control_minus_shuffled`；在 `primitive_only` 下，其下降可归因于动作元编号及其顺序/持续时间关系的联合贡献。若需继续分离编号、顺序、边界和持续时间贡献，应再做匹配的逐输入消融，而不是引入连续状态捷径。

## 输出文件含义

每个 `profile_<profile>/fold_<NN>_seed_<seed>/` 成员目录包含：

| 文件 | 含义 |
|---|---|
| `run_manifest.json` | 运行身份、数据协议审计、模型/损失配置、优化器参数、无 HAPPY 检查点声明、边界校准哈希和原命令。 |
| `boundary_calibration.json` | 仅由 train-old 拟合的原始运动学稳健缩放、q50/q90 阈值、train/validation 覆盖和模型帧对齐审计。 |
| `history.jsonl` | 每个完整 epoch 一行：调度后损失权重、训练/验证原始分量、总损失、有效码字数、学习率、最佳轮次。分量值本身未乘权重，解释总损失时必须同时读取权重。 |
| `checkpoint_last.pt` | 最近完整 epoch 的模型、优化器、学习率调度器、最佳键/轮次和 early-stop patience（提前停止耐心值）；用于恢复。 |
| `checkpoint_best.pt` | 按对应 profile 验证规则选中的模型、验证指标和运行身份；最终评价只加载它。 |
| `outer_test_primitive_trajectories.jsonl` | 每条 outer-test 完整试次的真值元数据、样本/帧数、硬标记序列、动作元区间及样本/秒范围，以及完成聚类和命名后的预测与正确性。 |
| `cgcd_confusion_heatmap.png` | 匈牙利命名后的 12 类 outer-test 行归一化混淆矩阵；每行表示召回比例。 |
| `activity_codebook_heatmap.png` | 真值活动×码字占用图；先使每条试次总质量为 1，再按活动平均，避免长试次支配。 |
| `fixed_trajectories_and_predictions.png` | 每类最多两个不同受试者代表试次；将完整硬动作元轨迹归一到 0–100% 横轴，并标注真值、受试者、试次、预测和帧数。 |
| `summary.json` | 选中轮次/哈希、CGCD 指标与映射、KMeans 审计、直接旧类头、顺序对照、码本/边界/片段诊断、图路径、传感器访问审计和解释限制。 |
| `complete.json` | 最后写入的完成标志；记录运行身份，并逐一校验 manifest、边界校准、历史、last/best 检查点、逐试次轨迹、三张图和 summary 的 SHA256。任一文件缺失或被改动，`--resume` 都会拒绝把该成员当成完成态。 |

CV（交叉验证）根目录额外产生：

| 文件 | 含义 |
|---|---|
| `cv_manifest.json` | profiles、folds、seeds、数据路径、全部训练参数和网格身份哈希。 |
| `aggregate_summary.json` | 全部成员行；先在折内平均 seeds，再以折为统计单位计算均值、标准差和折级 bootstrap（自助法）95% 区间；同时包含 J0-T−J0-U 的精确单侧 sign-flip（符号翻转）筛查。 |
| `per_run_metrics.csv` | 每个成员的 All/Old/New/H/macro-F1、顺序 H-score 下降、有效码字数、token-subject NMI（标记—受试者归一化互信息）、运行环境 SHA256、原始数据清单 SHA256、选中轮次和目录。 |
| `aggregate_metrics.png` | J0-U/J0-T 的 All、Old、New、H-score 折均值及 bootstrap 误差条。 |

CV 聚合前还会强制所有成员具有同一 Python/PyTorch/CUDA 运行环境身份和同一原始 USC-HAD 清单身份，防止中途换环境或数据后仍混入一个统计表。预先指定唯一主要推断终点为 H-score，方向为 J0-T 大于 J0-U；All/Old/New/macro-F1 的未校正 p 值只作探索性描述。七折训练集高度重叠，只有七个外测受试者对；即使是主要终点的 p 值和 bootstrap 区间也只能作为筛查证据，不是强确认性结论。

## 暂停与 `--resume` 恢复

没有专门的暂停信号处理器。使用 `Ctrl+C` 或 PyCharm Stop（停止）时，当前未完成 epoch 可能丢失；`checkpoint_last.pt` 采用临时文件后原子替换，最近一个完整 epoch 可恢复。

恢复时对完全相同的输出目录和参数重新运行，并加 `--resume`。程序会校验完整运行身份，使用 CPU 安全加载模型、优化器和调度器，从 `last_epoch+1` 继续，同时恢复最佳轮次、patience、提前停止状态，以及 Python、NumPy、Torch、CUDA 和训练 DataLoader（数据加载器）的随机数状态。多 worker 时关闭 persistent worker（持久工作进程），使恢复后的采样器消耗顺序与未中断训练一致。`checkpoint_last.pt` 内嵌本轮历史记录及其哈希；若恰好在检查点写入后、历史提交前中断，恢复会只补齐这一条，遇到缺口或冲突则拒绝猜测。若成员已有 `complete.json`，必须通过完整产物哈希校验后才会快速返回。

## PyCharm Linux/WSL 单行命令

在 PyCharm 的 Linux/WSL Terminal（终端）中直接运行完整 7 折、单 seed、两个 profile，并允许恢复：

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_one_stage_cv.py --data-root /mnt/d/WorkDir/DataSet/USC-HAD --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/one_stage_primitive_only_J0_U_T_7fold_seed50_v2 --profiles J0-U,J0-T --folds 1,2,3,4,5,6,7 --seeds 50 --epochs 80 --batch-size 8 --eval-batch-size 16 --num-workers 0 --device cuda --learning-rate 3e-4 --weight-decay 1e-4 --gradient-clip 5 --warmup-epochs 10 --early-stop-patience 20 --frame-size 128 --frame-stride 64 --codebook-size 32 --trajectory-input-mode primitive_only --checkpoint-selection common_unsupervised --temperature-start 2 --temperature-end 0.25 --trajectory-mask-ratio 0.15 --labelled-fraction 0.8 --anomaly-policy report --cluster-restarts 10 --order-shuffles 10 --minimum-segment-windows 2 --bootstrap-replicates 10000 --aggregate-seed 20260908 --deterministic --resume
```

在 Windows PowerShell 中调用 Ubuntu WSL 的等价单行命令：

```powershell
wsl.exe -d Ubuntu -- bash -lc "cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_one_stage_cv.py --data-root /mnt/d/WorkDir/DataSet/USC-HAD --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/one_stage_primitive_only_J0_U_T_7fold_seed50_v2 --profiles J0-U,J0-T --folds 1,2,3,4,5,6,7 --seeds 50 --epochs 80 --batch-size 8 --eval-batch-size 16 --num-workers 0 --device cuda --learning-rate 3e-4 --weight-decay 1e-4 --gradient-clip 5 --warmup-epochs 10 --early-stop-patience 20 --frame-size 128 --frame-stride 64 --codebook-size 32 --trajectory-input-mode primitive_only --checkpoint-selection common_unsupervised --temperature-start 2 --temperature-end 0.25 --trajectory-mask-ratio 0.15 --labelled-fraction 0.8 --anomaly-policy report --cluster-restarts 10 --order-shuffles 10 --minimum-segment-windows 2 --bootstrap-replicates 10000 --aggregate-seed 20260908 --deterministic --resume"
```

首次运行也可以保留 `--resume`：目录尚无检查点时会正常从第 1 轮开始。若改变任一网格身份参数，应使用新的 `--output-root`，不要在旧目录上强行恢复。

## 首轮比较与解释边界

- `B0`：历史 A2 编码器加固定窗口 KMeans-32 的两阶段基线。
- `J0-U`：当前单阶段模型，表征训练完全不使用活动标签。
- `J0-T`：与 J0-U 相同，只增加 80% train-old 完整试次轨迹交叉熵。

首要因果比较是 J0-T−J0-U：它只回答“旧类试次级轨迹监督是否有益”。J0 相对 B0 的差异同时包含训练方式、完整试次、可微码本和轨迹建模，不能把总差值归因于某一个组件。分类提高但顺序打乱不下降，也不能证明模型真正使用了轨迹顺序。在线码本扩张、自动类别数发现、归纳式新样本预测以及 HAPPY-CGCD 正式整合均属于后续工作。
