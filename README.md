# HHR：冻结动作元轨迹 HAR-CGCD

HHR 是独立的 wearable-sensor Human Activity Recognition（可穿戴传感器人体活动识别）与 Continual Generalized Category Discovery（连续广义类别发现，CGCD）研究项目。当前路线围绕 Motion Primitive（动作元）展开：先获得稳定的局部窗口表征，再把完整 activity trial（活动试次）表示为离散动作元轨迹，最终进行旧类保持和无标签新类发现。

## 当前实验规则（2026-09-19 起）

当前活动协议已经停止扩展七折网格，只使用 fixed split 01（固定划分 1）：训练受试者为 `1,3,4,5,6,7,8,9,12,14`，验证受试者为 `2,13`，outer-test（外层测试）受试者为 `10,11`。这里应称“固定划分 1”，不能称为 1-fold cross-validation（单折交叉验证），也不能把结果外推为 14 名受试者总体性能。

当前同时设置两个 encoder provenance（编码器来源）分支：`R0_reuse_0914v1` 严格复用 `window_codebook_batch_proxy_3arm_3fold2seed_20260914_v1` 中 `W128/S64、A2、fold 1、seed 0` 的 `motion_encoder_final.pt`；`R1_retrained_current` 使用相同数据划分和超参数，从 seed 0 的随机 ResNet1D 初始化重新完成 60 轮窗口训练和 30 轮 A2 训练。两条线各自运行 `run_seed=0,5,50,500`；这些 run seed 只控制 K64 动作元码本、静动态门控与下游聚类，不是四个端到端编码器种子。总计为 `2 个编码器来源 × 4 个下游种子 × 3 个读出臂 = 24 个评分单元`。

`R1` 使用当前源码重新训练，而 `R0` 是历史 checkpoint（检查点）；两个训练入口源码的身份哈希并不相同，因此两者差值是“历史已训练编码器 vs 当前重新训练编码器”的实用比较，不是纯粹的随机初始化单因素因果效应。固定划分只报告逐 run-seed 值、均值和算法随机性标准差，不计算跨折 bootstrap confidence interval（自助置信区间）。

当前实验报告三个配对实验臂：

| 实验臂 | 数据流 |
|---|---|
| `B0_global_trajectory` | 全部 120 条 outer 轨迹使用时长不变动作元描述器，再做全局 KMeans12 |
| `E1_gate_static_expert` | 无标签静态/动态门控；动态分支使用时长不变动作元轨迹专家；静态分支使用姿态、能量、带符号重力趋势和低权重动作元的独立专家 |
| `E2_gate_static_expert_soft_a025` | 完全复用 E1 的门控、分支 K、静态专家与静态预测；仅在动态分支加入 α=0.25 的 soft subject debiasing（软受试者去偏） |

E2 是三个模块的最终汇集臂：duration invariance（时长不变）、软受试者去偏、以及带 signed gravity（带符号重力）的静态独立专家。软去偏使用 `z'=L2Norm[z-0.25(zB^T)B]`；动态 outer 子集只负责无标签坐标/PCA32 拟合，干扰基 `B` 仅用 Offline old6 的 300 条源轨迹和 10 名源受试者身份拟合，rank（秩）不超过 4、解释率目标为 0.90。它不读取 outer 受试者身份。这里是动态专家范围内的新组合变体：其 PCA 拟合范围不是早期 A025 单因素实验的全部 120 条 outer，不能把两者称为逐字复现。

`E1-B0` 衡量门控、分支化和重力静态专家的整体变化；`E2-E1` 才是同门控、同 K、同静态预测条件下软去偏的配对增量。由于 B0 与 E1 的动态坐标拟合范围本来不同，`E1-B0` 不能被解释成纯重力单变量效应。

`K_dynamic + K_static = 12` 按无标签门控成员比例分配，完整公式为 `K_static=clip(round(12*N_static/120),1,12-K_old)`、`K_dynamic=12-K_static`，其中 CGCD 已知旧类数 `K_old=6` 提供动态分支下界。在固定 outer batch 的 `70/50` 成员下得到 `7/5`，而不是直接把真实静态类别数写入学习器。该估计成立还要求 USC-HAD 每个活动 trial 数相等，且门控近似按完整活动类形成纯分支；不平衡数据流或类内跨门控分裂时必须另行估计类别数，不能照搬。代码不再拟合或报告“仅门控、静态仍用普通轨迹”的第三臂。所有门控、分支变换和聚类都在 `raw_predictions.npz` 写盘并计算 SHA256 之前完成；活动真值只能在此后进入 scorer（评分器）。该实验仍是 known-K12 transductive batch GCD（已知总类数的传导式批量广义类别发现），不是多会话 Online CGCD（在线连续广义类别发现）。

PyCharm Linux/WSL 终端一行命令：

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_dual_encoder_static_dynamic_fixedsplit01.py --npz-path /mnt/d/WorkDir/HHR/processed/uschad_w128_s64_train17stats/uschad_windows.npz --reused-encoder-root /mnt/d/WorkDir/HHR/results/motion_primitive/window_codebook_batch_proxy_3arm_3fold2seed_20260914_v1 --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/duration_soft_a025_static_dynamic_dual_encoder_fixedsplit01_4runseed_20260919_v1 --encoder-seed 0 --run-seeds 0,5,50,500 --window-epochs 60 --window-batch-size 256 --window-eval-batch-size 1024 --window-learning-rate 0.1 --window-weight-decay 0.0005 --window-weak-scale-std 0.1 --window-strong-scale-std 0.2 --a2-epochs 30 --a2-trial-batch-size 8 --a2-source-encode-batch-size 1024 --a2-learning-rate 0.0001 --a2-minimum-learning-rate 0.000001 --a2-weight-decay 0.0001 --cp-context-windows 2 --encode-batch-size 1024 --gate-n-init 20 --kmeans-n-init 50 --kmeans-max-iter 300 --subject-nuisance-max-rank 4 --subject-nuisance-explained-variance 0.90 --subject-nuisance-projection-strength 0.25 --static-motion-primitive-pca-dim 8 --static-posture-weight 0.35 --static-gravity-weight 0.35 --static-energy-weight 0.20 --static-motion-primitive-weight 0.10 --num-workers 0 --device cuda --resume
```

当前 dual suite（双分支实验套件）的 `--resume` 会核验顶层 identity（身份）、两条 encoder checkpoint SHA256、重新训练的窗口/A2 完成标记、两个下游 suite identity、八个 member identity（成员身份）与精确 artifact SHA256（产物哈希）；顶层、suite 和 member 输出目录都有独占写锁。执行顺序是重新训练编码器，再依次运行两个下游分支，避免同一 GPU 并行争抢；中断后原样重跑上面一行即可。参数、数据、依赖源码或环境版本变化时必须改用新的 `--output-root`，不能复制其他实验的成员目录进行接管。

每个编码器分支的 `aggregate.json`、顶层 `branch_comparison.json` 与 `member_rows.csv` 中的 headline 指标统一标记为 `global_hungarian_upper_bound`（全局匈牙利匹配上界）：它用于比较三臂的无标签可聚类性，不是旧类身份固定的部署准确率。

## 历史严格 Online 主线（仅保留复现，不是当前默认规则）

以下七折、三会话、K32 内容及命令保留用于核验既有结果和 artifact（产物），不得作为新实验默认网格，也不应删除旧结果或改写其 manifest（清单）。

```text
旧六类窗口
  -> ResNet1D（一维残差网络）窗口级暖启动
  -> A2 动作编码器训练，取 final checkpoint（最终检查点；A2 不含 InfoNCE）
  -> 冻结 A2 content representation（内容表征）
  -> E0 固定窗口 w256/s128
  -> L2 -> trial 等权 PCA64 -> L2 -> KMeans32 -> 余弦硬量化
  -> 每个 trial 的完整离散 token 轨迹
  -> 3739 维 state descriptor（物理状态轨迹描述符）
  -> 仅用 Offline old-train 拟合：常量列过滤 -> z-score -> PCA32 -> L2
  -> 旧类原型与拒识门控
  -> 三个严格无标签 Online session（在线会话）
  -> 只追加 activity registry（活动类别注册表）和 Unknown buffer（未知样本缓冲区）
```

这条路线不是旧版“两个损失包同时叠加”的联合训练。它包含两个有先后关系的表征训练检查点：

1. 窗口级 ResNet1D 暖启动；
2. 从该折、该种子的最佳窗口检查点继续训练 A2；
3. A2 第 30 轮保存后完全冻结，E0、轨迹描述符和 Online CGCD 不再向编码器反向传播。

因此它应被描述为“顺序表征训练后冻结读出”，而不是 end-to-end joint optimization（端到端联合优化），也不是 Happy 原损失与动作元损失在同一次 backward（反向传播）中混合。

## 四个执行阶段

### 1. 窗口级 ResNet1D 暖启动

- 输入为旧类 `0..5` 的六轴 `w256/s128` 窗口；
- 每折只使用 10 名训练受试者训练，2 名验证受试者选点，2 名外层测试受试者完全不参与；
- ResNet1D 输出 256 维特征，`base_channels=64`、残差层数 `[2,2,2]`；
- 两个视图只做逐通道小幅缩放，标准差分别为 `0.10` 和 `0.20`，不加 jitter（抖动）、时间遮挡或随机裁剪；
- 损失为 `1.00×CE + 0.65×InfoNCE + 0.35×SupCon`。InfoNCE（信息噪声对比估计）温度为 `1.0`，SupCon（监督对比损失）温度为 `0.07`，分类 logits（分类输出）温度为 `0.1`；
- SGD（随机梯度下降）训练 60 轮，学习率 `0.1`、momentum（动量）`0.9`、weight decay（权重衰减）`5e-4`，使用 cosine schedule（余弦调度）；
- 只按验证窗口 macro-F1（宏平均 F1）选择 `model_best.pt`。

ResNet1D 初始参数是随机初始化的；但正式 A2 不从随机状态直接开始，而是从相同 fold/seed（折/随机种子）的 `model_best.pt` 开始。

### 2. A2 动作编码器

A2 从窗口最佳检查点严格加载完整 ResNet1D。其正式损失权重为：

| A2 项 | 权重 |
|---|---:|
| physical changepoint（物理变点） | 1.00 |
| content-boundary alignment（内容—边界对齐） | 0.10 |
| non-collapse（防坍缩） | 0.05 |
| masked temporal prediction（遮挡时间预测） | 0.50 |
| old-class trial auxiliary（旧类试次辅助分类） | 0.10 |
| window augmentation consistency / InfoNCE（窗口增强一致性） | **0.00** |
| cross-subject loss（跨受试者损失） | 0.00 |

- AdamW 训练 30 轮，学习率 `1e-4`、最小学习率 `1e-6`、权重衰减 `1e-4`；
- backbone BatchNorm（骨干批归一化）的运行统计冻结，affine parameters（仿射参数）仍可更新；
- EMA teacher（指数滑动平均教师）只作为时间预测目标；
- 下游固定使用第 30 轮 `motion_encoder_final.pt`，不用验证最优检查点，也不用 EMA 权重；
- E0 下游只读取 A2 的 `content` 输出。A2 的 `segmentation` 输出只参与 A2 训练约束，不参与 E0 切分。

A2 内部仍保留权重为 `0.1` 的旧类 trial auxiliary head（试次辅助头），它使用 `mean+q90` 形成训练期辅助监督。这是历史 A2 定义的一部分，但它既不是 E0 下游输入，也不是最终 CGCD 预测表示。最终分类读取的是动作元轨迹描述符，不能把该辅助头的结果报告为动作元分类结果。

### 3. 冻结 E0 与 Offline registry

E0 是固定窗口动作元方案，不是可学习边界或变长分段：

- 每个已保存窗口对应一个 token occurrence（动作元出现）；
- 不进行 run-length encoding（游程压缩）；
- trial 之间窗口数可以不同，所以轨迹长度可变；但每个内部 E0 单元仍由固定 `w256/s128` 栅格产生；
- 重叠窗口的持续时间采用窗口中心 Voronoi ownership（沃罗诺伊归属），避免把重叠区域重复计时。

码本只在 Offline old-train 上拟合：

```text
A2 content
  -> 行 L2 归一化
  -> 每个 trial 总权重相同的加权 PCA64
  -> 行 L2 归一化
  -> KMeans32（n_init=20, max_iter=300, algorithm=lloyd）
  -> 归一化中心的余弦最近邻硬分配
```

`K=32` 是 E0 codebook capacity（动作元码本容量）；actual used-K（实际使用码数）是某个数据划分真正出现的 token 类型数量。两者必须分开报告。

每条轨迹先形成 3739 维原始描述符，包括 token 频次、持续时间、四分位位置、转移、量化质量和六轴物理状态。随后仅在 Offline old-train 上拟合常量列过滤、z-score（标准化）、PCA32 和末端 L2；这些变换之后全部冻结。旧六类 prototype（原型）由 Offline old-train 拟合，拒识距离与距离比阈值由 Offline validation（离线验证集）校准。

### 4. 严格三会话 Online CGCD

Online 阶段不会更新以下任一组件：

- ResNet1D 或 A2 参数、BatchNorm 缓冲；
- E0 PCA64；
- KMeans32 中心或动作元编号；
- 3739 维描述符定义、常量列掩码、z-score 或 PCA32；
- 已注册旧类原型、门控阈值与 ID。

Online 唯一可变状态是：

- append-only activity registry（只追加活动类别注册表）；
- 跨 session 保留的 Unknown buffer。

每个 session 先用当前注册表路由未标注 incoming trials（流入试次）；只有被拒识的 trial 才进入未知池。未知池在不读取活动标签的情况下聚类，并通过最小 trial 支持、最小受试者支持、silhouette（轮廓系数）、bootstrap stability（自助稳定性）及候选间/候选—注册表间分离度门控。通过的候选获得顺序追加的新 registry ID；未通过的样本留在 Unknown buffer。旧 registry 行不能被覆盖或重排。

评估时先把 raw registry predictions（原始注册表预测）保存并计算 SHA256，随后 scorer（评分器）才从独立 `TruthStore` 读取真实标签。Online learner API（在线学习器接口）不接收 activity label 参数。

注意：Online 增长的是活动类别注册表，不是 E0 动作元码本。正式路线的 E0 始终是 K32。

## 历史 USC-HAD 七折协议

- 默认数据：`processed/uschad_w256_s128_train17stats/uschad_windows.npz`；
- 窗口 256 samples（采样点），stride（步长）128，采样率 100 Hz；
- “完整 trial”指该 trial 在 NPZ 内所有已保存完整窗口的有序集合；预处理时不足一个窗口的尾部不在 NPZ 内；
- 物理标签 `0..5` 是旧类，`6..11` 按两个一组进入三个 Online session，活动空间为 `6→8→10→12`；
- 固定 7-fold subject-disjoint cross-validation（7 折受试者无交叉交叉验证）：每折 10 名训练、2 名验证、2 名外层测试受试者；
- 每折传感器归一化只由该折训练受试者的旧类窗口重新拟合；
- Offline 试次数固定为 train/validation/outer-test=`300/60/60`；
- Online train 与 evaluation trial ID 严格不重叠，session 之间也不重复使用 incoming trial。

## 历史三层指标与 84 行统计

每个 fold/seed/session 都保存三套严格区分的评分层：

| 层 | 含义 | 是否使用测试标签对齐 | 用途 |
|---|---|---:|---|
| `direct_registry` | 直接使用 append-only registry ID；Unknown 保持 `-1` | 否 | 最接近直接部署的 ID 层 |
| `old_fixed_novel_hungarian` | 旧类 `0..5` 固定，只对 novel registry ID 做 Hungarian matching（匈牙利匹配） | 是，仅评分 | **主要 CGCD 指标** |
| `global_hungarian_upper_bound` | 允许全部 registry ID 全局匹配真实类 | 是，仅评分 | 乐观诊断上界，不能称为部署准确率 |

三层均报告 All/Old/New accuracy（全体/旧类/新类准确率）、H-score（旧/新准确率调和得分）、macro-F1 和 Unknown 比例。不能把三层数值混写。

完整网格为：

```text
7 folds × 4 seeds × 3 sessions = 84 session rows
```

84 行不是 84 个独立统计样本。正式汇总先在每个 held-out-subject fold（留出受试者折）内平均 4 个种子，再用 7 个 fold mean（折均值）计算均值、跨折标准差和 10,000 次 bootstrap 95% confidence interval（自助法 95% 置信区间）。统计单位是 fold，不是 seed 或 session row。

## 历史结果边界

旧 `joint VQ + learned boundary + GRU`（联合向量量化、学习边界与门控循环网络）代码仍保留为 legacy（历史对照），但不是默认入口、正式消融臂或当前成功标准。它与本路线在编码器训练、切分方式、码本更新和 Online 协议上均不同。

历史记录中的 `75.96%` 来自 batch oracle（批次先验上界）协议；它不满足当前 append-only、label-free Online registry 的约束，不能与 `old_fixed_novel_hungarian` 主指标直接并表，也不能用来宣称严格 CGCD 已达到 75.96%。若要比较，必须把旧方法重新放进完全相同的数据流、拒识、注册和三层评分协议。

## 可视化解释边界

正式可视化是 complete artifact（完成产物）之后的只读后处理，并验证 NPZ、预测、representation（表征）及 registry 哈希链。它会输出：

- Offline/Online actual used-K 摘要；
- 每个唯一观测 trial 的完整 token 序列，并叠加经 NPZ 哈希验证后重建的原始六轴信号；
- final-session activity×primitive 热力图；先求每个 trial 的 token fraction，再在受试者内平均，最后让受试者等权；
- `direct_registry`、`old_fixed_novel_hungarian`、`global_hungarian_upper_bound` 三层混淆热图；
- activity registry 与 Unknown buffer 的 session 变化。

图中必须区分：

- `primitive_occurrence_count`：该 trial 的 token 出现总数，即 E0 窗口数；
- `unique_primitive_type_count`：该 trial 实际用到的不同 token 类型数；
- `capacity-K=32`：全局动作元码本容量；
- `activity registry K`：当前已注册活动类别数。

KMeans token ID 存在置换不确定性。`P0..P31` 只在同一个 fold/seed member（折/种子成员）内有意义；未经中心对齐，禁止把不同折或不同种子的同编号 token 直接求平均。默认只为一个代表成员生成可视化。

## 无门控三臂批次 GCD 验证

这一独立消融用于比较时间分辨率与动作元码本容量，共包含以下三个 targeted arms（定向实验臂）：

| 配置 ID | 窗口 / 步长 | 动作元容量 |
|---|---:|---:|
| `w64_s32_k128` | 64 / 32 | 128 |
| `w128_s64_k64` | 128 / 64 | 64 |
| `w128_s64_k128` | 128 / 64 | 128 |

默认运行 `3 arms × 3 folds × 2 seeds = 18` 个代理成员。两个 W128 实验臂在同一 fold/seed 内共享同一个窗口暖启动编码器和 A2 编码器，只分别拟合 K64 与 K128 码本；直接报告 `W128/K128 − W128/K64`，用于估计固定窗口栅格下的端到端码本容量效应。另直接报告固定 K128 的 `W128/S64 − W64/S32` 窗口栅格对照；但窗口长度改变时 stride、单独训练的编码器以及 A2 固定 `cp-context-windows` 对应的物理时长也随之改变，所以它不是纯窗口长度主效应。W64/K128 与 W128/K64 同时改变窗口栅格和 K，不能解释成单一因素效应。

这项实验不使用拒识门控，也不进行多 Session Online（多会话在线）注册或更新。每个成员在 Offline old-train（离线旧类训练集）拟合编码器、PCA 和动作元码本，然后对留出受试者的 120 条全类别轨迹做一次无标签 12 簇分配。它是 Transductive Batch GCD（传导式批量广义类别发现）的轨迹可分性上界；Global Hungarian Alignment（全局匈牙利对齐）仅用于事后评分，不能作为可部署 CGCD 的结果。

主要输出为 `batch_proxy_runs.csv`（18 个成员）、`batch_proxy_summary.{csv,json}`（先在每折内平均两个种子，再以折为统计单位）、`isolated_paired_contrasts.{csv,json}`（上述两个定向对照；先求同折同种子差值，再在折内平均种子并对折均值 bootstrap）和 `complete.json`。该三臂 runner 严格锁定 `legacy_state_v1`；其他描述器必须使用下一节的专用消融 runner，以保证同 fold/seed 的全部 profile 复用同一份冻结码本。本 runner 按当前实验计划不重复 W256/K32，三臂汇总也不报告相对 W256/K32 的配对差值。已有 Session3 K12 的 W256/K32 本地结果采用 78-trial fit 后对 42-trial evaluation 评分，只能作为历史参照，不能填入当前 120-trial joint transductive batch summary 或配对统计；其他外部 W256 结果也只有在相同 120-trial proxy 协议、相同 fold/seed、数据和训练身份均核验一致后，才能另行作配对比较。PyCharm 的 Linux/WSL 终端一行命令如下：

每个成员的 `visualizations/trajectory_sequences.png` 使用 physical-time raster（物理时间栅格）：依据每个动作元的 `ownership_start_samples` / `ownership_end_samples_exclusive`，按 USC-HAD 的 100 Hz 采样率将动作元铺回公共时间轴，每列对应 0.01 秒，横轴单位为秒。因此 W64 与 W128 的轨迹图可按真实持续时间比较，而不会把“窗口个数不同”误读成动作快慢不同；白色只表示该 trial 的可观测区间已经结束。完整的窗口级 token 序列仍保存在 `trajectories_label_free.jsonl`，且动作元 ID 仍只在单个 fold/seed member 内有意义。

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_window_codebook_batch_proxy_cv.py --dataset-root /mnt/d/WorkDir/DataSet/USC-HAD --processed-root /mnt/d/WorkDir/HHR/processed --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/window_codebook_batch_proxy_3arm_3fold2seed_20260914_v1 --arms 64:32:128,128:64:64,128:64:128 --folds 1,2,3 --seeds 0,5 --window-epochs 60 --window-batch-size 256 --window-eval-batch-size 1024 --a2-epochs 30 --a2-trial-batch-size 8 --a2-source-encode-batch-size 1024 --encode-batch-size 1024 --device cuda --resume
```

缺失的 W64 与 W128 预处理缓存会自动生成并校验；`--resume` 允许中断后原命令继续，但参数、数据或源码身份变化时应换新的输出目录。默认只提高无梯度评估与编码批次；窗口训练 batch 保持 256、A2 trial batch 保持 8，以免把训练优化差异混入窗口/码本消融。

## W128/K64 重力方向、受试者去偏与绝对时长分离实验

该实验固定上一轮较优的 `A2 + E0 + W128/S64 + K64`，直接复用既有 3 折×2 种子的六个冻结 A2 检查点。正式验证拆为三个相互独立、完全对称的两臂实验；每个实验都只有 `legacy_state_v1` 基线和一个单变量处理臂。每个 fold/seed 只由 legacy 成员拟合一次动作元码本，处理臂加载同一个已完成基线成员；汇总前强制要求二者的 codebook state SHA256（码本状态哈希）完全一致。

第一部分 `gravity` 是 12-member（12 成员）严格两臂实验，只检验新增带符号重力趋势：

| profile | 带符号重力趋势 | 删除绝对时长字段 | 受试者去偏 |
|---|---:|---:|---:|
| `legacy_state_v1` | 否 | 否 | 否 |
| `gravity_signed_v1` | 是 | 否 | 否 |

第二部分 `subject_debias` 同样是 12-member（12 成员）严格两臂实验，只检验是否投影源域受试者干扰方向：

| profile | 带符号重力趋势 | 删除绝对时长字段 | 受试者去偏 |
|---|---:|---:|---:|
| `legacy_state_v1` | 否 | 否 | 否 |
| `subject_debiased_v1` | 否 | 否 | 是 |

第三部分 `duration` 仍是 12-member（12 成员）严格两臂实验，只检验删除绝对 trial 时长字段及使用相对量替代：

| profile | 带符号重力趋势 | 删除绝对时长字段 | 受试者去偏 |
|---|---:|---:|---:|
| `legacy_state_v1` | 否 | 否 | 否 |
| `duration_invariant_v1` | 否 | 是 | 否 |

`gravity_duration_subject_v1` 不进入上述三部分。只有三个单变量实验分别得到证据后，才适合用它做最终组合确认。

Signed vertical trend（带符号垂直趋势）先从重叠 raw windows 严格重建 trial，令三轴加速度的完整 trial 均值为重力参考 `g`、`u=g/||g||`，再计算 `a_v(t)=a(t)·u−mean(a·u)`。描述器保留 8 个 normalized-time phase bins（归一化时间相位区间）均值和一个 signed temporal moment（带符号时间矩）`mean[a_v(t)(0.5−(t+0.5)/T)]`。该 9 维块单独标准化，既绕过结构 PCA32，也不进入受试者干扰基的 SVD（奇异值分解）或投影，以固定距离权重 `0.15` 融合；重力向量 xyz 分量只作审计，不进入描述器。

Duration-invariant（时长不变）profile 删除 `log_child_count`、`log_duration_samples`、三个 `child_duration_seconds_*`、`quantization_distance_max`、独立 token-presence 和历史 parent-count 占位；保留统计改成 trial 内相对量。这里“时长不变”只表示描述器对已注册的统一 span 缩放保持不变，不代表固定窗口编码器无法间接保留速度或时长线索。

这里不能把整个变换称为 source-only（仅源域）：standardization/PCA coordinate transform（标准化／主成分坐标变换）仍由 120 条 outer 无标签描述器传导式拟合；只有 subject nuisance basis（受试者干扰基）利用 Offline old6 的 10 名训练受试者元数据。每名源域受试者都包含相同的 6 类×5 trials，所以在不读取活动标签的前提下对等权受试者中心做 SVD，再从 32 维结构主块移除最多 4 个方向。Outer 受试者 ID 不参与坐标变换、干扰基或聚类拟合，推理也不需要 ID；它只在 `raw_predictions.npz` 写入并完成 SHA256 冻结、随后 scorer truth join（评分器真值连接）之后用于诊断和可视化。`raw_predictions.npz`、`fit_manifest.json` 与 `trajectories_label_free.jsonl` 均不保存 outer 受试者 ID。该操作只能降低源域可辨识的受试者方向，不能保证未见受试者身份完全不可恢复；每个成员必须同时检查 `descriptor_bias_audit.json` 中 source/outer centroid dispersion（源域／外层中心离散度）和时长相关诊断。

这仍是已知活动簇数 K12、120-trial transductive batch GCD（120 试次传导式批量广义类别发现）代理，不是多 Session Online CGCD。`descriptor_targeted_contrasts.json/csv` 在三个目录中分别只给出 `gravity − legacy`、`subject-debias − legacy` 或 `duration-invariant − legacy`；统计先在同 fold/seed 内配对、再在折内平均种子，三折置信区间仅作探索性证据。

`--experiment-part all` 只负责编排：它依次执行 `gravity/`、`subject_debias/`、`duration/` 三个独立子目录，每个子目录仍有自己的 12-member 身份、汇总和完成清单；顶层 `suite_summary.json` 只汇总三项完成状态，不把处理臂组合。PyCharm Linux/WSL 终端一行命令：

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_trajectory_descriptor_ablation_cv.py --experiment-part all --npz-path /mnt/d/WorkDir/HHR/processed/uschad_w128_s64_train17stats/uschad_windows.npz --encoder-source-root /mnt/d/WorkDir/HHR/results/motion_primitive/window_codebook_batch_proxy_3arm_3fold2seed_20260914_v1/encoders/w128_s64 --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/w128_k64_descriptor_3factor_suite_3fold2seed_20260916_v2 --folds 1,2,3 --seeds 0,5 --signed-vertical-distance-weight 0.15 --subject-nuisance-max-rank 4 --subject-nuisance-explained-variance 0.90 --kmeans-n-init 50 --kmeans-max-iter 300 --encode-batch-size 1024 --bootstrap-seed 20260916 --bootstrap-replicates 10000 --device cuda --resume
```

### 时长不变＋软受试者去偏强度实验

新入口 `duration_soft_subject` 固定 `A2 + E0 + W128/S64 + K64`，比较 `duration_invariant_v1` 与三个 soft subject nuisance projection（软受试者干扰投影）强度。真正执行的公式为 `z' = z - alpha * (z B^T) B`，随后重新进行 L2 normalization（L2 归一化）；`B` 仍只用 Offline old6 的平衡源域受试者中心拟合，outer 受试者身份不进入任何预真值变换。`alpha` 固定为 `0.25/0.50/0.75`，而不是通过改变 rank 间接模拟软投影。

该入口一共运行 `5 profiles × 3 folds × 2 seeds = 30` 个成员：`legacy_state_v1` 只为每个 fold/seed 生成一次共享 K64 码本；`duration_invariant_v1` 是三个软投影臂的直接实验基线。汇总只把 `soft-alpha − duration-invariant` 作为目标对照，并强制核验四个非 legacy profile 在去偏前拥有完全相同的 transform state SHA256（变换状态哈希）。`--experiment-part all` 仍保持原来的三个独立单变量实验与 36 个成员，不会自动加入本组合实验。PyCharm Linux/WSL 终端一行命令：

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_trajectory_descriptor_ablation_cv.py --experiment-part duration_soft_subject --npz-path /mnt/d/WorkDir/HHR/processed/uschad_w128_s64_train17stats/uschad_windows.npz --encoder-source-root /mnt/d/WorkDir/HHR/results/motion_primitive/window_codebook_batch_proxy_3arm_3fold2seed_20260914_v1/encoders/w128_s64 --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/w128_k64_duration_soft_subject_alpha_grid_3fold2seed_20260916_v1 --folds 1,2,3 --seeds 0,5 --subject-nuisance-max-rank 4 --subject-nuisance-explained-variance 0.90 --kmeans-n-init 50 --kmeans-max-iter 300 --encode-batch-size 1024 --bootstrap-seed 20260916 --bootstrap-replicates 10000 --device cuda --resume
```

## 历史七折 PyCharm Linux/WSL 命令

下面命令只用于历史七折路线复现，不是当前默认命令：

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_full_cv.py --npz-path /mnt/d/WorkDir/HHR/processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/frozen_a2_e0_state_k32_strict_7fold4seed_20260912_v1 --folds 1,2,3,4,5,6,7 --seeds 0,5,50,500 --window-epochs 60 --window-batch-size 64 --window-eval-batch-size 256 --window-learning-rate 0.1 --window-weight-decay 0.0005 --window-weak-scale-std 0.1 --window-strong-scale-std 0.2 --a2-epochs 30 --a2-trial-batch-size 8 --a2-learning-rate 0.0001 --a2-minimum-learning-rate 0.000001 --a2-weight-decay 0.0001 --encode-batch-size 512 --num-workers 0 --old-distance-alpha 0.05 --old-ratio-alpha 0.05 --novel-distance-alpha 0.05 --minimum-cluster-trials 3 --minimum-cluster-subjects 2 --minimum-cluster-silhouette 0.20 --discovery-bootstrap-replicates 100 --minimum-bootstrap-stability 0.80 --minimum-registry-separation 0.10 --minimum-candidate-separation 0.10 --summary-bootstrap-seed 20260912 --device cuda --visualize --visual-fold 1 --visual-seed 0 --maximum-trial-plots 0 --visual-dpi 180 --resume
```

该入口按顺序完成 28 个窗口编码器、28 个 A2 编码器、28 个 Offline 冻结读出、28 个三会话 Online 成员、84 行统计汇总，以及 fold 1/seed 0 的完整可视化。它是顺序执行，预计运行时间主要由 56 次编码器训练决定。

## 历史七折入口的中断与 `--resume`

可以在终端用 `Ctrl+C` 中断；继续时原样重跑上面一行命令。

- `--resume` 只复用身份和哈希均验证通过的 complete member（完整成员）；
- 窗口/A2 的未完成成员会先移入该 encoder grid（编码器网格）的 `_interrupted` 目录，再从该成员起点重跑；
- Offline/Online 未完成成员会从成员级阶段重跑，不支持 mid-epoch resume（轮次中间恢复）；
- 已有 complete marker（完成标记）若损坏或与 artifact 不一致，程序会 fail closed（拒绝继续），不会静默覆盖；
- 输出根目录由 NPZ 哈希、全部关键参数和实现源码 SHA256 锁定。修改参数、数据或代码后必须使用新的 `--output-root`；
- 不要手动把其他实验成员复制到当前输出根目录，也不要删除 `grid_manifest.json` 后尝试接管旧目录。

## 主要产物

```text
<output-root>/
  grid_manifest.json
  encoders/
    window_pretrain/fold_XX_seed_Y/
    a2/fold_XX_seed_Y/motion_encoder_final.pt
  offline/fold_XX_seed_Y/
    e0_codebook.npz
    state_descriptor_transform.npz
    old_registry.json
  online/fold_XX_seed_Y/
    raw_predictions_session_*.npz
    discovery_session_*.json
    registry_session_*.json
    metrics_session_*.json
  online/online_runs.csv
  online/online_summary.{json,csv,md}
  visualizations/fold_01_seed_0/
  complete.json
```

算法细节、监督边界和数学定义见 [METHOD.md](METHOD.md)。
