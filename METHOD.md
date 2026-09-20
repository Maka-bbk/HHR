# HHR 方法定义：冻结动作元轨迹 HAR-CGCD

## 1. 研究问题与协议地位

HHR 验证如下假设：对 USC-HAD，一个完整 activity trial（活动试次）不应只压缩为单个 mean/max pooled feature（均值/最大池化特征）；先把按时间排序的局部窗口编码为 Motion Primitive（动作元），再利用动作元的种类、数量、持续时间、位置、顺序、转移和物理状态构造 activity trajectory（活动轨迹），可能更有利于 Continual Generalized Category Discovery（连续广义类别发现，CGCD）。

当前活动实现（2026-09-19 起）是 fixed split 01（固定划分 1）的 `A2 + E0 + W128/S64 + K64` 三臂验证；历史冻结式 `A2 + E0 + state + K32` 七折三会话路线保留用于复现，但不再是新实验默认入口。动作元模块存在、轨迹图清晰或某类只出现一个 token 都不能单独证明假设；证据必须来自预注册的配对消融和冻结预测后的分类指标。

### 1.1 当前固定划分与种子语义

- 内部仍使用 registered split id `fold=1` 核验数据和 checkpoint（检查点），但 CLI（命令行接口）不提供 folds（多折）参数；
- 训练/验证/outer-test（外层测试）受试者固定为 `10/2/2`，outer-test 仅为受试者 `10,11`；
- 编码器来源包含两条线：`R0_reuse_0914v1` 复用既有 `W128/S64、A2、fold 1、encoder_seed=0` 检查点；`R1_retrained_current` 从 seed 0 随机 ResNet1D 初始化重新训练相同的 60 轮窗口阶段和 30 轮 A2 阶段；
- 每条编码器线的 `run_seed=0,5,50,500` 只改变 Offline old6 K64 码本、无标签 GMM（高斯混合模型）门控和 KMeans（K 均值）聚类，不代表四个独立编码器种子；
- R0 是历史训练产物，R1 使用当前训练源码，因此二者比较不是纯随机初始化单因素实验，而是历史已训练编码器与当前重训编码器的来源比较；
- 单一受试者划分不能估计跨受试者总体 confidence interval（置信区间）。只报告四个下游随机种子的逐项值、均值与样本标准差。

### 1.2 当前三臂数据流

```text
两条 encoder 来源：R0 复用 0914v1；R1 从随机 ResNet1D 重新训练 warm-up -> A2
  -> 各自冻结 fold-1/seed-0 A2 encoder
  -> Offline old6 拟合 K64 动作元码本
  -> 120 条 outer trial 形成动作元轨迹
  -> B0：时长不变轨迹描述器 -> 全局 KMeans12
  -> E1：无标签静态/动态 GMM 门控
       -> dynamic：时长不变动作元轨迹专家
       -> static：姿态 + 能量 + 带符号重力 + 低权重动作元独立专家
       -> 按无标签门控成员比例分配 K_dynamic/K_static，总和固定为 12
  -> E2：完全复用 E1 的门控、K 分配、静态专家和静态预测
       -> 仅 dynamic 改为时长不变 + alpha=0.25 软受试者去偏
       -> dynamic outer 子集拟合无标签坐标/PCA32
       -> Offline old6 的 300 条源轨迹、10 名源受试者拟合 rank<=4 干扰基
       -> z'=L2Norm[z-0.25(zB^T)B]
  -> 先冻结 raw cluster IDs 及 SHA256
  -> 后打开 TruthStore，进行全局 Hungarian scoring（匈牙利评分）
  -> 顶层按同一 run_seed 配对比较 R1-R0
```

静态专家的距离预算预注册为：姿态 `0.35`、带符号重力 `0.35`、能量 `0.20`、动作元 `0.10`。带符号重力块使用 confidence-preserving normalization（保置信度归一化）：强信号被限幅，但安静试次的微小噪声不会被强制放大为单位长度。门控和专家 API 不接收 activity label（活动标签）或 outer subject ID（外层受试者编号）。E2 的干扰基只使用源域受试者元数据；其拟合 API 不接收活动标签，但源 cohort（队列）由注册协议预先限定为 old6，因此不能把整条流程描述成完全无标签。完整分配公式为 `K_static=clip(round(12*N_static/120),1,12-K_old)`、`K_dynamic=12-K_static`，其中已知旧类数 `K_old=6` 是动态分支下界。该估计同时假设 USC-HAD outer batch 每类 trial 数相等、门控近似按完整活动类形成纯分支；在不平衡、类跨分支或真实流式数据上不能直接使用。

该协议仍然是 known-K12 transductive batch GCD（已知总类数的传导式批量广义类别发现），不是 sequential Online CGCD（顺序在线连续广义类别发现）。此外，old6 全部属于人体动态类，而 new6 中五类属于人体静态类，因此静动态门控也是一个明显的 old/new shortcut（旧新类捷径）；即便 E1/E2 提升，也不能直接证明其可泛化到静动态混合的新旧类划分。`E1-B0` 同时包含门控、分支内坐标重拟合和静态物理专家三个变化，只能解释为“完整组合路线效果”，不能单独归因为静态专家的纯因果贡献；`E2-E1` 共享门控、分支 K 与静态预测，才隔离动态软去偏增量。E2 在 gate-selected dynamic subset（门控选出的动态子集）拟合坐标，而早期 A025 单因素实验在全部 120 条 outer 上拟合坐标，两者不是同一个实验范围。

## 2. 术语和三个不同的“K”

文档中必须区分：

- 当前 `E0 capacity-K=64`：固定动作元码本有 64 个中心；
- 历史 Online `E0 capacity-K=32`：旧七折路线固定使用 32 个中心；
- `actual used-K`：某个数据划分实际用到的不同动作元 token 数；
- `activity cluster K=12`：当前 batch GCD 三臂预先知道的总活动簇数；
- `activity registry K`：历史 Online 路线已经注册的活动类别原型数，初始为 6，随后最多依次增长。

Online 只允许第三项增长。把 activity registry 增长写成“动作元码本从 32 扩容”是错误的。

E0 的一个 primitive occurrence（动作元出现）对应一个已保存窗口；`unique_primitive_type_count` 才是该 trial 用到的不同 token 类型数量。两者也不能混写。

## 3. 数据、受试者划分与标签边界

### 3.1 当前输入窗口

当前输入为：

```text
processed/uschad_w128_s64_train17stats/uschad_windows.npz
```

- 六个通道：三轴加速度和三轴角速度；
- window size（窗口长度）128 samples，stride（步长）64 samples；
- 采样率 100 Hz，对应 1.28 秒窗口和 0.64 秒移动步长；
- 通过 `trial_global_ids`、`window_indices` 和 `window_start_indices` 恢复完整有序窗口轨迹；
- 不足 128 个采样点的原始尾段已在预处理时丢弃，因此“完整 trial”只指 NPZ 内所有已保存完整窗口。

NPZ、固定 A2 checkpoint 及其 fold-1 mean/std（均值/标准差）必须通过 SHA256 和 checkpoint metadata（检查点元数据）共同核验。NPZ 中可逆的历史归一化用于恢复门控和静态专家需要的原始六轴数值；outer 数据不重新拟合编码器归一化。

历史七折 Online 路线使用 `uschad_w256_s128_train17stats/uschad_windows.npz`，对应 256/128 栅格；下文历史章节中的 W256/K32 只属于该路线。

### 3.2 历史七折受试者协议

以下 7-fold subject-disjoint cross-validation（7 折受试者无交叉交叉验证）只描述历史 Online 路线：

- 10 名训练受试者；
- 2 名验证受试者；
- 2 名 outer-test（外层测试）受试者；
- 三组受试者两两不重叠，7 折覆盖全部 14 名受试者。

旧类为物理标签 `0..5`。每折 Offline old-train/validation/outer-test 的 trial 数固定为 `300/60/60`。

### 3.3 标签隔离

协议构造器需要真实标签来建立规定的 old/new stream（旧/新类数据流），scorer（评分器）也需要真实标签计算指标；这不等于 Online learner（在线学习器）可以读取标签。

学习器可见的 `SensorTrial`、`SessionStream` 和 `LabelFreeTrial` 只包含：

- `trial_id`；
- `subject_id`；
- `session_id`；
- 六轴窗口、时间位置或冻结描述符。

这些对象没有 activity label（活动标签）字段。Online 原始预测必须先持久化并计算 SHA256，之后 scorer 才能从独立 `TruthStore` 连接真实标签。评分产生的映射不能写回 registry（注册表）。

## 4. Stage A：窗口级 ResNet1D 暖启动

### 4.1 架构

每个六轴窗口由 ResNet1D（一维残差网络）编码：

```text
in_channels=6
feature_dim=256
base_channels=64
layers=[2,2,2]
dropout=0
```

参数从随机状态初始化。训练只使用 old-train 窗口，validation 仅用于选点，outer-test forward count（外层测试前向次数）必须为 0。

### 4.2 视图和损失

同一窗口产生 weak/strong views（弱/强视图）：

- weak：逐通道 scale standard deviation（缩放标准差）`0.10`；
- strong：逐通道缩放标准差 `0.20`；
- jitter、time mask（时间遮挡）和时间顺序变换均为 0。

窗口暖启动目标为：

```text
L_window = 1.00 L_CE + 0.65 L_InfoNCE(T=1.0) + 0.35 L_SupCon(T=0.07)
```

其中 CE 是 cross-entropy（交叉熵），InfoNCE 是信息噪声对比估计，SupCon 是 supervised contrastive loss（监督对比损失）。分类 logits temperature（分类输出温度）单独为 `0.1`，不能误写成 InfoNCE 温度。历史 cluster loss（聚类损失）权重为 0，不进入总损失。

优化器为 SGD（随机梯度下降）：学习率 `0.1`、momentum（动量）`0.9`、weight decay（权重衰减）`5e-4`；bias 和一维参数不衰减；60 轮 cosine schedule（余弦调度），最小倍率 `1e-3`；batch size（批大小）64。每轮计算 validation window macro-F1（验证窗口宏平均 F1），并保存最高值对应的 `model_best.pt`。

## 5. Stage B：A2 动作编码器

### 5.1 初始化和角色分离

A2 从同一 fold/seed 的窗口 `model_best.pt` 严格加载完整 ResNet1D。编码器输出至少分成两个角色：

- `content`：E0 码本唯一允许读取的局部运动内容；
- `segmentation`：只用于 A2 变点/边界训练约束，E0 下游不读取。

内容和分段残差支路末层零初始化，使 A2 初始点保留窗口暖启动表征。backbone BatchNorm（骨干批归一化）的 running statistics（运行统计）冻结，仿射参数仍参与训练。EMA teacher（指数滑动平均教师）只提供 masked temporal prediction（遮挡时间预测）目标。

### 5.2 A2 损失

正式 A2 总目标为：

```text
L_A2 = 1.00 L_changepoint
     + 0.10 L_content_boundary_alignment
     + 0.05 L_noncollapse
     + 0.50 L_temporal_prediction
     + 0.10 L_trial_auxiliary
```

明确为 0 的项：

```text
L_window_augmentation_InfoNCE = 0
L_cross_subject = 0
```

因此，“A2 不含 InfoNCE”只指 Stage B；Stage A 的窗口暖启动仍包含 InfoNCE，二者不能混淆。

`L_trial_auxiliary` 是 old-class（旧类）训练辅助项：在 A2 content 窗口上计算 `mean+q90` 并进行低权重 trial 分类。它用于维持历史 A2 的内容语义，不是 E0 下游表示、不是最终 CGCD 分类头，也不进入 Online learner。若把该权重改为 0，必须命名为新消融，不能仍宣称是正式 A2。

### 5.3 优化与固定输出

- AdamW，学习率 `1e-4`、最低 `1e-6`、权重衰减 `1e-4`；
- 30 轮，trial batch size 8；
- gradient clipping norm（梯度裁剪范数）5；
- EMA momentum（指数滑动平均动量）0.99；
- early stopping（提前停止）关闭；
- selection policy（选择规则）固定为 `final_epoch`。

下游只加载第 30 轮 `motion_encoder_final.pt` 的 student parameters（学生参数）；不加载 `motion_encoder_best.pt`，也不把 EMA teacher 当作码本编码器。

### 5.4 为什么不是“旧两阶段损失叠加”

Stage A 和 Stage B 确实是两个顺序优化过程，但不存在以下行为：

- 不在同一个 batch 中同时计算 Happy 原损失和动作元轨迹损失；
- 不在 E0/Online 阶段继续优化上述任一损失；
- 不在进入 CGCD 后重置再训练编码器；
- 不用多个预测头做 pooled/fused（池化/融合）结果。

准确表述是：窗口语义暖启动 → A2 局部动作表征训练 → 固定轮次冻结 → 非参数轨迹读出。

## 6. Stage C：冻结 E0/PCA64/KMeans32

### 6.1 Trial 等权局部空间

设 A2 content 为 `c_tj`，其中 `t` 是 trial，`j` 是窗口。首先逐行 L2 归一化。若 trial `t` 有 `n_t` 个窗口，则每个窗口的权重与 `1/n_t` 成正比，使每个 trial 对 PCA（主成分分析）和 KMeans（K 均值）拟合贡献相同总质量，避免长 trial 因窗口更多而主导码本。

在且仅在 Offline old-train 上执行：

```text
c -> row L2 -> weighted PCA64 -> row L2
  -> KMeans(K=32, n_init=20, max_iter=300, algorithm=lloyd)
```

KMeans 拟合时继续使用同一 trial-equal sample weights（试次等权样本权重）。推理时归一化 32 个中心，对每个窗口选择余弦相似度最大的中心：

```text
z_tj = argmax_k cosine(pca(c_tj), centre_k)
```

分配是 hard quantization（硬量化）；码本、PCA 和 token ID 从此冻结。

### 6.2 E0 的时间含义

E0 使用原始固定 `w256/s128` 栅格，每个窗口对应一个 token，不做相同 token 的游程合并。因此：

- trial token occurrence 数等于该 trial 保存窗口数；
- 不同 trial 可有不同轨迹长度；
- E0 不是变点检测，也不是 learned variable-length segmentation（学习式变长分段）。

由于窗口 50% 重叠，持续时间不能简单记为每个 token 256 samples。实现按相邻窗口中心中点划分 ownership（归属区间）：内部窗口通常拥有 128 samples，首尾窗口拥有边缘区域；所有区间首尾相接，其总和恰好等于 NPZ 可观测 trial span（试次跨度）。原始六轴窗口统计仍在完整重叠窗口上计算，ownership 只用于持续时间计权。

## 7. Stage D：3739 维 state trajectory descriptor

对 `K=32` 的 token 序列构造以下原始块：

| 块 | 维数 |
|---|---:|
| token count fraction（出现比例） | `K=32` |
| token ownership-duration fraction（归属持续时间比例） | `K=32` |
| 四个相对时间区间的 token 持续时间 | `4K=128` |
| any/peak/valley 三组有向转移矩阵 | `3K²=3072` |
| 长度、持续时间、量化质量及父元占位标量 | `11` |
| 六轴窗口状态的全局均值与标准差 | `48` |
| 每个 token 的六轴 mean/energy 条件统计 | `12K=384` |
| token presence mask（出现掩码） | `K=32` |

无 state 部分为：

```text
6K + 3K² + 11 = 3275
```

state 增量为：

```text
48 + 12K + K = 464
```

所以正式 raw descriptor（原始描述符）严格为：

```text
3275 + 464 = 3739 dimensions
```

E0 没有 peak/valley event（峰/谷事件），因此对应两个转移矩阵是全零占位块；随后会由训练集常量列过滤删除。保留它们是为了与已验证历史 state schema（状态结构）逐字段兼容。

这些统计是结构化 trajectory readout（轨迹读出），不是把 A2 窗口特征直接做单个 mean+max pooling。它显式保留 token 身份、相对时间、持续时间和一阶有向转移；但它仍是固定维摘要，因此更高阶长程顺序是否被充分保留需要后续消融验证。

## 8. Stage E：Offline-only descriptor transform 与旧类注册表

只用 300 条 Offline old-train trial 拟合：

1. 删除训练集标准差不超过阈值的常量列；
2. 对保留列拟合 z-score；
3. 用 SVD（奇异值分解）拟合最多 32 维 PCA；
4. 对输出逐行 L2 归一化。

validation、outer-test 和全部 Online trial 只能调用冻结 transform（变换），不能重新拟合列掩码、均值、尺度或 PCA。

旧类 `r=0..5` 的 prototype 是该类 old-train 单位描述符均值再 L2 归一化。validation 用于校准两个 split-conformal gate（分割保形门控）：

- 到本类 prototype 的余弦距离阈值；
- 最近距离与第二近距离之比阈值。

有限样本 `1-alpha` 分位点使用 `ceil((n+1)(1-alpha))` 阶统计量并封顶到 `n`，正式 `alpha=0.05`。旧类行、阈值、ID 和其 SHA256 anchor（锚点）随后永久冻结。

所有对象冻结后，Offline outer-test 只评估一次，用作旧类基线诊断，不参与任何选点或门控拟合。

## 9. Stage F：严格三会话 activity registry

### 9.1 固定 stream

新类顺序固定为 `(6,7) -> (8,9) -> (10,11)`，不随机打乱。每个 outer-test subject/class 有固定 trial 池；Online incoming 与 evaluation 不重叠，incoming trial 也不会跨 session 重用。

| Session | 可见类别数 | Incoming trials | Evaluation trials |
|---:|---:|---:|---:|
| 1 | 8 | 22 | 58 |
| 2 | 10 | 26 | 52 |
| 3 | 12 | 30 | 42 |

### 9.2 冻结与可变状态

| 组件 | Online 状态 |
|---|---|
| ResNet1D/A2 参数和 BatchNorm buffers（缓冲） | 冻结 |
| E0 PCA64 | 冻结 |
| KMeans32 中心、容量和旧 token ID | 冻结 |
| 3739 schema、列掩码、z-score、PCA32 | 冻结 |
| 旧类 registry 行、阈值和 ID | 冻结 |
| novel activity registry rows（新活动类别行） | 只追加 |
| Unknown buffer | 可跨 session 累积或缩小 |

Online 没有神经优化器，也不会同步更新动作元码本。

### 9.3 路由、发现和追加

每个 incoming trial 先按当前 registry 最近/第二近余弦距离路由。只有同时满足该最近行距离阈值和距离比阈值时才接收；否则输出 `Unknown=-1`。

本 session 新拒识 trial 与以前 Unknown buffer 合并，然后只在该未知池上执行 spherical KMeans（球面 K 均值）。正式协议预先知道“每个 session 最多引入两个新类”，当前候选聚类数固定请求为 2。这不使用具体 activity label，但仍是 class-increment prior（类别增量先验），所以当前方案不是未知新类数量的完全开放世界发现。

候选必须同时通过：

- 至少 3 条 trial；
- 至少 2 名受试者；
- mean silhouette 至少 `0.20`；
- bootstrap stability 至少 `0.80`；
- 到已有 registry 的最小余弦距离至少 `0.10`；
- 候选之间最小余弦距离至少 `0.10`。

发现 bootstrap（自助抽样）正式为 100 次。候选顺序由最小 trial ID 决定，不查询真实 activity；通过者从当前 `next_registry_id` 起顺序追加。新行半径使用 leave-one-out（留一）距离保形分位数，新行距离比阈值同样只用候选内部无标签几何校准。未通过或注册后仍被拒识的 trial 留在 Unknown buffer。

每个 successor registry（后继注册表）记录前一状态哈希；实现验证 representation hash、old anchor、旧行和已见 trial 集合的 append-only invariant（只追加不变量）。

## 10. 预测持久化与三层评分

每个 session 在 evaluation 前已经完成本 session 的 registry update。评估顺序严格为：

```text
label-free descriptors
  -> current registry routing
  -> raw_predictions_session_S.npz
  -> SHA256 + registry/representation binding
  -> TruthStore join
  -> scoring-only alignment and metrics
```

### 10.1 `direct_registry`

不做映射，旧类 `R0..R5` 保持已知语义，新 registry ID 按无标签注册先后存在。它不使用测试标签，是最接近部署的 ID 层；但新 registry ID 的数字通常不等于真实新类物理编号，因此其 novel accuracy 不能代替聚类质量。

### 10.2 `old_fixed_novel_hungarian`

旧类 ID 永远固定，只允许已注册 novel IDs 与真实 novel classes 做 Hungarian matching（匈牙利匹配）。该映射只在评分副本上应用，不回写 registry。这是正式主 CGCD 指标，但它仍使用测试标签完成新类编号对齐，不能称为完全无标签部署准确率。

### 10.3 `global_hungarian_upper_bound`

允许旧类和新类全部重排，是诊断表示几何的乐观上界。它可以掩盖旧类语义交换或 old/novel 混淆，不能作为主结果或部署性能。

三层分别保存带 Unknown 列的混淆矩阵、aligned predictions（对齐预测）和映射，并报告：

- All accuracy；
- Old accuracy；
- New accuracy；
- `H = 2*Old*New/(Old+New)`；
- macro-F1；
- Unknown count/fraction。

## 11. 历史 7 折 × 4 种子 × 3 会话统计

历史 grid（网格）固定为：

```text
folds = 1,2,3,4,5,6,7
seeds = 0,5,50,500
sessions = 1,2,3
```

因此落盘 `28×3=84` 行 session metrics（会话指标）。这 84 行有共享受试者和重复算法设置，不能直接当作 84 个独立样本。

每个 session/metric 的汇总步骤是：

1. 在每个 fold 内平均 4 个 seed；
2. 得到 7 个 held-out-subject fold means；
3. 报告 7 折均值和 sample standard deviation across folds（跨折样本标准差）；
4. 对 7 个 fold means 做 10,000 次有放回 bootstrap，报告 2.5%/97.5% 分位数。

统计推断单位是 held-out-subject fold，不是单 seed，也不是 84 行中的每一行。

## 12. 可视化协议

可视化只在 Online `complete.json` 之后运行；它不导入 registry update API，也不修改模型、PCA、码本、描述符或 registry artifact。加载器先验证：

- Offline manifest 与 complete hash；
- NPZ SHA256；
- 每个 raw prediction SHA256 及其 registry/representation 绑定；
- 三个 registry state 的 previous-state hash chain；
- scorer CSV 与 raw prediction 的 trial 顺序和数值；
- learner-facing trajectory JSONL 不含 activity label。

原始六轴曲线从经哈希验证的 NPZ 按 `trial_id` 重载并去归一化；代码不读取 NPZ 的 activity label 数组。真实类别名称只从预测冻结后的 scorer artifact 读取，用于图注和评分热图。

输出至少包括：

- actual used-K CSV/JSON/PNG；
- 每个唯一观测 trial 的完整离散序列索引，以及六轴信号 + token ownership 图；
- final-session activity×primitive heatmap（活动—动作元热力图）；
- 三层 session confusion heatmap（会话混淆热图）；
- 轨迹与三层预测对照；
- activity registry K、注册数和 Unknown buffer 变化。

Activity×primitive 不能按所有窗口直接求和，否则长 trial 和 trial 更多的受试者会过度加权。实现先对每条 trial 求 token fraction，再在 subject 内平均，最后让 subject 等权。

KMeans 的中心编号可任意置换。`P0..P31` 只在单个 fold/seed member 内可解释；跨折/种子平均同编号热图在没有中心对齐时没有统计含义。正式入口默认只为 fold 1/seed 0 生成解释图，跨成员比较应先用码本中心做无标签匹配并公开匹配方法。

## 13. 历史七折入口的中断、恢复与身份锁定

历史根入口是 `experiments/motion_primitive/run_full_cv.py`。输出根的 `grid_manifest.json` 固化：

- NPZ 解析路径和 SHA256；
- folds/seeds；
- 全部训练、registry、运行时和可视化参数；
- 核心实现文件 SHA256。

`--resume` 规则：

- 只复用 schema、身份和 artifact 哈希全部验证通过的 complete member；
- 同一输出根若参数、NPZ 或实现哈希改变，立即拒绝，必须换新输出目录；
- 未完成的窗口/A2 成员先安全移动到 `encoders/_interrupted` 后从成员起点重跑；
- 未完成的 Offline/Online 成员从该成员阶段重跑；
- 不支持 optimizer-level mid-epoch resume（优化器级轮次中恢复）；
- 损坏的 complete member 不会被 `--resume` 静默覆盖，而是 fail closed（拒绝继续）。

因此可以用 `Ctrl+C` 中断，之后原样重新执行 README 中的一行正式命令；不要删除身份 manifest 或手动混合其他实验的成员。

## 14. Legacy 与历史数值的可比性

旧 `joint VQ + learned boundary + GRU`（联合向量量化、学习边界与门控循环网络）路线仍可作为 legacy（历史代码）保留。下列“当前路线”均指本节所比较的历史冻结 Online 路线，不是文档顶部 2026-09-19 fixed-split 三臂实验：

- 旧路线端到端更新编码器、可学习 VQ 码本、边界头和 GRU；
- 历史冻结 Online 路线先完成固定 A2 表征，再用非参数 E0 K32 和结构化 state descriptor；
- 旧路线 Online 可更新神经参数或动作元码本；历史冻结 Online 路线只追加 activity registry；
- 旧路线评估协议与历史冻结路线的 raw-prediction-first、三层严格评分不同。

旧实验报告的 `75.96%` 属于 batch oracle（批次先验上界）结果，不是当前严格 Online registry 指标。即使数值更高，也不能与 `old_fixed_novel_hungarian` 直接比较。公平比较需要把 legacy 表征冻结后接入完全相同的 7×4×3 stream、拒识/Unknown buffer、append-only registry 和三层评分；否则差异可能来自 oracle 信息、数据单位或对齐方式，而不是动作元方法本身。

## 15. 历史冻结 Online 路线的已知限制和消融

- E0 边界受 1.28 秒步长限制，不是采样点级真实动作边界；
- 每个 E0 token 对应固定窗口，当前尚未验证变点式变长动作元是否优于 E0；
- 3739 维描述符保留一阶转移和四分位位置，但不完整保留任意长程顺序；
- A2 trial auxiliary 使用旧类标签，不能把整个 A2 称为纯自监督；
- Online 不读活动标签，但知道每个 session 最多新增两类，仍含 class-count prior；
- `subject_id` 用于候选跨受试者支持门控，是协议元数据而非分类输入；这降低单一受试者伪发现风险，但不保证完整域不变性；
- Sitting/Standing（坐下/站立）是否由 state 块改善，必须通过去 state 消融确认；
- Running Forward（向前跑）只有单一或高度稳定轨迹不是缺陷，只要其分类和跨受试者稳定性更好；
- actual used-K 更大不等于分类更好，dead code 更少也不是独立成功指标。

最小公平消融顺序应保持同 fold/seed 和同严格 Online registry：

1. A2 vs 窗口暖启动特征；
2. state on/off；
3. transition/order/duration 块逐项移除；
4. trial-equal PCA/KMeans vs window-equal；
5. 固定 E0 K32 vs 经预注册规则控制的动作元码本扩展；
6. 当前已知两新类先验 vs 无新类数先验的模型选择。

上述跨折成功标准只适用于历史七折 Online 路线。当前 fixed split 01 实验只能确认受试者 10/11 上的机制可行性和四个下游种子的算法稳定性，不能提供跨受试者总体结论。
