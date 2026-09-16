# Motion Primitive Trajectory Experiment：阶段 1 可行性诊断

## 1. 当前实验只回答什么

本目录实现的是 **USC-HAD 完整 trial 的动作元候选分解与跨 subject 序列关联诊断**，只回答两个问题：

1. 冻结的窗口编码器输出能否被离散为非坍缩的局部状态，使一个典型完整 trial 表示为多个动作元候选及其有序序列；
2. 在从未参与动作元码本拟合的 subjects 上，同类 activity 的动作元候选序列是否比异类 activity 的序列更接近。

这里的“动作元”严格指 **latent motion-state candidate（潜在动作状态候选）**。KMeans 必然会产生离散簇，因此“得到 token”本身不是证据；真正需要检查的是码本是否坍缩、trial 是否通常包含多个候选，以及这种表示能否在 held-out subjects 上保留同类关联。

本阶段不是完整 trajectory model，也不是 CGCD 实验；不包含 trajectory memory、novelty detection、online clustering、未知类发现或 Happy-CGCD 主训练流程改造。现有 trial pooling 分支实际使用的是 `mean + q=0.90 robust peak + MLP fusion`，并非字面意义上的普通 `mean + max`；本实验尚未证明动作元表示优于该 pooling 基线。

## 2. 数据与表示

- 数据：USC-HAD，14 subjects、12 activities、每个 subject/activity 5 个 trials，共 840 个 trials；
- 预处理文件：`processed/uschad_w256_s128_train17stats/uschad_windows.npz`；
- 输入窗口：`[6, 256]`，100 Hz，即 2.56 秒；stride 为 128 samples，即 1.28 秒，原始相邻窗口有 50% 重叠；
- 全数据包含 20,768 个窗口，并保留 `subject_id`、activity、trial、窗口序号和窗口起点，因此可以按原时间顺序还原每个 trial 的窗口序列；
- NPZ 不包含每条原始 trial 最后不足一个窗口的尾段，输出中的时长是 **observed span**，不是原始 trial 的精确总时长。

## 3. 方法

每个 fold/seed 独立执行以下流程：

1. 加载对应 subject-CV checkpoint，提取并冻结 ResNet1D 窗口编码器；编码器保持 `eval` 模式且所有参数 `requires_grad=False`。
2. 只使用该 fold 的训练 subjects、旧 6 类窗口计算 fold-specific normalization，并提取 256 维局部特征。
3. 对特征做 L2 normalization；仅在拟合集上训练 trial 等权的 weighted PCA-64；再次 L2 normalization。
4. 仅在训练 subjects 的旧 6 类上拟合 `K=32` KMeans。每个 trial 获得相同总权重，避免长 trial 因窗口更多而支配 PCA 和码本。
5. 将 held-out outer-test subjects 的每个窗口分配到最近码字，按 `window_start_indices` 排序，形成 raw token sequence；再将连续相同 token 压缩为 RLE sequence，同时保存每段的窗口数和 observed span。
6. 用 normalized Levenshtein distance 比较跨 subject 的 RLE 序列。主指标是异类平均距离减同类平均距离：

   `sequence margin = mean(distance_different_activity) - mean(distance_same_activity)`

   margin 大于 0 表示同类 activity 的序列更接近。另报告跨 subject 1-NN activity retrieval accuracy；它只是表示诊断，不是训练出的分类器。
7. 每个 run 执行 1,000 次 subject-blocked 标签置换：只在各 subject 内打乱 activity 标签，以保留每个 subject 的类别数量。最终汇总先在每个 fold 内平均 4 个 seeds，再以 7 个 folds 为独立块做精确单侧 sign-flip test。

主序列关联统计使用 held-out subjects 的全部 12 类；旧 6 类和 novel 6 类的分组结果仍保存在各 run 的明细文件中。novel 类只用于诊断，不能据此声称完成未知类发现。

### 3.1 可切换的动作元分段前端

`--primitive-segmentation` 现在提供两种互斥选择；两者下游都使用相同的 weighted PCA、KMeans `K=32`、窗口栅格序列控制和 subject-CV 评价：

- `fixed_window`：原实验路径，每个 2.56 秒输入窗口直接作为一个码本 observation；这是默认值，保持旧 KMeans32 基线。
- `ssl_feature_changepoint`：冻结原窗口编码器；只在 fit subjects / old 6 classes 的 embedding 上训练无标签 masked-denoising adapter；在 adapter 特征序列上计算局部左右均值的 cosine change score；阈值只由 fit trials 的 trial-equal score quantile 校准；变点之间的窗口特征均值形成变长 segment，再进入同一个 KMeans32。

`ssl_feature_changepoint` 中的“自监督”严格指新增的无标签 masked-denoising adapter，并不把原 checkpoint 重新表述为纯自监督模型。eval subjects 不参与 adapter、阈值、PCA 或 KMeans 拟合。每个 segment 的 token 会回填到它包含的原窗口，因而原 RLE、histogram、non-overlap、edge-trim、OOV 和 activity distance 输出继续使用相同的窗口时间栅格；真正的 detected segments 另存在 `segment_embeddings_and_tokens.npz` 和 trial JSONL 的 `primitive_segmentation.segments` 中。

默认 `per_trial` 权重仍保证每个 trial 对 adapter、PCA 和 KMeans 的总贡献相同；变长 segment 在 trial 内按其包含的原窗口数分配 KMeans 权重。这样保留旧 fixed-window 基线的时间占用口径，不把“自适应分段”与“短段/长段改为同权”混成一个实验变量。

当前 NPZ 仍由 2.56 秒窗口、1.28 秒 stride 构成；因此新增变点只能位于相邻窗口之间，边界网格分辨率为 1.28 秒，而且窗口感受野会跨越真实边界。该实验检验的是“embedding 序列上的自适应变长分段”，不是原始采样点级精确动作边界恢复。

## 4. 严格 split 约束

- 7-fold outer subject CV；每个 fold 使用 checkpoint 元数据中的训练 subjects 拟合 normalization、PCA 和 KMeans，使用 outer-test subjects 评估；
- 训练 subjects 与评估 subjects 无交集，fit/eval 的窗口和 trial 也无交集；
- 码本只看训练 subjects 的旧 6 类，评估端不回流任何 held-out subject 特征；
- 每个 seed 使用同 fold、同 seed 的冻结 checkpoint，避免 checkpoint 与码本 seed 错配；
- 4 个 seeds 是同一个 subject fold 的重复估计，**不是 4 个独立统计样本**。显著性结论的独立块数为 7，而不是 28；
- 已知标签冲突 `Subject14/a3t2.mat` 默认采用 `--anomaly-policy report`：保留当前预处理标签并显式报告，不静默删除。需要敏感性复核时可改为 `exclude`。

## 5. 控制实验

为避免把窗口重叠、trial 长度或 token 组成误认为顺序结构，代码同时计算：

- **non-overlap control**：每隔一个窗口取样，使相邻被选窗口不再重叠；
- **edge-trim control**：去掉 trial 首尾各 10% 的窗口，检查边界/起止姿态是否主导结果；
- **primitive histogram**：用 raw token 频率的 Jensen-Shannon distance，只保留候选组成和占用比例，不保留顺序；
- **window-count-only**：只根据 log window count 比较 trial，检查 activity 时长捷径；
- **order shuffle**：每条 trial 的 RLE token multiset 和 run count 不变，仅随机打乱 run 顺序；打乱后强制保持合法 RLE（相邻 token 不相同），每个 run 做 50 次；该采样含非均匀 fallback，因此是启发式 Monte Carlo 控制，不是严格的均匀条件置换检验；
- **OOV/shift diagnostic**：以拟合窗口到最近中心距离的 p95 为阈值，报告 held-out 窗口超过该阈值的比例。

需要注意：主 RLE edit distance 使用 token 顺序但不使用 run length；run length 和 observed span 已保存，尚未进入主距离函数。因此本阶段没有验证“持续时间建模”的价值。

## 6. 运行方式

在项目根目录 `C:\Users\BJ\Desktop\HCGCD\HHR` 下运行。以下命令执行完整的 7 folds × 4 seeds，共 28 runs：

```powershell
Set-Location C:\Users\BJ\Desktop\HCGCD\HHR

D:\APPS\Python\python.exe .\experiments\motion_primitive\run_subject_cv.py `
  --cv-root .\results\trial_pooling_subject_cv\mean_robust_max_q90_offline_online_7fold_4seed_20260830 `
  --output-root .\results\motion_primitive\subject_cv_k32_4seed_reproduce `
  --folds "1,2,3,4,5,6,7" `
  --seeds "0,5,50,500" `
  --primitive-num 32 `
  --pca-dim 64 `
  --label-permutations 1000 `
  --order-shuffles 50 `
  --batch-size 512 `
  --device auto `
  --anomaly-policy report
```

自监督特征变点与原 KMeans32 基线的主对比命令为：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_subject_cv.py `
  --cv-root .\results\trial_pooling_subject_cv\mean_robust_max_q90_offline_online_7fold_4seed_20260830 `
  --output-root .\results\motion_primitive\subject_cv_ssl_feature_changepoint_k32_20260902 `
  --folds "1,2,3,4,5,6,7" `
  --seeds "0,5,50,500" `
  --primitive-num 32 `
  --primitive-segmentation ssl_feature_changepoint `
  --pca-dim 64 `
  --ssl-feature-dim 64 `
  --ssl-epochs 25 `
  --ssl-learning-rate 0.001 `
  --ssl-mask-ratio 0.15 `
  --ssl-noise-std 0.02 `
  --changepoint-context-windows 2 `
  --changepoint-score-quantile 0.90 `
  --changepoint-min-segment-windows 2 `
  --label-permutations 1000 `
  --order-shuffles 50 `
  --batch-size 512 `
  --device auto `
  --anomaly-policy report
```

新分段参数必须使用新的 `output-root`；不要与旧 `fixed_window` runs 混放。若只做烟雾检查，可先把 `--folds` 和 `--seeds` 分别改为 `"1"` 和 `"50"`，并暂时降低 permutation/shuffle 数，但正式对比必须恢复上面的完整配置。

输出目录必须是新的目录，或者不包含同名 run 子目录。若只需复核并重新聚合现有 28-run 结果，可运行：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_subject_cv.py `
  --cv-root .\results\trial_pooling_subject_cv\mean_robust_max_q90_offline_online_7fold_4seed_20260830 `
  --output-root .\results\motion_primitive\subject_cv_seed0_k32_20260831 `
  --folds "1,2,3,4,5,6,7" `
  --seeds "0,5,50,500" `
  --primitive-num 32 `
  --pca-dim 64 `
  --label-permutations 1000 `
  --order-shuffles 50 `
  --batch-size 512 `
  --device auto `
  --anomaly-policy report `
  --skip-existing
```

新增单元测试可用以下命令运行：

```powershell
D:\APPS\Python\python.exe -m unittest discover -s tests -p "test_motion_primitive.py" -v
```

### 6.1 固定轨迹的四组分类消融

`run_trajectory_ablation.py` 直接读取已经完成的
`subject_cv_ssl_feature_changepoint_k32_20260902` 28个runs；不会重新训练编码器、
变点适配器、PCA或KMeans，也不会改变segment边界、token和RLE序列。每个run会保存
`trial_grid_hash`、`rle_token_hash`、`codebook_center_hash`和
`segment_boundary_hash`；组1还必须逐项精确复现原hard-RLE指标，否则实验中止。

四组只改变加权编辑距离的局部替换代价：

1. `g1_hard_rle`：原0/1硬码字RLE距离；
2. `g2_dynamic_soft`：用冻结KMeans32中心的cosine distance（余弦距离）进行软替换；
3. `g3_dynamic_state`：组2加独立state residual（状态残差），包括原量纲加速度均值、重力方向代理、加速度/陀螺仪标准差；
4. `g4_dynamic_state_duration_context`：组3再加入partition duration（分区持续时间）和前后token context（码字上下文）。

状态统计从`D:\workdir\dataset\USC-HAD`原始MAT的`sensor_readings`提取，并严格切
非重叠partition；同时逐trial用NPZ逆变换和重叠去重重建相同visible span（可见区间），
要求最大绝对误差不超过`1e-3`。同一个run不允许混用MAT与NPZ fallback（回退源）。
所有状态尺度和持续时间尺度只在当前fold的fit subjects、old 6 classes上拟合；eval标签
只在距离矩阵全部完成后用于评价。

本轮是`exploratory post-hoc experiment（探索性后验实验）`：state residual特征族是在
查看过早期outer-test的Sit/Stand诊断后确定的。因此工程gate通过也不能作为确认性统计
证据；后续仍需未触碰的subjects、trial留出或外部数据集复验。

G4另外执行三类shortcut/control（捷径与控制）检查：合法RLE顺序打乱、trial内持续时间
与token位置的错配、subject内跨trial总时长打乱；并单独报告`total visible duration only`
（仅总可见时长）1-NN。后两者用于区分“阶段持续时间结构”与“activity总时长捷径”。
old/novel准确率的主口径始终保留全部12类候选，只按query类别分组；仅在old或novel子集
内部找邻居的restricted-candidate（受限候选）结果只作为诊断，不能用于非退化gate。

在PyCharm Terminal中将工作目录设为`C:\Users\BJ\Desktop\HCGCD\HHR`，正式实验使用
下面这一行命令：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_trajectory_ablation.py --input-root D:\workdir\HCGCD\HHR\results\motion_primitive\subject_cv_ssl_feature_changepoint_k32_20260902 --fixed-window-root D:\workdir\HCGCD\HHR\results\motion_primitive\subject_cv_seed0_k32_20260831 --output-root .\results\motion_primitive\trajectory_readout_ablation_k32_20260902_v2 --npz-path D:\workdir\HCGCD\HHR\processed\uschad_w256_s128_train17stats\uschad_windows.npz --folds "1,2,3,4,5,6,7" --seeds "0,5,50,500" --expected-runs 28 --state-weight 0.25 --context-weight 0.15 --duration-weight 0.15 --control-shuffles 50 --sample-rate-hz 100 --old-class-count 6 --seed 20260902
```

首次运行可先做一个fold/seed烟雾验证，且必须换用新的output-root：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_trajectory_ablation.py --input-root D:\workdir\HCGCD\HHR\results\motion_primitive\subject_cv_ssl_feature_changepoint_k32_20260902 --fixed-window-root D:\workdir\HCGCD\HHR\results\motion_primitive\subject_cv_seed0_k32_20260831 --output-root .\tmp\trajectory_readout_ablation_smoke_new --npz-path D:\workdir\HCGCD\HHR\processed\uschad_w256_s128_train17stats\uschad_windows.npz --folds "1" --seeds "50" --expected-runs 1 --state-weight 0.25 --context-weight 0.15 --duration-weight 0.15 --control-shuffles 5 --sample-rate-hz 100 --old-class-count 6 --seed 20260902
```

新增测试命令为一行：

```powershell
D:\APPS\Python\python.exe -m unittest discover -s tests -p "test_trajectory_ablation.py" -v
```

聚合目录会直接生成：

- `ablation_run_metrics.csv`：28×4=112行逐run/组指标；
- `ablation_per_class.csv`：逐类别recall、precision和F1；
- `ablation_paired_summary.json`：以7个fold均值为统计单位的相邻组配对比较；
- `ablation_gate_report.json`：预先固定的工程晋级门槛；
- `preflight_protocol_audit.json`与`canonical_protocol_audit.json`：规范参数、split、trial grid、NPZ/MAT来源和指纹审计；任何一项不满足时所有`passed=false`；
- `fixed_window_noninferiority.json`：同fold/seed配对的G4−fixed-window非劣性分析，非劣界为−0.01；不再使用硬编码基线均值；
- `trajectory_sequences.csv`：逐trial列出自适应segment数、RLE动作元数、动作元ID序列、每元持续时间和样本边界；
- `mean_ablation_activity_distance_heatmaps.png`：四组平均活动距离热力图；
- `mean_ablation_confusion_heatmaps.png`：四组平均并列感知1-NN混淆热力图；
- 每个run的`fixed_trajectories_and_predictions.png`：共享token轨迹和G1–G4预测对照图；跨类最高概率并列显示为灰色`?`，不再任意取最小class id；
- 每个run的`duration_weighted_trajectories.png`：按相对持续时间绘制动作元宽度，并在右侧显示绝对总时长。

只有完整`folds=1..7 × seeds={0,5,50,500}`、K=32、old=6、100 Hz、权重
`0.25/0.15/0.15`、至少50次control shuffle、七组test subjects互斥且覆盖1–14、单一
canonical NPZ指纹、每fold四seed的trial grid一致、MAT/NPZ逐trial一致时，结果才具备
promotion eligibility（工程晋级资格）。单fold烟雾运行即使数值很好也会强制
`promotion_eligible=false`。

组2当前故意使用最低成本、无新增训练参数的码本中心余弦代价，用来单独检验“硬编号
替换是否是瓶颈”。原始局部波形的constrained DTW（受限动态时间规整）属于组2通过
后的下一项独立消融，不能在读取本轮outer-test结果后替换组2定义。

## 7. 输出文件

聚合目录 `results/motion_primitive/subject_cv_seed0_k32_20260831/` 中（目录名是早期 seed-0 运行遗留名称，当前内容已经包含 4 个 seeds）：

- `cv_summary.json`：最终 28-run 数值汇总、每 fold 的 4-seed 平均、7-fold sign-flip tests 和 gate 计数；
- `cv_runs.csv`：每个 fold/seed 的主要指标和状态；
- `activity_decomposition_summary.csv`：逐 activity 汇总多 primitive trial 比例、候选数和 RLE run 数；
- `mean_activity_sequence_distance_matrix.csv`：28 runs 平均的 activity-to-activity RLE 距离矩阵；
- `mean_activity_sequence_distance_heatmap.png`：上述矩阵的热力图；
- `fold_XX_seed_Y_k32/`：单次运行的完整可审计结果。

每个单次运行目录中最重要的文件为：

- `summary.json`：该 run 的 split、动作元统计、关联指标、顺序控制和结论；
- `split_audit.json`、`experiment_config.json`：split 安全检查和可复现实验配置；
- `primitive_codebook.npz`：PCA 参数和 KMeans centers；
- `window_embeddings_and_tokens.npz`：按原索引保存的编码特征、token 和最近中心距离；
- `segment_embeddings_and_tokens.npz`：segment 级特征、边界、token、距离和 fit/eval 角色；
- `segmentation_statistics.json`：分段数量、持续窗口数、train-only 变点阈值和自监督训练摘要；
- `ssl_feature_adapter.pt`：仅 `ssl_feature_changepoint` 生成的 train-only masked-denoising adapter；
- `trial_primitive_sequences.jsonl`：逐 trial 的 raw/RLE tokens、run lengths、observed spans 及控制序列；
- `primitive_statistics.json`：码本利用率、primitive coverage、trial 分解和 OOV 统计；
- `sequence_association_metrics.json`：序列、histogram、window count 及控制版本的关联与置换结果；
- `order_shuffle_control.json`：破坏顺序后的 50 次对照；
- `cross_subject_pairwise_distances.csv`：所有跨 subject trial pairs 的距离；
- `activity_sequence_distance_matrix.csv/.png`：单 run 的 activity 距离矩阵和热力图；
- `activity_trial_token_sequences.png`：按 activity 排列的 held-out trial 轨迹图；变点模式叠加显式 segment 边界；
- `feasibility_decision.json`：启发式 gate 及其解释边界；
- `sequence_metric_revision.json`：tie-aware 1-NN 与合法 RLE shuffle 的派生指标修订记录；旧版派生指标以 `*_legacy_*_v1.json` 保留；
- `run.log`：运行日志。

## 8. 最终 28-run 结果

结果来源：`results/motion_primitive/subject_cv_seed0_k32_20260831/cv_summary.json`。下表中的“±”是 28 runs 的描述性 sample standard deviation；由于同 fold 的 4 seeds 共享 subject split，不能把它用于声称 `n=28` 的独立显著性。

| 指标 | 28-run mean ± sample std | 解释 |
|---|---:|---|
| held-out codebook utilization | 0.9587 ± 0.0583 | 平均使用 95.87% 的 32 个码字，未见整体码本坍缩 |
| held-out max token share | 0.2044 ± 0.0735 | 最大单 token 占比平均 20.44%，没有单 token 主导 |
| effective K | 16.912 ± 3.985 | 使用分布并非均匀，但有效候选数明显大于 1 |
| RLE same-activity distance | 0.7029 ± 0.1225 | 跨 subject 同类 trial 的 normalized edit distance |
| RLE different-activity distance | 0.9496 ± 0.0207 | 跨 subject 异类 trial 的 normalized edit distance |
| RLE separation ratio | 1.3983 ± 0.2845 | 异类距离约为同类距离的 1.40 倍 |
| RLE margin | 0.2467 ± 0.1087 | 28/28 runs 为正；同类序列更接近 |
| RLE cross-subject 1-NN | 0.5284 ± 0.1100 | tie-aware 52.84%；12 类均衡随机机会约 8.33% |
| non-overlap RLE margin | 0.2506 ± 0.1065 | 去除窗口重叠后关联没有消失 |
| non-overlap 1-NN | 0.5051 ± 0.1125 | tie-aware 结果仍明显高于随机机会 |
| edge-trim RLE margin | 0.2507 ± 0.1053 | 去掉首尾各 10% 后关联没有消失 |
| histogram margin | 0.2548 ± 0.0853 | 只保留 token 组成时，margin 不低于完整 RLE 序列 |
| histogram 1-NN | 0.5305 ± 0.1121 | tie-aware 53.05%，与 RLE 序列的 52.84% 基本相同 |
| window-count-only margin | 0.0765 ± 0.0621 | trial 长度本身包含一定类别信息，但解释不了全部关联 |
| window-count-only 1-NN | 0.2368 ± 0.0906 | tie-aware 23.68%，高于随机机会，说明时长是必须保留的混杂因素 |
| observed − valid-order-shuffled margin | 0.0040 ± 0.0048 | 合法 RLE 对照下顺序增量仅占总体 margin 的约 1.6%，实质意义很弱 |
| observed − valid-order-shuffled 1-NN | 0.0080 ± 0.0198 | 顺序平均增加约 0.80 个百分点，且 run 间不稳定 |
| old-class OOV ratio | 0.4606 ± 0.1850 | held-out 旧类窗口有 46.06% 超过 in-sample fit-distance p95；提示严重分布/量化失配，但不能单独归因于 subject shift |
| novel-class OOV ratio | 0.8244 ± 0.1530 | novel 类为 82.44%；这里只能作为 shift 诊断 |

以每个 fold 内 4-seed 平均值为一个独立块，精确单侧 sign-flip test 得到：

- full RLE margin：7/7 folds 为正，`p = 0.0078125`；
- non-overlap margin：7/7 folds 为正，`p = 0.0078125`；
- edge-trim margin：7/7 folds 为正，`p = 0.0078125`；
- observed − order-shuffled margin：7/7 folds 为正，`p = 0.0078125`。

7 个块的精确 sign-flip test 最小单侧 p 值就是 `1/2^7 = 0.0078125`；因此应将结果理解为跨 folds 方向一致，而不是高精度估计很小的 p 值。各 fold 的 held-out subjects 不重叠，但训练 subjects 大量重叠，所以这个 CV sign-flip 仍是诊断性统计，不应被解释成 7 个完全独立实验的严格推断。

顺序增量虽然在 7 个 fold 平均值上同向，但均值只有 0.0040；且合法 RLE shuffle 的 fallback 不是所有合法排列上的均匀采样。因此这里最多说明“可能存在极弱顺序信号”，不能据其 p 值声称顺序结构已得到实质验证。

启发式 gate 计数为：

- sequence association gate：28/28 通过；
- stable motion-primitive candidate gate：仅 1/28 通过；
- per-run order evidence：8/28 通过；没有任何 fold 的 4 个 seeds 全部通过。

其中 stable gate 使用 `old-class OOV <= 0.20` 作为警告边界；这个 0.20 是宽松的诊断启发式阈值，并非经 validation 调参得到的决策阈值。OOV 阈值来自参与 PCA/KMeans 拟合的窗口的 in-sample 距离 p95，会混合聚类拟合乐观偏差、加权口径与分布变化，因此 stable gate 不能当作纯粹的 subject-invariance 检验。

### 8.1 trial 是否真的包含多个候选

对 28 个 `trial_primitive_sequences.jsonl` 的补充描述性审计显示：每个 fold/seed 有 120 个 held-out trials，共形成 3,360 个 trial–seed 表示；其中 2,491 个（74.14%）包含不止一个 primitive id。各 run 的 trial 内 unique primitive 数中位数为 1.5–4，RLE run count 中位数为 1.5–11。

因此可以说 **典型 trial 能被表示为多个动作状态候选**，但不能说每个 trial 都被分成多个候选；各 run 都存在只得到一个 primitive id 的 trial。分 activity 看，Walking Forward 的多 primitive 比例为 98.21%，而 Sitting 为 45.71%、Sleeping 仅 22.50%；静态活动不应被强行要求具有多段动作元。上述 3,360 个表示包含同一批 physical trials 在 4 个 seeds 下的重复编码，不是 3,360 个独立 trials，也不参与显著性检验。

## 9. 结论

当前结果对研究假设给出 **部分支持**：

- 支持：冻结局部编码特征经 train-only 离散化后没有明显整体坍缩；多数 trial 得到多个动作状态候选；同类 activity 的候选序列在 held-out subjects 间显著更接近，而且该方向在 7/7 folds、non-overlap 和 edge-trim 控制下保持一致。
- 不足：histogram 的 margin 和 1-NN 与有序 RLE 序列相当甚至略高；合法 RLE shuffle 后顺序 margin 增量仅 0.0040。因此目前的主要信号更像是 **primitive composition/occupancy**，而不是已被充分验证的复杂顺序或转移结构。
- 关键否定证据：held-out 旧类 OOV 平均 46.06%，stable candidate gate 只有 1/28 通过；这不支持“码字已经是稳定、跨 subject 不变的动作元”这一强结论。

## 10. 当前不能支持的结论

本实验不能据此声称：

- KMeans token 已对应可解释、可重复的语义动作元；
- 每个完整 trial 都会被可靠地分解为多个动作元；
- 动作元顺序、转移或持续时间已经被证明是主要判别信息；
- trajectory 表示优于现有 `mean + q0.90 robust peak` pooling；
- 对 novel classes 已完成发现、聚类或 novelty detection；
- Motion Primitive 方案已经改善 HAR-CGCD 或 Happy-CGCD。

## 11. 主要局限与下一阶段门槛

- 编码器来自已有监督训练 checkpoint，当前关联可能部分继承了窗口编码器的类别判别能力，不等同于发现了独立的自然动作基元；
- KMeans 的簇编号跨 fold/seed 不可直接对应；本阶段没有测量码字对齐、cluster stability 或语义一致性；
- 2.56 秒窗口对“primitive”可能过粗，且主 RLE edit distance 忽略 run duration；
- histogram 与 RLE 结果几乎相当，尚未证明顺序、转移或持续时间提供了超越组成的稳定增益；
- tie-aware window count 1-NN 达 23.68%，activity duration 是不可忽视的捷径变量；
- old-class OOV 很高，显示 held-out 量化失配；但该 in-sample p95 诊断不能单独识别 subject shift，subject-invariant codebook 仍未成立；
- `Subject14/a3t2.mat` 存在标签冲突，默认结果按当前文件名标签保留并报告；
- 仅验证了主要配置 `K=32, PCA=64`，尚未完成 K、窗口尺度、编码器层级和距离函数的系统敏感性分析；
- 只有 7 个独立 subject folds，统计分辨率有限；
- 尚未与 pooled trial feature 在完全相同 split 下做配对比较。

进入更完整 trajectory/CGCD 分支前，至少需要降低跨 subject OOV、验证码字跨 seed/fold 的稳定性，并证明顺序或 duration-aware 表示能稳定优于纯 histogram 和现有 pooling 基线。

## 12. 独立 Motion Primitive Encoder 实验分支（2026-09-03）

该分支只验证“重新训练局部编码器能否改善动作元边界及后续轨迹表示”，不改写
Happy-CGCD 主训练器，也不据单次结果声称 CGCD 已改善。完整链路为：

`raw trial -> overlapping windows -> dual-role encoder -> variable segments -> PCA64 -> KMeans32 -> trajectory/readout`

其中 dual-role encoder（双角色编码器）明确分开两个出口：segmentation head（分段头）只
产生边界；content head（内容头）只产生 segment mean（片段均值）并进入 PCA/KMeans。
两个出口都从旧 ResNet1D 的 identity residual（恒等残差）初始化；初始分段特征严格等于
归一化后的旧 backbone feature（骨干特征），因此 A0/A1 不再以随机分段头作为不公平基线。

训练目标为：

`L = λ_aug L_aug + λ_nc L_nc_seg + λ_cp L_cp_seg + λ_align L_cp_content + λ_pred L_pred + λ_trial L_trial`

- `L_aug`：同一窗口两种语义保持增强的一致性；正式 A1/A3/A4 使用 InfoNCE（噪声、共享三轴缩放、非循环轻微平移、短遮挡；A4 再加 3° SO(3) 旋转）；
- `L_nc_seg`：跨 trial 等权采样的 variance/covariance（方差/协方差）防坍缩；协方差使用非对角元素均值以避免随维度增大；不再白化旧 content 几何；
- `L_cp_seg`：raw kinematics（原始运动学）与 frozen legacy feature（冻结旧特征）共识锚点上的稳定边界压缩、变化/稳定排序及增强视图一致性；不按 trial 强制 top-k；
- `L_cp_content`：低权重把相同边界结构传递给最终进入 KMeans 的 content，防止“分段头切开、内容头两侧仍相同”的断链；
- `L_pred`：EMA teacher（指数移动平均教师）提供 clean-view masked temporal prediction（干净视图掩码时序预测）目标；EMA 不产生边界伪标签；
- `L_trial`：低容量 old6 trial auxiliary classifier（试验级辅助分类器），默认权重 0.1，仅提供活动语义，不替代轨迹 readout；
- gradient clipping（梯度裁剪）按 backbone、三个表示头、trial 辅助头和时序预测器分别执行，避免某一辅助头连带压低边界梯度。

正式消融定义如下；A0--A4 的身份参数冲突会直接拒绝运行，任意自由组合必须使用
`CUSTOM`：

| profile | 变点监督 | 同窗增强一致性 | 旋转 | content 边界对齐 |
|---|---:|---|---:|---:|
| A0 | 无 | 无 | 0° | 0 |
| A1 | 无 | InfoNCE | 0° | 0 |
| A2 | 有 | 无 | 0° | 0.1 |
| A3 | 有 | InfoNCE | 0° | 0.1 |
| A4 | 有 | InfoNCE | 3° | 0.1 |

InfoNCE 只允许每个 trial 抽一个窗口，避免同一 trial 重叠窗口成为最确定的 false
negative（假负样本）。不同 trial 的同类随机窗口仍可能是动作阶段不同的样本，本轮不按
activity label（活动标签）擅自把它们拉近或排除；用 `A1-A0`、`A3-A2` 判断 InfoNCE
净效应。如果两组都变差，再以 `CUSTOM + vicreg` 做 negative-free（无负样本）对照。

数据安全约束为：优化只使用 train subjects 的 old6；选择/日志只使用 offline validation
subjects 的 old6；outer-test sensor windows（外层测试传感器窗口）在编码器训练中的选择数
和前向次数都必须为 0。正式 checkpoint 使用 final epoch（最终轮），禁用 early stopping
（提前停止）以保证消融预算一致。每个 checkpoint 记录 state SHA256、NPZ SHA256、完整
训练参数和五个实现源码文件的 implementation fingerprint（实现指纹）。CV 在启动下游前
要求整个 fold/seed grid 只有一个去除路径、设备和 seed 后的训练身份；fixed-window 与
changepoint 必须使用完全相同的 checkpoint SHA256。

窗口仍为 256 samples、stride 为 128 samples、采样率为 100 Hz，因此窗口覆盖 2.56 秒，
边界网格分辨率为 1.28 秒；本轮损失不能产生亚秒级边界。一个 trial（例如稳定的 Running
Forward）只得到一个 segment 是允许且可能有利于分类的，不设置“至少切成多个元”的目标。

### 12.1 PyCharm PowerShell 一行命令

以下命令都假定 PyCharm Terminal 当前目录为
`C:\Users\BJ\Desktop\HCGCD\HHR`；每个 `--output-dir/--output-root` 必须是新的空目录。

先运行独立自检：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\train_motion_encoder.py --self-test
```

fold 06、seed 500 的正式 A3 训练：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\train_motion_encoder.py --source-checkpoint ".\results\trial_pooling_subject_cv\mean_robust_max_q90_offline_online_7fold_4seed_20260830\fold_06\window_pretrain\seed_500_offline\uschad\Old6_Ratio0.8_SampleUnitWindow_20260830-164143\checkpoints\model_best.pt" --npz-path ".\processed\uschad_w256_s128_train17stats\uschad_windows.npz" --output-dir ".\results\motion_primitive\encoder_training\fold_06_seed_500_A3_formal_v1" --ablation-profile A3 --window-aug-weight 1 --content-boundary-alignment-weight 0.1 --trial-weight 0.1 --segmentation-dim 0 --epochs 30 --trial-batch-size 8 --source-encode-batch-size 512 --learning-rate 0.0001 --minimum-learning-rate 0.000001 --weight-decay 0.0001 --gradient-clip-norm 5 --ema-momentum 0.99 --early-stopping-patience 0 --selection-policy final_epoch --deterministic --device cuda --seed 500
```

同一 checkpoint 的 KMeans32 fixed-window（固定窗）对照：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_experiment.py --checkpoint ".\results\motion_primitive\encoder_training\fold_06_seed_500_A3_formal_v1\motion_encoder_final.pt" --npz-path ".\processed\uschad_w256_s128_train17stats\uschad_windows.npz" --output-dir ".\results\motion_primitive\single_fold06_motion_encoder_fixed_window_A3" --primitive-segmentation fixed_window --primitive-num 32 --pca-dim 64 --embedding-normalization l2 --codebook-weighting per_trial --kmeans-n-init 20 --kmeans-max-iter 300 --old-class-count 6 --anomaly-policy report --sample-rate-hz 100 --label-permutations 1000 --order-shuffles 50 --batch-size 512 --device cuda --seed 500
```

同一 checkpoint 的 motion-encoder changepoint（动作编码器变点）实验：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_experiment.py --checkpoint ".\results\motion_primitive\encoder_training\fold_06_seed_500_A3_formal_v1\motion_encoder_final.pt" --npz-path ".\processed\uschad_w256_s128_train17stats\uschad_windows.npz" --output-dir ".\results\motion_primitive\single_fold06_motion_encoder_changepoint_A3" --primitive-segmentation motion_encoder_changepoint --primitive-num 32 --pca-dim 64 --embedding-normalization l2 --codebook-weighting per_trial --kmeans-n-init 20 --kmeans-max-iter 300 --old-class-count 6 --anomaly-policy report --sample-rate-hz 100 --label-permutations 1000 --order-shuffles 50 --batch-size 512 --changepoint-context-windows 2 --changepoint-score-quantile 0.90 --changepoint-min-segment-windows 2 --device cuda --seed 500
```

当且仅当同一 profile 的 7 folds × 4 seeds 共 28 个正式编码器都已训练完，可分别运行
两个 CV wrapper（CV 封装器）；它们会在执行前拒绝混合参数、源码版本或 smoke checkpoint：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_subject_cv.py --cv-root ".\results\trial_pooling_subject_cv\mean_robust_max_q90_offline_online_7fold_4seed_20260830" --motion-encoder-root ".\results\motion_primitive\encoder_training" --expected-encoder-profile A3 --output-root ".\results\motion_primitive\subject_cv_motion_encoder_fixed_window_A3" --folds "1,2,3,4,5,6,7" --seeds "0,5,50,500" --primitive-num 32 --primitive-segmentation fixed_window --pca-dim 64 --label-permutations 1000 --order-shuffles 50 --batch-size 512 --device cuda --anomaly-policy report
```

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_subject_cv.py --cv-root ".\results\trial_pooling_subject_cv\mean_robust_max_q90_offline_online_7fold_4seed_20260830" --motion-encoder-root ".\results\motion_primitive\encoder_training" --expected-encoder-profile A3 --output-root ".\results\motion_primitive\subject_cv_motion_encoder_changepoint_A3" --folds "1,2,3,4,5,6,7" --seeds "0,5,50,500" --primitive-num 32 --primitive-segmentation motion_encoder_changepoint --pca-dim 64 --label-permutations 1000 --order-shuffles 50 --batch-size 512 --changepoint-context-windows 2 --changepoint-score-quantile 0.90 --changepoint-min-segment-windows 2 --device cuda --anomaly-policy report
```

最后运行完整 trajectory readout（轨迹读出）比较；该入口要求 28-run 规范网格，不能用
单 fold smoke 冒充正式聚合：

```powershell
D:\APPS\Python\python.exe .\experiments\motion_primitive\run_trajectory_ablation.py --input-root ".\results\motion_primitive\subject_cv_motion_encoder_changepoint_A3" --fixed-window-root ".\results\motion_primitive\subject_cv_motion_encoder_fixed_window_A3" --output-root ".\results\motion_primitive\trajectory_readout_motion_encoder_A3_v1" --input-protocol motion_encoder_v1 --expected-encoder-profile A3 --npz-path ".\processed\uschad_w256_s128_train17stats\uschad_windows.npz" --folds "1,2,3,4,5,6,7" --seeds "0,5,50,500" --expected-runs 28 --state-weight 0.25 --context-weight 0.15 --duration-weight 0.15 --control-shuffles 50 --sample-rate-hz 100 --old-class-count 6 --seed 20260902
```

每个下游 run 会直接生成 `activity_sequence_distance_heatmap.png` 和
`activity_trial_token_sequences.png`；`segmentation_statistics.json` 额外记录最终 segment 数、
每 trial 的 segment 范围，以及检测边界后相邻 segment 是否仍被 KMeans 分成不同 token 的
比例。只有 `motion encoder -> changepoint -> PCA/KMeans32 -> G3/G4` 相对同 checkpoint 的
fixed-window 配对分类结果更好，才支持“新编码器改善 CGCD 分类”的下一步假设。

## 13. Session-2 无标签二级码本确认实验（2026-09-03）

该实验仍是 trajectory clustering proxy（轨迹聚类代理），不是正式 Happy-CGCD。它严格
复现 fold06/seed500 的在线 Session 2：Session 1 增量训练 22 条、Session 2 增量训练 26
条、累计无标签训练 48 条、Session 2 测试 52 条；未来 Session 类别 10/11 不得进入任何
特征或拟合集合。粗码本候选只按累计训练 trial 中的非重叠持续时间覆盖率自动选择，不使用
activity label（动作类别标签）。

四个 arm（实验臂）为：

- `U0_coarse`：不拆分粗码字；
- `U1_residual`：只按编码器量化残差拆为两个 child；
- `U2_gravity`：只按原始加速度的重力方向拆为两个 child；
- `U3_joint`：残差距离与重力角距离等权融合。

每个 arm 都使用 `duration-normalized histogram -> train-only KMeans K=10`，再冻结测试预测。
All/Old/New/F1 共用该 arm 在完整测试集上的一次 global Hungarian alignment（全局匈牙利
对齐）。因此本实验验证的是码字组成/持续时间分类，不验证顺序或转移关系。

最终可追溯结果只使用 `*_v5_confirmatory`；v1--v4 均为开发或过渡结果。冻结实现指纹为：

- runner SHA256：`2ae54a78b7121aa32e84acff1523b51cb3f5bad20a10af4050d94a25c05d98f9`；
- helper SHA256：`f43717d572f7fe0bd040182b502aecda766a402603a98f08567ec092e50dd004`。

### 13.1 主要分类结果

`H` 为 Old/New accuracy 的调和平均，只是由已报告指标派生，未参与模型选择。

| Encoder/分段 | arm | All | Old | New | H | macro-F1 | Sit/Stand BAcc |
|---|---|---:|---:|---:|---:|---:|---:|
| A0/fixed | U0 | 61.54% | 58.33% | 68.75% | 63.11% | 58.11% | — |
| A0/fixed | U2 | 67.31% | 61.11% | 81.25% | 69.76% | 66.26% | 90.00% |
| A3/fixed | U0 | 63.46% | 61.11% | 68.75% | 64.71% | 56.42% | — |
| A3/fixed | U2 | 67.31% | 61.11% | 81.25% | 69.76% | 65.39% | 90.00% |
| A0/changepoint | U0 | 63.46% | 63.89% | 62.50% | 63.19% | 55.92% | — |
| A0/changepoint | U2 | 55.77% | 47.22% | 75.00% | 57.95% | 54.90% | 90.00% |
| A3/changepoint | U0 | 69.23% | 69.44% | 68.75% | 69.10% | 64.84% | — |
| A3/changepoint | U2 | 69.23% | 63.89% | 81.25% | 71.53% | 68.01% | 90.00% |

A3 相对 A0 的 subject×activity 分层 paired trial bootstrap（受试者×动作分层配对试验级
自助法）结果为：fixed/U0 的 All 差值 `+1.92 pp`，95% percentile interval（百分位区间）
`[-7.69,+11.54] pp`；changepoint/U0 为 `+5.77 pp`，区间 `[-3.85,+15.38] pp`。
二者都跨过 0，不能据此确认 A3 编码器提升分类。changepoint/U2 的 A3−A0 虽为
`+13.46 pp [5.77,21.15]`，但 A0/changepoint/U2 相对 A0/fixed/U2 本身下降 11.54 pp，
而 A3/changepoint/U2 相对 A3/fixed/U2 只增加 1.92 pp；因此该差异主要反映 A0 变点方案
退化，不能当作 A3 的净收益。

### 13.2 重力负对照与解释边界

正式结果对每组执行 500 次 within-subject × split-role gravity shuffle（受试者内×数据角色
重力置换）：保持每名受试者在累计训练/测试池各自的重力边际分布，破坏重力与原 trial
轨迹的对应；每次重新拟合 U2/U3 二级码本与训练集 KMeans，全部 null prediction（零假设
预测）冻结后才接入标签。原始 null cluster/child assignment 已无标签落盘为
`gravity_shuffle_null_assignments.npz`。

U2 的总体 All accuracy 相对置换零分布在四组均未达到 0.05：A0/fixed `p=.0639`、
A3/fixed `p=.1457`、A0/changepoint `p=.5529`、A3/changepoint `p=.0938`。New accuracy
虽比 null mean 高约 12 pp，四组上侧 p 值仍为 `.0639--.0938`。所以重力旁路的全局
分类收益尚未通过确认门槛。

Sit/Stand 的 8 条测试 trial 在 U2 下为 7/8、balanced accuracy（平衡准确率）90%；但该
类别对是看过同一 fold 既往结果后选出的 exploratory/post-hoc（探索性/事后）终点，未做
多重比较校正。总体 Sit/Stand 置换 p 值为 A0/fixed `.0080`、A3/fixed `.0080`、
A0/changepoint `.0180`、A3/changepoint `.0778`；Subject 4 只有 3 条并未单独通过，较强
证据主要来自 Subject 5 的 5 条。因此可以说“重力姿态是有物理解释的可用线索”，不能说
“已解决跨受试者 Sit/Stand”或“该收益来自 A3 编码器”。

U3 等权融合不成立：它没有稳定改善 All/New，并在 A3 两种分段下使 All 相对 shuffle null
更差。A3 候选训练集中实际同时包含 dynamic（动态）Walking Right、static Sitting 和
static Standing 三种潜在子模态；强制 K=2 且残差/重力等权无法表达三级结构。下一最小实验
应改为 hierarchical gated split（层次门控拆分）：先用 residual 区分 dynamic/static，只对
置信度足够的 static child 使用 gravity 区分 Sit/Stand；不应继续调一个全局等权系数。

本结果也支持“单 segment 完全可接受”：A3/changepoint 的 120 条 held-out trial 中 85%
只有一个 segment，Session-2 的 8 条 Sit/Stand 原轨迹也都是单 token；U2 仅通过可更新
child codebook 提供判别。因此当前证据支持“稳定状态码字＋二级码本”，不支持“必须多段化”
或“顺序/转移已经带来收益”。

### 13.3 PyCharm Linux/WSL 一行复现命令

以下命令假定项目位于 `/mnt/c/Users/BJ/Desktop/HCGCD/HHR`，并使用 Linux 环境中的
`python`。输出目录必须不存在；下面使用 `*_v5_reproduction`，不会覆盖正式 v5 结果。

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python -m unittest tests.test_online_secondary_codebook tests.test_online_secondary_confirmations tests.test_sit_stand_probe tests.test_motion_encoder_training_protocol -v
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_secondary_codebook.py --run-dir ./results/motion_primitive/single_fold06_A0_formal_v1_fixed_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_secondary_codebook_A0_fixed_fold06_seed500_v5_reproduction --seed 500 --gravity-shuffles 500 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_secondary_codebook.py --run-dir ./results/motion_primitive/single_fold06_A3_formal_v1_fixed_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_secondary_codebook_A3_fixed_fold06_seed500_v5_reproduction --seed 500 --gravity-shuffles 500 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_secondary_codebook.py --run-dir ./results/motion_primitive/single_fold06_A0_formal_v1_changepoint_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_secondary_codebook_A0_changepoint_fold06_seed500_v5_reproduction --seed 500 --gravity-shuffles 500 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_secondary_codebook.py --run-dir ./results/motion_primitive/single_fold06_A3_formal_v1_changepoint_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_secondary_codebook_A3_changepoint_fold06_seed500_v5_reproduction --seed 500 --gravity-shuffles 500 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/analyze_online_secondary_confirmations.py --a0-fixed-dir ./results/motion_primitive/online_secondary_codebook_A0_fixed_fold06_seed500_v5_reproduction --a3-fixed-dir ./results/motion_primitive/online_secondary_codebook_A3_fixed_fold06_seed500_v5_reproduction --a0-changepoint-dir ./results/motion_primitive/online_secondary_codebook_A0_changepoint_fold06_seed500_v5_reproduction --a3-changepoint-dir ./results/motion_primitive/online_secondary_codebook_A3_changepoint_fold06_seed500_v5_reproduction --output-dir ./results/motion_primitive/online_secondary_confirmation_v5_reproduction --bootstrap-replicates 10000 --seed 20260903
```

## 14. A3 residual-to-gravity hierarchical gate（残差到重力的层次门控）探索实验

本实验只回答一个最小问题：A3 编码器学到的 residual（残差）能否先安全地区分
static/dynamic proxy（静态/动态代理），再只对静态分支用 gravity（重力方向）细分，并最终
改善 Session-2 的 trajectory-clustering proxy（轨迹聚类代理）分类。它没有修改正式
Happy-CGCD，也没有验证顺序/转移特征。

### 14.1 冻结协议与独立检查

- 上游强制为 K32/PCA64、L2 normalization（L2 归一化）、cosine assignment（余弦分配）、
  per-trial weighting（逐 trial 加权）、KMeans `n_init=20/max_iter=300`、old classes=6；
- 四组强制共用 seed=500、同一个原始 NPZ、同一个 legacy source checkpoint（旧编码器源
  检查点）、同一套编码器实现和同一套变点参数 `context=2/q=.90/min_segment=2`；
- A0/A3 各自的 fixed/changepoint 必须使用同一 motion-encoder checkpoint（动作编码器
  检查点），而 A0 与 A3 检查点必须不同；
- gate（门控）阈值为 `radius_q=.95/token_fraction=.50/energy_ratio=2.0/energy_gap=.25`。
  这些阈值是在 fold06 开发结果后确定的 exploratory heuristic（探索性启发式），不是
  confirmatory threshold（确认性阈值）；
- 若能量条件不成立，G3 fail-closed（失败即关闭），直接复用 G0 的 raw/aligned prediction
  （原始/对齐预测），同时再次逐 trial 校验二者完全一致；
- 相关联合回归测试为 62/62 通过。

### 14.2 fold06/seed500 结果

| Encoder/分段 | arm | All | Old | New | macro-F1 | Sit/Stand BAcc | gate |
|---|---|---:|---:|---:|---:|---:|---|
| A0/fixed | G0 | 61.54% | 58.33% | 68.75% | 58.11% | — | — |
| A0/fixed | G3 | 61.54% | 58.33% | 68.75% | 58.11% | 0.00% | disabled |
| A3/fixed | G0 | 63.46% | 61.11% | 68.75% | 56.42% | — | — |
| A3/fixed | G3 | **73.08%** | **66.67%** | **87.50%** | **73.80%** | **100.00%** | enabled |
| A0/changepoint | G0 | 63.46% | 63.89% | 62.50% | 55.92% | — | — |
| A0/changepoint | G3 | 63.46% | 63.89% | 62.50% | 55.92% | 0.00% | disabled |
| A3/changepoint | G0 | 69.23% | 69.44% | 68.75% | 64.84% | — | — |
| A3/changepoint | G3 | 71.15% | 63.89% | 87.50% | 71.25% | **100.00%** | enabled |

A0 的两个分段版本均只有约 `1.285` 的两簇能量比和 `0.070` 的能量差，因此 G3 正确关闭
并逐项复现 G0。A3/fixed 与 A3/changepoint 的两簇能量比分别为 `25.11/25.17`，能量差
为 `1.837/1.835`，门控均开启；13 条训练候选中有 11 条进入置信静态分支。两种 A3
分段都把测试集的 3 条 Sitting 与 5 条 Standing 全部路由并达到 8/8，且 Subjects 4/5
各自也都是 100%。动态候选 Walking Right/Walking Forward 全部保留粗 token。

但 A3 的 13 条训练候选实际是 7 Sitting、5 Standing 和仅 1 条 Walking Right；residual
K-medoids（残差 K 中心点聚类）的两簇支持为 `12:1`。这个单一动态样本对约 25 倍的能量
比影响很大，而当前门控只约束静态簇至少两条，没有约束动态簇最小支持。逐条留一敏感性
检查中，移除该 Walking Right 后能量比降至约 `2.91`，高置信静态训练样本由 11 条降为
2 条。另一方面，A0 的 12 条候选全部是 Sitting/Standing；当前 fail-closed 无法区分
“候选确实全静态，可直接重力细分”和“残差结构不可判定”，因此 A3−A0/G3 也混入了门控
策略差异。该结果是强机制线索，但尚不是稳定的码本发现证据。

paired trial bootstrap（配对试验级自助法）的主要结果为：A3/fixed G3 相对自身 G0 的
All 差值 `+9.62 pp [5.77,13.46]`，A3/changepoint G3 相对自身 G0 只有
`+1.92 pp [-3.85,7.69]`；后者同时出现 Old `-5.56 pp` 与 New `+18.75 pp` 的权衡。
A3/fixed G3 相对 A0/fixed G3 为 `+11.54 pp [1.92,19.23]`；A3/changepoint G3 相对
A0/changepoint G3 为 `+7.69 pp [0.00,15.38]`。这些区间只描述 Subjects 4/5 内 52 条
测试 trial 的条件稳定性，不能替代受试者级泛化证据。

当前证据支持“A3 residual 对静态/动态门控有用，并能与重力子码本组成有效分类表示”；
不支持“编码器单独已经证明提升 CGCD”。理由是：A3 的 G0 encoder-only（仅编码器）差值
仍较小且区间跨 0；G3 还使用原始加速度/陀螺仪来命名静态簇并使用重力完成 Sit/Stand
细分；所有 arm 都会重新拟合 global KMeans readout（全局 K 均值读出），所以未改变
轨迹的动态动作仍会因全局质心竞争改变最终预测。A3/changepoint 中 Walking Right 从
G0 的 3/6 降到 G3 的 0/6，正是这一问题，而不是动态 trial 被门控误拆。

因此，不应立即扩展到 7 folds × 4 seeds。下一最小实验应同时做两项保护：冻结 G0 粗
读出，只允许门控命中的静态 trial 进入局部 child readout（子读出），验证能否保留非静态
预测并获得 Sit/Stand 的净增益；同时加入 residual child minimum support（残差子簇最小
支持数）、all-static bypass（全静态旁路）和 leave-one-out stability（逐条留一稳定性）
消融。二者都通过后，再进入多折多种子验证。

### 14.3 PyCharm Linux/WSL 一行复现命令

每条命令均为单行；输出目录必须不存在。下面使用 `*_linux_reproduction_v1`，不会覆盖
本次 `*_exploratory_v1` 结果。

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python -m unittest tests.test_hierarchical_gate tests.test_online_secondary_codebook tests.test_online_secondary_confirmations tests.test_sit_stand_probe tests.test_motion_encoder_training_protocol -v
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_hierarchical_gate.py --run-dir ./results/motion_primitive/single_fold06_A0_formal_v1_fixed_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_hierarchical_gate_A0_fixed_fold06_seed500_linux_reproduction_v1 --seed 500 --static-radius-quantile 0.95 --minimum-token-fraction 0.50 --minimum-motion-energy-ratio 2.0 --minimum-motion-energy-gap 0.25 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_hierarchical_gate.py --run-dir ./results/motion_primitive/single_fold06_A3_formal_v1_fixed_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_hierarchical_gate_A3_fixed_fold06_seed500_linux_reproduction_v1 --seed 500 --static-radius-quantile 0.95 --minimum-token-fraction 0.50 --minimum-motion-energy-ratio 2.0 --minimum-motion-energy-gap 0.25 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_hierarchical_gate.py --run-dir ./results/motion_primitive/single_fold06_A0_formal_v1_changepoint_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_hierarchical_gate_A0_changepoint_fold06_seed500_linux_reproduction_v1 --seed 500 --static-radius-quantile 0.95 --minimum-token-fraction 0.50 --minimum-motion-energy-ratio 2.0 --minimum-motion-energy-gap 0.25 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_hierarchical_gate.py --run-dir ./results/motion_primitive/single_fold06_A3_formal_v1_changepoint_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_hierarchical_gate_A3_changepoint_fold06_seed500_linux_reproduction_v1 --seed 500 --static-radius-quantile 0.95 --minimum-token-fraction 0.50 --minimum-motion-energy-ratio 2.0 --minimum-motion-energy-gap 0.25 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/analyze_online_hierarchical_gate.py --a0-fixed-dir ./results/motion_primitive/online_hierarchical_gate_A0_fixed_fold06_seed500_linux_reproduction_v1 --a3-fixed-dir ./results/motion_primitive/online_hierarchical_gate_A3_fixed_fold06_seed500_linux_reproduction_v1 --a0-changepoint-dir ./results/motion_primitive/online_hierarchical_gate_A0_changepoint_fold06_seed500_linux_reproduction_v1 --a3-changepoint-dir ./results/motion_primitive/online_hierarchical_gate_A3_changepoint_fold06_seed500_linux_reproduction_v1 --output-dir ./results/motion_primitive/online_hierarchical_gate_paired_fold06_seed500_linux_reproduction_v1 --bootstrap-replicates 10000 --seed 20260903
```

## 15. Frozen readout＋可变 K32/K34 码本最小验证

本实验固定使用 fixed window（固定窗口），旧类训练得到的 32 个父中心和编号永久冻结。
Online（在线）阶段只在某个父 token 内部同时通过 trial/subject support（试次/受试者支持）、
silhouette（轮廓系数）、Leave-One-Trial-Out（逐试次留一）、Leave-One-Subject-Out
（留一受试者）和 subject-confound（受试者混杂）检查时，才追加两个 child token；
否则保持 K=32。单个新类只出现一个 token 不是扩容条件。

三条实验臂共用一次 K=10 trial-level coarse readout（试次级粗读出）：

- `F0_frozen_K32`：冻结 K32 基线；
- `F1_registered_K32_or_K34`：历史 residual-to-gravity（残差到重力）机制参考；
- `F2_adaptive_K32_or_K34`：无标签、自适应、append-only（只追加）主实验。

`F1` 仅用于确认已知局部机制；主比较是 `F2-F0`。分析时还需比较
`(A3F2-A3F0)-(A0F2-A0F0)`，以判断 A3 编码器是否为自适应扩容提供额外价值。
输出目录必须不存在。

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python -m unittest tests.test_adaptive_codebook tests.test_hierarchical_gate_v2 tests.test_online_hierarchical_gate_v2 tests.test_frozen_hierarchical_readout tests.test_hierarchical_gate tests.test_online_secondary_codebook -v
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_hierarchical_gate_v2.py --run-dir ./results/motion_primitive/single_fold06_A0_formal_v1_fixed_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_adaptive_codebook_A0_fixed_fold06_seed500_v2 --seed 500 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/run_online_hierarchical_gate_v2.py --run-dir ./results/motion_primitive/single_fold06_A3_formal_v1_fixed_k32 --npz-path ./processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-dir ./results/motion_primitive/online_adaptive_codebook_A3_fixed_fold06_seed500_v2 --seed 500 --bootstrap-resamples 10000
```

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && python experiments/motion_primitive/analyze_online_hierarchical_gate_v2.py --run-dir ./results/motion_primitive/online_adaptive_codebook_A0_fixed_fold06_seed500_v2 --run-dir ./results/motion_primitive/online_adaptive_codebook_A3_fixed_fold06_seed500_v2 --output-dir ./results/motion_primitive/online_adaptive_codebook_paired_fold06_seed500_v2 --old-class-count 6 --bootstrap-replicates 10000 --seed 20260904 --expected-folds "6" --expected-seeds "500"
```

## 16. Peak/Valley Segmentation＋Hierarchical Motion Primitives（峰谷分段＋层次动作元）独立实验

本分支是 isolated CGCD clustering proxy（独立 CGCD 聚类代理），不替换 Happy-CGCD 主训练器。
它只验证两件事：采样点级变长分段是否比历史固定窗口更有利于 CGCD 分类；由相邻子元及其
峰/谷事件组成的父元是否提供超出“切得更多”或“偶然相邻”的增量信息。分类是唯一主要判据，
所以某类 trial 最终只有一个稳定动作元并不构成失败。

### 16.1 已固定的协议事实

#### A2 编码器及损失

A2 是本实验的主编码器，A0/A3 只作为编码器消融。A2 明确关闭 InfoNCE（信息噪声对比估计）
和同窗口增强一致性，`window_augmentation=0.0`；保留以下有效损失权重：

| 损失项 | 权重 | 作用 |
|---|---:|---|
| changepoint loss（变点损失） | 1.0 | 使表征响应冻结的原始信号变点锚点 |
| content-boundary alignment（内容—边界对齐） | 0.1 | 约束内容表征与边界证据相容 |
| temporal prediction（时序预测） | 0.5 | 保留局部时间动力学 |
| trial auxiliary classification（试次辅助分类） | 0.1 | 保留旧类动作判别信息 |
| non-collapse regularization（防坍缩正则） | 0.05 | 防止表征退化为常量 |
| cross-subject loss（跨受试者损失） | 0.0 | 本轮不直接施加受试者对齐 |

因此，A2 含自监督目标，但由于 `trial auxiliary classification` 使用 offline（离线）旧类监督，
不能称为纯 self-supervised learning（自监督学习）编码器。其 backbone batch normalization
（骨干网络批归一化）保持冻结，模型选择固定为最终 epoch（训练轮次），避免在同一实验网格中
按测试结果挑选检查点。

#### 实验臂与对照

所有变长片段默认先重采样到编码器输入长度，再取冻结 content embedding（内容嵌入）；子元
码本固定为 KMeans-32（32 中心 K 均值码本）。父元是非破坏性 overlay（叠加标注），不会
删除、替换或重编号原始子元序列。

| 实验臂 | 定义 | 回答的问题 |
|---|---|---|
| E0 | 历史 `256 samples / stride 128 / KMeans-32` tokenization（离散化）＋当前统一轨迹读出对照 | 新方法是否优于已有 K32，而非优于新造的固定短片段 |
| E1 | 在规范运动强度包络的峰/谷处细切，不做逐轴支持投票，也不做编码特征确认 | 仅靠尽可能细的峰谷边界是否有用 |
| E2 | E1 候选边界再经过 multi-axis voting（多轴投票）和冻结编码器特征变化确认 | 去除单轴毛刺和弱语义边界后是否改善分类 |
| E3 | E2 子元之上，达到出现次数门槛就注册相邻二元父元 | 朴素“高频组合即父元”是否有效 |
| E4 | E2 子元之上，同时通过跨 trial/subject（试次/受试者）支持、NPMI（归一化逐点互信息）、MDL（最小描述长度）和 LOSO（留一受试者）稳定性门控后才注册父元 | 受约束、可跨受试者复现的父元是否提供增量价值 |
| C1 | 与 E2 逐 trial 匹配片段数、最短片段约束和峰/谷事件数量，但随机放置边界 | E2 的收益是否只是片段数量造成的 |
| C2 | 与 E4 匹配父元数量、事件类型及出现/试次/受试者支持度的非参考负模式 | E4 的收益是否只是增加父元特征维数造成的 |

C1 不匹配真实片段长度分布；C2 不是均匀随机父元。二者分别是 matched random boundary
control（匹配随机边界对照）和 support-matched negative motif control（支持度匹配负模式
对照），其解释范围不能扩大为“排除了全部随机因素”。

#### `state` 与 `no_state`

每个实验臂同时产生两个 trajectory descriptor（轨迹描述符）。二者都保留子元数量与持续
时间、位置四分位数、任意/峰/谷转移、量化距离、边界比例，以及父元数量、跨度和位置。
`state` 是主要读出，额外加入每段原始物理统计的 trial 全局加权均值/标准差，以及按子元
编号聚合的均值、对数能量和 presence（是否出现）；这条旁路包含重力方向与姿态状态信息。
`no_state` 删除该物理状态旁路，用来检验 Sitting/Standing（坐下/站立）的可分性是否依赖
重力/姿态，而不是动作元轨迹本身。

E0 的离散编号与按编号聚合的非时长物理状态都使用同一个原始 256-sample 重叠窗口；所有
持续时间特征及权重则使用 window-centre Voronoi ownership cell（窗口中心沃罗诺伊归属区间），以免重叠窗口
重复计算时间。因此 E0 严格复现的是历史 PCA64＋KMeans-32 编号路径，不是历史实验的完整
trajectory readout（轨迹读出）或最终 Hungarian（匈牙利）指标。

#### 数据拟合和标签访问顺序

1. 对每个 outer subject fold（外层受试者折），分段阈值、PCA（主成分分析）、KMeans-32
   子元码本和父元目录只使用训练受试者的旧六类；held-out（留出）受试者不得参与这些拟合。
2. label-aware benchmark split builder（标签感知的基准划分构造器）在 raw prediction freeze
   （原始预测冻结）之前运行。它会在封闭的基准协议内部读取 activity label（动作标签），但
   这些标签只用于重建并校验固定的 Session-2 `48/52` 划分；构造器向预测链路只返回 trial ID
   和 subject ID（受试者编号），不返回动作标签。
3. 预测器侧将留出受试者的动作标签统一替换为 `-1`。Session-2 的 48 条 cumulative
   online-train（累计在线训练）trial 只用于无标签拟合 K=10 试次级聚类读出；与其分离的
   52 条 test trial 只用于预测，预测器在冻结前不接收它们的动作标签。
4. 先把 52 条测试 trial 的 raw cluster predictions（原始簇预测）写入独立文件并记录 SHA256
   哈希；只有完成这个 freeze point（冻结点）后，scoring truth（评分真值）才从原始数据源
   接入评分阶段。
5. 测试标签只在最后用于一次 global Hungarian matching（全局匈牙利匹配），将任意簇编号
   对齐到动作编号并计算指标；此前的分段、码本、父元、描述符和预测都不得访问这些标签。

这个顺序防止标签进入表示学习，但最后一步仍使用同一测试集标签做簇编号对齐。因此所得数值
是标准无监督聚类评估，不等价于部署时无需标签即可输出语义类别的正式在线分类器。

### 16.2 主要终点与统计单位

主要终点是 H-score（旧类/新类准确率调和平均）；同时报告 All/Old/New accuracy（全体/旧类/
新类准确率）和 macro-F1（宏平均 F1）。本轮正式网格运行前预先指定的唯一主比较是
`A2/E2(state) - A0/E0(state)`：它同时检验建议路线相对历史直接编码＋固定窗口 K32 路线的
端到端净变化。原有 8 项次要配对比较组成一个固定 Holm correction（Holm 多重比较校正）
家族，包括 A2 内 E2−E0、E4−E2、E2−C1、E4−C2、`A2/E2−A0/E2`、
`A2/E2−A3/E2`，以及两个 `state-no_state` 比较。这里把 A2/A3 对比固定写成 A2−A3；
正值表示在其余编码器约束及 E2 分段相同的条件下，去掉 InfoNCE 后 H-score 提高。

另报告不进入上述 8 项 Holm 家族的 exploratory fixed-window replication（探索性固定窗口
复核）`A2/E0−A3/E0`。它检查“去掉 InfoNCE”的方向在历史固定窗口 E0 下是否仍一致；其
exact sign-flip p value（精确符号翻转 p 值）和 bootstrap interval（自助法区间）均只作描述，
不能把它解释成额外的确认性检验。

同时报告两个 exploratory factorial interaction（探索性析因交互）：
`(A2/E2-A2/E0)-(A0/E2-A0/E0)` 回答“E2 分段在 A2 上的增益是否超过其在 A0 上的
增益”，用于检查 changepoint supervision＋content-boundary alignment（变点监督＋内容—
边界对齐）与 E2 分段是否存在交互；`(A2/E2-A3/E2)-(A2/E0-A3/E0)` 则回答去掉 InfoNCE
的效应是否依赖 E2 分段。两个交互都不属于上述主检验，也不纳入固定 8 项 Holm 家族，
只能作为后续实验设计依据。

四个随机种子不是四个独立受试者。必须先在每个 held-out-subject fold（留出受试者折）内
平均种子，再以 7 个互不重复的留出受试者对折作为统计推断单位；配对 H-score 差值使用 two-sided exact
sign-flip test（双侧精确符号翻转检验）。在 `n=7` 时，双侧精确 p 值的理论下限为
`2/2^7=0.015625`；对 8 项家族做 Holm 校正时，最小可能校正 p 值为
`8×0.015625=0.125`，所以该 8 项家族在当前样本分辨率下不可能达到 `α=0.05`。代码仍输出
校正值，目的在于如实显示统计分辨率，而不是制造“可显著”的错觉。

bootstrap interval（自助法区间）通过重采样 7 个折均值生成，但由于折间训练受试者大量重叠
且只有 7 个单位，只作为描述性不确定度范围，不能当作确认性置信区间或绕过精确检验分辨率。
trial 级结果和 28 个 fold×seed 运行也只用于描述与稳定性审计，不能把样本量写成 28 或 52
来夸大显著性。

#### 跨受试者轨迹诊断（仅描述性）

每个 profile×fold×seed×arm×`state/no_state` 都在 raw prediction freeze（原始预测冻结）后
计算五项 cross-subject trajectory diagnostic（跨受试者轨迹诊断）：

| 指标 | 定义与方向 |
|---|---|
| distance margin（距离间隔） | 不同动作跨受试者平均距离减去同动作跨受试者平均距离；正值越大越好 |
| `P(same<different)` | 随机同动作距离小于随机异动作距离的概率，距离并列按 0.5 计；高于 0.5 越多越好 |
| rank separation effect（秩分离效应） | `2×P(same<different)−1`；正值越大越好 |
| tie-aware cross-subject 1NN accuracy（并列感知跨受试者最近邻准确率） | 每条 trial 只在其他受试者中找最近邻；多个等距最近邻按其正确率均值计分，越高越好 |
| cluster-subject NMI（簇—受试者归一化互信息） | 原始试次簇与受试者身份的 NMI；越低表示簇携带的受试者身份越少 |

这些指标会读取评分真值，因此只能诊断“同动作跨受试者是否更近”和“簇是否编码受试者身份”；
它们不会回流到分段、码本、描述符或预测器，也不是主要分类检验。任一 arm/读出缺少诊断、
诊断标记为 unavailable（不可用）、数值非有限或概率/秩效应不一致时，CV 汇总会 fail closed。

汇总顺序固定为：先在同一个 held-out-subject fold 内平均不同 seed，再用 7 个 fold 均值计算
描述性 mean/std（均值/标准差）及 fold bootstrap interval（折级自助法区间）。四个 seed 不是
独立样本，不能把 `7×4=28` 写成推断样本量；这些 bootstrap 区间也不能替代主要 H-score
检验或用来宣称确认性显著。

### 16.3 待实验验证的研究假设

- 若 E2 同时优于 E0 和 C1，才支持“峰谷边界本身具有分类价值”；仅观察到分段更多或轨迹
  更复杂不构成证据。
- 若 E4 优于 E2 且优于 C2，才支持“受门控父元提供独立的层次组合信息”；E3 优于 E2 只能
  支持高频相邻模式有用，不能证明跨受试者稳定性。
- 若 A2/E2 优于 A0/E2，只能把增益归因于 A2 相对 A0 新增的“变点监督＋内容—边界对齐”
  约束包，不能归因于去掉 InfoNCE，因为 A0 和 A2 都未使用 InfoNCE；若要进一步分离变点
  损失与 alignment（对齐）本身，仍需新增编码器消融。
- 若 A2/E2 优于 A3/E2，才支持“在 E2 下去掉 InfoNCE 有利”；`A2/E0−A3/E0` 用于固定
  窗口复核，而对应 difference-in-differences（差分中的差分）只判断该效应是否依赖 E2。
  单独比较 A2/E2 与 A0/E0 会同时改变编码器和分段，存在混杂。
- 若 `state` 明显优于 `no_state`，尤其集中在 Sitting/Standing，只能说明物理状态旁路有效；
  不能据此声称纯动作元轨迹已经解决该类别对。
- 同类不同受试者轨迹更相似、异类轨迹更远，以及“上升—下降/下降—上升可组成父元”目前
  都是待检验假设，不是由分段算法定义自动保证的事实。

### 16.4 已知限制

- 峰/谷定义在六轴导数归一化后的规范运动强度包络上，表示“运动强度上升后下降/下降后
  上升”，不等同于人体关节轨迹的生物力学峰谷。
- 固定 K=32 只验证分段与层次结构，不验证在线可变码本数量；速度、佩戴方向、传感器位置、
  trial 长度和活动周期数仍可能造成受试者混杂。
- E3 容易偏向长 trial 和高频周期动作；E4 的 NPMI/MDL/LOSO 门槛仍是人为设定，父元叠加
  也不等于已经学得端到端层次表示。
- E4 同时使用 supporter-conditioned LOSO（支持者条件留一受试者稳定率）和 effectful LOSO
  （有效删除留一受试者稳定率）：前者只检查支持该候选父元的受试者，后者还检查虽不支持候选
  父元、但其删除会改变支持度或 NPMI 边缘分布的受试者；无同类事件且不改变任何判据的
  leave-out（留出）是显式 no-op（无操作），仅写入审计记录，不进入任一稳定率分母。
- `cross_subject=0.0` 表示没有直接受试者不变性约束；subject-equal weighting（受试者等权）
  和 E4 的多受试者门控只能降低部分偏差，不能保证域不变。
- 互不重复的留出受试者对折只有 7 个，统计功效有限；在正式多折结果产生前，不应根据少数
  示例图、单折或单种子结果宣称优于历史路线。

### 16.5 输出文件类别

单折运行应保存以下可审计产物：协议身份、数据哈希和 split audit（划分审计）；带哈希的
分段器、子元码本与父元目录状态；无标签 Session-2 清单和冻结原始簇预测；逐 trial 的片段
数量、边界、持续时间、峰/谷事件、child/parent（子元/父元）原序列及最终预测；父元支持度
与门控明细；描述符和活动距离矩阵。评分真值接入后另输出 post-truth diagnostic（真值后接
诊断）：跨受试者同类/异类距离效应、tie-aware cross-subject 1-NN（并列感知跨受试者最近
邻）及 raw-cluster–subject NMI（原始簇—受试者归一化互信息）；这些量不回流到模型或预测。

可视化包括代表性六轴原始波形及边界、子元/父元轨迹序列、动作距离热力图、混淆矩阵、
片段持续时间/数量分布和代表动作元波形。多折汇总另保存逐运行指标、折内种子均值、跨折
汇总、预先指定的后续主比较、探索性固定窗口复核、探索性析因交互、8 项 Holm 校正、总体指标图及
平均动作距离热力图。跨受试者诊断另输出
`cross_subject_diagnostics_by_run.csv`、
`cross_subject_diagnostics_by_fold_after_seed_average.csv` 和
`cross_subject_diagnostics_across_folds.csv`，分别对应单次运行、折内种子均值和跨折描述汇总。
单折 `experiment_result.json` 中的 generated-files（生成文件）清单不是文件名字符串列表，
而是逐文件记录相对路径、字节数和 SHA256（安全哈希算法 256 位）摘要；CV 断点续跑会拒绝
绝对路径、`..`、目录逃逸、大小/摘要不符，以及缺失当前实验臂必需产物的旧结果。所有具体
文件名以每次运行的 `experiment_result.json`、`summary.json` 和该内容寻址清单为准。

成员 request identity（请求身份）会对 runner、动作元算法及其所有影响结论的 transitive
dependency（传递依赖）记录源码 SHA256，包括特征准备、冻结读出、轨迹消融、编码器定义和
USC-HAD 加载器；任一依赖变化都会使旧成员身份失配并 fail closed（失败即关闭）。CV wrapper
（交叉验证封装器）只影响调度与汇总，不进入成员源码指纹，以允许修正纯汇总逻辑而不重跑
昂贵成员；CV 到单折 runner 的命令行接线另有独立哈希，整个网格身份仍记录 CV wrapper 指纹。

CV 不再让单折子进程直接写正式成员目录。每次执行都在正式目录同父级创建唯一的
`.<final-name>.staging-<UUID>` staging directory（暂存目录名）；子进程成功后，CV 先按上述
内容寻址清单完成身份、必需文件、大小和摘要校验，再以同文件系统 atomic rename（原子
重命名）发布为正式目录，并在发布后复核一次。进程中断、校验失败或发布冲突都会保留暂存
目录，不会把半成品伪装成可 `--skip-existing` 的正式结果；下次重试生成新的 UUID，只检查
正式目录，所以旧暂存目录不会阻塞续跑。代码不会自动删除旧暂存目录、空正式目录或任何
可疑残留；若正式路径已经存在但不能通过完整校验，会失败并要求人工检查。

缺失 A2 编码器时采用相同的“暂存训练→检查点语义及完整性校验→原子发布”顺序。编码器
暂存目录放在 `encoder_root` 外的同文件系统父级，避免递归 checkpoint finder（检查点查找器）
把一次已写完但尚未发布的中断暂存误认为 canonical checkpoint（规范检查点）；只有发布后的
目录才会进入正式编码器网格。训练或验证失败同样保留暂存证据，不覆盖已有规范目录。

### 16.6 PyCharm Linux/WSL 运行命令

先运行 A3、fold 1、seed 0 的全臂预检。预检使用独立输出目录，不能作为正式网格成员复用：

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_peak_valley_hierarchy_cv.py --cv-root /mnt/c/Users/BJ/Desktop/HCGCD/HHR/results/trial_pooling_subject_cv/mean_robust_max_q90_offline_online_7fold_4seed_20260830 --npz-path /mnt/c/Users/BJ/Desktop/HCGCD/HHR/processed/uschad_w256_s128_train17stats/uschad_windows.npz --encoder-root /mnt/c/Users/BJ/Desktop/HCGCD/HHR/results/motion_primitive/encoder_training --output-root /mnt/c/Users/BJ/Desktop/HCGCD/HHR/results/motion_primitive/peak_valley_hierarchy_A3_fold01_seed0_preflight_20260906_v1 --profiles A3 --arms E0,E1,E2,E3,E4 --folds 1 --seeds 0 --python /home/bj/miniconda3/envs/hhr/bin/python --device cuda --batch-size 512 --no-train-missing-encoders --skip-existing --matched-random-controls --old-class-count 6 --primitive-num 32 --pca-dim 64 --child-feature-source content_embedding --trial-cluster-count 10 --sample-rate-hz 100 --smooth-seconds 0.15 --prominence-mad 1.5 --extrema-distance-seconds 0.2 --vote-tolerance-seconds 0.1 --minimum-axes 2 --minimum-segment-seconds 0.25 --feature-confirm-quantile 0.5 --feature-context-seconds 0.5 --shape-points 64 --parent-min-occurrences 10 --parent-min-trials 6 --parent-min-subjects 2 --parent-min-npmi 0 --parent-min-mdl-gain 0 --parent-min-loso-stability 0.8 --bootstrap-replicates 10000 --analysis-seed 20260906
```

预检通过后运行 A0/A2/A3、7 folds × 4 seeds 正式网格。A2 没有规范检查点时会按固定配置
训练；中断后原命令可直接重跑，`--skip-existing` 只跳过通过内容哈希和身份审计的完整成员：

```bash
cd /mnt/c/Users/BJ/Desktop/HCGCD/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_peak_valley_hierarchy_cv.py --cv-root /mnt/c/Users/BJ/Desktop/HCGCD/HHR/results/trial_pooling_subject_cv/mean_robust_max_q90_offline_online_7fold_4seed_20260830 --npz-path /mnt/c/Users/BJ/Desktop/HCGCD/HHR/processed/uschad_w256_s128_train17stats/uschad_windows.npz --encoder-root /mnt/c/Users/BJ/Desktop/HCGCD/HHR/results/motion_primitive/encoder_training --output-root /mnt/c/Users/BJ/Desktop/HCGCD/HHR/results/motion_primitive/peak_valley_hierarchy_A0_A2_A3_7fold4seed_20260906_v1 --profiles A0,A2,A3 --arms E0,E1,E2,E3,E4 --folds 1,2,3,4,5,6,7 --seeds 0,5,50,500 --python /home/bj/miniconda3/envs/hhr/bin/python --device cuda --batch-size 512 --train-missing-encoders --skip-existing --matched-random-controls --old-class-count 6 --primitive-num 32 --pca-dim 64 --child-feature-source content_embedding --trial-cluster-count 10 --sample-rate-hz 100 --smooth-seconds 0.15 --prominence-mad 1.5 --extrema-distance-seconds 0.2 --vote-tolerance-seconds 0.1 --minimum-axes 2 --minimum-segment-seconds 0.25 --feature-confirm-quantile 0.5 --feature-context-seconds 0.5 --shape-points 64 --parent-min-occurrences 10 --parent-min-trials 6 --parent-min-subjects 2 --parent-min-npmi 0 --parent-min-mdl-gain 0 --parent-min-loso-stability 0.8 --bootstrap-replicates 10000 --analysis-seed 20260906
```

## 17. D 盘单阶段主线：HAPPY 式联合动作元原型（2026-09-09）

本节覆盖第 16 节之前的历史脚本，但不修改它们的实验事实。当前
`D:\WorkDir\HHR` 主入口为 `run_one_stage_cv.py`；默认协议是：

```text
完整 trial
-> ResNet1D 局部编码器（随机初始化，不加载 HAPPY checkpoint）
-> K=32 随机归一化可学习动作元原型
-> 确定性直通硬编号（反向软后验温度退火）
-> 边界、持续时间、顺序和转移轨迹
-> trial-level CGCD
```

ResNet1D 与动作元原型从第一个 epoch 进入同一个 AdamW 优化器；不存在独立编码器预热，
也不在未训练特征上执行 KMeans++。HAPPY 只提供“归一化特征与归一化原型联合学习”的
架构参考，不提供窗口分类、InfoNCE、SupCon、DINO 损失或 checkpoint。默认参数身份为
`local_encoder=resnet1d`、`codebook_init=learnable`、`codebook_update=gradient`；
`kmeans++`、`random+ema`、`light_cnn` 和 Gumbel 仅作为显式消融。

`loss_ramp_epochs=10` 只逐渐增加若干损失权重，不冻结编码器、不重置模型或优化器；旧参数名
`--warmup-epochs` 仅保留为同一参数的兼容别名。完整协议和当前 PyCharm Linux/WSL 单行命令
以项目根目录的 `ONE_STAGE_FRAMEWORK.md` 为准。
