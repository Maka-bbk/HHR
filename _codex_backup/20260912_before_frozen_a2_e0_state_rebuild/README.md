# HHR：动作元轨迹 HAR-CGCD

HHR 是只面向可穿戴传感器的 HAR-CGCD（人体活动识别连续广义类别发现）项目。唯一目标是利用 Motion Primitive（动作元）及其变长 trajectory（轨迹）提高旧类保持、新类发现和整体分类效果。

HHR **不是 Happy 的复现工程**。Happy 只提供候选机制：ResNet1D（一维残差网络）局部编码器、按 trial（试次）组织监督与受试者划分的方式，以及 online anti-forgetting（在线抗遗忘）思想。新 HHR 不复制 Happy 的完整试次池化分类路径；是否保留任何借鉴机制，只由 HHR 内同协议的动作元消融决定。

当前仍是实验验证阶段。代码实现完成不等于动作元路线已经改善 HAR-CGCD；最终结论必须来自完整多折、多种子的分类指标和动作元内部消融。

## 当前唯一主线：A2-MP + E0 + state

历史实验中，A2 编码目标、E0 固定窗口和 state（物理状态）轨迹描述的组合最值得保留。新项目把它改造成一次联合优化的 `A2-MP`：保留 A2 的变点、内容—边界、防坍缩与时间预测目标，删除 A2 原有的池化试次辅助头，改由变长动作元轨迹直接承担分类监督。这是**有历史依据的起始路线**，并不是已经在新代码上得到多折验证的最终最优结论。

```text
完整 activity trial（活动试次）
  -> 按时间顺序恢复 6 轴局部窗口
  -> 共享 ResNet1D（初始化方案待实验决定）
  -> 局部窗口特征
  -> 可学习码本 + 可学习边界
  -> 变长动作元 run（连续片段）
  -> 种类/持续时间/顺序/转移/位置/物理状态
  -> GRU（门控循环单元）-> trajectory head（轨迹分类头）
  -> 一个动作元轨迹损失 -> 一个 optimizer（优化器）-> 每个 batch 一次 backward（反向传播）
```

ResNet1D 是随机初始化还是从历史窗口编码器 warm-start（热启动）目前**未定**，代码不会替实验者暗选。每次正式运行必须显式传入 `--encoder-initialization random` 或 `--encoder-initialization warmstart`；后者还必须提供匹配当前 fold/seed（折/随机种子）与归一化协议的编码器检查点。无论采用哪种初始化，码本、边界头、轨迹编码器和分类头都在同一次连续训练中更新，不存在训练中途重置模型或优化器的第二阶段。

动作元分段由最终边界决定，而不是把每次码字变化都强制切开。每个 run 输入轨迹编码器的内容包括：由 run 内码字分配加权得到的固定维码本嵌入、进入边界强度、进入转移强度、持续时间、相对位置，以及由六轴窗口统计投影得到的物理状态。完整 `K` 维分配只保留给 VQ（向量量化）损失、占用诊断和可视化，不直接作为 GRU 输入；因此 online 增加码本行时不会扩张或重置轨迹输入层。某个完整动作只有一个 run 是允许的；分类与新类发现效果才是目标，动作元数量不是越多越好。

`K=32` 表示初始码本容量，不表示每次运行一定拆出 32 种动作元。结果必须区分 capacity-K（容量）与 used-K（实际被有效窗口使用的码字数），并报告 dead code（死码）、每条 trial 的 run 数及类/受试者占用偏差。

这里仍允许两种局部 aggregation（聚合）：ResNet1D 在**单个窗口内部**汇聚时间特征，以及一个 primitive run 在**自身边界内部**汇聚所含窗口。二者都不会跨越整条 trial，也不会删除 run 之间的顺序，因此不属于完整试次池化。

## 数据协议

- 数据集为 USC-HAD；当前公开训练入口读取预处理后的 `uschad_windows.npz`；
- 默认窗口 256 samples（采样点），stride（步长）128；100 Hz 下分别对应 2.56 秒和 1.28 秒；
- 不足一个完整窗口的 trial 尾部不会进入当前 NPZ。“完整 trial”指该 trial 所有**已保存完整窗口**的有序集合，不代表保留了原始尾段；
- 默认旧类为物理标签 `0..5`，三次在线会话扩展为 `6 -> 8 -> 10 -> 12` 类；
- subject-disjoint cross-validation（受试者无交叉交叉验证）固定 7 折；每折 10 名训练、2 名验证、2 名外层测试受试者；
- 每折归一化统计只由训练受试者旧类窗口重新计算；
- online（在线）训练和测试 trial 按 trial ID 排除重叠。活动标签用于构造规定的基准数据流与最终评分，但不会传入在线优化损失。

## 唯一现行实验臂

公开 CLI 使用下列动作元 profile（实验臂）：

| profile | 作用 |
|---|---|
| `motion_primitive_joint` | 以变长动作元轨迹直接完成 offline/online HAR-CGCD |

历史 trial-pooling（试次池化）、pooled/fused（池化/融合）分支只属于旧结果，不是新 HHR 的实验组成、候选输出或成功标准；现行 CLI 不再接受其 profile 名称。

## 当前固定训练事实与待决变量

- offline（离线）训练 100 epochs、batch size（批大小）16；
- SGD（随机梯度下降）学习率 `0.01`；随机初始化时编码器倍率为 `1.0` 且冻结 epoch 数为 `0`，warm-start 的倍率与是否短暂冻结必须作为显式消融记录；
- 两个 trial views（试次视图）都保留完整时间范围；`view[0]` 是严格不增强的 clean anchor（干净锚点），只有 `view[1]` 使用小幅通道缩放，不随机裁掉动作阶段；
- 完整 trial 的旧类监督只施加在 trajectory head 上；训练目标包含轨迹 CE（交叉熵）、轨迹跨受试者 SupCon（监督式对比学习）、A2-MP 局部目标、VQ（向量量化）及边界正则；
- offline clustering（离线聚类蒸馏）与实例级 InfoNCE（信息噪声对比估计）默认关闭；
- 初始码本 `K=32`；动作元总权重从第 1 个 epoch 就是 `1.0`，过程中不重置模型状态；
- 物理变点锚点须同时满足 `score >= q75` 与严格的 `score > max(median + 3×1.4826×MAD, 0.01)` null gate（零变化门控），允许整条低动态或等幅变化 trial 没有变点；`0.01` 与 `3×MAD` 只是预注册消融起点，并非已验证最优值；
- 轨迹 CE、轨迹跨受试者 SupCon、VQ（向量量化）commitment/codebook（承诺/码本）损失，以及小权重的最短持续时间与转移预算约束共同训练；
- 普通随机 batch 不保证出现同类不同受试者正对；每个 epoch 与最终 summary 会记录 eligible-anchor（可用锚点）比例、覆盖 batch 比例和计数，在未验证类别—受试者感知采样器前不得宣称每批都完成了去偏；
- 只按 validation trajectory macro-F1（验证集轨迹宏平均 F1）选点；外层测试不参与选点；
- 新实验只生成并汇总 trajectory 预测；现行 checkpoint、选点与报告协议均不包含 pooled/fused 字段。

完整方法、online 数据流和解释边界见 [METHOD.md](METHOD.md)。

## 新主线入口

- `experiments/motion_primitive/train_offline.py`：单个 offline 成员；
- `experiments/motion_primitive/run_offline_cv.py`：offline 多折多种子编排；
- `experiments/motion_primitive/run_online.py`：单个三会话 online 成员；
- `experiments/motion_primitive/run_online_cv.py`：online 多折多种子编排；
- `experiments/motion_primitive/export_trajectory_visuals.py`：动作元序列、活动—码本热力图、混淆矩阵和原始六轴轨迹导出；
- `models/motion_primitives.py`：中性的可学习码本、成对边界头与软转移／持续时间组件；
- `models/motion_primitive_cgcd.py` 与 `experiments/motion_primitive/joint_losses.py`：中性模型与损失接口。

`offline_trainer.py` / `offline_cv_runner.py` 与 `online_runner.py` /
`online_cv_runner.py` 是四个公开入口的现行实现；规范源码哈希只覆盖本项目实际导入的模块。

## PyCharm Linux/WSL 运行

参数检查：

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_offline_cv.py --help
```

初始化结论尚未确定，因此不存在可省略初始化参数的“默认正式命令”。随机初始化分支的 offline 7 folds × 4 seeds（7 折 × 4 随机种子）命令为：

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_offline_cv.py --npz-path /mnt/d/WorkDir/HHR/processed/uschad_w256_s128_train17stats/uschad_windows.npz --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/offline_a2mp_e0_state_random_7fold4seed_v3 --profiles motion_primitive_joint --folds 1,2,3,4,5,6,7 --seeds 0,5,50,500 --encoder-initialization random --device cuda --resume
```

warm-start 分支必须使用不同输出目录，并显式使用 `--encoder-initialization warmstart --encoder-warmstart-root <严格匹配网格的检查点根目录>`。在初始化方案确定前不要把任一分支称为正式主结果。`--resume` 只续跑实验网格；单成员不支持 mid-epoch resume（中途 epoch 恢复）。

offline 完成后运行 online 网格：

```bash
cd /mnt/d/WorkDir/HHR && /home/bj/miniconda3/envs/hhr/bin/python experiments/motion_primitive/run_online_cv.py --offline-cv-root /mnt/d/WorkDir/HHR/results/motion_primitive/offline_a2mp_e0_state_random_7fold4seed_v3 --output-root /mnt/d/WorkDir/HHR/results/motion_primitive/online_a2mp_e0_state_random_adaptive_7fold4seed_v3 --profiles motion_primitive_joint --folds 1,2,3,4,5,6,7 --seeds 0,5,50,500 --codebook-expansion residual_adaptive --device cuda --resume
```

当前 online 主命令显式使用 `residual_adaptive`：从 Offline `K=32` 出发，每个 session（会话）仅凭无标签局部特征残差在 `ΔK=0..4` 中选择，因此码本长度可不变也可逐会话增长。这里的 `max-delta=4` 是**每会话上限**而不是全程上限；三次会话的理论最大路径为 `32→36→40→44`。`none` 和 `fixed_delta` 是配对消融，必须使用独立输出目录；参数即使有程序默认值，正式实验也应显式写出，避免不同策略混入同一网格。

## 结果判定

HHR 的完成标准不是与历史框架数值一致，而是动作元路线自身的 HAR-CGCD 分类价值。项目外保存的历史结果只能作为迁移排错和外部基线；动作元路线没有带来稳定分类收益就不算完成：

- 主要报告 All/Old/New accuracy（全体/旧类/新类准确率）与 H-score（调和得分）；
- 主要在线对齐为 `constrained_old_fixed`（旧类语义固定、只对齐新类）；另报 `standard_global_hungarian`（全局匈牙利匹配上界）和 `direct_head`（不借测试标签对齐的直接分类头）；
- trajectory 是唯一有效分类输出；以旧 HHR 版本和“去顺序、去持续时间、固定边界、固定码本”等动作元内部消融作同 fold/seed（折/随机种子）配对比较；
- 单折单种子只用于 smoke test（冒烟测试）；码本大小、run 数量或可视化更复杂也不是成功证据。

## 外部历史边界

旧的试次池化、固定 KMeans32（K 均值 32 簇）、自监督变点、峰谷层次动作元、二级码本和层次门控代码及结果保存在本项目之外，不作为可调用入口。只有数据划分、初始化、监督范围和评估对齐一致的历史结果，才允许作为外部基线与当前主线并表。
