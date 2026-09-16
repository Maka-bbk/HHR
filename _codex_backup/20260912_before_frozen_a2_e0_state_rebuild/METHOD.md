# HHR 方法定义：动作元轨迹 HAR-CGCD

## 1. 研究目标与非目标

HHR 研究 Motion-Primitive Trajectory HAR-CGCD（基于动作元轨迹的人体活动识别连续广义类别发现）。唯一优化目标是提高旧类保持、新类发现和总体活动分类，而不是复现 Happy、追求相同源码结构或对齐其逐张量数值。

Happy 提供的是可验证候选，不是约束 HHR 的模板。当前只借用 ResNet1D（一维残差网络）、按 trial（试次）组织监督与受试者划分的方式，以及 online anti-forgetting（在线抗遗忘）思想；不引入其完整试次池化分类路径。任何候选若在公平动作元消融中降低 HAR-CGCD，就应调整或删除。

## 2. 核心假设

一个 activity trial（活动试次）通常包含局部阶段结构。把所有窗口压缩为单个 pooled feature（池化特征）可能丢失动作元的种类、持续时间、顺序与转移；先形成变长 motion primitive run（动作元连续片段），再编码 activity trajectory（活动轨迹），可能提升跨受试者稳定性和类别区分度。

这是待验证假设，不是实现动作元模块后自动成立的事实；某类活动只有一个 run 也完全合法。若单 run 已能稳定区分类别，就不应为了“细致”而强制过分段。

## 3. 数据与监督边界

当前主入口使用 USC-HAD 窗口级 NPZ，并按 `trial_global_id` 与 `window_start_indices` 恢复有序试次；默认窗口 256、步长 128，不完整尾窗已在预处理阶段丢弃。因此模型看到的是一个 trial 的全部已保存完整窗口，而不是原始采样点无损全长信号。

7-fold subject-disjoint cross-validation（7 折受试者无交叉交叉验证）中，每折为 10 名训练、2 名验证、2 名外层测试受试者；归一化统计只拟合训练受试者旧类。offline（离线）训练只使用旧类完整试次标签；subject ID（受试者编号）只构造同类跨受试者对比正样本掩码，不作为分类输入。

online（在线）基准构造器必须读取标签才能生成规定的旧/新类数据流，最终评估也读取测试标签；在线优化 API 只接收未标注的两个 trial view（试次视图），不接收活动标签。这个区别必须在论文和结果表中公开。

## 4. 单模型动作元架构：A2-MP + E0 + state

该路线由历史上较优的 A2 编码目标、E0 固定窗口和 state 轨迹描述启发；新项目不复用其“两段训练后再读出”的执行方式。`A2-MP` 表示把 A2 中有价值的局部表征约束直接并入动作元轨迹的一次联合训练，并用轨迹分类替换原 A2 的池化试次辅助分类。历史排序支持选择这个起点，但不能替代对新联合模型的重新验证。

### 4.1 共享局部编码器

每个按序窗口由同一个 ResNet1D 编码为局部特征；编码器只接收轨迹分类和动作元学习目标的梯度，不接收完整试次池化分类梯度。

初始化目前是一个未决实验变量，而不是框架事实。所有正式运行必须显式选择 `--encoder-initialization random` 或 `--encoder-initialization warmstart`；warm-start 只加载严格匹配 fold/seed、窗口、通道和归一化协议的 ResNet1D 权重。两种分支必须写入不同结果目录并作配对比较，不能把随机初始化失败自动归因于动作元，也不能把 warm-start 的收益误写成动作元本身的收益。

### 4.2 局部聚合与完整试次池化的边界

新 HHR 允许两个局部 aggregation（聚合）操作：

- ResNet1D 只在一个局部窗口内部汇聚时间特征，得到该窗口的稳定表示；
- 一个 primitive run（动作元连续片段）只在自身边界内汇聚所含窗口，得到 run-level（动作元级）表示。

二者都保留 run 之间的种类、持续时间、顺序和转移，不会把完整 trial 压成单个向量，因此不属于 complete-trial pooling（完整试次池化）。跨整个 trial 的 mean+q90 pooling（均值与 90% 分位数池化）、pooled head（池化分类头）和 fused prediction（融合预测）不进入新项目的现行训练、选点或评估。

### 4.3 可学习码本与边界

局部特征被归一化后映射到可学习 codebook（码本）；默认 `K=32`。训练前向使用确定性 hard one-hot（硬独热）编号，反向使用软后验梯度的 straight-through estimator（直通估计器）；评估同样使用硬码本分配。

相邻窗口的最终边界概率由两类证据共同决定：

```text
final_boundary_logit = learned_boundary_logit + token_change_evidence
```

`token_change_evidence` 来自相邻码字分布变化，但码字改变并不强制切分；边界头可以把不同子码合并为同一父级 run，也可在硬码字相同处主动切开。阈值化后的最终边界才定义变长 run。

物理变点伪锚点还必须同时满足 `score >= q75` 与严格的 `score > max(median + 3×1.4826×MAD, 0.01)` null gate（零变化门控）；因此低幅噪声或所有相邻变化等幅的 trial 可以完全没有 change anchor（变化锚点），不会再由单纯分位数强制造出边界。stable/change（稳定/变化）掩码还被强制设为互斥。`0.01` absolute floor（绝对下限）与 `3×MAD`（三倍中位绝对偏差）只是预注册消融起点，当前不能称为 USC-HAD 上的最优参数。

### 4.4 Run-level 轨迹

每个 run 聚合其包含窗口的信息，当前轨迹编码器真正读取：

- run 内码字分配对归一化码本向量的加权和，即固定 `D=feature_dim` 维 codebook embedding（码本嵌入）；它保留 run 内可能包含的多个子码信息，但不会把随 `K` 增长的完整 `K` 维分布直接送入 GRU；
- 进入该 run 的边界与转移强度；
- 由最终边界产生的持续时间；
- trial 内相对位置及其正弦/余弦位置；
- 六轴窗口统计，经可学习投影形成 physical state（物理状态）特征。

run 序列经 GRU（门控循环单元）得到 trajectory embedding（轨迹嵌入）与 trajectory logits（轨迹预测）。物理状态旁路用于保留 Sitting/Standing（坐下/站立）这类低动态类别所需的姿态线索；其增益必须通过 `run_state_dim=0` 消融验证，不能提前归因为“纯轨迹已经解决静态类别”。

这里 `K` 只是码本容量；一次运行实际拆出的动作元种类数应报告为 used-K（被至少一个有效窗口使用的码字数），并同时报告 dead-code（死码）数、每类/每受试者占用和每条 trial 的 run 数。Online 扩展 `K` 时只增加码本行，固定维 run embedding、GRU 输入层和既有分类坐标系均不扩维；否则单纯增加候选码字就会无故改变旧轨迹表示。

## 5. Offline 单阶段优化

“单阶段”指一个模型、一个优化器、一条连续 epoch 轨迹，每个 batch 只组合一次总损失并调用一次反向传播；它不等于取消 CGCD 的 offline/online 时间阶段。

现行动作元路线使用一个损失接口：

```text
L_total = w_motion(epoch) * [
              L_trajectory_CE
            + 0.25 * L_trajectory_cross_subject_SupCon
            + 1.00 * L_physical_changepoint
            + 0.10 * L_content_boundary_alignment
            + 0.05 * L_noncollapse
            + 0.50 * L_masked_temporal_prediction
            + 0.25 * L_VQ_commitment
            + 0.25 * L_VQ_codebook
            + 0.02 * L_minimum_duration
            + 0.02 * L_transition_budget
          ]
```

其中 CE 是 cross-entropy（交叉熵）；SupCon 是 supervised contrastive learning（监督式对比学习）；VQ 是 vector quantization（向量量化）。四个新增的 A2-MP 局部项分别约束物理变点、内容变化与边界的一致性、表征防坍缩和遮挡位置的时间预测。当前 `w_motion` 从第 1 个 epoch 即为 `1.0`；若未来启用 ramp（渐增权重），它只能是同一连续优化中的消融，不能重建模型或重置优化器。

旧类分类监督存在于轨迹 CE 中，监督单位是完整动作元轨迹，而不是单窗口或完整试次池化向量。`w_motion` 不是在池化与动作元之间做权衡，只是统一控制动作元目标的连续课程。

两个视图默认都是 full trial（完整试次），不会随机裁掉动作阶段；`view[0]` 是严格不加噪声、不缩放的 clean anchor（干净锚点），只有 `view[1]` 使用小幅通道缩放。同一试次的两个视图是正样本；有 subject ID 时，额外正样本只取同类别、不同受试者试次。该约束针对类内跨受试者稳定性，但仍依赖 offline 旧类标签，不能称为无监督去偏。普通随机 batch 未必含跨受试者同类正对，因此训练历史与 summary 必须记录 eligible-anchor（可用锚点）比例、覆盖 batch 比例及计数；同一 trial 的第二视图不计为跨受试者覆盖。

以下损失默认关闭，仅作显式消融：offline self-distillation clustering（离线自蒸馏聚类）、实例级 InfoNCE（信息噪声对比估计）、轨迹蒸馏、轨迹视图一致性、强制码本利用率、分配熵、码本排斥和边界一致性。实例级 InfoNCE 会把其他同类 trial 当负样本，可能与同类跨受试者稳定目标冲突，因此不进入默认主线。

## 6. Offline 选点与可归因性

每个 `motion_primitive_joint` 成员只按 validation trajectory macro-F1（验证集轨迹宏平均 F1）保存 checkpoint；外层测试指标只在该选点固定后计算，不参与选点。trajectory 是新项目唯一有效的分类输出。

最终动作元内部消融至少包括：

- 完整动作元轨迹；
- 打乱 run 顺序，检测顺序信息是否真正有用；
- 去掉持续时间或转移输入，检测对应轨迹属性的边际贡献；
- 固定边界、固定码本或禁用物理状态旁路，定位分类收益来源。

项目外历史结果中的 pooled/fused 字段不得进入新项目汇总，也不得替代 trajectory 结果。

## 7. Online 数据流

每次 online session（在线会话）引入两个新类，轨迹分类头依次从 6 扩为 8、10、12 行。当前动作元实现采用的候选抗遗忘机制包括：

- 用 KMeans（K 均值）初始化新轨迹分类行；
- 训练轨迹 self-distillation（自蒸馏）与 grouped MeMax（分组最大熵）；
- 对旧轨迹进行 logit/feature distillation（预测/特征蒸馏），并继续更新 VQ 项；
- 对同一批未标注窗口约束旧/新编码器局部特征，并锚定会话开始前已有的码本行；两项损失可独立消融；
- 每会话训练后运行无增强、无活动标签的 primitive-retention audit（动作元保持审计），将旧码本行相似度、同窗口特征相似度、旧码分配一致率和新码使用率同时固化到 JSON、checkpoint 与汇总中。

这些是从既有强 HAR-CGCD 路线借来的候选，不是必须永久保留的“Happy 损失包”。需要逐项消融其对 Old、New 和 H-score（调和得分）的影响；实例级 online InfoNCE（信息噪声对比估计）因可能造成同类排斥，不进入当前动作元默认路线。

动作元码本扩展有三种机制层级：不扩展、固定增量、基于未标注局部特征残差的自适应增量。自适应策略只允许读取在线局部特征、当前码本和无标签统计，并在候选 `delta K` 之间以窗口数、不同 trial 数、不同 subject 数、失真改善与复杂度惩罚选择；测试标签不得进入扩容决策。不同 trial/subject 门控用于阻止一个长 trial 的大量重叠窗口伪造新动作元支持。存在代码路径不等于已得到分类收益，必须与固定 `K=32` 做配对实验。

轨迹分类器不直接读取随 `K` 变化的完整分配向量，而读取由分配对码本向量加权得到的固定 `feature_dim` 维动作元嵌入。因此扩容只增加码本行，不改变 GRU 输入宽度；在旧窗口不使用新码时，旧 token、轨迹 embedding（嵌入）与旧轨迹 logits（输出）必须在连续三次扩容前后保持一致，online checkpoint 也必须能够据自身 architecture（架构）字段在全新模型对象中严格重载。

所有物理变点、边界融合、最短持续时间、残差支持度和扩容惩罚门槛目前都是待验证超参数，而非 USC-HAD 上已证明最优的常数；正式比较应冻结参数网格并记录选择数据，避免用外层测试结果反调门槛。

## 8. Online 评估层

每个 head（分类头）和 session 同时输出：

- `constrained_old_fixed`：固定旧类语义行，只对新类行做 Hungarian matching（匈牙利匹配）；这是当前主要 CGCD 指标，但新类对齐仍使用测试标签；
- `standard_global_hungarian`：允许全局重排类别行的乐观聚类上界；
- `direct_head`：不使用测试标签拟合映射，最接近直接部署语义预测。

三层指标不能混写。全局匈牙利结果不能表述为部署准确率；`constrained_old_fixed` 也不是完全无标签部署，因为新类编号仍通过测试标签对齐。

## 9. 研究闭环与通过条件

主假设只有在以下证据同时成立时才得到支持：

1. 轨迹分类器实际读取 run-level 变长序列，而非逐窗口伪轨迹；
2. 码本、边界头、ResNet1D 和轨迹编码器均收到有效梯度，且分段未全部合并或无意义爆炸；
3. trajectory-only 明显高于对应随机水平，并在跨受试者测试中稳定；
4. 当前动作元路线相对旧 HHR 版本及动作元内部去顺序/去持续时间/固定边界等消融，在主要 HAR-CGCD 指标上形成跨折、跨种子的稳定成对改善，重点看 H-score，且不能以严重牺牲 Old 或 New 一侧换取表面总分；若只有更清晰的动作元图而没有分类收益，则核心假设尚未通过；
5. 先在每折内汇总 seeds（随机种子），再以受试者折为统计单位，不能挑选单次最好结果。

可视化只用于解释：应报告每个 trial 的 run 数、原始 token/run 序列、边界、持续时间、活动—码本热力图和混淆矩阵；图像不能替代分类指标。

## 10. 当前限制

- 边界只能落在相邻已保存窗口之间；默认时间栅格 1.28 秒，且相邻 2.56 秒窗口有 50% 重叠，因此不是采样点级真实动作边界；
- 码本编号是潜在运动状态，不自动等价于可解释的生物力学动作元；
- run-level 物理状态来自已归一化窗口统计，并非直接使用设备坐标系原始重力向量；
- 当前普通随机 batch 不保证每批都含“同类、不同受试者”正对；因此跨受试者 SupCon 的 batch/anchor 覆盖率必须随训练结果报告，未实现并验证 class-subject-aware sampler（类别—受试者感知采样器）前，不得声称每个 batch 都完成了受试者去偏。即便之后使用平衡采样，它也只能减少部分偏差，不能保证完整域不变性；
- online 自适应码本扩容、静态类别区分和跨受试者轨迹一致性均待多折验证；
- 当前公开 API 不提供历史 profile 别名或 pooled/fused 输出；历史结果若需比较，应在项目外单独完成协议审计，不能混入现行结果目录。

## 11. 外部历史边界

旧的试次池化、冻结 KMeans32 轨迹诊断、自监督特征变点、峰谷层次分段、二级码本和层次门控实现不随 clean project（干净项目）部署。它们只能作为外部历史证据回看假设来源，不得进入现行数据流或成功标准。
