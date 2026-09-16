# HHR：基于动作元轨迹的 CGCD

本项目的中心任务是 Motion-Primitive Trajectory CGCD（基于动作元轨迹的连续广义类别发现）：将 USC-HAD 的完整变长 activity trial（活动试次）分解为动作元序列，再利用动作元种类、数量、边界、持续时间、顺序和转移关系形成轨迹表示，用于旧类识别与新类发现。

当前主实验是 J0 单阶段可行性验证：随机初始化的 ResNet1D（一维残差网络）与 K=32 可学习归一化动作元原型从第一个 epoch（训练轮次）进入同一个 AdamW 优化器，共同学习局部动态、变长边界、离散码本和完整试次轨迹。默认不在随机编码器特征上执行 KMeans++，也不由 EMA（指数移动平均）独占更新码本；这两条旧路径仅保留为显式消融。项目不加载 HAPPY/A2 检查点，也不使用窗口级活动分类、InfoNCE（信息噪声对比估计）或 SupCon（监督式对比学习）。

主入口：

```text
experiments/motion_primitive/run_one_stage_cv.py
```

主结果必须使用：

```text
--trajectory-input-mode primitive_only
--checkpoint-selection common_unsupervised
--local-encoder resnet1d
--codebook-init learnable
--codebook-update gradient
```

`primitive_only` 的分类入口只读取离散动作元编号及其边界、转移和持续时间。`state_only` 与 `primitive_plus_state` 仅用于 attribution ablation（归因消融），不能替代或混入动作元 CGCD 主结果。仓库内保留的 HAPPY、固定窗口、池化及旧码本脚本只作为历史基线与对照，不定义本项目的中心方法。

当前 J0 是研究筛查实验；online adaptive codebook（在线自适应码本扩展）、自动类别数发现和归纳式单样本部署仍属于后续阶段，不能从 J0 结果中提前宣称已经解决。

完整协议、损失定义、输出解释、恢复语义和 PyCharm WSL 单行命令见 [ONE_STAGE_FRAMEWORK.md](ONE_STAGE_FRAMEWORK.md)。
