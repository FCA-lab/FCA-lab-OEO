# 训练规模与规则决策实验

独立入口 `training_size_experiment.py`，配置 `training_size_config.py`。
目的：评价训练规模变化与规则数量、唯一预测可靠性、覆盖率、候选范围的关系；不预设单调性或性能优势。

默认数据 Iris/Wine/WDBC；数据字典支持公共数值 CSV 和类别表格格式，可登记其它独立样本数据集，不限 UCI。类别表格设置 feature_kind=categorical（直接 one-hot，不遍历 bins）。不自动处理混合类型、缺失值、实体重复或时间划分；这些数据须先准备与任务相符的表格，不能无声改成随机样本划分。

每种子、每类随机打乱，留出 ceil(类别样本数×0.3) 作为固定测试集；剩余为训练池。每类训练池按比例向上取整取前缀，从而嵌套且各类均被保留。记录实际比例及每类数量。某类无训练对象、或两种比例得到同一子集时明确报错。标签只用于分层；编码仅在当前训练子集拟合，不能把结果当作固定属性空间下的理论验证。

main 中默认：比例 [0.2,0.4,0.6,0.8,1.0]，max_bins=[5,8,10]，种子 [11,37,53]，uniform 投票，全部四类规则，仅 RC-OEO。总计135任务。配置可直接增删数据集；终端只覆盖显式参数。

公共 evaluate_fold/runner 负责训练、候选指标、四类规则统计、时间、检查点与调度保护。结果位于 results/classification/training_size/，不覆盖其它实验。folds 保留实际二值属性数，training_design 工作表与独立 CSV 保留请求/实际训练比例、训练池、训练/测试数量及各类别数量。per_seed/summary 按实际 train_budget 分组；一个训练设置只提取一次规则。样本预算为实际总数，train_per_class 留空，因为各类不强制等量。

```bash
cd /home/charles/projects/FCA-lab-OEO
conda activate fca
PYTHONPATH=src python -m oeo_lab.experiments.classification.training_size_experiment --dry-run
PYTHONPATH=src python -m oeo_lab.experiments.classification.training_size_experiment --n-jobs 3
```

默认串行；并行数由运行时决定，沿用公共 RAM/swap 保护。相同参数重启复用成功任务；修改 n_jobs 不改变任务身份。不要同时运行同一输出目录的两个汇总进程。

原 TabPFN 小样本划分使用“每类相同数量预算”；本实验使用“每类训练池比例”，因此不直接调用其预算生成函数，但保持相同的逐类随机排列与固定留出原则。原模块及公共实验代码保持不变。
