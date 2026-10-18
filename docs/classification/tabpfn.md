# 小样本分类实验

## 目的与边界
比较 RC-OEO、TabPFN v2、随机森林、XGBoost 和 CatBoost 在有限标签条件下的表现。
首先使用已有 Iris、Wine、WDBC；这不是 Nature 原论文数据集和流程的精确复现，也不能将本实验数值与论文表格直接比较。
参考模型来源：https://github.com/PriorLabs/TabPFN 。必须使用原版 `tabpfn-v2-classifier.ckpt`，不得将新版本默认模型称为 Nature 2025 模型。

## 协议
每个种子按类留出 30%（向上取整）作为固定测试集。训练每类 5、10、20、50、100 个对象，使用同一随机排列的前缀，保证嵌套。
不足的预算写入 tabpfn_skipped.json，不缩小预算、不复制样本。不同种子的测试集可能重叠，不视为独立样本。
所有模型接收同样的二值条件属性；分箱和编码仅在对应训练子集拟合。
网格完整报告，不根据测试结果挑参数。一个 RC-OEO 任务只拟合一次，三种投票共享规则。
这是样本级划分；如新增数据有患者、实体或重复测量，需先审核分组方案。

## 配置和输出
运行参数位于 tabpfn_experiment.main；数据集、模型、路径等位于独立 tabpfn_config.py。
输出 tabpfn_results.xlsx、tabpfn_index.json、tabpfn_skipped.json 和检查点。与 UCI、BABD 原结果分开。
沿用总体正确率、Macro-P/R/F1、覆盖率、已分类准确率、候选集合命中率和平均大小、规则数以及 fit/predict 时间。
TabPFN 是预训练模型，fit 可能主要保存上下文，需结合 fit 和 predict 时间分析；预训练成本和模型加载不在这些计时内。
checkpoint 记录依赖版本与权重 SHA256。缺失权重、依赖或超出 v2 的 10000 训练行、500 二值属性、10 类限制均显式报错。

## 运行
先准备兼容的 tabpfn/torch 环境，并将官方 v2 权重放到 artifacts/tabpfn/tabpfn-v2-classifier.ckpt，或使用 OEO_TABPFN_MODEL 指定路径。
当前实现不自动安装依赖或下载权重。

```bash
cd /home/charles/projects/FCA-lab-OEO
conda activate fca
PYTHONPATH=src python -m oeo_lab.experiments.classification.tabpfn_experiment --dry-run
PYTHONPATH=src python -m oeo_lab.experiments.classification.tabpfn_experiment --n-jobs 1
```

依赖未准备时，可用 --models rc-oeo random_forest 检查流程，但此结果不能作为已完成 TabPFN 对比。
