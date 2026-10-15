# Elliptic 交易二分类实验

独立入口 `elliptic_experiment.py`、配置 `elliptic_config.py`。
不调用 Bitcoin/BABD/UCI 实验入口；仅复用编码、模型、评价、调度和存储。

## 数据

需要原始 Elliptic 文件，放到 `data/raw/public/elliptic/`：
- elliptic_txs_features.csv：无表头，167列（ID、时间步、165数值特征）。
- elliptic_txs_classes.csv：有表头 txId,class。

标签1=illicit，2=licit，unknown仅排除，不映射为合法。内部编码合法0、非法1。
一对一按ID合并，拒绝重复ID、缺失ID、不匹配标签文件、非法字段和非有限特征。
2026-10-15 已从 PyTorch Geometric 官方加载器使用的镜像下载原始交易文件。
镜像：https://data.pyg.org/datasets/elliptic/ （文件名追加 .zip）
来源与SHA256记录在 data/raw/public/elliptic/audit.json，数据文件不进入Git。
校验：203769条交易；合法42019、非法4545、未知157205；已标注交易ID无重复。
默认时间划分训练29894条（非法3462），测试16670条（非法1083）。

## 默认协议（可在main中修改）

训练时间步1–34，测试35–49；时间步不作为模型输入。
local采用原始列2–94，共93个数值特征（发布的94个本地特征包含时间步）。
all采用列2–166，共165个数值特征；包含邻居聚合信息，只作为显式选项。
发布的特征并非原始事件流重算特征，时间留出不证明所有输入在交易发生时可得。
本实验定位为离线非法标签识别，不声称实时部署性能。
来源：https://medium.com/elliptic/the-elliptic-data-set-opening-up-machine-learning-on-the-blockchain-e0a343d99a14
论文：https://arxiv.org/abs/1908.02591

全部模型使用相同二值特征，仅训练数据拟合分箱和编码。
分箱[5,10,20,30,50]；种子[11,37,53]；默认uniform投票。
全部规则、仅正规则(A1/A2)、仅负规则(B1/B2)共用一次fit。
随机种子仅改变随机模型，不改变时间划分；确定性RC-OEO重复结果不视为独立测试样本。
固定参数不使用测试标签选择最优；如果调参，另划早期验证时间段。

## 结果

results/classification/elliptic_results.xlsx，独立索引elliptic_index.json。
原有通用指标、候选集合指标、规则审计和耗时不变。
增加illicit_tp/fp/fn、illicit_precision/recall/f1、illicit_average_precision。
延迟预测的非法样本计入FN；AP使用全部测试对象的连续分数，包括延迟对象。
RC-OEO排序分数=非法类得分减合法类得分；基线使用非法类概率或决策分数。
AP是average precision阶梯积分，不命名为梯形PR-AUC；不是概率校准指标。
计分和预测均计入predict_seconds；预处理、规则审计单独进行。
时间边界及特征组进入任务ID和汇总分组，不混合不同协议。
每种子只有一个时间留出，跨种子平均反映随机模型变化，不是时间泛化置信区间。
二分类没有1<候选数<2的部分缩减情况，相关部分集合指标为空/零属正常。

## 运行（D30工程目录，fca环境）

```bash
PYTHONPATH=src python -m oeo_lab.experiments.classification.elliptic_experiment --dry-run
PYTHONPATH=src python -m oeo_lab.experiments.classification.elliptic_experiment --models rc-oeo random_forest --max-bins 5 --seeds 11 --n-jobs 1
```

数据下载及全量结构检查已完成；正式实验尚未启动。人工小背景测试已通过。

2026-10-15 真实数据冒烟检查通过：按既定时间划分分别抽取200条训练、200条测试，93个本地特征；RC-OEO一次拟合验证全部/正/负三种组合，随机森林一次拟合；四行结果全部完成，检查点与Excel汇总通过。临时结果已自动清理，不进入正式论文结果。
