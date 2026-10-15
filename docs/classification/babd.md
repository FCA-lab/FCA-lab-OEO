# BABD 独立实验（2026-10-15）

入口 `oeo_lab.experiments.classification.babd_experiment`，配置 `babd_config.py`。
原 `bitcoin_experiment` 默认恢复自建数据；BABD 不导入其入口或配置。

## 两套流程

- balanced：原类别 0,1,2,3,5,6,11；每个种子先留出每类1000个测试地址。
  每类训练50,100,200,500,1000,2000,4000个，总数350至28000；按同一排列前缀取样。
- imbalanced：清洗后11类；分层5折，每折测试集固定。
  训练总预算500,1000,2000,5000,10000,20000,30000。
  每类至少一个样本，按原训练池比例缺额分配余下名额；精确预算、嵌套、不重复采样。
- 数据不足直接报错，不静默删类或缩小预算。地址不交叉，不声称实体不交叉。

## 参数

main中的defaults集中管理实验类型、模型、分箱、种子、训练规模、测试规模、折数和并行数。
命令行只覆盖显式同名参数。当前分箱网格5,10,20,30,50，种子11,37,53；默认uniform。
`--voting uniform nonzero support` 可在一次规则提取后比较三种投票。
八模型均接收相同二值属性，编码只拟合当前训练子集。新SVM不额外标准化；原入口行为不变。
模型固定参数见配置；缺少依赖时正式运行报错，dry-run仅检查数据和任务，不检查可选模型依赖。

## 输出与恢复

同目录下 babd_balanced_results.xlsx、babd_imbalanced_results.xlsx，分别索引。
共享检查点目录，任务ID绑定实验、索引、预算、数据哈希、模型及源文件哈希。
逐折保留train_budget、train_per_class、实际训练对象数、二值属性数、训练/预处理/预测时间、
分类指标、覆盖率、候选集合指标及逐规则审计。不同训练预算独立汇总。
均衡实验每种子一个留出测试集；不均衡实验种子内合并互不交叉测试折，再跨种子求均值和标准差。
fit_seconds仅模型拟合；train_seconds包含训练预处理；predict_seconds不含测试编码和规则审计。
这些是已实现的公共指标；ROC-AUC、峰值内存、特征稳定性与新的消融网格尚未加入本入口。

## 命令（D30工程根目录，fca环境）

```bash
PYTHONPATH=src python -m oeo_lab.experiments.classification.babd_experiment --dry-run --models rc-oeo --max-bins 5 --seeds 11 --per-class 50 --train-sizes 500
PYTHONPATH=src python -m oeo_lab.experiments.classification.babd_experiment --experiments balanced --models rc-oeo random_forest --max-bins 5 --seeds 11 --per-class 50 --n-jobs 1
```

第二条会实际训练；当前仅验证小型夹具及真实数据任务矩阵，不代表正式实验完成。
TabPFN和Elliptic独立入口仍待分别核对基准版本、数据协议后实现。
