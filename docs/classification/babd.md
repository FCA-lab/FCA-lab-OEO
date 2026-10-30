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

The second command runs training. The paper results are complete (840 balanced and 4,200 imbalanced tasks); see [the artifact map](paper_artifacts.md). The dated recovery notes below describe the 2026-10-15 state, not current progress. TabPFN is exploratory and is not required by either paper. For raw input preparation, see [the BABD preparation guide](babd_preparation.md).

## 2026-10-15 模型接口修复与恢复审核

- CatBoost多分类predict可返回(n,1)列。公共评价在模型输出处仅压掉单列维度，仍拒绝多列概率矩阵或长度错误。
- LightGBM训练/预测显式使用相同feature_i列名和顺序；未改变二值特征值。
- 工作进程捕获任务异常后保存failed并以1退出，控制台同时显示检查点status及退出码。
- 日志增加train预算，不再把BABD的所有预算误显示成只有fraction=1.0。
- 22项相关测试通过，含实际CatBoost和LightGBM；LightGBM数组/具名输入的标签和概率等价检查通过。
- 中断时840项：99成功、15失败、726未完成。一次性兼容审核核对原任务、环境、包版本和源码指纹，
  仅允许本次评价接口/日志修复，迁移99个成功结果为新ID检查点副本；原文件不改。
  新文件reuse_audit保留原任务、原检查点哈希及复制理由，耗时为原测量，未冒充重新运行。
  审核索引：results/classification/checkpoints/babd_adapter_reuse_audit.json。
- 使用相同数据、网格、模型和环境恢复时，99项可复用；失败和未完成共741项重跑。
  再修改源码/环境/实验参数仍可能使检查点失效，不全局放宽缓存校验。
