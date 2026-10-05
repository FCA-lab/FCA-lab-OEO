# OEO 分类论文实验说明

分类器的规则提取与预测已修改；分类实验的指标、划分和调度流程仍待按下述约定落实。下述设计约定不代表分类实验已完成。


## Rule completeness and efficiency

From the project root in the D30 `fca` environment:

```sh
PYTHONPATH=src python -m oeo_lab.experiments.classification.completeness --dry-run
PYTHONPATH=src python -m oeo_lab.experiments.classification.completeness
```

Settings are in `src/oeo_lab/experiments/classification/completeness_config.py`. Defaults use Balloons, Lenses, Iris, Wine and WDBC, with independent completeness settings. `--datasets balloons lenses`, `--n-jobs`, `--timeout`, and `--results-root` override the relevant settings. Numerical data use the existing zero-bin encoder; --max-bins overrides the configured grid (currently values >= 3). Actual grids and time limits come from the independent configuration and entry-point arguments; do not infer current values from this document. A null timeout means no task time limit; any timeout is incomplete, not a pass. This is an extraction experiment on the complete decision context, not a predictive evaluation or cross-validation experiment.

Categorical columns are encoded separately, with the decision column and ID columns excluded from condition attributes. No object rows are deduplicated. The direct route calls classifier `fit`; the reference route constructs all concepts with 3WOC and extracts rules from every class/non-class eligible extent. Empty-support literals are retained. Rules are expanded and deduplicated by `(type, attribute index, class index)` for comparison. `covered` checks full-route rules are included in direct-route rules; both differences and an independent support-containment oracle are recorded. A pass requires matching sets, agreement with that oracle, and valid nonduplicate generated concepts. This finite experiment is not a mathematical proof of concept-lattice completeness.

Direct time includes estimator creation and fit (background construction, support counts, and compact rule extraction). Full construction time includes background and builder initialization plus start; full extraction time includes concept decoding, class masks, and expanded rule extraction. Full total is the sum of those two times. Direct rule expansion, semantic checks, comparisons and saving are outside timing. Thus the time columns describe the actual routes and their output representations, not identical low-level operations. Basic concept entry counts include repeated extents; unique basic counts also appear separately. All times are seconds; execution settings, input/source hashes and package versions are stored with each task.

The configured `completeness_results.xlsx` has `results` and `differences` sheets. JSON checkpoints retain both rule sets, class/attribute mapping and oracle differences, enabling independent recounting. Rerun the same command to resume; successful computation with a comparison failure remains visible and exits nonzero. Failed/timeout attempts retry; prior records are preserved before retry. Changed input, code or execution settings change task IDs. Do not run two writers concurrently against the same output directory. Tests use temporary output directories and do not leave permanent test results.


### Completeness module structure

`parse_args` validates CLI overrides; `check_environment` calls the shared `common/provenance.py` helper; `build_tasks` identifies inputs and encoding variants; `load_dataset` excludes decision/ID columns and delegates encoding to `preprocess_features`. `extract_direct_rules` and `extract_full_lattice_rules` preserve the two timing scopes. `compare_context` verifies concepts and rules outside those timings. `run_experiment` manages checkpoints and reporting. Chinese docstrings explain these boundaries.

The independent completeness configuration specifies source patterns and package names. Classification config.py is not imported or hashed by completeness. Dataset/preprocessing settings are embedded in tasks, and execution settings in provenance, so unrelated classification setting changes do not invalidate completeness checkpoints. This refactor changes the computation source hash once; existing checkpoints remain on disk but are not reused as the new implementation's results.


## 分类性能实验：已确定的设计（2026-10-05）

### 数据集与实验目的

- Iris、Wine、WDBC：通用分类性能的补充验证。
- Bitcoin：行业分类的主要应用实验，分别采用地址级划分和实体级分组划分。
- 分类实验分别使用 `uci_config.py` 和 `bitcoin_config.py`；完备性实验使用独立的 `completeness_config.py`。两类实验的分箱网格、重复次数和运行设置分别管理。

### 数据划分与预处理

1. 不对训练或测试样本去重。各模型共享相同的划分及测试样本。
2. Bitcoin 地址级协议按地址划分，并尽量保持类别比例；其测试地址可能来自训练中已出现的实体。
3. Bitcoin 实体级协议以可靠的地址—实体映射作为分组依据。同一实体的所有地址必须位于同一折，训练实体与测试实体集合的交集必须为空。实体标识用于分组，不作为条件特征；分类目标仍为行业类别。
4. 实体不是行业类别。例如，交易所 A 与交易所 B 是两个实体，都可以属于“交易所”类别。实体级协议评估未见实体的行业分类能力。
5. 划分前统计每个类别的独立实体数，划分后核查实体交集及各折类别分布。如果无法同时满足实体互斥和类别覆盖，应报告限制并重新确定折数或划分方案，不得静默退回地址级划分。缺失或不可靠的实体标识处理方式仍待确定。
6. 先划分数据，再仅用训练数据拟合分箱、编码及其他预处理。测试数据沿用训练边界和编码。决策类别与原始 ID 不进入条件特征。
7. 分箱数及其他模型参数如需选择，只能在训练数据内部完成；禁止根据外部测试集的表现选择参数。
8. UCI 和 Bitcoin 的折数、重复种子及分箱设置分别由独立配置确定；当前值见各入口说明。

### 延迟分类与评价指标

设测试样本总数为 N，给出类别的样本数为 A，其中正确分类数为 C。

| 指标 | 计算口径 | A=0 时 |
|---|---|---|
| 分类覆盖率 | A/N | 0 |
| 已分类样本准确率 | C/A | NaN |
| 全测试集正确分类比例 | C/N | 0 |
| 全测试集宏平均 F1 | 对真实任务类别分别计算 F1 后等权平均 | 0 |

宏平均 F1 的类别集合由数据集的任务类别定义，不以某个模型实际预测出的类别为准。对每类 n，F1_n=2TP_n/(2TP_n+FP_n+FN_n)；分母为零时取 0。defer 样本计入其真实类别的 FN，不计为任何真实类别的 FP；defer 不作为额外类别参与宏平均。缺类折需显式报告，不能通过缩小宏平均的类别集合掩盖。

普通基线若对全部样本给出类别，其覆盖率为 100%，两个准确率指标相同。可补充延迟率 1-A/N，但不能只报告排除延迟样本后的准确率。保留 N、A、C 等计数，便于复核；后续需明确跨折汇总方式，不将含 NaN 的平均值误报为所有折的结果。

### 方法、时间与规则数量

- 本文方法采用 uniform、nonzero、support 三种投票策略。
- 同一训练折、同一编码设置只提取一次规则，三种策略共享规则及支持计数，分别预测。
- 记录训练时间、预测时间和四类展开并去重后的规则数量。共享的提取时间只测一次，策略的预测时间分别测量。
- 最终基线模型清单、模型参数、预处理是否计入训练时间、重复计时与跨折汇总方式仍待讨论，不能把旧配置的默认值视为最终实验方案。

### 实现与核查状态

本节记录已同意的实验设计。分类评估代码仍须逐项接入并验证，尤其是 defer 计分、实体交集检查、训练集内预处理及三种策略共享规则。正式结果应保留实际配置、划分信息与恢复检查点；文档中的设计与代码实现状态分开记录。


### Bitcoin 类别范围（已确认）

分类性能实验仅纳入原始 `cate_id` 为 1、2、3、4 的样本，第 5 类不纳入。地址级与实体级协议、所有模型均使用同一类别清单 `BITCOIN_CLASS_IDS`。原始文件保留完整数据，加载时过滤；内部标签仍按 `cate_id - 1` 映射为 0—3。当前文件过滤后为 25,866 个地址、285 个不同 label_id；这属于预先确定的实验范围，不根据模型表现选择类别。


## UCI 通用分类实验（新入口）

入口为 `uci_experiment.py`，独立配置为 `uci_config.py`。Bitcoin 的旧入口暂保留，尚未迁移到新任务循环；UCI 不调用其地址加载、四类过滤或实体划分逻辑。

```sh
cd /home/charles/projects/FCA-lab-OEO
conda activate fca
PYTHONPATH=src python -m oeo_lab.experiments.classification.uci_experiment --dry-run
PYTHONPATH=src python -m oeo_lab.experiments.classification.uci_experiment
```

当前实验矩阵：Iris、Wine、WDBC；5 折，种子 11/37/53；max_bins=10/20/30。固定模型参数，不依据外层测试结果挑选最优分箱。可通过 `--datasets`、`--models`、`--max-bins`、`--seeds`、`--folds`、`--n-jobs`、`--timeout`、`--results-root` 覆盖参数。默认串行，实验任务可并行；各模型内部限制为单线程。分箱复用当前零值专用编码器，其实际编码列数随训练折变化；并非每个特征必然形成 max_bins 个有效箱。

模型在配置的 MODELS 中登记：多数类、决策树（二值背景）、随机森林（raw/context）、XGBoost（raw/context）、RBF-SVM（raw，训练内标准化）、OEO（三种投票）。所有模型共享同一折索引；raw 模型每折只训练一次，不随分箱网格重复。OEO 每折每分箱 fit 一次，三种策略共享规则与训练支持计数。依赖缺失报错，不静默删模型。

`datasets.load_numeric_dataset` 排除 ID/类别列并返回原始数值特征与完整标签映射；`splits.make_splits(protocol='stratified')` 生成索引；`evaluate.evaluate_fold` 执行训练与评价；`runner.run_classification_tasks` 调用公共进程调度及检查点；`runner.summarize_classification` 输出报表。控制台即时显示任务开始、完成计算、失败与检查点复用信息。

结果默认位于 `results/classification/uci_results.xlsx`：

- folds：每折/策略的四项指标、计数、耗时及延迟原因。
- per_seed：将一个种子的所有互斥测试折合并，累加 N/A/C 和每类 TP/FP/FN 后计算 OOF 指标。
- summary：对完整种子的 OOF 指标取均值、样本标准差，并记录有效种子数；未完成组不输出部分结果的汇总性能。
- tasks：成功/失败/待完成状态。
- rules：OEO 每条规则的类型、属性、类别、训练支持计数、测试触发数与正确数；零测试触发时正确率为空，不强行计零。

检查点保存划分索引、数据哈希、实际模型参数、环境/源码记录、每类混淆计数、训练分箱边界及规则质量记录，不另存冗余 predictions.csv。uci_index.json 记录当前任务矩阵。修改实验参数或源码产生新任务身份；重新执行相同命令复用成功检查点，失败记录先归档再重试。一个输出文件只反映本次请求的任务矩阵，子集运行需用临时输出路径以免覆盖完整报表。

时间口径：训练分箱及编码为 preprocessing_fit_seconds；模型 fit 为 fit_seconds（SVM 内含训练标准化）；两者之和为 train_seconds。测试编码单列为 preprocessing_test_seconds，模型 predict 单列为 predict_seconds。规则审计和指标计算不计入 fit/predict。三种 OEO 策略行重复引用同一次训练耗时，不应将三行相加当成三次训练。每个种子按折累加时间，种子间汇总；并发设置必须随时间结果一起报告。

全部 defer 时 classified_accuracy 在 JSON 保存为 null、Excel 为空；summary 同时报告 defined_seeds，不能把跳过未定义值的平均误作全部种子的平均。训练特征和标签不去重；所有编码/标准化只拟合训练折。测试规则精度仅用于分析，不用于规则筛选。

### UCI 补充分箱实验（2026-10-05）

模型展示及配置名称统一为 `rc-oeo`（RC-OEO）；内部 `estimator=oeo` 保留以兼容已有实现。当前分箱网格为 `[10,20,30]`（按后续决定移除 3、5；已有历史结果和检查点保留），仍使用 5 折、种子 `[11,37,53]`，固定模型参数，完整报告各分箱结果。

决策树、random_forest_context、xgboost_context 与 rc-oeo 使用相同训练折拟合的离散二值形式背景。random_forest、xgboost 使用原始数值输入，svm_rbf 使用训练折拟合的标准化数值输入；这些属于不同预处理流程的整体比较，不应称为同一形式背景上的比较。所有模型共享划分。

运行：`PYTHONPATH=src python -m oeo_lab.experiments.classification.uci_experiment --n-jobs 8`。既有基线检查点继续复用；模型键改名使原 OEO 任务不直接复用，RC-OEO 各分箱任务重新运行，旧检查点保留。

## Bitcoin 应用分类入口

`bitcoin_config.py` 独立配置数据、类别、特征、模型、分箱、种子和输出；`bitcoin_experiment.py` 生成任务并调用共享调度。`bitcoin_data.py` 审核数据，`evaluate.py` 执行训练折内预处理和预测，`runner.py` 保存检查点与工作簿。

保留类别 1—4、8 个数值特征，排除 label_cnt 和全部标识字段，不去重。缺失标识、重复地址或实体跨类别均报错。实体标识的语义可靠性仍依赖数据来源。address 使用 StratifiedKFold，entity 使用 StratifiedGroupKFold；均共享各模型的折索引。两种协议分别汇总，不混合求均值。默认沿用 Bitcoin 的 250/300/350/400/450/500 分箱、5 折和 11/37/53/79/97/101 种子。当前支持嵌套训练比例，具体抽样方案见下节。

```bash
PYTHONPATH=src python -m oeo_lab.experiments.classification.bitcoin_experiment --dry-run
PYTHONPATH=src python -m oeo_lab.experiments.classification.bitcoin_experiment --n-jobs 8
```

结果：`results/classification/bitcoin_results.xlsx`，检查点索引 `checkpoints/bitcoin_index.json`。工作簿沿用 folds/per_seed/summary/tasks/rules；protocol 区分协议，entity_overlap 记录实体交叉审计。中断后同参数重跑恢复。共享评估代码改动会改变源码指纹，因此历史 UCI 工作簿保留，但新 UCI 运行不会冒用旧代码检查点。小规模验证应使用临时 --results-root，避免覆盖正式报表。

### 入口参数优先级

UCI 与 Bitcoin 入口按“配置默认值 → 脚本 experiment_args → 终端显式参数”的顺序覆盖。仅指定 --n-jobs 不会清除脚本中的 max_bins 或 models；显式 --max-bins 则替换脚本分箱列表。可追加 --dry-run 核对任务参数而不训练。

## Bitcoin 嵌套训练比例：已实现

训练比例列表为 `fractions = [0.2, 0.4, 0.6, 0.8, 1.0]`，放在 `bitcoin_experiment.py` 的 `if __name__ == '__main__':` 入口中，并允许终端 `--fractions` 覆盖。参数解析、嵌套子集生成、任务身份和分比例汇总均已接入。

### 含义与目的

fraction 表示当前训练折中的地址抽样比例，不是整个数据集比例，也不是实体比例。对同一外层划分，20% 训练集包含于 40% 训练集，依次包含于 60%、80%、100% 训练集。扩大比例时保留已有样本并增加新样本，减少独立抽样造成的样本替换影响，便于研究训练数据量对分类效果、规则数量及成本的影响；不保证性能单调提高。

### 执行顺序

1. 过滤到类别 1—4，生成地址级或实体级的外层交叉验证划分。
2. 固定测试折，所有训练比例使用同一测试折。
3. 在训练折内，对每一类仅随机排列一次；各比例取该类排列的相应前缀，形成嵌套子集。抽样随机流与外层划分随机流分离，固定种子并可复现。整数样本数存在舍入，结果记录实际数量。
4. 同一协议、种子、折和比例下，所有模型及分箱设置共享训练子集。
5. 每个训练子集独立拟合分箱、编码及标准化；测试折只应用对应训练子集确定的变换。不能用完整训练折预先拟合后供小比例复用。
6. 实体级协议先按实体划分，再抽取训练地址。因此训练/测试实体仍互斥，但各比例的训练实体数量可能变化，需要一并记录。
7. 按协议、训练比例、模型、分箱数和投票策略分别汇总；记录 fraction、实际训练地址数和训练实体数。每个比例独立合并各测试折的 OOF 计数，不能跨比例合并测试计数。

### 参数速查

| 参数 | 控制内容 | 与其他参数的关系 |
|---|---|---|
| protocols | 地址级或实体级划分 | 分开报告结果 |
| seeds、folds | 重复次数和交叉验证折数 | 每个种子内每个地址恰好测试一次 |
| fractions | 使用训练折的地址比例 | 测试折固定，训练子集嵌套 |
| max_bins | 分箱数上限 | 只影响形式背景输入模型 |
| models | 参与比较的模型配置 | 同一实验条件共享样本索引 |
| voting | RC-OEO 的三种投票策略 | 同一训练任务提取规则一次，分别预测 |
| n_jobs | 同时运行的任务数 | 不改变实验设计，模型内部单线程 |

使用时主要在入口选择 fractions、max_bins、models、protocols 和 n_jobs；数据路径、模型超参数及输出名称由独立配置文件管理。运行前检查控制台列出的实际参数与任务数。训练比例会使任务数增加：在原有全部比例无关设置不变且采用 5 个比例时，训练任务数为原来的 5 倍。每类样本数向下取整且至少保留一个，100% 恢复完整训练折。
