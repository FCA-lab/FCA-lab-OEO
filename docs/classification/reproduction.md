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

UCI、Bitcoin 与 completeness 入口均按“配置默认值 → 脚本 experiment_args → 终端显式参数”的顺序覆盖。仅指定 --n-jobs 不会清除脚本中的 max_bins 或 models；显式 --max-bins 则替换脚本分箱列表。可追加 --dry-run 核对任务参数而不训练。

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

## 实验章节写作备忘：RC-OEO 的条件性性能优势（2026-10-05）

来源：本次 `results/classification/uci_results.xlsx`，20—100 分箱网格，1,800 个训练任务均成功。以下数值为三个种子的平均总体正确率。结果文件可能被后续实验覆盖，正式写作时应按对应检查点及任务配置复核。

### 可采用的结论

在相同离散二值表示、相同分箱设置和共享训练/测试划分下，RC-OEO 在 Wine 的多个分箱设置以及 WDBC 的部分细粒度分箱设置中取得了高于所选基线的分类表现。其优势与数据集及离散化粒度有关，尚未扩展到与原始数值输入强基线的整体比较。应表述为“条件性的性能优势”，不能写成“没有优势”或“全面领先”。

### 支撑数值

下表 RC-OEO 固定使用均匀投票，不逐行挑选投票策略。单位为百分比。

| 数据集 | max_bins | RC-OEO | 决策树 | RF-context | XGBoost-context |
|---|---:|---:|---:|---:|---:|
| Wine | 40 | 94.57 | 70.97 | 89.14 | 83.90 |
| Wine | 100 | 96.44 | 64.98 | 86.89 | 73.60 |
| WDBC | 100 | 92.44 | 77.62 | 90.63 | 70.06 |

Wine 和 WDBC 在 100 箱时 RC-OEO 的覆盖率均为 100%，上述优势不是剔除延迟样本后计算条件准确率形成的。Wine 在 30—100 箱多个设置下超过同输入三种基线；WDBC 在较细分箱下超过决策树和 XGBoost-context，90、100 箱时也超过 RF-context。Iris 仅在部分设置下 Macro-F1 略高于决策树，没有整体领先。

### 必须保留的边界

- 原始数值输入的强基线仍更好，例如 Wine 原始特征随机森林总体正确率 98.31%，高于 RC-OEO 的 96.44%。同输入比较和完整流程比较分开报告。
- 当前结论是观察到的均值优势，尚未进行统计显著性检验，不能使用“统计显著优于”。
- 展示完整分箱敏感性结果，不凭测试结果选取最有利分箱作为无偏最终性能；若要选择参数，应使用训练数据内部验证。
- 定稿前核查细分箱时基线下降的原因及跨种子波动。不能未经验证将优势归因为某一种理论机制。
- 该结论限于当前 UCI 结果，不能外推至尚待分析的 Bitcoin 实验。

## 实验章节写作定位：理论方法与通用分类潜力（已确认）

论文重点是 OEO 理论方法及其规则提取、分类实现。实验用于检验有效性、通用性和适用边界，不将全面超过所有强分类器作为预设目标。以重复实验的平均表现为主要比较依据，使用“平均表现高于”“取得较好效果”等措辞；不声称未经检验的统计显著性。无需为这一写作定位额外引入显著性检验要求。

### 论述层次

1. 有效性：相同离散二值输入下，RC-OEO 在部分数据集和分箱设置中取得较好的平均分类效果，并超过所选基线。结合直接规则提取的理论性质与效率实验，支撑方法贡献。
2. 通用性潜力：UCI 结果表明该流程能够应用于 Bitcoin 之外的分类任务，为其作为通用分类方法提供初步证据；不能据此宣称对所有分类问题均有效。
3. 改进空间：离散化粒度、特征表示、规则权重与预测机制仍可改进。后续新方法提高精度是研究方向，不作为当前已验证的成果。

### 信息损失的准确表述

数值特征离散化会丢失箱内数值差异；若后续二值编码完整保留区间归属，通常不会再增加这部分信息损失。因此应使用“离散化导致的信息损失”，而不是笼统认定二值编码本身损失大量信息。当前比较中模型及预处理均可能不同，尚不能把与原始数值输入模型的全部性能差距归因于离散化。

### 可用于实验章节的总结段落

在相同离散二值表示下，RC-OEO 在部分数据集和参数设置中取得了优于所选基线的平均分类表现，验证了所提出规则提取与预测流程的有效性。通用数据集上的结果进一步表明，该方法具有应用于其他分类任务的潜力。同时，其性能对离散化粒度具有一定敏感性，为进一步改进特征表示和预测机制提供了方向。

正式写作时，将该段与前述具体数值及完整分箱敏感性结果对应；同输入比较与原始数值输入的完整流程比较分别报告，保留不存在一致优势的结果。

### 完备性实验的实际背景规模

成功任务的检查点 metrics 和 completeness_results.xlsx 的 results 表记录 n_objects（实际对象数）、n_features_raw（去除决策及 ID 列后的原始特征数）、n_attributes（实际二值属性数）、density（二值背景中 1 的数量除以对象数与属性数之积）。空矩阵的 density 留空；历史 objects、attributes 数值列暂保留兼容。规模统计不计入规则提取及构格耗时。旧检查点不自动补写这些字段，修改后新任务按新源码指纹生成。

### 完备性实验数据集维护（2026-10-06）

当前仅使用 Balloons、Lenses、Iris；Wine、WDBC 从完备性实验移除，不影响 UCI 分类实验。`completeness_config.py` 的 DATASETS 是唯一数据集登记清单，默认运行其中全部条目。增删数据集只需修改这一处，不再维护额外名称列表。`--datasets iris` 可临时选择子集，未登记名称会在参数解析阶段明确报错。

数值数据集默认使用 COMPLETENESS['preprocessing']，无需另写数据集名称；需要特殊设置时，在该 DATASETS 条目加入 `"preprocessing": {"max_bins": [5,10], "min_data_in_bin": 3}`。入口/终端 --max-bins 优先于这些默认值。类别型数据不受分箱参数影响。入口现有 max_bins=[5,10,20] 保留，因此默认生成 Balloons、Lenses 各一个任务和 Iris 三个分箱任务，共五个配对任务。

## 分类审查修复（2026-10-06）

- 编码器在训练时移除无训练支持的候选区间，保留已有训练分组，并重新编号。测试取值总归属一个区间，超范围进入首/末箱；只有一个箱时全部归入该箱。模型仍保留可能由补属性产生的零支持规则。
- RC-OEO fit 严格要求二值输入，保存 n_features_in_，布尔输入转换为 int8 后构造补背景。
- 分类及完备性任务身份忽略并行数、超时、内存等待等调度设置，但仍保存这些信息。算法源码、数据、划分和模型参数变化仍使旧检查点失效。此次源码更改意味着不能直接复用修复前结果；不要伪造旧记录为新算法结果。
- 超大规则表自动拆成 rules、rules_2 等，每表不超过一百万条数据。仍需留意汇总时总体内存开销。
- 旧 run.py 拒绝执行并提示新入口，避免误用旧预测/评估流程。

### 修复记录与验证状态（2026-10-06）

本次审查修复已落实，并在对应代码旁增加原因和兼容性注释：

| 修改位置 | 原问题 | 当前行为 |
|---|---|---|
| preprocessing/encoding.py | 未见取值落入无训练样本的候选区间，编码列丢弃后整组全零 | 训练时合并空候选区间；每特征唯一激活，首末箱接收超范围值 |
| algorithms/classification/classifier.py | fit 接受非二值值，1-X 无法正确表达补背景 | 拒绝非二值输入；布尔值转 int8；记录 n_features_in_ |
| experiments/common/checkpoints.py 及三类入口/汇总 | 修改 n_jobs 等调度参数导致检查点失效 | 统一 task_id/same_task 语义，保留算法与数据身份校验，原执行设置仍保存在检查点 |
| experiments/classification/runner.py | 全量规则写单张 Excel 表可能超限 | 每百万条拆分 rules、rules_2 等，原检查点完整保留 |
| experiments/classification/run.py | 旧入口可能误用旧实验流程 | 明确停止并提示新入口 |

验证：42 项分类测试通过，包含新取值区间唯一归属、单箱/正负/零值背景、非二值拒绝、布尔输入、调度变更复用与实验变更失效、规则分表。未启动完整实验。既有性能结论基于修复前版本，实际受影响样本范围尚未量化；后续正式报告应注明版本并以核查后的结果为准。此次源码变化会更新任务指纹，不能将历史检查点伪装为新版结果。代码旁增加注释也可能因整文件哈希导致指纹改变。

## 实验章节写作备忘：直接规则提取的有效性与计算实用性（2026-10-07）

### 证据定位

完备性与效率实验直接对应本文“避免完整概念构造、直接提取目标规则”的方法贡献。与 UCI 分类及 Bitcoin 应用实验共同构成证据链：规则提取正确且完备 → 获取成本低 → 所得规则能够用于实际预测。无需以分类精度全面领先作为这一贡献成立的前提。

### 本次核查的实测记录

以下来自当前完备性检查点；正式写作前复核 comparison、规则差集、独立支持集核查字段及实验版本。

| 数据集/设置 | 背景大小 | 完整概念数 | 直接提取时间（秒） | 完整构格时间（秒） |
|---|---|---:|---:|---:|
| Balloons | 20×8 | 82 | 0.00146322 | 0.00937973 |
| Lenses | 24×9 | 190 | 0.00157711 | 0.05743774 |
| Iris，max_bins=5 | 150×16 | 10,424 | 0.00431653 | 720.07149950 |

对应任务 ID：Balloons 1d781de3a7335ff30847fd02；Lenses fcb7969cd9524cce2d8471ed；Iris 5 箱 4a11b246239ce9a93b7d7e6c。构格时间不包含之后的规则提取与核查，应与 full_total_seconds 区分。

核查时 Iris 10 箱（150×36）、20 箱（150×76）的完整比较尚无成功检查点，两个子进程已运行约 22 小时。另行诊断的直接提取时间约为 0.009483 秒、0.025849 秒，分别得到 57、138 条展开规则。这些是单独诊断值，不是原比较任务完成后的正式结果；进程运行时长也不能冒充已完成的构格耗时。此处为当时快照，后续任务状态可能变化。

### 写作边界

1. 已完成且通过规则差集、概念与独立支持集核查的背景，能够支持直接提取的正确性和完备性；实验不替代一般性理论证明。
2. 避免枚举完整概念集合带来的计算节省，是计算实用性的直接证据；不据此宣称分类精度领先或优于所有其他提取算法。
3. 未完成的完整路线只能作为可扩展性限制的观察，不能标注完备性验证通过，也不能提供精确加速比。
4. 当前计时是单次记录，不能称为重复运行均值。正式效率表应明确计时口径与重复次数；不以极大的单次加速比代替解释。
5. 数据支持的是所测试背景上的结果与进一步应用潜力，不能外推为任意规模背景的性能保证。

### 可用于实验章节的表述（需与最终状态对应）

在已完成并通过对照验证的形式背景上，所提出的方法直接获得了全部目标规则，并避免了完整概念构造带来的大量计算开销。Iris 的实验表明，即使对象数量不大，完整概念构造仍可能产生较高成本，而直接规则提取能够快速完成。对于更细分箱设置，完整路线在核查时仍未完成，单独的直接提取诊断则已完成。这些结果支持了该方法在规则获取方面的有效性与计算实用性，并为进一步应用于实际分类任务提供依据。其预测效果与应用价值由后续通用分类及 Bitcoin 实验分别评价。


### 2026-10-09：完备性实验分阶段保存
直接提取完成即原子保存到配置指定的 {task_id}.direct.json，记录耗时、规则数、规则、类别、背景规模和编码信息。最终任务检查点仍由调度器保存，失败或超时不覆盖直接阶段文件。汇总读取阶段文件，direct_status=ok 不表示整体比较通过。完整路线未完成不填完整耗时和比较结果。工作簿在调度结束或正常中断清理后汇总，不由多个进程并发写入。恢复未完成任务会重跑直接提取并更新阶段文件，不恢复构造内存状态。正在运行的旧进程不热更新。


### COVID-19 Surveillance：仅用于规则完备性实验
补充小规模完整比较案例，不进入 UCI 分类性能实验。原始14行、7个二值类别特征，Categories为决策列；PUS/PIM/PWS分别8/5/1个对象。跳过CSV表头，条件特征完整独热编码为14属性，决策列不进入条件背景。Zhang2025构造复现包含3个类别属性，背景为14×17；本文为14×14，因此概念数不能直接混同。保留Iris已完成与未完成记录，新增案例不能替代Iris未完成设置的完备性结论。

## 2026-10-15：UCI 三类实验与候选集合评价

`uci_experiment.py` 的 `main()` 集中设置实验类型、数据集、模型、
分箱网格、投票策略、消融固定分箱/投票、规则组合、随机种子和折数。
数据路径、模型超参数、规则组合定义、并行/超时和输出文件名保留在
`uci_config.py`。终端仅覆盖显式指定的参数。

三个函数负责各自实验矩阵，随后合并并交给公共调度器运行：

- `model_comparison`：模型对比；RC-OEO 固定投票，使用全部规则。
- `binning_voting_experiment`：RC-OEO 分箱与投票分析，使用全部规则。
- `rule_ablation_experiment`：固定分箱和投票，比较全部、仅支持、仅排除及逐类移除。

当前消融默认 `max_bins=20`、`uniform`，可在 main 修改；不得根据测试结果选择。
原始数值输入模型不随分箱重复运行。`--models` 只控制模型对比组，
另两组始终使用 RC-OEO；单独选择实验使用 `--experiments`。

同一数据集/种子/折/模型/分箱设置合为一个训练任务，编码器只在训练折拟合，
RC-OEO 只提取一次完整规则；不同投票及规则组合复用该模型。
同一预测组合跨实验共享计算，结果通过 `study`、`rule_group` 区分。
每行训练时间为共享训练任务的测量值，不能跨投票、规则组合或实验组相加作为总耗时。
检查点保存完整训练任务的结果，不缓存训练后的模型；已完成任务可以恢复，
中断任务或修改预测组合的任务需要重跑。新任务标识与源码哈希防止复用旧预测规则结果。

结果写入 `results/classification/uci_three_studies.xlsx`，保留旧 `uci_results.xlsx`。
工作表为 folds、per_seed、summary、tasks、rules；不新增逐样本预测文件。
同一种子先合并各折原始计数，再算 OOF 指标，最后对种子求均值/标准差。
不完整组合不报告最终均值。

记录 overall_correct_rate（defer 不算正确）、macro_precision、macro_recall、macro_f1、
coverage、classified_accuracy；defer 对真实类别计入 FN，不作为额外真实类别。
还记录候选集合命中率 candidate_hit_rate、平均大小 mean_candidate_size、
部分缩小比例 partial_reduction_rate（1 < 集合大小 < 类别数），及该子集的
partial_hit_rate。条件分母为零时保存 null/Excel 空白。
普通基线返回单元素集合。候选命中率必须结合集合大小解释。
记录完整规则数、启用规则总数及四类数量，以及预处理、训练、预测时间。
规则数量为按类别和规则类型区分的去重单文字规则数量。

```bash
cd /home/charles/projects/FCA-lab-OEO
conda activate fca
# 先检查任务矩阵，不训练。
PYTHONPATH=src python -m oeo_lab.experiments.classification.uci_experiment --dry-run
# 一次运行三类实验，同一设置共享训练。
PYTHONPATH=src python -m oeo_lab.experiments.classification.uci_experiment
# 也可单独运行一类；跨不同启动的变体计划不保证复用训练。
PYTHONPATH=src python -m oeo_lab.experiments.classification.uci_experiment --experiments ablation
```

## WDBC 二分类及细分箱补充实验分析（2026-10-15）

### 结果来源与完成性

- 原结果：`results/classification/uci_three_studies.xlsx`。
- 补充结果：`results/classification/wdbc_extended_bins/uci_three_studies.xlsx`。
- 补充实验：max_bins=150、200，种子11、37、53，每种子5折。
  RC-OEO、决策树、随机森林、XGBoost均使用相同二值输入。
  模型比较固定uniform；RC-OEO另记录nonzero、support投票。
- 120个训练任务全部成功，210条折结果无重复实验键。
  comparison与sensitivity中的uniform结果复用同一次预测，不能视为独立重复实验。
- 下列数值为先按种子合并五折计数，再跨三个种子平均。
  overall_correct_rate将延迟决策计为未正确；classified_accuracy仅评价已唯一分类样本。

### 同输入基线比较

| max_bins | 模型 | 总体正确率 | Macro-F1 |
|---|---|---:|---:|
| 150 | RC-OEO（uniform） | 91.10% | 90.45% |
| 150 | 决策树 | 76.27% | 73.59% |
| 150 | 随机森林 | 84.65% | 81.71% |
| 150 | XGBoost | 63.39% | 44.56% |
| 200 | RC-OEO（uniform） | 91.10% | 90.44% |
| 200 | 决策树 | 75.69% | 72.75% |
| 200 | 随机森林 | 84.12% | 80.89% |
| 200 | XGBoost | 63.09% | 43.15% |

RC-OEO相对于随机森林的总体正确率优势分别为6.44、6.97个百分点。
可支持：在WDBC的这两个细分箱设置、相同二值输入和本次固定模型参数下，
RC-OEO的总体正确率与Macro-F1均优于所比较的三个基线。
这不是统计显著性结论，也不说明优于原始数值输入模型或各基线调优后的最优性能。

### 分箱敏感性与覆盖率

| max_bins | RC-OEO总体正确率（uniform） | 唯一决策覆盖率 |
|---|---:|---:|
| 100 | 92.15% | 98.65% |
| 150 | 91.10% | 98.95% |
| 200 | 91.10% | 98.95% |

100增至150/200后，覆盖率略增，但正确率下降约1.05个百分点。
补充实验并未提高RC-OEO自身的最好观察精度。
150和200的实际二值属性数分别平均4470和4522.8（按训练折平均），仅增加约1.18%。
参数是上限，受训练样本数、不同取值和最小箱样本数约束；不能按150/200倍数推断实际维度。
两组均值相同不代表逐折预测相同；核查发现正确数、覆盖数和属性数并非逐折完全一致。
可支持：继续细化分箱的收益有限；在已测试细分箱区间内，RC-OEO较基线保持了更稳定的分类表现。
原因解释仍是实验观察及可能机制，不能仅凭维度变化证明因果。

### 投票策略

| max_bins | 投票 | 总体正确率 | 覆盖率 | Macro-F1 |
|---|---|---:|---:|---:|
| 150 | uniform / nonzero | 91.10% | 98.95% | 90.45% |
| 150 | support | 91.27% | 99.53% | 90.37% |
| 200 | uniform / nonzero | 91.10% | 98.95% | 90.44% |
| 200 | support | 91.39% | 99.59% | 90.48% |

支持度加权提高了覆盖率及总体正确率，但Macro-F1没有一致提高。
仅能表述为有限改善，不能宣称一种权重全面优于其他权重。

### 训练与决策成本

`fit_seconds_mean`是每个种子五折训练时间之和，再跨种子平均，不是单折训练时间。
RC-OEO在150/200分箱下分别约3014.96/3139.67秒；随机森林约6.94/7.55秒。
RC-OEO对应predict_seconds_mean约0.588/0.552秒，同样为五折之和的跨种子均值。
拟合耗时不含训练预处理；预测耗时不含测试编码及规则审计。
不能跨投票行累加共享的fit时间；并行环境下的墙钟测量也不等同于独占机器的算法基准。
可支持：这些输入设置下的分类优势伴随明显更高的规则提取成本，不能宣称训练效率占优。

### 二分类与负规则作用：引用已有消融，不新增实验

原WDBC消融固定max_bins=20、uniform：七种组合（all、support_only、exclusion_only、
without_A1、without_A2、without_B1、without_B2）各3种子×5折，共105条，全部完成。
105条结果均与检查点一致，15个训练划分的当前编码边界与保存边界一致，分类器源码一致。
所有组合的汇总指标相同：总体正确率83.95%、Macro-F1 88.83%、覆盖率87.17%、
已分类样本准确率96.31%。汇总相同不直接等于逐样本预测完全相同。
可支持：在该WDBC二分类设置下，仅负规则能够获得与完整规则相同的汇总分类表现，
说明排除证据能够独立承担分类作用；不能据此宣称负规则更优或有独立的增量收益。
互斥且穷尽的二分类中，排除一类等价于支持另一类，可作为解释这一观察的理论背景。
新增150/200实验使用全部规则，不能用于单独归因负规则贡献。
二分类的候选集合只有1或2类，不存在部分缩减区间1<|C|<2；重点报告覆盖率及条件准确率。

### 论文写作安排

1. 在分箱敏感性部分展示20–200完整网格，报告相对优势、自身性能平台及训练代价。
2. 在二分类消融分析引用max_bins=20的既有完整结果，无需新增二分类函数或补跑。
3. 150/200是在观察原结果后开展的扩展敏感性检查，应如实表述为补充实验；
   不作为独立未见测试集验证，不只选择有利设置，也不将观察到的最高测试精度作为无偏选参结果。
4. 可用结论：本文方法在部分高维二值表示下具有竞争力；负规则可独立支持二分类。
   不可扩展为：普遍最优、显著优于所有方法、训练更快或负规则一定更优。
