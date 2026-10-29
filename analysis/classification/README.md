# 论文实验绘图

本模块仅保留最终论文引用的六幅实验图，共用一个入口，不重新训练模型。

| 图名（PDF、PNG、CSV 共用文件名） | 绘图函数 |
|---|---|
| `uci_model_comparison_grid` | `plot_comparison_grid` |
| `uci_rule_ablation_combined` | `plot_ablation_combined` |
| `uci_voting_model_comparison` | `plot_voting_comparison` |
| `uci_candidate_set_sizes` | `plot_candidate_sizes` |
| `babd_behavior_grid` | `plot_babd_behavior` |
| `babd_runtime_comparison` | `plot_babd_timing` |

在工程根目录执行 `python analysis/classification/plot_all.py`，默认生成以上六幅图。可用 `--only 图名 ...` 选择子集，`--uci`、`--babd`、`--babd-imbalanced` 指定工作簿，`--output` 指定输出目录，`--dpi` 设置分辨率。默认输出位置为 `results/figures/classification/`；同名文件会被覆盖。

- `plot_all.py`：配置数据筛选、种子、图形及输出，生成运行清单。
- `result_data.py`：读取已有 `per_seed` 结果，检查完整性并整理绘图数据。
- `figures.py`：六个绘图函数，不读取实验工作簿。

均值与样本标准差按随机种子计算，误差棒不是置信区间。规则消融展示各配置的准确率与覆盖率，不展示与完整规则集的差值。候选集图使用已记录的候选计数；BABD 的候选集大小和真实类别保留率统计全部测试对象。时间图排除预处理，拟合时间按折平均（如适用），预测时间按每 1,000 个测试对象归一化。

每幅图导出 PDF、PNG 和同名 CSV；完整运行保存 `manifest.json`，子集运行保存 `manifest_selected.json`。检查运行清单中的成功/跳过状态，不把旧文件误认成本次生成结果。详细参数及版本对应见[复现说明](../../docs/classification/reproduction.md)与[论文对应清单](../../docs/classification/paper_artifacts.md)。

未被论文采用的试选图入口和专用函数已移除，原实现可在 Git 提交 `1757a97` 中查看。既有图片、CSV、实验结果及检查点未删除。论文中的概念结构示意图不由本模块生成。
