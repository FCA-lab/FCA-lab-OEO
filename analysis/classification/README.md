# 分类实验绘图

在工程根目录运行：

```bash
conda activate fca
python analysis/classification/plot_all.py
```

默认读取已完成的 UCI 三类实验、BABD balanced 工作簿，输出到
`results/figures/classification/`。可指定 `--uci 文件 --babd 文件 --output 目录 --dpi 300`。

所有论文图设置位于 `plot_all.main()`：数据源、种子、固定分箱、指标、过滤条件、标题、坐标轴名称、字号、尺寸、配色、输出格式和分辨率。
新增论文图只需在 jobs 中添加配置。非均衡 BABD 结果可用新 job 明确指定对应文件、dataset 和 protocol，不与均衡数据混合。

- `figures.py`：性能曲线、候选集合分布、消融差值热图、计算代价四个公共函数。
- `result_data.py`：直接消费实验已有 per_seed 汇总，不重新计算折指标，不修改实验代码。
- `plot_all.py`：组装并导出全部配置的图。

每幅图保存 PDF、PNG 和实际绘图 CSV；manifest.json 记录筛选参数、源文件 SHA256、版本和成功/跳过状态。缺失文件、未完成种子/折或未固定的混合设置会明确报错，入口继续处理其它图，最终以非零退出码提示未全部生成。检查 manifest 后再将图用于论文；不要把旧输出文件当作本轮成功结果。

曲线为种子等权均值，误差线为样本 SD，不是置信区间；种子测试集可能重叠。消融先按种子配对求差，再平均，不根据测试成绩挑选分箱参数。默认 bins=20 为配置值，可在 main 修改。

候选集合图使用现有计数恢复“唯一类别／部分候选／全部类别”，不是逐大小直方图。现有结果未存储逐大小命中数，不能从平均大小推断。公共绘图函数可以接受后续提供的精确大小分布。

时间使用 per_seed 的 fit_seconds 和 predict_seconds（同一种子内各折的总和），不包含模型加载和任务排队时间；各投票共享同一次 fit，不是独立训练。正式解释计时需考虑并行负荷。
