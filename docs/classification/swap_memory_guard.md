# 公共任务调度与 RAM + swap 保护（2026-10-18）

所有调用 `oeo_lab.experiments.common.runner.run_tasks` 的实验共用
`common/runner.py` 的任务分配和 `common/resource_guard.py` 的资源判断。
阈值在 `common/scheduling_config.py`：预留 10 GiB，暂停后增加 5 GiB 恢复余量。
实验配置仅保留任务网格、并行上限、启动间隔和超时等自身参数。

有效余量 = MemAvailable − (SwapTotal − SwapFree)。首次准入至少 10 GiB；
因内存不足暂停后，至少 15 GiB 才恢复新任务分配。
这是一种保守准入预算，swap 和 RAM 中可能包含同一页；不是精确物理用量。
达到阈值只停止补充工作进程，现有工作进程继续运行，因此不能保证运行中绝不发生 OOM。
内存信息不可读时暂停准入；不自动执行 swapoff 或清理缓存。

科学任务指纹继续包含数据、划分、模型及计算源码；公共调度文件及资源参数
不参与计算身份。历史源码版本仅根据已核验的精确哈希映射复用。
构造实验的原有签名只包含计算模块及并行数，不包含公共调度源码。

运行中的 Python 父进程不会热更新；下次启动或守护自动恢复使用共享策略。
