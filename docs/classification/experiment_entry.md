# 统一实验入口

```bash
cd /path/to/FCA-lab-OEO
conda activate fca-oeo
python run_experiment.py
```

菜单支持输入一个或多个编号（空格分隔），或 all：
1 完备性、2 UCI 分类、3 BABD 分类、4 训练规模、5 Elliptic 二分类。

## 默认公共守护模式

```bash
python run_experiment.py elliptic
python run_experiment.py --batch uci elliptic
python run_experiment.py all
```

多组实验依次执行，每组保留原 main 和配置文件中的默认设置。
BABD 包含哪些子实验也以其原设置为准。统一入口不复制守护实现，
调用已安装的公共 experiment-watch。启动后输出任务名称、tmux、状态、
停止命令和日志目录。关闭 SSH 或查看日志的 tmux 不停止后台任务。
不要另行启动同一实验的重复计算；参数不同会产生不同守护实例。

异常退出由公共守护程序按自身重试间隔和次数上限重新启动批次。
恢复时批次从第一组进入，各实验自行复用已有检查点；这不是单独保存
批次进度。某组非零退出后不启动后续组；所有组零退出后守护正常完成。
如需取消自动恢复并停止任务，使用启动输出中的 experiment-watch stop 命令。
已完成的同名实例再次启动仍视为完成；需要重新计算时应明确调整实验设置或管理原实例。

## 直接运行及参数覆盖

```bash
python run_experiment.py elliptic --no-watch
python run_experiment.py babd --experiments imbalanced --n-jobs 30
python run_experiment.py --batch uci elliptic --n-jobs 30
python run_experiment.py all --dry-run
python run_experiment.py elliptic --help
```

--no-watch 仅控制入口，不传给实验、不影响检查点标识。
--help 和 --dry-run 自动直接执行，不注册守护任务。
批量附加参数会原样传给每组实验，只使用各组共同支持的参数；
各组不同的参数在各自 main/config 中设置，或分别启动。
all --dry-run 会读取各组数据并构建任务，数据缺失会报错，非纯帮助命令。

公共守护工具不可用时明确报错，可用 --no-watch。
顺序执行限制的是实验组数；组内并行数和内存保护仍由各实验管理。
原模块命令保持可用，且不经过此默认守护入口。
