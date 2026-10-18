"""规则完备性与效率实验的独立配置；不依赖分类性能实验配置。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
RESULTS_ROOT = str(PROJECT_ROOT / 'results' / 'classification')
# 增删数据集只修改 DATASETS；默认运行其中所有条目。
# 列索引从 0 开始；决策列和 ID 列不进入条件背景。
# 数值数据集可在条目内添加 preprocessing 覆盖公共预处理默认值。
DATASETS = {
    "balloons": {"path": "raw/uci/balloons/adult+stretch.data", "format": "csv",
                 "decision_column": 4, "id_columns": [], "feature_kind": "categorical"},
    "lenses": {"path": "raw/uci/lenses/lenses.data", "format": "whitespace",
               "decision_column": 5, "id_columns": [0], "feature_kind": "categorical"},
    "covid19": {"path": "raw/uci/covid19_surveillance/Surveillance.csv", "format": "csv",
                "header": True, "decision_column": 7, "id_columns": [],
                "feature_kind": "categorical"},
    "iris": {"path": "raw/uci/iris/iris.data", "format": "csv",
             "decision_column": 4, "id_columns": [], "feature_kind": "numerical"}
}

COMPLETENESS = {
    "preprocessing": {"max_bins": [3], "min_data_in_bin": 3},
    "repeats": 1,
    "execution": {"n_jobs": 1, "task_timeout_sec": None},
    "routes": ["direct", "full_lattice"],
    "retain_zero_support": True,
    "deduplicate_rules": True,
    "rule_key": ["type", "attribute", "class"],
}

EXECUTION = {
    "n_jobs": 1, "tasks_per_worker": 1, "resume": True,
    "keep_partial": True, "worker_ramp_sec": 0.2,
    "memory_wait_timeout_sec": 300, "task_timeout_sec": None,
}

OUTPUT = {
    "root": RESULTS_ROOT,
    "files": {"completeness": "completeness_results.xlsx"},
    "checkpoint_dir": "checkpoints", "checkpoint_index": "index.json",
    "checkpoint_template": "{task_id}.json",
    "direct_checkpoint_template": "{task_id}.direct.json",  # 直接提取完成即保存。
}

# 运行记录范围：算法及运行代码变化会生成新任务标识。
SOURCE_PATTERNS = [
    'src/fca_core/*.py',
    'src/oeo_lab/algorithms/construction/*.py',
    'src/oeo_lab/algorithms/classification/classifier.py',
    'src/oeo_lab/preprocessing/encoding.py',
    'src/oeo_lab/preprocessing/categorical.py',
    'src/oeo_lab/experiments/classification/completeness.py',
    'src/oeo_lab/experiments/common/*.py',
]
PROVENANCE_PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'threadpoolctl', 'openpyxl']


def dataset_path(name):
    """允许通过 OEO_DATA_ROOT 指定数据根目录。"""
    root = Path(os.environ.get('OEO_DATA_ROOT', PROJECT_ROOT / 'data'))
    return root / DATASETS[name]['path']
