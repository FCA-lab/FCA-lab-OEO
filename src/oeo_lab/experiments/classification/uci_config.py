"""UCI 通用分类实验配置；与 Bitcoin 和完备性实验相互独立。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path(os.environ.get('OEO_DATA_ROOT', PROJECT_ROOT / 'data'))
DATASETS = {
    'iris': {'path': 'raw/uci/iris/iris.data', 'decision_column': 4, 'id_columns': []},
    'wine': {'path': 'raw/uci/wine/wine.data', 'decision_column': 0, 'id_columns': []},
    'wdbc': {'path': 'raw/uci/wdbc/wdbc.data', 'decision_column': 1, 'id_columns': [0]},
}
# 首轮默认参数；各分箱设置完整报告，不根据外层测试集选择最优设置。
MAX_BINS = [10, 20, 30]
MIN_DATA_IN_BIN = 3
FOLDS = 5
SEEDS = [11, 37, 53]
VOTING = ['uniform', 'nonzero', 'support']
# 同一模型的 raw/context 版本作为不同实验项；所有参数均在此登记。
MODELS = {
    'dummy': {'estimator': 'dummy', 'input': 'raw', 'params': {'strategy': 'most_frequent'}},
    'decision_tree': {'estimator': 'decision_tree', 'input': 'context', 'params': {}},
    'random_forest': {'estimator': 'random_forest', 'input': 'raw', 'params': {'n_estimators': 100}},
    'random_forest_context': {'estimator': 'random_forest', 'input': 'context', 'params': {'n_estimators': 100}},
    'xgboost': {'estimator': 'xgboost', 'input': 'raw', 'params': {'n_estimators': 300, 'tree_method': 'hist', 'eval_metric': 'mlogloss'}},
    'xgboost_context': {'estimator': 'xgboost', 'input': 'context', 'params': {'n_estimators': 300, 'tree_method': 'hist', 'eval_metric': 'mlogloss'}},
    'svm_rbf': {'estimator': 'svm_rbf', 'input': 'raw', 'params': {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}},
    'rc-oeo': {'estimator': 'oeo', 'input': 'context', 'params': {}},
}
EXECUTION = {'n_jobs': 1, 'min_free_gb': 2.0, 'worker_ramp_sec': 0.2,
             'memory_wait_timeout_sec': 300, 'task_timeout_sec': None, 'resume': True}
OUTPUT = {'root': str(PROJECT_ROOT / 'results/classification'),
          'results': 'uci_results.xlsx', 'checkpoint_dir': 'checkpoints',
          'index': 'uci_index.json', 'template': '{task_id}.json'}
PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'threadpoolctl', 'openpyxl']
SOURCE_PATTERNS = ['src/fca_core/*.py', 'src/oeo_lab/algorithms/construction/*.py',
    'src/oeo_lab/algorithms/classification/classifier.py', 'src/oeo_lab/preprocessing/encoding.py',
    'src/oeo_lab/experiments/common/*.py',
    *['src/oeo_lab/experiments/classification/' + f for f in
      ('uci_experiment.py', 'datasets.py', 'splits.py', 'baselines.py', 'evaluate.py', 'runner.py')]]
