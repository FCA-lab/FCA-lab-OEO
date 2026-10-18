"""Bitcoin 应用分类配置；地址级和实体级协议独立汇总。"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path(os.environ.get('OEO_DATA_ROOT', PROJECT_ROOT / 'data'))
DATASET = {'kind': 'bitcoin', 'path': 'raw/bitcoin/getFeatures.csv',
    'decision_column': 'cate_id', 'group_column': 'label_id', 'address_column': 'address',
    'class_ids': [1, 2, 3, 4],
    'feature_columns': ['avgUseGap', 'balance', 'total_received', 'received_cnt',
                        'avg_receive', 'total_sent', 'sent_cnt', 'avg_sent']}
PROTOCOLS = ['address', 'entity']
MAX_BINS = [250, 300, 350, 400, 450, 500]
MIN_DATA_IN_BIN = 3
FOLDS = 5
SEEDS = [11, 37, 53, 79, 97, 101]
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
EXECUTION = {'n_jobs': 10, 'worker_ramp_sec': 0.2,
             'memory_wait_timeout_sec': 300, 'task_timeout_sec': None, 'resume': True}
OUTPUT = {'root': str(PROJECT_ROOT / 'results/classification'),
          'results': 'bitcoin_results.xlsx', 'checkpoint_dir': 'checkpoints',
          'index': 'bitcoin_index.json', 'template': '{task_id}.json'}
PACKAGES = ['numpy', 'pandas', 'scikit-learn', 'threadpoolctl', 'openpyxl']
SOURCE_PATTERNS = ['src/fca_core/*.py', 'src/oeo_lab/algorithms/construction/*.py',
    'src/oeo_lab/algorithms/classification/classifier.py', 'src/oeo_lab/preprocessing/encoding.py',
    'src/oeo_lab/experiments/common/*.py',
    *['src/oeo_lab/experiments/classification/' + f for f in
      ('bitcoin_experiment.py', 'bitcoin_config.py', 'bitcoin_data.py', 'datasets.py', 'splits.py', 'baselines.py', 'evaluate.py', 'runner.py')]]

# BABD仅提供地址标识，不能把地址当成控制实体并声称实体泛化。
DATASETS = {
    'legacy': DATASET,
    'babd13': {'kind':'bitcoin', 'path':'raw/public/babd13/BABD-13-clean.csv',
        'decision_column':'label', 'group_column':'account', 'address_column':'account',
        'group_kind':'address', 'class_ids':[0,1,2,3,4,5,6,9,10,11,12],
        'feature_manifest':'raw/public/babd13/cleaning_report.json'},
}
