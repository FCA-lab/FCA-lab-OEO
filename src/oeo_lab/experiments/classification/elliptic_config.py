"""Elliptic 交易实验的路径、字段及模型配置；运行网格见 main。"""
import os
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[4]
DATA_ROOT=Path(os.environ.get('OEO_DATA_ROOT', PROJECT_ROOT/'data'))
DATASET={'kind':'elliptic','features_path':'raw/public/elliptic/elliptic_txs_features.csv',
         'classes_path':'raw/public/elliptic/elliptic_txs_classes.csv'}
# 原文件：第0列交易ID，第1列时间步，第2..94列本地特征，第95..166列聚合特征。
FEATURE_COLUMNS={'local':list(range(2,95)), 'all':list(range(2,167))}
MIN_DATA_IN_BIN=3
MODELS={
 'rc-oeo':{'estimator':'oeo','input':'context','params':{}},
 'random_forest':{'estimator':'random_forest','input':'context','params':{'n_estimators':100}},
 'logistic':{'estimator':'logistic','input':'context','params':{'max_iter':1000}},
 'xgboost':{'estimator':'xgboost','input':'context','params':{'n_estimators':300,'tree_method':'hist','eval_metric':'logloss'}}}
RULE_GROUPS={'all':['A1','A2','B1','B2'],'positive':['A1','A2'],'negative':['B1','B2']}
EXECUTION={'n_jobs':1,'min_free_gb':2.0,'worker_ramp_sec':0.2,
 'memory_wait_timeout_sec':300,'task_timeout_sec':None,'resume':True}
OUTPUT={'root':str(PROJECT_ROOT/'results/classification'),'results':'elliptic_results.xlsx',
 'index':'elliptic_index.json','checkpoint_dir':'checkpoints','template':'{task_id}.json'}
PACKAGES=['numpy','pandas','scikit-learn','threadpoolctl','openpyxl']
SOURCE_PATTERNS=['src/fca_core/*.py','src/oeo_lab/algorithms/construction/*.py',
 'src/oeo_lab/algorithms/classification/classifier.py','src/oeo_lab/preprocessing/encoding.py',
 'src/oeo_lab/experiments/common/*.py', *['src/oeo_lab/experiments/classification/'+f for f in
 ('elliptic_config.py','elliptic_experiment.py','evaluate.py','runner.py','baselines.py','datasets.py')]]
