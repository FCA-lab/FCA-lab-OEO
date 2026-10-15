"""BABD 独立配置；实验网格在 babd_experiment.main 中设置。"""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path(os.environ.get('OEO_DATA_ROOT', PROJECT_ROOT / 'data'))
DATASET = {'kind': 'bitcoin', 'path': 'raw/public/babd13/BABD-13-clean.csv',
    'decision_column': 'label', 'group_column': 'account', 'address_column': 'account',
    'group_kind': 'address', 'feature_manifest': 'raw/public/babd13/cleaning_report.json'}
CLASS_IDS = {'balanced': [0,1,2,3,5,6,11], 'imbalanced': [0,1,2,3,4,5,6,9,10,11,12]}
MIN_DATA_IN_BIN = 3
MODELS = {
    'rc-oeo': {'estimator':'oeo','input':'context','params':{}},
    'logistic': {'estimator':'logistic','input':'context','params':{'max_iter':1000}},
    'random_forest': {'estimator':'random_forest','input':'context','params':{'n_estimators':100}},
    'xgboost': {'estimator':'xgboost','input':'context','params':{'n_estimators':300,'tree_method':'hist','eval_metric':'mlogloss'}},
    'catboost': {'estimator':'catboost','input':'context','params':{'iterations':300,'verbose':False,'allow_writing_files':False}},
    'lightgbm': {'estimator':'lightgbm','input':'context','params':{'n_estimators':300,'verbosity':-1}},
    'svm_rbf': {'estimator':'svm_rbf','input':'context','scale':False,'params':{'C':1.0,'gamma':'scale'}},
    'mlp': {'estimator':'mlp','input':'context','params':{'hidden_layer_sizes':(100,),'max_iter':500}},
}
EXECUTION = {'n_jobs':1,'min_free_gb':2.0,'worker_ramp_sec':0.2,
    'memory_wait_timeout_sec':300,'task_timeout_sec':None,'resume':True}
OUTPUT = {'root':str(PROJECT_ROOT/'results/classification'),'checkpoint_dir':'checkpoints',
    'template':'{task_id}.json'}
OUTPUT_FILES = {s:{'results':f'babd_{s}_results.xlsx','index':f'babd_{s}_index.json'}
                for s in CLASS_IDS}
PACKAGES = ['numpy','pandas','scikit-learn','threadpoolctl','openpyxl']
SOURCE_PATTERNS = ['src/fca_core/*.py','src/oeo_lab/algorithms/construction/*.py',
    'src/oeo_lab/algorithms/classification/classifier.py','src/oeo_lab/preprocessing/encoding.py',
    'src/oeo_lab/experiments/common/*.py', *['src/oeo_lab/experiments/classification/'+f for f in
    ('babd_config.py','babd_experiment.py','bitcoin_data.py','datasets.py','splits.py','baselines.py','evaluate.py','runner.py')]]
