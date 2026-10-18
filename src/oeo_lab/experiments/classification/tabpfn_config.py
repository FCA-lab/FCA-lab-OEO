"""独立小样本实验配置；运行网格在 main 中设置。"""
import os
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path(os.environ.get('OEO_DATA_ROOT', PROJECT_ROOT/'data'))
# 可直接增删数据集；数值 CSV 无表头，ID 和标签不作为特征。
DATASETS = {
 'iris': {'path':'raw/uci/iris/iris.data','decision_column':4,'id_columns':[]},
 'wine': {'path':'raw/uci/wine/wine.data','decision_column':0,'id_columns':[]},
 'wdbc': {'path':'raw/uci/wdbc/wdbc.data','decision_column':1,'id_columns':[0]},
}
MODELS = {
 'rc-oeo': {'estimator':'oeo','input':'context','params':{}},
 'tabpfn_v2': {'estimator':'tabpfn_v2','input':'context','params':{'device':'cpu','n_estimators':8}},
 'random_forest': {'estimator':'random_forest','input':'context','params':{'n_estimators':100}},
 'xgboost': {'estimator':'xgboost','input':'context','params':{'n_estimators':300,'tree_method':'hist','eval_metric':'mlogloss'}},
 'catboost': {'estimator':'catboost','input':'context','params':{'iterations':300,'verbose':False,'allow_writing_files':False}},
}
# 使用显式本地 v2 权重；禁止自动下载默认的新模型。运行时记录内容哈希。
MODEL_PATH = Path(os.environ.get('OEO_TABPFN_MODEL', PROJECT_ROOT/'artifacts/tabpfn/tabpfn-v2-classifier.ckpt'))
MIN_DATA_IN_BIN = 3
EXECUTION = {'n_jobs':1,'worker_ramp_sec':0.2,
 'memory_wait_timeout_sec':300,'task_timeout_sec':None,'resume':True}
OUTPUT = {'root':str(PROJECT_ROOT/'results/classification'), 'results':'tabpfn_results.xlsx',
 'checkpoint_dir':'checkpoints','index':'tabpfn_index.json','template':'{task_id}.json',
 'skipped':'tabpfn_skipped.json'}
PACKAGES = ['numpy','pandas','scikit-learn','threadpoolctl','openpyxl']
SOURCE_PATTERNS = ['src/fca_core/*.py','src/oeo_lab/algorithms/construction/*.py',
 'src/oeo_lab/algorithms/classification/classifier.py','src/oeo_lab/preprocessing/encoding.py',
 'src/oeo_lab/experiments/common/*.py', *['src/oeo_lab/experiments/classification/'+f for f in
 ('tabpfn_config.py','tabpfn_experiment.py','tabpfn_adapter.py','datasets.py','baselines.py','evaluate.py','runner.py')]]
