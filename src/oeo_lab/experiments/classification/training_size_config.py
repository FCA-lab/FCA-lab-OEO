"""训练规模实验：数据登记独立于 UCI，运行网格在 main 中配置。"""
import os
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[4]
DATA_ROOT=Path(os.environ.get('OEO_DATA_ROOT',PROJECT_ROOT/'data'))
# 任意符合公共加载器格式的数值/类别表格均可登记，不限数据来源。
# 同实体重复测量或时间预测不能直接使用本模块的逐样本随机划分。
DATASETS={
    'iris': {'path':'raw/uci/iris/iris.data','decision_column':4,'id_columns':[]},
    'wine': {'path':'raw/uci/wine/wine.data','decision_column':0,'id_columns':[]},
    'wdbc': {'path':'raw/uci/wdbc/wdbc.data','decision_column':1,'id_columns':[0]},
}
MODELS={'rc-oeo':{'estimator':'oeo','input':'context','params':{}}}
MIN_DATA_IN_BIN=3
EXECUTION={'n_jobs':1,'worker_ramp_sec':0.2,'memory_wait_timeout_sec':300,
           'task_timeout_sec':None,'resume':True}
OUTPUT={'root':str(PROJECT_ROOT/'results/classification/training_size'),
        'results':'training_size_results.xlsx','checkpoint_dir':'checkpoints',
        'index':'training_size_index.json','template':'{task_id}.json'}
PACKAGES=['numpy','pandas','scikit-learn','threadpoolctl','openpyxl']
SOURCE_PATTERNS=['src/fca_core/*.py','src/oeo_lab/algorithms/construction/*.py',
    'src/oeo_lab/algorithms/classification/classifier.py',
    'src/oeo_lab/preprocessing/encoding.py','src/oeo_lab/preprocessing/categorical.py',
    'src/oeo_lab/experiments/common/*.py',
    *['src/oeo_lab/experiments/classification/'+f for f in
      ('training_size_config.py','training_size_experiment.py','datasets.py','uci_data.py',
       'baselines.py','evaluate.py','runner.py')]]
