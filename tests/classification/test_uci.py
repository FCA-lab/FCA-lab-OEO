import hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from oeo_lab.experiments.classification import evaluate as e
from oeo_lab.experiments.classification import baselines, uci_experiment as u


def task_for(path):
    return {'dataset':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'decision_column':2,'id_columns':[]},
        'split':{'seed':11,'fold':1,'train_indices':list(range(20)),'test_indices':list(range(20,24))},
        'model':{'estimator':'oeo','input':'context','params':{}},'max_bins':3,
        'min_data_in_bin':3,'voting':['uniform','nonzero','support']}


def test_training_only_fit_shared_once_and_test_outliers(tmp_path):
    path=tmp_path/'data.csv'
    path.write_text('\n'.join(f'{i+1},{(i%5)+1},{i%2}' for i in range(20))+'\n'+
                    '\n'.join(f'{1000+i},{2000+i},{i%2}' for i in range(4)))
    task=task_for(path)
    real=e.LeakFreeContextEncoder
    fits=[]
    class SpyEncoder(real):
        def fit(self,X,y=None):
            fits.append(X.copy())
            return super().fit(X,y)
    original=baselines.create_model
    models=[]
    def factory(spec,seed):
        m=original(spec,seed)
        from unittest.mock import Mock
        m.fit=Mock(wraps=m.fit)
        models.append(m)
        return m
    with patch.object(e,'LeakFreeContextEncoder',SpyEncoder),patch.object(baselines,'create_model',factory):
        result=e.evaluate_fold(task)
    assert len(fits)==1 and len(fits[0])==20 and fits[0].to_numpy().max()<1000
    assert models[0].fit.call_count==1
    assert len(result['results'])==3
    for row in result['results']:
        assert row['n_test']==4
        assert row['defer_no_evidence']+row['defer_tie']==row['n_deferred']


def test_all_models_same_split_and_raw_not_repeated():
    args=u.parse_args(['--datasets','iris','--seeds','11','--folds','3','--max-bins','3','5'])
    tasks=u.build_tasks(args,{'test':True})
    assert len(tasks)==36  # 3 folds * (4 raw + 4 context * 2 bins)
    for fold in [1,2,3]:
        rows=[t for t in tasks if t['split']['fold']==fold]
        assert len({tuple(t['split']['test_indices']) for t in rows})==1
        assert sum(t['model_name']=='random_forest' for t in rows)==1
        assert sum(t['model_name']=='rc-oeo' for t in rows)==2
