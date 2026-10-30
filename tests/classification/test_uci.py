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


def test_all_models_same_split_and_raw_not_repeated(synthetic_uci_data):
    args=u.parse_args(['--experiments','comparison','--datasets','iris','--seeds','11','--folds','3','--max-bins','3','5'])
    tasks=u.build_tasks(args,{'test':True})
    assert len(tasks)==36  # 3 folds * (4 raw + 4 context * 2 bins)
    for fold in [1,2,3]:
        rows=[t for t in tasks if t['split']['fold']==fold]
        assert len({tuple(t['split']['test_indices']) for t in rows})==1
        assert sum(t['model_name']=='random_forest' for t in rows)==1
        assert sum(t['model_name']=='rc-oeo' for t in rows)==2


def test_three_studies_share_fit_and_prediction(tmp_path, synthetic_uci_data):
    """三类实验的共同组合仅训练/预测一次，并保留各自报表归属。"""
    args=u.parse_args(['--datasets','iris','--models','rc-oeo','--seeds','11',
                      '--folds','3','--max-bins','3','--ablation-bins','3'])
    tasks=u.build_tasks(args,{'test':True})
    assert len(tasks)==3
    task=tasks[0]
    assert len(task['prediction_variants'])==9  # 3 voting + 6 additional masks
    shared=next(v for v in task['prediction_variants'] if v['voting']=='uniform' and v['rule_group']=='all')
    assert set(shared['studies'])=={'comparison','sensitivity','ablation'}
    original=baselines.create_model
    models=[]
    def factory(spec,seed):
        from unittest.mock import Mock
        m=original(spec,seed)
        m.fit=Mock(wraps=m.fit)
        m.predict=Mock(wraps=m.predict)
        models.append(m)
        return m
    with patch.object(baselines,'create_model',factory):
        result=e.evaluate_fold(task)
    assert models[0].fit.call_count==1
    assert models[0].predict.call_count==9
    assert len(result['results'])==11
    for row in result['results']:
        assert row['defer_no_evidence']+row['defer_tie']==row['n_deferred']
        assert 1 <= row['mean_candidate_size'] <= 3
        assert row['candidate_hit_rate'] >= row['overall_correct_rate']
        assert row['active_rule_count']==sum(row['rules_'+k] for k in ('A1','A2','B1','B2'))
    support=next(r for r in result['results'] if r['rule_group']=='support_only')
    assert support['rules_B1']==support['rules_B2']==0


def test_candidate_metrics():
    row=e.evaluate_candidates([0,1,2,0],[(0,),(0,1),(0,1,2),(1,2)],3)
    assert row['candidate_hits']==3
    assert row['candidate_size_sum']==8
    assert row['partial_count']==2 and row['partial_hits']==1
    assert row['partial_hit_rate']==.5
    assert e.evaluate_candidates([0],[(0,)],3)['partial_hit_rate'] is None


def test_report_groups_and_pending_folds(tmp_path, synthetic_uci_data):
    from oeo_lab.experiments.classification.runner import summarize_classification
    from oeo_lab.experiments.common.checkpoints import save
    import pandas as pd
    args=u.parse_args(['--datasets','iris','--models','rc-oeo','--seeds','11',
                      '--folds','2','--max-bins','3','--ablation-bins','3'])
    tasks=u.build_tasks(args,{'test':True})
    out=tmp_path/'report.xlsx'
    for task in tasks:
        save(tmp_path/(task['id']+'.json'), {'task':task,'status':'ok','result':e.evaluate_fold(task)})
    assert summarize_classification(tasks,tmp_path,out,'{task_id}.json')
    summary=pd.read_excel(out,sheet_name='summary')
    assert len(summary)==11 and summary.complete.all()
    assert set(summary.study)=={'comparison','sensitivity','ablation'}
    seed=pd.read_excel(out,sheet_name='per_seed')
    assert (seed.n_test==synthetic_uci_data).all()
    assert (seed.candidate_hit_rate==seed.candidate_hits/seed.n_test).all()
    (tmp_path/(tasks[-1]['id']+'.json')).unlink()
    assert not summarize_classification(tasks,tmp_path,out,'{task_id}.json')
    summary=pd.read_excel(out,sheet_name='summary')
    assert len(summary)==11 and not summary.complete.any()
    assert summary.macro_f1_mean.isna().all()
