import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from oeo_lab.experiments.classification import elliptic_config as config
from oeo_lab.experiments.classification.elliptic_experiment import build_tasks,load_elliptic_dataset,temporal_split
from oeo_lab.experiments.classification.evaluate import evaluate_fold,evaluate_illicit
from oeo_lab.experiments.classification.runner import summarize_classification
from oeo_lab.experiments.common.checkpoints import save


def test_illicit_defer_is_false_negative():
    r=evaluate_illicit([1,1,0,0],[1,None,1,0],[4,3,2,1])
    assert (r['illicit_tp'],r['illicit_fp'],r['illicit_fn'])==(1,1,1)
    assert r['illicit_recall']==.5
    assert r['illicit_average_precision']==1


def test_transaction_pipeline(tmp_path,monkeypatch):
    # 无表头特征文件，特意逆序标签以验证ID连接，unknown不当合法处理。
    n=18
    frame=pd.DataFrame(np.tile(np.arange(167),(n,1)))
    frame[0]=np.arange(100,100+n);frame[1]=[1]*8+[35]*10
    frame[2]=np.arange(n)%2
    frame.to_csv(tmp_path/'features.csv',header=False,index=False)
    labels=pd.DataFrame({'txId':frame[0], 'class':['1','2']*8+['unknown','unknown']})
    labels.iloc[::-1].to_csv(tmp_path/'classes.csv',index=False)
    monkeypatch.setattr(config,'DATA_ROOT',tmp_path)
    monkeypatch.setattr(config,'DATASET',{'kind':'elliptic','features_path':'features.csv','classes_path':'classes.csv'})
    args=SimpleNamespace(feature_set='local',seeds=[11],train_end=34,test_start=35,test_end=49,
        models=['rc-oeo','logistic'],max_bins=[5],voting=['uniform'],rule_groups=['all','positive','negative'])
    tasks=build_tasks(args,{})
    assert len(tasks)==2
    assert tasks[0]['split']==tasks[1]['split']
    X,y,_,time,ids=load_elliptic_dataset(tasks[0]['dataset'])
    assert len(y)==16 and X.shape[1]==93
    assert list(y[:2])==[1,0]
    assert max(time[tasks[0]['split']['train_indices']])<min(time[tasks[0]['split']['test_indices']])
    for task in tasks:
        result=evaluate_fold(task)
        assert len(result['results'])==(3 if task['model_name']=='rc-oeo' else 1)
        assert all(0<=r['illicit_average_precision']<=1 for r in result['results'])
        save(tmp_path/(task['id']+'.json'),{'task':task,'status':'ok','result':result})
    assert summarize_classification(tasks,tmp_path,tmp_path/'results.xlsx','{task_id}.json')
    summary=pd.read_excel(tmp_path/'results.xlsx',sheet_name='summary')
    assert len(summary)==4 and summary.complete.all()
    assert 'illicit_f1_mean' in summary
    bad=dict(tasks[0]['dataset'],feature_columns=[1,2])
    with pytest.raises(ValueError):load_elliptic_dataset(bad)
    with pytest.raises(ValueError):temporal_split(y,time,11,35,34,49)
    labels.loc[0,'class']='unexpected';labels.to_csv(tmp_path/'classes.csv',index=False)
    with pytest.raises(ValueError):load_elliptic_dataset(tasks[0]['dataset'])
