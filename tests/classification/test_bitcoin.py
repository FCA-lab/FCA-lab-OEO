"""Bitcoin 特有的数据约束和任务划分测试。"""
import hashlib
import pandas as pd
import pytest
from oeo_lab.experiments.classification import bitcoin_experiment as exp
from oeo_lab.experiments.classification.bitcoin_data import load_bitcoin_dataset


def fixture_spec(tmp_path):
    rows=[{'cate_id':c,'label_id':f'{c}-{g}','address':f'{c}-{g}-{a}',
           'value':float(g+a),'label_cnt':999} for c in [1,2,3,4,5] for g in range(4) for a in range(2)]
    p=tmp_path/'data.csv';pd.DataFrame(rows).to_csv(p,index=False)
    return {'path':str(p),'kind':'bitcoin','decision_column':'cate_id','group_column':'label_id',
            'address_column':'address','class_ids':[1,2,3,4],'feature_columns':['value'],
            'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}


def test_data_and_shared_splits(tmp_path,monkeypatch):
    spec=fixture_spec(tmp_path)
    X,y,labels,groups=load_bitcoin_dataset(spec)
    assert len(X)==32 and len(set(groups))==16 and list(X.columns)==['value']
    assert labels==[1,2,3,4] and set(y)=={0,1,2,3}
    monkeypatch.setattr(exp.config,'DATASET',spec)
    args=exp.parse_args(['--models','dummy','rc-oeo','--max-bins','10','20','--folds','2','--seeds','11','--fractions','1'])
    tasks=exp.build_tasks(args,{})
    assert len(tasks)==12
    for protocol in ['address','entity']:
        for fold in [1,2]:
            subset=[t for t in tasks if t['protocol']==protocol and t['split']['fold']==fold]
            assert all(t['split']==subset[0]['split'] for t in subset)
            if protocol=='entity':
                split=subset[0]['split']
                assert not set(groups[split['train_indices']]) & set(groups[split['test_indices']])
                assert split['entity_overlap']==0


@pytest.mark.parametrize('issue',['duplicate','mixed_entity','leaking_feature'])
def test_invalid_data_rejected(tmp_path,issue):
    spec=fixture_spec(tmp_path)
    if issue=='leaking_feature':spec['feature_columns']=['label_cnt']
    else:
        frame=pd.read_csv(spec['path'])
        if issue=='duplicate':frame.loc[1,'address']=frame.loc[0,'address']
        else:frame.loc[8,'label_id']=frame.loc[0,'label_id']
        frame.to_csv(spec['path'],index=False)
        spec['sha256']=hashlib.sha256(open(spec['path'],'rb').read()).hexdigest()
    with pytest.raises(ValueError):load_bitcoin_dataset(spec)


def test_nested_training_subsets():
    import numpy as np
    y=np.repeat([0,1],10); groups=np.arange(20)
    split={'seed':11,'fold':1,'train_indices':list(range(8))+list(range(10,18)),
           'test_indices':[8,9,18,19]}
    rows=list(exp.training_subsets(split,y,groups,[0.2,0.4,1.0]))
    previous=set()
    for fraction,current,count in rows:
        selected=set(current['train_indices'])
        assert previous <= selected
        assert current['test_indices']==split['test_indices']
        assert not selected & set(split['test_indices'])
        assert set(y[current['train_indices']])=={0,1}
        previous=selected
    assert previous==set(split['train_indices'])
    assert rows==list(exp.training_subsets(split,y,groups,[0.2,0.4,1.0]))


@pytest.mark.parametrize('values',[['0'],['1.1'],['nan'],['0.2','0.2']])
def test_invalid_fractions(values):
    with pytest.raises(SystemExit):exp.parse_args(['--fractions',*values])
