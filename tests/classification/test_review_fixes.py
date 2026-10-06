import copy
import numpy as np
import pandas as pd
import pytest
from oeo_lab.preprocessing.encoding import LeakFreeContextEncoder
from oeo_lab.algorithms.classification.classifier import RC_3Woc
from oeo_lab.experiments.common.checkpoints import task_id, completed, save
from oeo_lab.experiments.classification.runner import write_rule_sheets

@pytest.mark.parametrize('values', [[1,2,3,4,5,6],[-6,-5,-4,-3,-2,-1],[-3,-2,-1,1,2,3],[0]*6,[2]*6])
def test_all_test_values_have_one_bin(values):
    e=LeakFreeContextEncoder(['a'],3,3).fit(pd.DataFrame({'a':values}))
    out=e.transform(pd.DataFrame({'a':[-100,0,100]}))
    assert out.sum(axis=1).eq(1).all()
    assert out.iloc[0,0]==1 and out.iloc[-1,-1]==1
    assert e.transform(pd.DataFrame({'a':values})).sum(axis=1).eq(1).all()

def test_nonbinary_training_rejected_and_bool_supported():
    with pytest.raises(ValueError,match='binary'):RC_3Woc().fit([[2,0],[0,1]],[0,1])
    m=RC_3Woc().fit([[True,False],[False,True]],[0,1])
    assert m.n_features_in_==2
    assert list(m.predict([[1,0],[0,1]]))==[0,1]

def test_checkpoint_reuse_ignores_only_scheduling(tmp_path):
    a={'model':{'p':1},'provenance':{'source_hash':'abc','settings':{'n_jobs':10,'task_timeout_sec':None}}}
    a['id']=task_id(a)
    b=copy.deepcopy(a);b['provenance']['settings']['n_jobs']=8;b['id']=task_id(b)
    assert a['id']==b['id']
    p=tmp_path/'record.json';save(p,{'task':a,'status':'ok'})
    assert completed(p,b)
    b['model']['p']=2
    assert not completed(p,b) and task_id(b)!=a['id']
    b=copy.deepcopy(a);b['provenance']['source_hash']='def'
    assert not completed(p,b)

def test_excel_rule_sheet_partition(tmp_path):
    p=tmp_path/'rules.xlsx'
    with pd.ExcelWriter(p,engine='openpyxl') as w:write_rule_sheets(w,[{'a':i} for i in range(5)],2)
    sheets=pd.read_excel(p,sheet_name=None)
    assert list(sheets)==['rules','rules_2','rules_3']
    assert sum(len(s) for s in sheets.values())==5
