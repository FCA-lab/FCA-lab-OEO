"""绘图统计口径：配对差值、候选计数和不完整结果保护。"""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis/classification'))
from result_data import candidate_data, ablation_data, curve_data, select_results


def test_candidate_counts():
    f=pd.DataFrame([dict(seed=11,n_test=10,n_classified=4,n_correct=3,
        partial_count=5,partial_hits=4,candidate_hits=8)])
    d=candidate_data(f)
    assert d['hit'].tolist()==[.3,.4,.1]
    assert d['miss'].tolist()==[.1,.1,0]
    assert (d.hit+d['miss']).sum()==pytest.approx(1)


def test_ablation_paired_and_missing():
    f=pd.DataFrame([dict(dataset='x',seed=s,rule_group=g,score=v)
        for s,g,v in [(11,'all',.8),(37,'all',.4),(11,'without_A1',.7),(37,'without_A1',.6)]])
    assert ablation_data(f,'score','all').iloc[0,0]==pytest.approx(.05)
    with pytest.raises(ValueError): ablation_data(f.iloc[:-1],'score','all')


def test_incomplete_seeds_rejected(tmp_path):
    p=tmp_path/'results.xlsx'
    pd.DataFrame([dict(dataset='x',seed=11,complete=True)]).to_excel(p,sheet_name='per_seed',index=False)
    with pytest.raises(ValueError,match='Incomplete'):select_results(p,{},[11,37])


def test_no_silent_parameter_pooling():
    f=pd.DataFrame(dict(max_bins=[5,5],voting=['uniform','uniform'],model=['a','b'],score=[.5,.9]))
    with pytest.raises(ValueError,match='model'):curve_data(f,'max_bins','score','voting')
