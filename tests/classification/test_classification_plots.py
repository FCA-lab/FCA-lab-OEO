"""绘图统计口径：不完整结果与混合参数保护。"""
import sys
from pathlib import Path
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'analysis/classification'))
from result_data import curve_data, select_results


def test_incomplete_seeds_rejected(tmp_path):
    p=tmp_path/'results.xlsx'
    pd.DataFrame([dict(dataset='x',seed=11,complete=True)]).to_excel(p,sheet_name='per_seed',index=False)
    with pytest.raises(ValueError,match='Incomplete'):select_results(p,{},[11,37])


def test_no_silent_parameter_pooling():
    f=pd.DataFrame(dict(max_bins=[5,5],voting=['uniform','uniform'],model=['a','b'],score=[.5,.9]))
    with pytest.raises(ValueError,match='model'):curve_data(f,'max_bins','score','voting')
