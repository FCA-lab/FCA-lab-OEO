import numpy as np
from oeo_lab.experiments.classification.tabpfn_experiment import nested_splits

def test_fixed_test_nested_train_and_explicit_skips():
    y=np.repeat([0,1,2],20)
    splits,skipped=nested_splits(y,[11,37],[2,5,15],0.3)
    assert len(splits)==4 and len(skipped)==2
    for seed in [11,37]:
        items=[(b,s) for b,s in splits if s['seed']==seed]
        small,large=items[0][1],items[1][1]
        assert small['test_indices']==large['test_indices']
        assert set(small['train_indices'])<set(large['train_indices'])
        assert not set(large['train_indices'])&set(large['test_indices'])
        assert np.bincount(y[large['train_indices']]).tolist()==[5,5,5]
    assert nested_splits(y,[11,37],[2,5,15],0.3)==(splits,skipped)
