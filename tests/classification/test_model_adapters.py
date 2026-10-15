import warnings
import numpy as np
import pytest
from oeo_lab.experiments.classification.evaluate import normalize_class_predictions

def test_single_column_only():
    assert normalize_class_predictions([[1],[2]],2).tolist()==[1,2]
    with pytest.raises(ValueError):normalize_class_predictions([[.2,.8],[.4,.6]],2)
    with pytest.raises(ValueError):normalize_class_predictions([1],2)


def test_installed_catboost_lightgbm(tmp_path):
    import pandas as pd
    from oeo_lab.experiments.classification.evaluate import evaluate_fold
    from oeo_lab.experiments.classification.elliptic_experiment import file_hash
    X=np.random.default_rng(11).normal(size=(90,4)); y=np.tile([0,1,2],30)
    frame=pd.DataFrame(X);frame[4]=y;p=tmp_path/'data.csv';frame.to_csv(p,index=False,header=False)
    task={'dataset':{'path':str(p),'sha256':file_hash(p),'decision_column':4,'id_columns':[]},
      'split':{'train_indices':list(range(60)),'test_indices':list(range(60,90)),'seed':11},
      'max_bins':5,'min_data_in_bin':3,'voting':['uniform']}
    for name,params in [('catboost',{'iterations':3,'verbose':False,'allow_writing_files':False}),
                        ('lightgbm',{'n_estimators':3,'verbosity':-1})]:
        task['model']={'estimator':name,'input':'context','params':params}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); result=evaluate_fold(task)
        assert result['results'][0]['n_test']==30
        assert not any('valid feature names' in str(w.message) for w in caught)
