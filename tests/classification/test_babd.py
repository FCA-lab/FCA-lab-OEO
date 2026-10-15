import numpy as np
import pytest
from oeo_lab.experiments.classification.babd_experiment import balanced_splits, imbalanced_splits

@pytest.mark.parametrize('balanced',[True,False])
def test_nested_fixed_disjoint(balanced):
    y=np.repeat(np.arange(3),[30,40,50]);groups=np.arange(len(y)).astype(str)
    def run():
        return list(balanced_splits(y,groups,[11],[2,5,10],5) if balanced else
                    imbalanced_splits(y,groups,[11],[6,15,30],3))
    rows=run()
    assert rows==run()
    previous={}
    for budget,per_class,s in rows:
        tr=set(s['train_indices']);te=set(s['test_indices']);key=(s['seed'],s['fold'])
        assert len(tr)==budget and not tr&te
        assert all(c['train'] and c['test'] for c in s['class_counts'])
        if balanced:assert all(c['train']==per_class and c['test']==5 for c in s['class_counts'])
        if key in previous:assert previous[key][0]<=tr and previous[key][1]==te
        previous[key]=(tr,te)


def test_insufficient_budget_fails():
    y=np.repeat(np.arange(3),10);g=np.arange(30)
    with pytest.raises(ValueError):list(balanced_splits(y,g,[11],[9],3))
    with pytest.raises(ValueError):list(imbalanced_splits(y,g,[11],[2],2))


def test_end_to_end_both_reports(tmp_path, monkeypatch):
    import json
    import pandas as pd
    from types import SimpleNamespace
    from oeo_lab.experiments.classification import babd_config as config
    from oeo_lab.experiments.classification.babd_experiment import build_tasks
    from oeo_lab.experiments.classification.evaluate import evaluate_fold
    from oeo_lab.experiments.classification.runner import summarize_classification
    from oeo_lab.experiments.common.checkpoints import save
    frame=pd.DataFrame({'account':[f'a{i}' for i in range(60)],
                        'label':np.repeat([0,1],30),'x':np.tile(np.arange(30),2)})
    frame.to_csv(tmp_path/'data.csv',index=False)
    (tmp_path/'features.json').write_text(json.dumps({'features':['x']}))
    monkeypatch.setattr(config,'DATA_ROOT',tmp_path)
    monkeypatch.setattr(config,'DATASET',dict(config.DATASET,path='data.csv',feature_manifest='features.json'))
    monkeypatch.setattr(config,'CLASS_IDS',{'balanced':[0,1],'imbalanced':[0,1]})
    args=SimpleNamespace(seeds=[11],per_class=[3,5],test_per_class=5,
        train_sizes=[6,10],folds=2,models=['rc-oeo','logistic'],max_bins=[5],voting=['uniform','support'])
    for experiment in ('balanced','imbalanced'):
        tasks=build_tasks(args,experiment,{})
        assert len({t['id'] for t in tasks})==len(tasks)
        for task in tasks:
            result=evaluate_fold(task)
            assert result['n_train']==task['train_budget']
            assert result['fit_seconds']>=0
            save(tmp_path/(task['id']+'.json'),{'task':task,'status':'ok','result':result})
        output=tmp_path/(experiment+'.xlsx')
        assert summarize_classification(tasks,tmp_path,output,'{task_id}.json')
        summary=pd.read_excel(output,sheet_name='summary')
        assert set(summary.train_budget)=={6,10}
        assert summary.complete.all()
        assert set(summary.expected_seeds)=={1}
        folds=pd.read_excel(output,sheet_name='folds')
        assert (folds.n_train==folds.train_budget).all()
