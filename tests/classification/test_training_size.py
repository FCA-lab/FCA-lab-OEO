import numpy as np
import pandas as pd
import pytest
from oeo_lab.experiments.classification import training_size_experiment as e
from oeo_lab.experiments.common.checkpoints import task_id


def test_fixed_holdout_and_nested_prefixes():
    y=np.array([0]*40+[1]*60)
    rows=e.nested_fraction_splits(y,[11,37],[.2,.4,.6,.8,1],.3)
    assert len(rows)==10
    for seed in [11,37]:
        selected=[r for r in rows if r['split']['seed']==seed]
        test=selected[0]['split']['test_indices'];previous=set()
        for r in selected:
            split=r['split'];train=set(split['train_indices'])
            assert split['test_indices']==test
            assert not train.intersection(test)
            assert previous<=train
            assert set(y[list(train)])=={0,1}
            assert r['actual_fraction']==len(train)/70
            previous=train
        assert len(previous)==70 and len(test)==30
    assert rows==e.nested_fraction_splits(y,[11,37],[.2,.4,.6,.8,1],.3)


def test_small_class_and_collapsed_grid():
    with pytest.raises(ValueError,match='no training pool'):
        e.nested_fraction_splits([0,1,1],[11],[.2,1],.3)
    with pytest.raises(ValueError,match='identical subsets'):
        e.nested_fraction_splits([0]*3+[1]*3,[11],[.2,.4],.3)


def test_defaults_and_scheduling_identity(synthetic_uci_data):
    a=e.main(['--dry-run'])
    b=e.main(['--dry-run','--n-jobs','3'])
    assert len(a)==135
    assert [task_id(t) for t in a]==[task_id(t) for t in b]
    assert {t['max_bins'] for t in a}=={5,8,10}


def test_result_pipeline_and_resume_in_process(tmp_path, monkeypatch, synthetic_uci_data):
    # 单元测试用进程内适配器执行两个微型任务，不测试真实调度准入。
    # 公共内存阈值不变；正式入口仍完整调用公共调度器。
    from oeo_lab.experiments.common import runner
    from oeo_lab.experiments.common.checkpoints import completed
    calls=[]
    def synchronous(tasks, worker, directory, settings, template):
        for task in tasks:
            path=directory/template.format(task_id=task['id'])
            if not completed(path, task):
                calls.append(task['id'])
                runner._execute(worker,task,path)
    monkeypatch.setattr(runner,'run_tasks',synchronous)
    args=['--datasets','iris','--fractions','.2','1','--seeds','11',
          '--max-bins','5','--results-root',str(tmp_path)]
    tasks=e.main(args)
    workbook=tmp_path/'training_size_results.xlsx'
    seeds=pd.read_excel(workbook,sheet_name='per_seed')
    folds=pd.read_excel(workbook,sheet_name='folds')
    design=pd.read_excel(workbook,sheet_name='training_design')
    assert len(seeds)==2 and seeds.complete.all()
    assert set(seeds.study)=={'training_size'}
    assert all(c in seeds for c in ['rules_A1','rules_A2','rules_B1','rules_B2','candidate_hit_rate'])
    assert folds.attributes.notna().all()
    assert design.training_fraction.tolist()==[.2,1]
    paths=[tmp_path/'checkpoints'/f"{t['id']}.json" for t in tasks]
    before=[p.read_bytes() for p in paths]
    e.main(args+['--n-jobs','2'])
    assert before==[p.read_bytes() for p in paths]
    assert len(calls)==2
