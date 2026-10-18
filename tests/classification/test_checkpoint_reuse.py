"""Regression checks for BABD restart compatibility (no experiment execution)."""
import copy
import hashlib
import json
from pathlib import Path

from oeo_lab.experiments.common.checkpoints import (
    completed, recover_compatible_checkpoints, same_task, save, task_id)
from oeo_lab.experiments.common.resume_identity import computation_hash, PREFIX


def task():
    return {'id':'0'*24, 'name':'babd_imbalanced','max_bins':5,
            'model':{'name':'rc-oeo'}, 'dataset':{'sha256':'data'},
            'split':{'seed':11,'train_indices':[1,2],'test_indices':[3]},
            'provenance':{'packages':{'numpy':'1'},'settings':{'n_jobs':8,'rc_oeo_share':0.2},
                'sources':{PREFIX+'classification/babd_config.py':'unknown-config',
                           PREFIX+'classification/baselines.py':'model-code'},
                'source_hash':'raw-audit'}}


def test_scheduling_only():
    a=task(); b=copy.deepcopy(a)
    b['provenance']['settings'].update(n_jobs=32,rc_oeo_share=0.75,scheduler='babd_rc_oeo')
    assert same_task(a,b) and task_id(a)==task_id(b)


def test_calculation_changes_invalidate():
    a=task()
    for path,value in [(['max_bins'],10),(['dataset','sha256'],'new-data'),
        (['split','seed'],53),(['split','train_indices'],[1,3]),
        (['model','name'],'other'),(['provenance','packages','numpy'],'2'),
        (['provenance','sources',PREFIX+'classification/baselines.py'],'new-model-code'),
        (['provenance','settings','voting'],'support')]:
        b=copy.deepcopy(a); parent=b
        for key in path[:-1]: parent=parent[key]
        parent[path[-1]]=value
        assert not same_task(a,b), path


def test_config_semantics():
    path=PREFIX+'classification/babd_config.py'
    assert computation_hash(path,'EXECUTION={"n_jobs":8}\nMODELS=[1]') == computation_hash(
        path,'EXECUTION={"n_jobs":32}\nMODELS=[1]')
    assert computation_hash(path,'EXECUTION={}\nMODELS=[1]') != computation_hash(
        path,'EXECUTION={}\nMODELS=[2]')


def test_runner_semantics():
    path=PREFIX+'classification/runner.py'
    a='def classification_worker(t):\n return 1\ndef run_classification_tasks():\n return 1'
    b=a.replace('run_classification_tasks():\n return 1','run_classification_tasks():\n return 2')
    assert computation_hash(path,a)==computation_hash(path,b)
    assert computation_hash(path,a)!=computation_hash(path,a.replace('worker(t):\n return 1','worker(t):\n return 2'))


def test_old_filename_recovery(tmp_path):
    old=task(); current=copy.deepcopy(old)
    current['id']=task_id(current)
    old_path=tmp_path/('a'*24+'.json')
    save(old_path,{'task':old,'status':'ok','result':{'score':0.8}})
    assert recover_compatible_checkpoints([current],tmp_path)==1
    new_path=tmp_path/(current['id']+'.json')
    assert completed(new_path,current)
    stored=json.loads(new_path.read_text())
    assert stored['task']==old and stored['result']=={'score':0.8}
    assert stored['reused_from']==old_path.name
    assert old_path.exists()
    assert recover_compatible_checkpoints([current],tmp_path)==0


def test_failed_and_unknown_code_not_reused(tmp_path):
    old=task(); current=copy.deepcopy(old); current['id']=task_id(current)
    save(tmp_path/('a'*24+'.json'),{'task':old,'status':'failed'})
    changed=copy.deepcopy(old)
    changed['provenance']['sources'][PREFIX+'classification/babd_config.py']='other-unknown'
    save(tmp_path/('b'*24+'.json'),{'task':changed,'status':'ok','result':{}})
    (tmp_path/('c'*24+'.json')).write_text('{broken')
    assert recover_compatible_checkpoints([current],tmp_path)==0


def test_audited_legacy_hashes():
    from oeo_lab.experiments.common.resume_identity import legacy_aliases
    a=task(); b=copy.deepcopy(a)
    for path,mapping in legacy_aliases().items():
        raw,canonical=next(iter(mapping.items()))
        a['provenance']['sources'][path]=raw
        b['provenance']['sources'][path]='current-raw'
        b['provenance'].setdefault('computation_sources',{})[path]=canonical
    assert same_task(a,b)
    b['provenance']['computation_sources'][PREFIX+'classification/runner.py']='changed-calculation'
    assert not same_task(a,b)


def test_no_mutation():
    a=task(); before=copy.deepcopy(a)
    task_id(a)
    assert a==before


def test_provenance_preserves_non_utf8_bytes(tmp_path):
    from oeo_lab.experiments.common.provenance import collect_provenance
    config=tmp_path/(PREFIX+'classification/babd_config.py')
    config.parent.mkdir(parents=True)
    config.write_text('EXECUTION={"n_jobs":8}')
    other=config.with_name('other.py')
    other.write_bytes(b'# coding: latin-1\n# \xb0\n')
    result=collect_provenance(tmp_path,[PREFIX+'classification/*.py'],[],{})
    key=str(other.relative_to(tmp_path))
    assert result['sources'][key]==result['computation_sources'][key]
    assert result['source_hash']
