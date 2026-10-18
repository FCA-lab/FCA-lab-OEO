"""小样本分类：固定测试集、嵌套训练预算、相同二值输入。"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from . import tabpfn_config as config
from .datasets import load_numeric_dataset
from .runner import run_classification_tasks
from ..common.checkpoints import task_id, digest
from ..common.provenance import collect_provenance


def nested_splits(y, seeds, budgets, test_fraction):
    """测试集按类固定留出；增加预算仅扩展训练前缀，不改变测试对象。"""
    splits, skipped = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        pools, tests = [], []
        for label in np.unique(y):
            ids = rng.permutation(np.flatnonzero(y == label))
            n = max(1, int(np.ceil(len(ids)*test_fraction)))
            tests.extend(ids[:n].tolist()); pools.append(ids[n:])
        for budget in sorted(budgets):
            if any(len(pool)<budget for pool in pools):
                skipped.append(dict(seed=seed,train_per_class=budget,
                    reason='Insufficient training objects after fixed test holdout',
                    available_per_class=[len(pool) for pool in pools]))
                continue
            train = sorted(np.concatenate([pool[:budget] for pool in pools]).tolist())
            test = sorted(tests)
            assert not set(train)&set(test)
            split = dict(seed=seed,fold=1,train_indices=train,test_indices=test)
            splits.append((budget,split))
    return splits, skipped


def build_tasks(args, provenance, models):
    tasks, skipped = [], []
    for name in args.datasets:
        spec = copy.deepcopy(config.DATASETS[name])
        path = config.DATA_ROOT/spec['path']
        spec.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        X,y,labels = load_numeric_dataset(spec)
        splits, omissions = nested_splits(y,args.seeds,args.per_class,args.test_fraction)
        skipped.extend(dict(dataset=name,**item) for item in omissions)
        for budget,split in splits:
            for model_name in args.models:
                model=models[model_name]
                if model['input']!='context':raise ValueError('All models must use the same binary input')
                for bins in args.max_bins:
                    task=dict(experiment='tabpfn_small_sample',name=name, dataset=spec,
                        split=split,split_sha256=digest(split),train_budget=len(split['train_indices']),
                        train_per_class=budget,model_name=model_name,model=model,max_bins=bins,
                        min_data_in_bin=config.MIN_DATA_IN_BIN,voting=args.voting,provenance=provenance)
                    task['id']=task_id(task);tasks.append(task)
    return tasks, skipped


def main(argv=None):
    # 网格仅在这里设置；终端只覆盖显式给出的参数。默认串行，避免多个大模型争用内存。
    defaults=dict(datasets=list(config.DATASETS),models=list(config.MODELS),
        per_class=[5,10,20,50,100],max_bins=[5,10,20,50],seeds=[11,37,53],
        test_fraction=0.3,voting=['uniform','nonzero','support'],n_jobs=1)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=list(config.DATASETS))
    p.add_argument('--models',nargs='+',choices=list(config.MODELS))
    for key in ('per-class','max-bins','seeds'):p.add_argument('--'+key,nargs='+',type=int)
    p.add_argument('--test-fraction',type=float)
    p.add_argument('--voting',nargs='+',choices=['uniform','nonzero','support'])
    p.add_argument('--n-jobs',type=int)
    p.add_argument('--results-root',default=config.OUTPUT['root'])
    p.add_argument('--dry-run',action='store_true')
    p.set_defaults(**defaults);a=p.parse_args(argv)
    for key in ('datasets','models','per_class','max_bins','seeds','voting'):
        values=getattr(a,key)
        if not values or len(values)!=len(set(values)):p.error(f'{key}: values must be nonempty and distinct')
    if min(a.per_class)<1 or min(a.max_bins)<3 or not 0<a.test_fraction<1 or a.n_jobs<1:
        p.error('Invalid budget, bins, test fraction or concurrency')
    if any(s<0 or s>=2**32 for s in a.seeds):p.error('Invalid random seed')
    models=copy.deepcopy(config.MODELS);packages=list(config.PACKAGES)
    if 'tabpfn_v2' in a.models:
        path=config.MODEL_PATH
        if not a.dry_run and not path.is_file():
            p.error(f'Missing explicit TabPFN v2 checkpoint: {path}; set OEO_TABPFN_MODEL')
        models['tabpfn_v2']['params'].update(model_path=str(path),
            model_sha256=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'not-installed')
        packages += [] if a.dry_run else ['tabpfn','torch']
    if not a.dry_run:
        packages += [m for m in ('xgboost','catboost') if m in a.models]
    settings=dict(config.EXECUTION,n_jobs=a.n_jobs)
    provenance=collect_provenance(config.PROJECT_ROOT,config.SOURCE_PATTERNS,packages,settings)
    tasks,skipped=build_tasks(a,provenance,models)
    print(f'Small sample: tasks={len(tasks)}, skipped budgets={len(skipped)}, '
          f'per_class={a.per_class}, bins={a.max_bins}, seeds={a.seeds}, n_jobs={a.n_jobs}',flush=True)
    for item in skipped:print('[skip]',item,flush=True)
    if not a.dry_run:
        output=dict(config.OUTPUT,root=a.results_root)
        root=Path(a.results_root);root.mkdir(parents=True,exist_ok=True)
        (root/output['skipped']).write_text(json.dumps(skipped,indent=2))
        if not tasks:raise ValueError('No feasible tasks')
        run_classification_tasks(tasks,settings,output)
    return tasks,skipped

if __name__=='__main__':
    main(sys.argv[1:])
