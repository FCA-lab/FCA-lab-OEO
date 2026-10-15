"""BABD 公共地址实验：均衡独立留出与不均衡分层交叉验证。"""
import argparse
import hashlib
import json
import sys
import numpy as np
from . import babd_config as config
from .bitcoin_data import load_bitcoin_dataset
from .splits import make_splits
from .baselines import create_model
from .runner import run_classification_tasks
from ..common.checkpoints import task_id, digest
from ..common.provenance import collect_provenance


def parse_args(argv, defaults):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments', nargs='+', choices=['balanced','imbalanced'])
    p.add_argument('--models', nargs='+', choices=list(config.MODELS))
    for name in ('max-bins','seeds','per-class','train-sizes'):
        p.add_argument('--'+name, nargs='+', type=int)
    p.add_argument('--test-per-class', type=int)
    p.add_argument('--folds', type=int)
    p.add_argument('--n-jobs', type=int)
    p.add_argument('--voting', nargs='+', choices=['uniform','nonzero','support'])
    p.add_argument('--results-root', default=config.OUTPUT['root'])
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(**defaults)
    a = p.parse_args(argv)
    for name in ('max_bins','seeds','per_class','train_sizes'):
        values = getattr(a,name)
        if not values or len(values)!=len(set(values)):
            p.error(f'{name}: provide distinct values')
    if min(a.max_bins)<3 or min(a.per_class+a.train_sizes)<1 or a.test_per_class<1 or a.folds<2 or a.n_jobs<1:
        p.error('Invalid bins, sample sizes, folds or n-jobs')
    if any(s<0 or s>=2**32 for s in a.seeds):p.error('Invalid seed')
    return a


def _split(y, groups, train, test, seed, fold):
    """保存模型共享的索引和类计数；地址交叉直接拒绝。"""
    train, test = sorted(map(int,train)), sorted(map(int,test))
    if set(train)&set(test) or set(groups[train])&set(groups[test]):
        raise ValueError('Training/test address overlap')
    counts = [{'label':int(c),'train':int(np.sum(y[train]==c)),
               'test':int(np.sum(y[test]==c))} for c in np.unique(y)]
    if any(not c['train'] or not c['test'] for c in counts):
        raise ValueError('Every class must occur in training and test data')
    return dict(seed=int(seed),fold=fold,train_indices=train,test_indices=test,
                class_counts=counts,entity_overlap=0)


def balanced_splits(y, groups, seeds, per_class, test_per_class):
    """同一种子固定测试集；每类训练样本按同一随机排列取前缀。"""
    for seed in seeds:
        rng=np.random.default_rng(seed)
        ordered=[rng.permutation(np.flatnonzero(y==c)) for c in np.unique(y)]
        if any(len(ids)<test_per_class+max(per_class) for ids in ordered):
            raise ValueError('Insufficient samples for balanced test and largest training budget')
        test=np.concatenate([ids[:test_per_class] for ids in ordered])
        for size in sorted(per_class):
            train=np.concatenate([ids[test_per_class:test_per_class+size] for ids in ordered])
            yield len(train),size,_split(y,groups,train,test,seed,1)


def imbalanced_splits(y, groups, seeds, train_sizes, folds):
    """先固定测试折，再按比例缺额逐个分配训练名额，保证精确预算和嵌套性。

    每类先保留一个对象；后续优先补足相对于原训练池比例的缺额。
    因此极小类别不会消失，比例允许整数舍入误差，不进行过采样。
    """
    plan=make_splits(y,protocol='address',groups=groups,n_splits=folds,seeds=seeds)
    for base in plan['splits']:
        pool=np.asarray(base['train_indices'])
        rng=np.random.default_rng(np.random.SeedSequence([base['seed'],base['fold'],731]))
        ordered=[rng.permutation(pool[y[pool]==c]) for c in np.unique(y)]
        capacity=np.array([len(ids) for ids in ordered])
        if min(train_sizes)<len(ordered) or max(train_sizes)>len(pool):
            raise ValueError('Training budget must cover all classes and fit the training pool')
        counts=np.ones(len(ordered),dtype=int)
        chosen=[int(ids[0]) for ids in ordered]
        proportions=capacity/capacity.sum()
        for size in sorted(train_sizes):
            while len(chosen)<size:
                deficit=(len(chosen)+1)*proportions-counts
                deficit[counts==capacity]=-np.inf
                c=int(np.argmax(deficit))
                chosen.append(int(ordered[c][counts[c]]));counts[c]+=1
            yield size,None,_split(y,groups,chosen,base['test_indices'],base['seed'],base['fold'])


def build_tasks(args, experiment, provenance):
    spec=dict(config.DATASET,class_ids=config.CLASS_IDS[experiment])
    path=config.DATA_ROOT/spec['path']
    spec['path']=str(path)
    with path.open('rb') as stream:spec['sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    manifest=config.DATA_ROOT/spec.pop('feature_manifest')
    spec['feature_columns']=json.loads(manifest.read_text())['features']
    if set(spec['feature_columns']) & {'account','SW','label'}:
        raise ValueError('Identifiers or labels found in feature list')
    X,y,labels,groups=load_bitcoin_dataset(spec)
    print(f'BABD {experiment}: objects={len(y)}, features={X.shape[1]}, classes={labels}',flush=True)
    splits=(balanced_splits(y,groups,args.seeds,args.per_class,args.test_per_class)
            if experiment=='balanced' else imbalanced_splits(y,groups,args.seeds,args.train_sizes,args.folds))
    tasks=[]
    for budget,per_class,split in splits:
        split_hash=digest(split)
        for name in dict.fromkeys(args.models):
            model=config.MODELS[name]
            if model['input']!='context':raise ValueError('BABD models must share binary context input')
            for bins in args.max_bins:
                task=dict(experiment='babd_'+experiment,name='babd_'+experiment,
                    protocol='balanced_holdout' if experiment=='balanced' else 'address',
                    dataset=spec,split=split,split_sha256=split_hash,train_budget=budget,
                    train_per_class=per_class,model_name=name,model=model,max_bins=bins,
                    min_data_in_bin=config.MIN_DATA_IN_BIN,voting=args.voting,provenance=provenance)
                task['id']=task_id(task);tasks.append(task)
    return tasks


def _run(args, experiment):
    settings=dict(config.EXECUTION,n_jobs=args.n_jobs)
    packages=list(config.PACKAGES)
    for name in args.models:
        estimator=config.MODELS[name]['estimator']
        if estimator in ('xgboost','catboost','lightgbm'):packages.append(estimator)
        if not args.dry_run:create_model(config.MODELS[name],args.seeds[0])
    # dry-run 只检查数据与任务矩阵；真正运行会验证所有选中模型的依赖。
    provenance=collect_provenance(config.PROJECT_ROOT,config.SOURCE_PATTERNS,
        config.PACKAGES if args.dry_run else packages,settings)
    tasks=build_tasks(args,experiment,provenance)
    print(f'{experiment}: tasks={len(tasks)}, bins={args.max_bins}, seeds={args.seeds}, '
          f'n_jobs={args.n_jobs}, voting={args.voting}',flush=True)
    if not args.dry_run:
        output=dict(config.OUTPUT,**config.OUTPUT_FILES[experiment],root=args.results_root)
        run_classification_tasks(tasks,settings,output)
    return tasks


def run_balanced_experiment(args):
    return _run(args,'balanced')


def run_imbalanced_experiment(args):
    return _run(args,'imbalanced')


def main(argv=None):
    # 运行网格仅在这里设置；命令行显式参数覆盖相应设置。
    defaults=dict(experiments=['balanced','imbalanced'],models=list(config.MODELS),
        max_bins=[5,10,20,30,50],seeds=[11,37,53],
        per_class=[50,100,200,500,1000,2000,4000],test_per_class=1000,
        train_sizes=[500,1000,2000,5000,10000,20000,30000],folds=5,
        voting=['uniform'],n_jobs=1)
    args=parse_args(argv,defaults)
    for experiment in dict.fromkeys(args.experiments):
        if experiment=='balanced':run_balanced_experiment(args)
        else:run_imbalanced_experiment(args)


if __name__=='__main__':
    main(sys.argv[1:])
