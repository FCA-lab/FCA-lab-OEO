"""Bitcoin 应用分类入口；模型共享划分，实体级协议禁止实体交叉。"""
import argparse
import numpy as np
import hashlib
import sys
from pathlib import Path
from . import bitcoin_config as config
from .bitcoin_data import load_bitcoin_dataset
from .splits import make_splits
from .baselines import create_model
from .runner import run_classification_tasks
from ..common.provenance import collect_provenance
from ..common.checkpoints import digest


def parse_args(argv=None):
    """显式参数覆盖配置；不依据测试结果搜索分箱参数。"""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocols',nargs='+',choices=['address','entity'],default=config.PROTOCOLS)
    p.add_argument('--models',nargs='+',choices=list(config.MODELS),default=list(config.MODELS))
    p.add_argument('--max-bins',nargs='+',type=int,default=config.MAX_BINS)
    p.add_argument('--seeds',nargs='+',type=int,default=config.SEEDS)
    p.add_argument('--fractions',nargs='+',type=float,default=[0.2,0.4,0.6,0.8,1.0])
    p.add_argument('--folds',type=int,default=config.FOLDS)
    p.add_argument('--n-jobs',type=int,default=config.EXECUTION['n_jobs'])
    p.add_argument('--timeout',type=float,default=config.EXECUTION['task_timeout_sec'])
    p.add_argument('--results-root',default=config.OUTPUT['root'])
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.n_jobs<1 or a.folds<2 or any(b<3 for b in a.max_bins):p.error('n-jobs>=1, folds>=2, max-bins>=3 required')
    if a.timeout is not None and a.timeout<=0:p.error('timeout must be positive')
    if len(set(a.seeds))!=len(a.seeds):p.error('Duplicate seeds')
    if any(not 0<f<=1 for f in a.fractions) or len(set(a.fractions))!=len(a.fractions):
        p.error('fractions must be unique values in (0,1]')
    return a


def training_subsets(split, y, groups, fractions):
    """每类一次随机排序，取前缀形成嵌套训练地址子集；测试折固定。"""
    train=np.asarray(split['train_indices'],dtype=int)
    rng=np.random.default_rng(np.random.SeedSequence([split['seed'],split['fold'],731]))
    ordered={int(c):rng.permutation(train[y[train]==c]) for c in np.unique(y[train])}
    test=split['test_indices']
    for fraction in fractions:
        # 各类向下取整且至少一个；排序恢复原行序，100% 与原训练折一致。
        chosen=sorted(int(i) for ids in ordered.values()
                      for i in ids[:max(1,int(np.floor(len(ids)*fraction)))])
        current=dict(split,train_indices=chosen,
            class_counts=[{'label':c,'train':int(np.sum(y[chosen]==c)),
                           'test':int(np.sum(y[test]==c))} for c in ordered],
            entity_overlap=len(set(groups[chosen]) & set(groups[test])))
        yield fraction,current,len(set(groups[chosen]))


def build_tasks(args, provenance):
    """模型无关地生成一次划分；raw 基线不随分箱数重复，OEO 三策略合为一任务。"""
    path=config.DATA_ROOT/config.DATASET['path']
    spec=dict(config.DATASET,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    X,y,labels,groups=load_bitcoin_dataset(spec)
    print(f'Bitcoin: samples={len(y)}, features={X.shape[1]}, classes={labels}, entities={len(set(groups))}',flush=True)
    tasks=[]
    for protocol in dict.fromkeys(args.protocols):
        plan=make_splits(y,protocol=protocol,groups=groups,n_splits=args.folds,seeds=args.seeds)
        for split in plan['splits']:
            for fraction,sampled,entity_count in training_subsets(split,y,groups,args.fractions):
                for model_name in dict.fromkeys(args.models):
                    model=config.MODELS[model_name]
                    bins=list(dict.fromkeys(args.max_bins)) if model['input']=='context' else [None]
                    for b in bins:
                        task={'experiment':'bitcoin_classification','name':'bitcoin','protocol':protocol,
                              'dataset':spec,'split':sampled,'fraction':fraction,
                              'n_train_entities':entity_count,'split_sha256':plan['assignment_sha256'],
                              'inventory':plan['inventory'],'model_name':model_name,'model':model,
                              'max_bins':b,'min_data_in_bin':config.MIN_DATA_IN_BIN,
                              'voting':config.VOTING,'provenance':provenance}
                        task['id']=digest(task)[:24];tasks.append(task)
    return tasks


def main(argv=None):
    args=parse_args(argv)
    settings=dict(config.EXECUTION,n_jobs=args.n_jobs,task_timeout_sec=args.timeout)
    packages=list(config.PACKAGES)
    if any(config.MODELS[n]['estimator']=='xgboost' for n in args.models):packages.append('xgboost')
    provenance=collect_provenance(config.PROJECT_ROOT,config.SOURCE_PATTERNS,packages,settings)
    for name in args.models:
        spec=config.MODELS[name]
        if spec['input'] not in ('raw','context'):raise ValueError('Invalid input representation')
        if spec['estimator']=='oeo' and spec['input']!='context':raise ValueError('OEO requires context input')
        create_model(spec,args.seeds[0])  # 缺失模型依赖在批量执行前报错。
    tasks=build_tasks(args,provenance)
    print(f"任务={len(tasks)}, folds={args.folds}, seeds={args.seeds}, max_bins={args.max_bins}, "
          f"fractions={args.fractions}, n_jobs={args.n_jobs}, timeout={args.timeout}, output={args.results_root}",flush=True)
    if args.dry_run:return
    run_classification_tasks(tasks,settings,dict(config.OUTPUT,root=args.results_root))


if __name__=='__main__':
    # 脚本设置作为默认值；终端只覆盖同名参数，其他设置保留。
    fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    max_bins = config.MAX_BINS
    models = list(config.MODELS)
    protocols = config.PROTOCOLS
    n_jobs = config.EXECUTION['n_jobs']
    experiment_args = ['--fractions', *map(str,fractions), '--max-bins', *map(str,max_bins), '--models', *models,
                       '--protocols', *protocols, '--n-jobs', str(n_jobs)]
    main(experiment_args + sys.argv[1:])
