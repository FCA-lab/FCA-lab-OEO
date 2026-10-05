"""UCI 通用分类入口；所有模型共享重复分层交叉验证索引。"""
import argparse
import hashlib
import sys
from pathlib import Path
from . import uci_config as config
from .datasets import load_numeric_dataset
from .splits import make_splits
from .baselines import create_model
from .runner import run_classification_tasks
from ..common.provenance import collect_provenance
from ..common.checkpoints import digest


def parse_args(argv=None):
    """显式参数覆盖配置；不依据测试结果搜索分箱参数。"""
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=list(config.DATASETS),default=list(config.DATASETS))
    p.add_argument('--models',nargs='+',choices=list(config.MODELS),default=list(config.MODELS))
    p.add_argument('--max-bins',nargs='+',type=int,default=config.MAX_BINS)
    p.add_argument('--seeds',nargs='+',type=int,default=config.SEEDS)
    p.add_argument('--folds',type=int,default=config.FOLDS)
    p.add_argument('--n-jobs',type=int,default=config.EXECUTION['n_jobs'])
    p.add_argument('--timeout',type=float,default=config.EXECUTION['task_timeout_sec'])
    p.add_argument('--results-root',default=config.OUTPUT['root'])
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(argv)
    if a.n_jobs<1 or a.folds<2 or any(b<3 for b in a.max_bins):p.error('n-jobs>=1, folds>=2, max-bins>=3 required')
    if a.timeout is not None and a.timeout<=0:p.error('timeout must be positive')
    if len(set(a.seeds))!=len(a.seeds):p.error('Duplicate seeds')
    return a


def build_tasks(args, provenance):
    """模型无关地生成一次划分；raw 基线不随分箱数重复，OEO 三策略合为一任务。"""
    tasks=[]
    for name in dict.fromkeys(args.datasets):
        path=config.DATA_ROOT/config.DATASETS[name]['path']
        spec=dict(config.DATASETS[name],path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        X,y,labels=load_numeric_dataset(spec)
        plan=make_splits(y,protocol='stratified',n_splits=args.folds,seeds=args.seeds)
        print(f'{name}: samples={len(y)}, features={X.shape[1]}, classes={len(labels)}',flush=True)
        for split in plan['splits']:
            for model_name in dict.fromkeys(args.models):
                model=config.MODELS[model_name]
                bins=list(dict.fromkeys(args.max_bins)) if model['input']=='context' else [None]
                for b in bins:
                    task={'experiment':'uci_classification','name':name,'dataset':spec,
                          'split':split,'model_name':model_name,'model':model,'max_bins':b,
                          'min_data_in_bin':config.MIN_DATA_IN_BIN,'voting':config.VOTING,
                          'provenance':provenance}
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
          f"n_jobs={args.n_jobs}, timeout={args.timeout}, output={args.results_root}",flush=True)
    if args.dry_run:return
    run_classification_tasks(tasks,settings,dict(config.OUTPUT,root=args.results_root))


if __name__ == '__main__':
    # 脚本设置作为默认值；终端只覆盖显式提供的同名参数。
    max_bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # 可写多个值，例如 '3', '5', '10'；仅影响数值数据集。
    models = [
        'dummy',
        'decision_tree',
        'random_forest',
        'random_forest_context',
        'xgboost',
        'xgboost_context',
        'svm_rbf',
        'rc-oeo',
    ]
    n_jobs = 18

    experiment_args = [
        '--max-bins', *map(str, max_bins),
        '--models', *models,
        '--n-jobs', str(n_jobs),
    ]
    main(experiment_args + sys.argv[1:])
