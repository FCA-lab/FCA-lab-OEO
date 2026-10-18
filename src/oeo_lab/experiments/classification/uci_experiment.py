"""UCI 通用分类入口；所有模型共享重复分层交叉验证索引。"""
import argparse
import hashlib
import sys
from . import uci_config as config
from .datasets import load_numeric_dataset
from .splits import make_splits
from .baselines import create_model
from .runner import run_classification_tasks
from ..common.provenance import collect_provenance
# 2026-10-06：任务 ID 排除调度参数，调整并行数可继续复用成功任务。
from ..common.checkpoints import task_id


def parse_args(argv=None, defaults=None):
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
    p.add_argument('--experiments', nargs='+', choices=['comparison','sensitivity','ablation'],
                   default=['comparison','sensitivity','ablation'])
    p.add_argument('--voting', nargs='+', choices=config.VOTING, default=config.VOTING)
    p.add_argument('--fixed-voting', choices=config.VOTING, default='uniform')
    p.add_argument('--ablation-bins', type=int, default=20)
    p.add_argument('--rule-groups', nargs='+', choices=list(config.RULE_GROUPS),
                   default=list(config.RULE_GROUPS))
    if defaults:
        p.set_defaults(**defaults)
    a=p.parse_args(argv)
    if a.n_jobs<1 or a.folds<2 or any(b<3 for b in a.max_bins):p.error('n-jobs>=1, folds>=2, max-bins>=3 required')
    if a.timeout is not None and a.timeout<=0:p.error('timeout must be positive')
    if len(set(a.seeds))!=len(a.seeds):p.error('Duplicate seeds')
    if a.ablation_bins < 3:p.error('ablation-bins>=3 required')
    # argparse 不校验列表形式的默认值；main 中手改的参数同样必须检查。
    for field, allowed in [('datasets',config.DATASETS),('models',config.MODELS),
                           ('experiments',('comparison','sensitivity','ablation')),
                           ('voting',config.VOTING),('rule_groups',config.RULE_GROUPS)]:
        values=getattr(a,field)
        if not values or any(v not in allowed for v in values):p.error(f'Invalid {field}: {values}')
    if not a.seeds or not a.max_bins:p.error('seeds and max-bins cannot be empty')
    if a.fixed_voting not in config.VOTING:p.error('Invalid fixed-voting')
    for name in a.rule_groups:
        if not set(config.RULE_GROUPS[name]) <= {'A1','A2','B1','B2'}:
            p.error(f'Invalid rule types in {name}')
    return a


def model_comparison(args):
    """实验一：各模型与分箱设置对比；RC-OEO 使用固定投票和全部规则。"""
    return [{'study':'comparison', 'model':name, 'bins':b,
             'voting':args.fixed_voting, 'rule_group':'all'}
            for name in dict.fromkeys(args.models)
            for b in (args.max_bins if config.MODELS[name]['input']=='context' else [None])]


def binning_voting_experiment(args):
    """实验二：RC-OEO 分箱与投票策略；始终启用全部规则。"""
    return [{'study':'sensitivity', 'model':'rc-oeo', 'bins':b,
             'voting':v, 'rule_group':'all'} for b in args.max_bins for v in args.voting]


def rule_ablation_experiment(args):
    """实验三：固定分箱和投票，仅切换预测时启用的规则组合。"""
    return [{'study':'ablation', 'model':'rc-oeo', 'bins':args.ablation_bins,
             'voting':args.fixed_voting, 'rule_group':g} for g in args.rule_groups]


def group_requests(requests, categorical=False):
    """按模型和输入合并任务；类别数据不遍历无意义的分箱网格。"""
    grouped = {}
    for r in requests:
        r = dict(r, bins=None) if categorical else r
        model = config.MODELS[r['model']]
        is_oeo = model['estimator']=='oeo'
        key = (r['model'],r['bins'])
        variants = grouped.setdefault(key,{})
        vkey = (r['voting'],r['rule_group']) if is_oeo else ('baseline','baseline')
        variant = variants.setdefault(vkey, {'voting':vkey[0], 'rule_group':vkey[1],
            'rule_types':list(config.RULE_GROUPS[r['rule_group']]) if is_oeo else [], 'studies':[]})
        if r['study'] not in variant['studies']:variant['studies'].append(r['study'])
    return grouped


def categorical_provenance(provenance):
    """只给新增类别任务记录新增依赖，保留旧数值任务的来源文件集合。"""
    from copy import deepcopy
    result = deepcopy(provenance)
    for relative in ('src/oeo_lab/preprocessing/categorical.py',
                     'src/oeo_lab/experiments/classification/uci_data.py'):
        value = hashlib.sha256((config.PROJECT_ROOT / relative).read_bytes()).hexdigest()
        result.setdefault('sources', {})[relative] = value
        result.setdefault('computation_sources', {})[relative] = value
    from ..common.checkpoints import digest
    result['source_hash'] = digest(result['sources'])
    return result


def build_tasks(args, provenance):
    """先合并三类实验，再按训练设置组任务；每个任务仅编码、训练一次。

    同一预测组合属于多个实验时，计算一次并在报表中注明各实验归属。
    缓存粒度为整个训练任务：中断的任务重跑，完成的任务可恢复。
    """
    functions = {'comparison':model_comparison, 'sensitivity':binning_voting_experiment,
                 'ablation':rule_ablation_experiment}
    requests = [r for study in dict.fromkeys(args.experiments) for r in functions[study](args)]
    tasks=[]
    for name in dict.fromkeys(args.datasets):
        dataset = config.DATASETS[name]
        categorical = dataset.get('feature_kind', 'numerical') == 'categorical'
        path=config.DATA_ROOT/dataset['path']
        spec=dict(dataset,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if categorical:
            from .uci_data import load_uci_dataset
            X,y,labels=load_uci_dataset(spec)
        else:
            X,y,labels=load_numeric_dataset(spec)
        folds = dataset.get('folds', args.folds)
        plan=make_splits(y,protocol='stratified',n_splits=folds,seeds=args.seeds)
        grouped = group_requests(requests, categorical)
        source = categorical_provenance(provenance) if categorical else provenance
        print(f'{name}: samples={len(y)}, features={X.shape[1]}, classes={len(labels)}, '
              f'folds={folds}, encoding={"one-hot" if categorical else "binning + one-hot"}',flush=True)
        for split in plan['splits']:
            for (name_model,b),variants in grouped.items():
                task={'experiment':'uci_classification_v2','name':name,'dataset':spec,
                      'split':split,'model_name':name_model,'model':config.MODELS[name_model],
                      'max_bins':b,'min_data_in_bin':config.MIN_DATA_IN_BIN,
                      'voting':list(dict.fromkeys(v['voting'] for v in variants.values())),
                      'prediction_variants':list(variants.values()),'provenance':source}
                task['id']=task_id(task);tasks.append(task)
    return tasks


def main(argv=None):
    # 常改的实验参数集中于 main；终端参数只覆盖显式指定项。
    # 消融固定值应事先确定，不根据测试集结果选择。
    experiments = ['comparison', 'sensitivity', 'ablation']
    datasets = list(config.DATASETS)
    models = list(config.MODELS)
    max_bins = [20, 30, 40, 50, 60, 70, 80, 90, 100]
    voting = ['uniform', 'nonzero', 'support']
    fixed_voting = 'uniform'
    ablation_bins = 20
    rule_groups = list(config.RULE_GROUPS)
    seeds = [11, 37, 53]
    folds = 5
    args=parse_args(argv, dict(experiments=experiments, datasets=datasets, models=models,
        max_bins=max_bins, voting=voting, fixed_voting=fixed_voting,
        ablation_bins=ablation_bins, rule_groups=rule_groups, seeds=seeds, folds=folds))
    settings=dict(config.EXECUTION,n_jobs=args.n_jobs,task_timeout_sec=args.timeout)
    active_models = set(args.models) if 'comparison' in args.experiments else set()
    if set(args.experiments) & {'sensitivity','ablation'}:active_models.add('rc-oeo')
    packages=list(config.PACKAGES)
    if any(config.MODELS[n]['estimator']=='xgboost' for n in active_models):packages.append('xgboost')
    provenance=collect_provenance(config.PROJECT_ROOT,config.SOURCE_PATTERNS,packages,settings)
    for name in active_models:
        spec=config.MODELS[name]
        if spec['input'] not in ('raw','context'):raise ValueError('Invalid input representation')
        if spec['estimator']=='oeo' and spec['input']!='context':raise ValueError('OEO requires context input')
        create_model(spec,args.seeds[0])  # 缺失模型依赖在批量执行前报错。
    tasks=build_tasks(args,provenance)
    print(f"任务={len(tasks)}, folds={args.folds}, seeds={args.seeds}, max_bins={args.max_bins}, "
          f"n_jobs={args.n_jobs}, timeout={args.timeout}, output={args.results_root}",flush=True)
    print(f"实验={args.experiments}, voting={args.voting}, fixed_voting={args.fixed_voting}, "
          f"ablation_bins={args.ablation_bins}, rule_groups={args.rule_groups}", flush=True)
    print(f"训练任务={len(tasks)}, 独立预测组合={sum(len(t['prediction_variants']) for t in tasks)}", flush=True)
    if args.dry_run:return
    run_classification_tasks(tasks,settings,dict(config.OUTPUT,root=args.results_root))


if __name__ == '__main__':
    main(sys.argv[1:])
