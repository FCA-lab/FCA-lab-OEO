"""Elliptic 非法交易二分类：固定时间留出与正负规则消融。"""
import argparse
import hashlib
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from . import elliptic_config as config
from ..common.checkpoints import digest, task_id
from ..common.provenance import collect_provenance


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def load_elliptic_dataset(spec):
    """按交易ID一对一连接。unknown 不进入监督训练/评价；非法编码为1。"""
    for key in ('features','classes'):
        if file_hash(spec[key+'_path'])!=spec[key+'_sha256']:
            raise ValueError('Elliptic data changed after task creation')
    features=pd.read_csv(spec['features_path'],header=None,dtype={0:str})
    labels=pd.read_csv(spec['classes_path'],dtype=str)
    if features.shape[1]!=167 or not {'txId','class'}<=set(labels):
        raise ValueError('Expected original Elliptic: 167 feature-file columns and txId/class labels')
    if features[0].isna().any() or labels.txId.isna().any() or features[0].duplicated().any() or labels.txId.duplicated().any():
        raise ValueError('Missing/duplicate transaction IDs')
    if not set(labels['class'])<={'1','2','unknown'}:
        raise ValueError('Unexpected Elliptic labels')
    if set(features[0])!=set(labels.txId):raise ValueError('Transaction IDs do not match')
    frame=features.merge(labels,left_on=0,right_on='txId',validate='one_to_one',sort=False)
    frame=frame[frame['class'].isin(['1','2'])].reset_index(drop=True)
    columns=spec['feature_columns']
    if not columns or len(set(columns))!=len(columns) or any(c<2 or c>166 for c in columns):
        raise ValueError('Feature selection must exclude IDs and time')
    X=frame[columns].apply(pd.to_numeric,errors='raise')
    time=pd.to_numeric(frame[1],errors='raise').to_numpy()
    if not np.isfinite(X.to_numpy()).all() or not np.isfinite(time).all() or not np.all(time==time.astype(int)) or np.any((time<1)|(time>49)):
        raise ValueError('Nonfinite features or invalid time steps')
    y=frame['class'].map({'2':0,'1':1}).to_numpy(dtype=int)
    return X,y,['licit','illicit'],time.astype(int),frame.txId.to_numpy()


def temporal_split(y,time,seed,train_end,test_start,test_end):
    """时间边界仅由参数确定；不随机打乱测试集，不使用未来标签调参。"""
    if not 1<=train_end<test_start<=test_end<=49:raise ValueError('Invalid temporal boundaries')
    train=np.flatnonzero(time<=train_end);test=np.flatnonzero((time>=test_start)&(time<=test_end))
    if set(y[train])!={0,1} or set(y[test])!={0,1}:raise ValueError('Both periods must contain both known classes')
    return dict(seed=seed,fold=1,train_indices=train.tolist(),test_indices=test.tolist(),
        entity_overlap=None,class_counts=[{'label':c,'train':int(sum(y[train]==c)),
        'test':int(sum(y[test]==c))} for c in (0,1)])


def parse_args(argv,defaults):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',choices=list(config.MODELS))
    p.add_argument('--max-bins',nargs='+',type=int)
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--rule-groups',nargs='+',choices=list(config.RULE_GROUPS))
    p.add_argument('--voting',nargs='+',choices=['uniform','nonzero','support'])
    p.add_argument('--feature-set',choices=list(config.FEATURE_COLUMNS))
    for n in ('train-end','test-start','test-end','n-jobs'):p.add_argument('--'+n,type=int)
    p.add_argument('--results-root',default=config.OUTPUT['root'])
    p.add_argument('--dry-run',action='store_true')
    p.set_defaults(**defaults);a=p.parse_args(argv)
    if not 1<=a.train_end<a.test_start<=a.test_end<=49 or a.n_jobs<1 or min(a.max_bins)<3:
        p.error('Invalid time boundaries, n-jobs or bins')
    for values in (a.seeds,a.max_bins,a.models,a.rule_groups,a.voting):
        if len(set(values))!=len(values):p.error('Repeated parameter values')
    if any(s<0 or s>=2**32 for s in a.seeds):p.error('Invalid seed')
    return a


def build_tasks(args,provenance):
    spec=dict(config.DATASET,feature_columns=config.FEATURE_COLUMNS[args.feature_set])
    for key in ('features','classes'):
        path=config.DATA_ROOT/spec[key+'_path']
        if not path.is_file():raise FileNotFoundError(f'Original Elliptic transaction file required: {path}')
        spec[key+'_path']=str(path);spec[key+'_sha256']=file_hash(path)
    X,y,_,time,ids=load_elliptic_dataset(spec)
    tasks=[]
    for seed in args.seeds:
        split=temporal_split(y,time,seed,args.train_end,args.test_start,args.test_end)
        for name in args.models:
            model=config.MODELS[name]
            if model['input']!='context':raise ValueError('All models must use the same binary context')
            for bins in args.max_bins:
                variants=([{'voting':v,'rule_group':g,'rule_types':config.RULE_GROUPS[g],
                    'studies':['illicit_detection']} for v in args.voting for g in args.rule_groups]
                    if model['estimator']=='oeo' else [{'voting':'baseline','rule_group':'baseline',
                    'rule_types':[],'studies':['illicit_detection']}])
                task=dict(experiment='elliptic_transactions',name='elliptic',protocol='temporal',dataset=spec,
                    split=split,split_sha256=digest(split),model_name=name,model=model,max_bins=bins,
                    min_data_in_bin=config.MIN_DATA_IN_BIN,voting=args.voting,prediction_variants=variants,
                    positive_label=1,feature_set=args.feature_set,train_end=args.train_end,
                    test_start=args.test_start,test_end=args.test_end,provenance=provenance)
                task['id']=task_id(task);tasks.append(task)
    print(f'Elliptic: known={len(y)}, illicit={int(sum(y==1))}, features={X.shape[1]}, '
          f'train={len(split["train_indices"])}, test={len(split["test_indices"])}, tasks={len(tasks)}',flush=True)
    return tasks


def main(argv=None):
    # 修改运行网格的位置；其余路径、字段、模型参数在独立配置文件。
    defaults=dict(models=list(config.MODELS),max_bins=[5,10,20,30,50],seeds=[11,37,53],
        feature_set='local',train_end=34,test_start=35,test_end=49,
        rule_groups=['all','positive','negative'],voting=['uniform'],n_jobs=1)
    args=parse_args(argv,defaults)
    packages=list(config.PACKAGES)
    if not args.dry_run and 'xgboost' in args.models:packages.append('xgboost')
    settings=dict(config.EXECUTION,n_jobs=args.n_jobs)
    provenance=collect_provenance(config.PROJECT_ROOT,config.SOURCE_PATTERNS,packages,settings)
    tasks=build_tasks(args,provenance)
    if args.dry_run:return
    from .baselines import create_model
    from .runner import run_classification_tasks
    for name in args.models:create_model(config.MODELS[name],args.seeds[0])
    run_classification_tasks(tasks,settings,dict(config.OUTPUT,root=args.results_root))


if __name__=='__main__':main(sys.argv[1:])
