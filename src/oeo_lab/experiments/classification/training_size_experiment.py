"""固定测试集、分层嵌套训练比例；评估规则与决策随训练规模的变化。"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from . import training_size_config as config
from .uci_data import load_uci_dataset
from .runner import run_classification_tasks
from ..common.checkpoints import task_id, digest
from ..common.provenance import collect_provenance


def nested_fraction_splits(y, seeds, fractions, test_fraction):
    """每类固定留出 ceil(n*test_fraction)，训练按同一随机排列取前缀。

    训练比例按每类训练池 ceil(pool*fraction) 取整，至少保留一个对象；
    因此实际比例可能略大于请求比例，完整记录数量。不同请求若得到
    相同样本数量则报错，避免把同一次训练当作独立规模点。
    """
    y=np.asarray(y)
    if y.ndim!=1 or not len(y) or pd.isna(y).any():raise ValueError('Invalid labels')
    if not 0<test_fraction<1:raise ValueError('test_fraction must be in (0,1)')
    if not fractions or len(set(fractions))!=len(fractions) or any(not 0<f<=1 for f in fractions):
        raise ValueError('fractions must be distinct and in (0,1]')
    if not seeds or len(set(seeds))!=len(seeds) or any(not 0<=s<2**32 for s in seeds):
        raise ValueError('Invalid seeds')
    classes=np.unique(y)
    if len(classes)<2:raise ValueError('At least two classes required')
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); pools=[];tests=[]
        for label in classes:
            indices=rng.permutation(np.flatnonzero(y==label))
            n_test=int(math.ceil(len(indices)*test_fraction))
            if n_test>=len(indices):raise ValueError(f'Class {label}: no training pool remains')
            pools.append(indices[n_test:]);tests.extend(indices[:n_test].tolist())
        test=sorted(tests);pool_size=sum(map(len,pools));seen=set()
        for fraction in sorted(fractions):
            counts=tuple(int(math.ceil(len(pool)*fraction)) for pool in pools)
            if counts in seen:raise ValueError('Fractions collapse to identical subsets; reduce grid')
            seen.add(counts)
            train=sorted(np.concatenate([p[:n] for p,n in zip(pools,counts)]).tolist())
            assert not set(train)&set(test)
            inventory=[dict(label=int(label),pool=len(pool),train=n,
                            test=int(sum(y[test]==label)))
                       for label,pool,n in zip(classes,pools,counts)]
            rows.append(dict(fraction=float(fraction),pool_size=pool_size,
                actual_fraction=len(train)/pool_size,
                split=dict(seed=int(seed),fold=1,train_indices=train,test_indices=test,
                           class_counts=inventory,entity_overlap=None)))
    return rows


def build_tasks(args, provenance):
    tasks=[]
    for name in args.datasets:
        spec=copy.deepcopy(config.DATASETS[name]);path=config.DATA_ROOT/spec['path']
        # 本模块当前支持数值/类别表格，不能隐式把实体或时间协议换成随机划分。
        if spec.get('kind') not in (None,'tabular'):
            raise ValueError('Register a tabular dataset with an appropriate independent-sample protocol')
        spec.update(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        X,y,labels=load_uci_dataset(spec)
        splits=nested_fraction_splits(y,args.seeds,args.fractions,args.test_fraction)
        bins=[None] if spec.get('feature_kind')=='categorical' else args.max_bins
        for design in splits:
            split=design['split']
            for model_name in args.models:
                model=config.MODELS[model_name]
                if model['estimator']!='oeo':raise ValueError('This study currently evaluates RC-OEO only')
                for b in bins:
                    task=dict(experiment='training_size_v1',name=name,dataset=spec,split=split,
                        split_sha256=digest(split),train_budget=len(split['train_indices']),
                        train_per_class=None,training_fraction=design['fraction'],
                        actual_training_fraction=design['actual_fraction'],train_pool_size=design['pool_size'],
                        test_fraction=args.test_fraction,model_name=model_name,model=model,
                        max_bins=b,min_data_in_bin=config.MIN_DATA_IN_BIN,voting=args.voting,
                        prediction_variants=[dict(voting=v,rule_group='all',
                            rule_types=['A1','A2','B1','B2'],studies=['training_size']) for v in args.voting],
                        provenance=provenance)
                    task['id']=task_id(task);tasks.append(task)
        print(f'{name}: samples={len(y)}, raw_features={X.shape[1]}, classes={len(labels)}, '
              f'training settings={len(splits)*len(bins)}',flush=True)
    return tasks


def write_design(tasks, output):
    """额外设计工作表记录规模与类别计数；保留公共汇总器的指标及分组逻辑。"""
    records=[]
    for t in tasks:
        records.append(dict(task_id=t['id'],dataset=t['name'],seed=t['split']['seed'],
            max_bins=t['max_bins'],model=t['model_name'],training_fraction=t['training_fraction'],
            actual_training_fraction=t['actual_training_fraction'],train_pool_size=t['train_pool_size'],
            n_train=t['train_budget'],n_test=len(t['split']['test_indices']),
            n_classes=len(t['split']['class_counts']),
            class_counts=json.dumps(t['split']['class_counts'],ensure_ascii=False)))
    frame=pd.DataFrame(records)
    # 独立 CSV 在执行前写出，中断后仍可查看完整实验设计。
    root=Path(output['root']);root.mkdir(parents=True,exist_ok=True)
    path=root/'training_size_design.csv';temp=path.with_suffix('.tmp.csv')
    frame.to_csv(temp,index=False);os.replace(temp,path)
    return frame


def parse_args(argv, defaults):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=list(config.DATASETS))
    p.add_argument('--models',nargs='+',choices=list(config.MODELS))
    p.add_argument('--fractions',nargs='+',type=float)
    for name in ('max-bins','seeds'):p.add_argument('--'+name,nargs='+',type=int)
    p.add_argument('--test-fraction',type=float)
    p.add_argument('--voting',nargs='+',choices=['uniform','nonzero','support'])
    p.add_argument('--n-jobs',type=int)
    p.add_argument('--results-root',default=config.OUTPUT['root'])
    p.add_argument('--dry-run',action='store_true')
    p.set_defaults(**defaults);a=p.parse_args(argv)
    for key in ('datasets','models','fractions','max_bins','seeds','voting'):
        values=getattr(a,key)
        if not values or len(set(values))!=len(values):p.error(f'{key} must be nonempty and distinct')
    if any(not 0<f<=1 for f in a.fractions) or not 0<a.test_fraction<1:
        p.error('Invalid training/test fractions')
    if min(a.max_bins)<3 or a.n_jobs<1 or any(not 0<=s<2**32 for s in a.seeds):
        p.error('Invalid bins, concurrency or seed')
    return a


def main(argv=None):
    # 实验网格只在这里修改；终端仅覆盖显式提供的参数。
    defaults=dict(datasets=list(config.DATASETS),models=['rc-oeo'],
        fractions=[.2,.4,.6,.8,1.0],max_bins=[5,8,10],seeds=[11,37,53],
        test_fraction=.3,voting=['uniform'],n_jobs=config.EXECUTION['n_jobs'])
    args=parse_args(argv,defaults)
    settings=dict(config.EXECUTION,n_jobs=args.n_jobs)
    provenance=collect_provenance(config.PROJECT_ROOT,config.SOURCE_PATTERNS,config.PACKAGES,settings)
    tasks=build_tasks(args,provenance)
    print(f'Training-size tasks={len(tasks)}, fractions={args.fractions}, bins={args.max_bins}, '
          f'seeds={args.seeds}, n_jobs={args.n_jobs}',flush=True)
    if not args.dry_run:
        output=dict(config.OUTPUT,root=args.results_root)
        design=write_design(tasks,output)
        run_classification_tasks(tasks,settings,output)
        # 只扩展本实验的结果文件，不修改其它实验及公共汇总代码。
        with pd.ExcelWriter(Path(output['root'])/output['results'],engine='openpyxl',
                            mode='a',if_sheet_exists='replace') as writer:
            design.to_excel(writer,sheet_name='training_design',index=False)
    return tasks


if __name__=='__main__':
    main(sys.argv[1:])
