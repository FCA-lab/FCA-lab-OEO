"""读取实验既有 per_seed 汇总；不重新训练、不混合实验设置。"""
import pandas as pd

IDENTITY = ['study','rule_group','dataset','protocol','model','max_bins','voting',
            'train_budget','train_per_class','fraction','feature_set','train_end','test_start','test_end']


def select_results(data_file, filters, seeds):
    frame = pd.read_excel(data_file, sheet_name='per_seed')
    for key, value in filters.items():
        if key not in frame:
            raise ValueError(f'Missing filter column: {key}')
        frame = frame[frame[key].isin(value if isinstance(value, list) else [value])]
    frame = frame[frame.seed.isin(seeds)].copy()
    if frame.empty:
        raise ValueError('No results match the configured filters')
    keys = [k for k in IDENTITY if k in frame]
    if frame.duplicated(keys + ['seed']).any():
        raise ValueError('Duplicate experimental setting/seed')
    # 缺折或缺种子时不画残缺均值；调用入口记录失败，继续生成其它独立图。
    for _, group in frame.groupby(keys, dropna=False):
        if not group.complete.eq(True).all() or set(group.seed) != set(seeds):
            raise ValueError('Incomplete folds/seeds in selected settings')
    return frame


def validate_dimensions(frame, varying):
    for key in IDENTITY:
        if key in frame and key not in varying and frame[key].nunique(dropna=False) > 1:
            raise ValueError(f'Fix or facet this experimental setting: {key}')


def curve_data(frame, x, y, series):
    validate_dimensions(frame, {x, series, 'train_per_class'} if x == 'train_budget' else {x, series})
    if frame[x].isna().any() or frame[y].isna().any():
        raise ValueError(f'Undefined values for {x}/{y}')
    return frame.groupby([series, x], dropna=False)[y].agg(mean='mean', std='std', n='count').reset_index()


def comparison_grid_data(frame, datasets, models, bins, metrics):
    """保留模型/数据集分面；每个网格点要求完整且一致的种子集合。"""
    rows = []
    expected_seeds = set(frame.seed)
    for dataset in datasets:
        for model in models:
            part = frame[frame.dataset.eq(dataset) & frame.model.eq(model)]
            if part.empty or set(part.max_bins) != set(bins):
                raise ValueError(f'Missing or unexpected bin grid: {dataset}/{model}')
            if part.duplicated(['max_bins', 'seed']).any():
                raise ValueError(f'Duplicate seed: {dataset}/{model}')
            if any(set(g.seed) != expected_seeds for _, g in part.groupby('max_bins')):
                raise ValueError(f'Unpaired seeds: {dataset}/{model}')
            for metric in metrics:
                data = curve_data(part, 'max_bins', metric, 'model')
                rows.append(data.assign(dataset=dataset, metric=metric))
    return pd.concat(rows, ignore_index=True)


def ablation_grid_data(frame, datasets, rule_groups, metrics, seeds, numerical_bins=20):
    """Equal-seed means and paired differences; undefined accuracy is never zero-filled."""
    rows = []
    expected = set(seeds)
    for dataset in datasets:
        part = frame[frame.dataset.eq(dataset)]
        validate_dimensions(part, {'rule_group'})
        if set(part.rule_group) != set(rule_groups):
            raise ValueError(f'Missing or unexpected rule groups: {dataset}')
        if dataset in ('balloons', 'lenses'):
            valid_bins = part.max_bins.isna().all()
        else:
            valid_bins = part.max_bins.eq(numerical_bins).all()
        if not valid_bins:
            raise ValueError(f'Unexpected binning configuration: {dataset}')
        for rule, group in part.groupby('rule_group'):
            if set(group.seed) != expected or group.seed.duplicated().any():
                raise ValueError(f'Unpaired seeds: {dataset}/{rule}')
        for metric in metrics:
            pivot = part.pivot(index='seed', columns='rule_group', values=metric).reindex(seeds)
            if metric == 'classified_accuracy':
                counts = part.pivot(index='seed', columns='rule_group', values='n_classified').reindex(seeds)
                if (pivot.isna() != counts.eq(0)).any().any():
                    raise ValueError(f'Unexpected undefined accuracy: {dataset}')
            elif pivot.isna().any().any():
                raise ValueError(f'Undefined metric: {dataset}/{metric}')
            for rule in rule_groups:
                values = pivot[rule]
                delta = (values - pivot['all']) * 100
                complete = values.notna().all()
                paired = delta.notna().all()
                row = dict(dataset=dataset, rule_group=rule, metric=metric,
                    study='ablation', model='rc-oeo', voting='uniform',
                    max_bins=part.max_bins.iloc[0], seeds=','.join(map(str,seeds)),
                    n_seeds=len(seeds), n_defined=int(values.count()),
                    mean_pct=values.mean()*100 if complete else float('nan'),
                    sd_pct=values.std(ddof=1)*100 if complete else float('nan'),
                    reference_mean_pct=pivot['all'].mean()*100 if pivot['all'].notna().all() else float('nan'),
                    delta_pp=delta.mean() if paired else float('nan'),
                    delta_sd_pp=delta.std(ddof=1) if paired else float('nan'))
                row.update({f'seed_{seed}_pct':values.loc[seed]*100 for seed in seeds})
                rows.append(row)
    return pd.DataFrame(rows)


def voting_comparison_data(frame, datasets, bins, metrics, seeds):
    specs=[('DT','comparison','decision_tree','baseline','baseline'),
           ('RF','comparison','random_forest_context','baseline','baseline'),
           ('XGBoost','comparison','xgboost_context','baseline','baseline'),
           ('RC-OEO: uniform','sensitivity','rc-oeo','uniform','all'),
           ('RC-OEO: nonzero','sensitivity','rc-oeo','nonzero','all'),
           ('RC-OEO: support-weighted','sensitivity','rc-oeo','support','all')]
    pieces=[]
    for dataset in datasets:
        for label,study,model,voting,rules in specs:
            part=frame[frame.dataset.eq(dataset)&frame.study.eq(study)&frame.model.eq(model)&
                       frame.voting.eq(voting)&frame.rule_group.eq(rules)]
            if set(part.max_bins)!=set(bins) or part.duplicated(['max_bins','seed']).any():
                raise ValueError(f'Invalid bin grid: {dataset}/{label}')
            if any(set(g.seed)!=set(seeds) for _,g in part.groupby('max_bins')):
                raise ValueError(f'Incomplete seeds: {dataset}/{label}')
            for metric in metrics:
                if model!='rc-oeo' and metric=='coverage' and not part[metric].eq(1).all():
                    raise ValueError('Baseline coverage is not uniformly 100%')
                piece=curve_data(part,'max_bins',metric,'model')
                pieces.append(piece.assign(dataset=dataset,series=label,metric=metric,
                    study=study,voting=voting,rule_group=rules,seeds=','.join(map(str,seeds))))
    return pd.concat(pieces,ignore_index=True)


def candidate_size_data(frame, datasets, seeds):
    """Exact size groups for three-class tasks, normalized within each seed."""
    rows=[]
    for dataset in datasets:
        if dataset not in ('iris','wine','lenses'):
            raise ValueError('Exact size reconstruction is restricted to verified three-class datasets')
        part=frame[frame.dataset.eq(dataset)]
        validate_dimensions(part,set())
        if set(part.seed)!=set(seeds) or part.seed.duplicated().any():
            raise ValueError(f'Incomplete or duplicate seeds: {dataset}')
        if not (part.max_bins.isna().all() if dataset=='lenses' else part.max_bins.eq(20).all()):
            raise ValueError('Unexpected binning configuration')
        for _,v in part.iterrows():
            counts=[v.n_classified,v.partial_count,v.n_test-v.n_classified-v.partial_count]
            hits=[v.n_correct,v.partial_hits,v.candidate_hits-v.n_correct-v.partial_hits]
            if sum((i+1)*n for i,n in enumerate(counts))!=v.candidate_size_sum or hits[2]!=counts[2]:
                raise ValueError('Candidate-size/count identity failed')
            for size,(count,hit) in enumerate(zip(counts,hits),1):
                if not 0<=hit<=count or v.n_test<=0:raise ValueError('Invalid candidate counts')
                rows.append(dict(dataset=dataset,seed=int(v.seed),candidate_size=size,
                    n_test=int(v.n_test),count=int(count),retained_count=int(hit),excluded_count=int(count-hit),
                    total_pct=100*count/v.n_test,retained_pct=100*hit/v.n_test,
                    excluded_pct=100*(count-hit)/v.n_test,
                    conditional_retention_pct=100*hit/count if count else float('nan'),
                    study='comparison',model='rc-oeo',rule_group='all',voting='uniform',max_bins=v.max_bins))
    return pd.DataFrame(rows)


def babd_behavior_data(frame):
    """Aggregate matched per-seed metrics without mixing the two BABD protocols."""
    metrics = ['rule_count', 'classified_accuracy', 'coverage',
               'mean_candidate_size', 'candidate_hit_rate']
    rows = []
    for experiment, group in frame.groupby('experiment'):
        expected = ([350,700,1400,3500,7000,14000,28000] if experiment == 'balanced'
                    else [500,1000,2000,5000,10000,20000,30000])
        if sorted(group.train_budget.unique()) != expected:
            raise ValueError('Unexpected BABD training-budget grid')
        if group[metrics].isna().any().any():
            raise ValueError('Undefined BABD metric')
        for budget, part in group.groupby('train_budget'):
            row = dict(experiment=experiment, train_budget=budget, max_bins=20,
                       model='rc-oeo', voting='uniform', n=len(part),
                       seeds=','.join(map(str, sorted(part.seed.unique()))),
                       n_classes=7 if experiment == 'balanced' else 11)
            for metric in metrics:
                row[metric+'_mean'] = part[metric].mean()
                row[metric+'_std'] = part[metric].std(ddof=1)
                for entry in part.itertuples():
                    row[metric+'_seed_'+str(entry.seed)] = getattr(entry, metric)
            rows.append(row)
    return pd.DataFrame(rows)


def babd_timing_data(frame):
    """Per-seed fold-mean fit time and prediction seconds per 1,000 test objects."""
    d=frame.copy()
    if not d.complete.eq(True).all() or (d.completed_folds<=0).any() or (d.n_test<=0).any():
        raise ValueError('Incomplete timing records')
    d['fit_per_fold']=d.fit_seconds/d.completed_folds
    d['predict_per_1000']=1000*d.predict_seconds/d.n_test
    rows=[]
    for (experiment,model,budget),g in d.groupby(['experiment','model','train_budget']):
        if set(g.seed)!={11,37,53} or len(g)!=3:
            raise ValueError('Expected three unique seeds')
        row=dict(experiment=experiment,model=model,train_budget=budget,max_bins=20,n=3)
        for metric in ['fit_per_fold','predict_per_1000']:
            if g[metric].isna().any() or (g[metric]<=0).any():
                raise ValueError('Invalid runtime')
            row[metric+'_mean']=g[metric].mean()
            row[metric+'_std']=g[metric].std(ddof=1)
            for r in g.itertuples():row[metric+'_seed_'+str(int(r.seed))]=getattr(r,metric)
        rows.append(row)
    result=pd.DataFrame(rows)
    if len(result)!=42:raise ValueError('Expected 2 protocols x 3 models x 7 budgets')
    return result
