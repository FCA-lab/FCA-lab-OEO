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


def candidate_data(frame):
    """从已有计数精确恢复三组分布，不能据均值推断逐大小直方图。"""
    validate_dimensions(frame, set())
    rows = []
    for _, r in frame.iterrows():
        counts = [r.n_classified, r.partial_count, r.n_test-r.n_classified-r.partial_count]
        hits = [r.n_correct, r.partial_hits, r.candidate_hits-r.n_correct-r.partial_hits]
        for label, count, hit in zip(['Unique class','Partial set','All classes'], counts, hits):
            if not 0 <= hit <= count or r.n_test <= 0:
                raise ValueError('Inconsistent candidate counts')
            rows.append(dict(category=label, hit=hit/r.n_test, miss=(count-hit)/r.n_test))
    # 每个种子先归一化，再等权平均，与 per_seed 统计口径一致。
    return pd.DataFrame(rows).groupby('category', sort=False)[['hit','miss']].mean().reset_index()


def ablation_data(frame, metric, reference):
    validate_dimensions(frame, {'dataset','rule_group'})
    rows = []
    for dataset, group in frame.groupby('dataset', sort=False):
        pivot = group.pivot(index='seed', columns='rule_group', values=metric)
        if reference not in pivot or pivot.isna().any().any():
            raise ValueError('Missing reference or unpaired ablation measurements')
        delta = pivot.subtract(pivot[reference], axis=0).drop(columns=reference)
        for variant in delta:
            rows.append(dict(dataset=dataset, rule_group=variant, delta=delta[variant].mean()))
    return pd.DataFrame(rows).pivot(index='dataset',columns='rule_group',values='delta')
