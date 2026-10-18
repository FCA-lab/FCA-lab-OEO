"""Compare direct rule extraction against extraction from all OEO concepts."""
import argparse
import contextlib
import hashlib
from functools import partial
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import completeness_config as config
from ..common.provenance import collect_provenance
from oeo_lab.preprocessing.encoding import LeakFreeContextEncoder
# 2026-10-06：任务生成与恢复/汇总采用一致的身份比较，排除调度参数。
from ..common.checkpoints import task_id, same_task, digest, save, completed
from ..common.runner import run_tasks
from fca_core.context import formal_context
from fca_core.data_source import data_set
from oeo_lab.algorithms.classification.classifier import RC_3Woc
from oeo_lab.algorithms.construction.three_woc import NewThreeWayForObj

TYPES = ('A1', 'A2', 'B1', 'B2')


def load_dataset(spec):
    """Fit the extraction context on all rows; this is not prediction evaluation."""
    path = Path(spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Dataset changed after task creation')
    frame = pd.read_csv(path, header=None, dtype=str,
                        sep=r'\s+' if spec['format'] == 'whitespace' else ',')
    if frame.empty or frame.isna().any().any():
        raise ValueError('Empty data or missing values')
    # 有表头的数据显式跳过表头，列仍用位置编号，兼容既有配置。
    if spec.get('header', False):
        frame = frame.iloc[1:].reset_index(drop=True)
        if frame.empty:
            raise ValueError('Dataset has a header but no data rows')
    decision = spec['decision_column']
    y = frame[decision].to_numpy()
    features = frame.drop(columns=[decision] + spec['id_columns'])
    # 完备性实验使用整个数据集构造背景；预测实验须在训练折内拟合。
    X, metadata = preprocess_features(features, spec)
    metadata['n_features_raw'] = int(features.shape[1])  # 已排除决策列和 ID 列。
    return X, y, metadata


def preprocess_features(features, spec):
    """依据特征类型选择编码方式，不改动样本数及类别。"""
    if spec['feature_kind'] == 'numerical':
        return encode_numerical(features, spec)
    if spec['feature_kind'] == 'categorical':
        return encode_categorical(features)
    raise ValueError('Unknown feature kind')


def encode_numerical(features, spec):
    """复用现有分箱器；记录边界与列映射，便于重建同一背景。"""
    features = features.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Nonfinite numerical features')
    if spec['max_bins'] < 3:
        raise ValueError('The legacy zero-bin encoder requires max_bins >= 3 here')
    encoder = LeakFreeContextEncoder(col_names=list(features.columns),
                                     max_bin=spec['max_bins'],
                                     min_data_in_bin=spec['min_data_in_bin'])
    encoded = encoder.fit_transform(features)
    # JSON-safe final infinite bound is represented by null.
    metadata = {'kind': 'numerical', 'max_bins': spec['max_bins'],
                'min_data_in_bin': spec['min_data_in_bin'],
                'columns': [str(c) for c in encoder.columns_],
                'bin_upper_bounds': {str(c): [float(b) if np.isfinite(b) else None for b in bounds]
                                     for c, bounds in encoder.bins_.items()}}
    return encoded.to_numpy(dtype=np.int8), metadata


def encode_categorical(features):
    """复用公共类别编码；完备性实验在整个背景上拟合。"""
    from oeo_lab.preprocessing.categorical import encode_categorical as encode
    return encode(features)


def expand_rules(rule_set):
    """把四组压缩前件展开为 (类型, 属性索引, 类别索引)，并去重。"""
    return {(kind, int(a), int(n)) for n, groups in rule_set.items()
            for kind, attributes in zip(TYPES, groups) for a in attributes}


def full_rules(concepts, class_masks, attribute_count):
    """Extract from EVERY eligible concept, including the empty extent."""
    rules = set()
    universe = 0
    for mask in class_masks:
        universe |= mask
    for extent, positive, negative in concepts:
        for n, target in enumerate(class_masks):
            for boundary, kinds in ((target, ('A1', 'A2')),
                                    (universe ^ target, ('B1', 'B2'))):
                if extent & ~boundary == 0:
                    for intent, kind in zip((positive, negative), kinds):
                        rules.update((kind, a, n) for a in range(attribute_count)
                                     if intent >> a & 1)
    return rules


def extract_direct_rules(X, y):
    """直接路线：计时覆盖模型初始化及 fit；展开规则在计时后完成。"""
    start = time.perf_counter()
    model = RC_3Woc().fit(X, y)
    seconds = time.perf_counter() - start
    return model, seconds


def extract_full_lattice_rules(X, y, classes):
    """完整路线：分别统计概念构造和遍历所有合格概念提取规则的时间。"""
    start = time.perf_counter()
    builder = NewThreeWayForObj(formal_context(data_set(X.tolist()), 1), 1)
    builder.start()
    construction_seconds = time.perf_counter() - start
    start = time.perf_counter()
    concepts = [(int(c[0]), int(c[1][0]), int(c[1][1]))
                for bucket in builder.ref.values() for c in bucket.values()]
    class_masks = [sum(1 << i for i, label in enumerate(y) if label == n)
                   for n in classes]
    rules = full_rules(concepts, class_masks, X.shape[1])
    extraction_seconds = time.perf_counter() - start
    return builder, concepts, class_masks, rules, construction_seconds, extraction_seconds


def compare_context(X, y, on_direct_complete=None):
    """在同一背景执行两条路线；计时之外核验支持集、概念及规则差集。"""
    X = np.asarray(X)
    y = np.asarray(y)
    if X.ndim != 2 or not X.size or not np.isin(X, [0, 1]).all():
        raise ValueError('A nonempty binary context is required')
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError('Exactly one decision per object is required')
    # 算法内部保持单线程；任务级并行由公共 runner 控制。
    with threadpool_limits(limits=1), open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        model, direct_seconds = extract_direct_rules(X, y)
        # 直接路线完成即落盘；完整构造失败或中断不会丢失该阶段结果。
        if on_direct_complete is not None:
            rules = expand_rules(model.RuleSet_)
            on_direct_complete({
                'metrics': dict(n_objects=int(X.shape[0]), n_attributes=int(X.shape[1]),
                                density=float(np.count_nonzero(X) / X.size),
                                direct_seconds=direct_seconds, direct_rules=len(rules),
                                **{'direct_' + kind: sum(r[0] == kind for r in rules)
                                   for kind in TYPES}),
                'classes': model.classes_.tolist(), 'direct_rules': sorted(rules)})
        builder, concepts, class_masks, reference, construction_seconds, extraction_seconds = (
            extract_full_lattice_rules(X, y, model.classes_))
    # Normalization, independent semantic checks and reporting are outside timing.
    direct = expand_rules(model.RuleSet_)
    universe = (1 << len(X)) - 1
    supports = [sum(1 << i for i in range(len(X)) if X[i, a]) for a in range(X.shape[1])]
    oracle = set()
    for n, target in enumerate(class_masks):
        for a, support in enumerate(supports):
            for kind, antecedent, boundary in (
                ('A1', support, target), ('A2', universe ^ support, target),
                ('B1', support, universe ^ target), ('B2', universe ^ support, universe ^ target)):
                if antecedent & ~boundary == 0:
                    oracle.add((kind, a, n))
    invalid = 0
    for extent, positive, negative in concepts:
        ep = sum(1 << a for a, s in enumerate(supports) if s & ~extent == 0)
        en = sum(1 << a for a, s in enumerate(supports) if (universe ^ s) & ~extent == 0)
        recovered = 0
        for a, s in enumerate(supports):
            if positive >> a & 1: recovered |= s
            if negative >> a & 1: recovered |= universe ^ s
        invalid += int(positive != ep or negative != en or recovered != extent)
    base_unique = len({(tuple(sorted(x)), tuple(sorted(a)), tuple(sorted(b)))
                       for x, (a, b) in builder.base_OEOs})
    missing, extra = reference - direct, direct - reference
    row = dict(objects=len(X), attributes=X.shape[1], classes=len(model.classes_),
               n_objects=int(X.shape[0]), n_attributes=int(X.shape[1]),
               density=float(np.count_nonzero(X) / X.size) if X.size else None,
               context_sha256=digest({'X': X.tolist(), 'y': y.tolist()}),
               direct_seconds=direct_seconds, construction_seconds=construction_seconds,
               full_extraction_seconds=extraction_seconds,
               full_total_seconds=construction_seconds + extraction_seconds,
               basic_concept_entries=len(builder.base_OEOs), basic_concepts_unique=base_unique,
               full_concepts=len(concepts), invalid_concepts=invalid,
               duplicate_concepts=len(concepts)-len(set(concepts)),
               direct_rules=len(direct), full_rules=len(reference),
               missing_rules=len(missing), extra_rules=len(extra),
               covered=not missing, direct_oracle_match=direct == oracle,
               full_oracle_match=reference == oracle)
    row['comparison'] = ('pass' if not missing and not extra and not invalid
                         and row['duplicate_concepts'] == 0 and direct == oracle == reference else 'fail')
    for kind in TYPES:
        row['direct_' + kind] = sum(r[0] == kind for r in direct)
        row['full_' + kind] = sum(r[0] == kind for r in reference)
    return {'metrics': row, 'classes': model.classes_.tolist(),
            'direct_rules': sorted(direct), 'full_rules': sorted(reference),
            'missing_rules': sorted(missing), 'extra_rules': sorted(extra),
            'direct_oracle_difference': sorted(direct ^ oracle),
            'full_oracle_difference': sorted(reference ^ oracle)}


def execute(task, checkpoint_directory=None):
    """单个工作进程入口：预处理不计入规则提取时间。"""
    X, y, names = load_dataset(task['dataset'])
    def record_direct(result):
        result['metrics']['n_features_raw'] = names['n_features_raw']
        result['attributes'] = names
        # 单独的原子检查点，避免被公共调度器的失败/超时记录覆盖。
        path = Path(checkpoint_directory) / config.OUTPUT['direct_checkpoint_template'].format(task_id=task['id'])
        save(path, {'task': task, 'status': 'direct_complete', 'result': result})
        print(f"[直接提取完成] {task['name']} bins={task['dataset'].get('max_bins')} "
              f"time={result['metrics']['direct_seconds']:.6f}s "
              f"rules={result['metrics']['direct_rules']}; 已保存，开始完整构造",
              file=sys.__stdout__, flush=True)
    result = compare_context(X, y, record_direct if checkpoint_directory is not None else None)
    result['metrics']['n_features_raw'] = names['n_features_raw']
    result['attributes'] = names
    return result


def summarize(tasks, directory, output):
    """从检查点生成结果与差集表；未完成任务不写为通过。"""
    rows, differences = [], []
    for task in tasks:
        path = directory / config.OUTPUT['checkpoint_template'].format(task_id=task['id'])
        record = json.loads(path.read_text()) if path.exists() else {}
        if not same_task(record.get('task'), task):
            record = {}
        row = {'task_id': task['id'], 'dataset': task['name'], 'max_bins': task['dataset'].get('max_bins'),
               'repeat': task.get('repeat', 0), 'status': record.get('status', 'pending'), 'error': record.get('error')}
        # 任务未全部完成时仍汇总直接路线；不填造完整耗时或比较结论。
        direct_path = directory / config.OUTPUT['direct_checkpoint_template'].format(task_id=task['id'])
        direct_record = json.loads(direct_path.read_text()) if direct_path.exists() else {}
        row['direct_status'] = 'pending'
        if (same_task(direct_record.get('task'), task)
                and direct_record.get('status') == 'direct_complete'):
            row.update(direct_record['result']['metrics'])
            row['direct_status'] = 'ok'
        if row['status'] == 'ok':
            row['direct_status'] = 'ok'
            result = record['result']
            row.update(result['metrics'])
            for key in ('missing_rules', 'extra_rules', 'direct_oracle_difference', 'full_oracle_difference'):
                for kind, attribute, label in result[key]:
                    differences.append({'dataset': task['name'], 'difference': key,
                                        'type': kind, 'attribute': attribute,
                                        'class_index': label, 'class_label': result['classes'][label]})
        rows.append(row)
    tmp = output.with_name(output.stem + '.tmp.xlsx')
    with pd.ExcelWriter(tmp, engine='openpyxl') as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='results', index=False)
        pd.DataFrame(differences, columns=['dataset', 'difference', 'type', 'attribute',
                                         'class_index', 'class_label']).to_excel(writer, sheet_name='differences', index=False)
    os.replace(tmp, output)
    return all(row['status'] == 'ok' and row.get('comparison') == 'pass' for row in rows)


def parse_args(argv=None):
    """解析并验证命令行；显式参数优先于独立配置。"""
    defaults = dict(config.EXECUTION, **config.COMPLETENESS['execution'])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=list(config.DATASETS),
                        default=list(config.DATASETS), help='Default: all registered datasets')
    parser.add_argument('--results-root', type=Path, default=Path(config.OUTPUT['root']))
    parser.add_argument('--n-jobs', type=int, default=defaults['n_jobs'])
    parser.add_argument('--timeout', type=float, default=defaults['task_timeout_sec'])
    parser.add_argument('--max-bins', type=int, nargs='+',
                        help='Override numerical dataset bin grids; categorical datasets are unchanged')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if not args.datasets:
        parser.error('No datasets registered in DATASETS')
    if args.n_jobs < 1 or (args.timeout is not None and args.timeout <= 0):
        parser.error('n-jobs and timeout must be positive')
    if args.max_bins is not None and any(v < 3 for v in args.max_bins):
        parser.error('--max-bins must be >= 3 for the current zero-bin encoder')
    settings = dict(defaults, n_jobs=args.n_jobs, task_timeout_sec=args.timeout)
    if (config.COMPLETENESS['routes'] != ['direct', 'full_lattice'] or
        not config.COMPLETENESS['retain_zero_support'] or not config.COMPLETENESS['deduplicate_rules'] or
        config.COMPLETENESS['rule_key'] != ['type', 'attribute', 'class']):
        parser.error('This experiment requires both routes and all deduplicated single-literal rules')
    return args, settings


def check_environment(settings):
    """检查依赖与源码，记录本次实际使用的运行环境。"""
    return collect_provenance(config.PROJECT_ROOT, config.SOURCE_PATTERNS,
                              config.PROVENANCE_PACKAGES, settings)


def build_tasks(args, provenance):
    """按数据集 × 分箱值 × 重复编号生成任务；输入哈希防止错误续跑。"""
    tasks = []
    for name in dict.fromkeys(args.datasets):
        spec = dict(config.DATASETS[name], path=str(config.dataset_path(name)))
        spec['sha256'] = hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()
        grid = [None]
        if spec['feature_kind'] == 'numerical':
            # 公共默认值 → 数据集局部设置 → 命令行分箱覆盖。
            preprocessing = dict(config.COMPLETENESS['preprocessing'])
            preprocessing.update(spec.get('preprocessing', {}))
            grid = args.max_bins if args.max_bins is not None else preprocessing['max_bins']
            if not grid or any(not isinstance(v, int) or v < 3 for v in grid):
                raise ValueError('Numerical completeness max_bins must be integers >= 3')
            spec['min_data_in_bin'] = preprocessing['min_data_in_bin']
        repeats = config.COMPLETENESS['repeats']
        if not isinstance(repeats, int) or repeats < 1:
            raise ValueError('repeats must be a positive integer')
        for bins in dict.fromkeys(grid):
            for repeat in range(repeats):
                variant = dict(spec)
                if bins is not None:
                    variant['max_bins'] = bins
                task = {'experiment': 'rule_completeness', 'name': name, 'dataset': variant,
                        'repeat': repeat, 'provenance': provenance}
                task['id'] = task_id(task)
                tasks.append(task)
    return tasks


def run_experiment(tasks, settings, out):
    """保存任务索引、恢复中间结果；无论是否中断均汇总已有记录。"""
    out.mkdir(parents=True, exist_ok=True)
    directory = out / config.OUTPUT['checkpoint_dir']
    index = directory / config.OUTPUT['checkpoint_index']
    snapshot = json.loads(index.read_text()) if index.exists() else {'tasks': []}
    snapshot['tasks'] = [t for t in snapshot['tasks'] if t.get('experiment') != 'rule_completeness'] + tasks
    save(index, snapshot)
    # A mismatched/incomplete old record must not mask a crashed retry.
    for task in tasks:
        direct_path = directory / config.OUTPUT['direct_checkpoint_template'].format(task_id=task['id'])
        if not settings['resume'] and direct_path.exists():
            direct_path.unlink()
        path = directory / config.OUTPUT['checkpoint_template'].format(task_id=task['id'])
        if path.exists() and (not settings['resume'] or not completed(path, task)):
            old = json.loads(path.read_text())
            archive = path.with_name(path.stem + '.previous-' + str(time.time_ns()) + '.json')
            save(archive, old)
            path.unlink()
    try:
        worker = partial(execute, checkpoint_directory=str(directory))
        run_tasks(tasks, worker, directory, settings, config.OUTPUT['checkpoint_template'])
    finally:
        passed = summarize(tasks, directory, out/config.OUTPUT['files']['completeness'])
    if not passed:
        raise SystemExit('Incomplete or failed comparison; inspect workbook and checkpoints')
    print(out/config.OUTPUT['files']['completeness'])


def main(argv=None):
    """主流程只负责组织步骤，具体逻辑交给职责明确的函数。"""
    args, settings = parse_args(argv)
    provenance = check_environment(settings)
    tasks = build_tasks(args, provenance)
    print(f'{len(tasks)} paired tasks: ' + ', '.join(t['name'] for t in tasks), flush=True)
    if args.dry_run:
        for task in tasks:
            print(f"  {task['name']}: max_bins={task['dataset'].get('max_bins', 'categorical')}, repeat={task['repeat']}")
        return
    run_experiment(tasks, settings, args.results_root)


if __name__ == '__main__':
    # 本次实验的运行参数：在此修改即可；未列出的选项仍读取配置文件。
    max_bins = [5, 8, 10]
    experiment_args = [
    '--max-bins', *map(str, max_bins),
    '--n-jobs', '20',
    ]
    # 先读取脚本设置，终端仅覆盖显式提供的同名参数。
    main(experiment_args + sys.argv[1:])
