"""Evaluate current classification tasks and retain additive metric counts."""
import time
import numpy as np
import pandas as pd
from oeo_lab.preprocessing.encoding import LeakFreeContextEncoder

def evaluate_predictions(y_true, y_pred, labels):
    """统一评价普通分类与延迟分类；None 专用于 defer，不能作为真实类别。

    labels 必须传入整个任务的类别集合，不能按当前折或预测结果缩减。
    返回四项指标、原始计数和每类 TP/FP/FN；不把 defer 作为额外类别。
    空测试集与未知标签视为输入错误；全部 defer 时条件准确率为 NaN。
    """
    truth = np.asarray(y_true, dtype=object)
    prediction = np.asarray(y_pred, dtype=object)
    classes = np.asarray(labels, dtype=object)
    if truth.ndim != 1 or prediction.ndim != 1 or classes.ndim != 1:
        raise ValueError('Labels and predictions must be one-dimensional')
    if not len(truth) or len(truth) != len(prediction) or not len(classes):
        raise ValueError('Nonempty, equally sized truth/prediction arrays are required')
    # 不通过字符串转换统一类别，避免把整数标签与同名字符串混为一类。
    lookup = {}
    for i, label in enumerate(classes):
        if label is None or pd.isna(label):
            raise ValueError('Task classes cannot contain None or missing values')
        if label in lookup:
            raise ValueError('Duplicate task class')
        lookup[label] = i
    actual, predicted = [], []
    for true, pred in zip(truth, prediction):
        if true is None or pd.isna(true) or true not in lookup:
            raise ValueError(f'Unknown or missing true class: {true!r}')
        actual.append(lookup[true])
        if pred is None:
            predicted.append(-1)  # 内部索引，不占用任何用户类别值。
        elif pd.isna(pred) or pred not in lookup:
            raise ValueError(f'Unknown or missing predicted class: {pred!r}')
        else:
            predicted.append(lookup[pred])
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    classified = predicted >= 0
    total = len(actual)
    accepted = int(classified.sum())
    correct = int((actual == predicted).sum())
    per_class = []
    for i, label in enumerate(classes):
        tp = int(((actual == i) & (predicted == i)).sum())
        fp = int(((actual != i) & (predicted == i)).sum())
        # defer 对真实类别贡献 FN，但不对任何类别贡献 FP。
        fn = int(((actual == i) & (predicted != i)).sum())
        denominator = 2 * tp + fp + fn
        per_class.append({'label': label, 'tp': tp, 'fp': fp, 'fn': fn,
                          'support': int((actual == i).sum()),
                          'deferred': int(((actual == i) & ~classified).sum()),
                          'f1': 2 * tp / denominator if denominator else 0.0})
    return {
        'n_test': total, 'n_classified': accepted, 'n_correct': correct,
        'n_deferred': total - accepted,
        'coverage': accepted / total,
        'classified_accuracy': correct / accepted if accepted else float('nan'),
        'overall_correct_rate': correct / total,
        'macro_f1': float(np.mean([row['f1'] for row in per_class])),
        'macro_precision': float(np.mean([r['tp']/(r['tp']+r['fp']) if r['tp']+r['fp'] else 0.0 for r in per_class])),
        'macro_recall': float(np.mean([r['tp']/(r['tp']+r['fn']) if r['tp']+r['fn'] else 0.0 for r in per_class])),
        'per_class': per_class,
    }


def _rule_audit(model, X_test, y_test):
    """逐条单文字项规则统计支持度及测试触发正确数，不用测试结果筛选规则。"""
    records = []
    for n, groups in model.RuleSet_.items():
        for kind, attributes in zip(('A1','A2','B1','B2'), groups):
            for a in sorted(attributes):
                positive_literal = kind in ('A1','B1')
                trigger = X_test[:, a] == (1 if positive_literal else 0)
                correct = y_test == model.classes_[n]
                if kind in ('B1','B2'):
                    correct = ~correct
                support = (model.support_pos_ if positive_literal else model.support_neg_)[a]
                count = int(trigger.sum())
                right = int((trigger & correct).sum())
                records.append({'type':kind, 'attribute':int(a), 'class':int(model.classes_[n]),
                    'train_support':int(support), 'test_triggers':count, 'test_correct':right,
                    'test_precision':right/count if count else None})
    return records


def _defer_reasons(model, X_test, predictions, voting):
    """只诊断 defer 样本：无有效贡献，或有贡献但最高分并列。"""
    empty = ties = 0
    for sample, prediction in zip(X_test, predictions):
        if prediction is not None:
            continue
        present = set(np.flatnonzero(sample))
        evidence = False
        for U,V,Q,R in model.RuleSet_.values():
            for kind, (attributes, supports) in zip(('A1','A2','B1','B2'), ((U & present,model.support_pos_),
                    (V-present,model.support_neg_), (Q & present,model.support_pos_),
                    (R-present,model.support_neg_))):
                if kind not in model.rule_types:continue
                if any(voting == 'uniform' or supports[a] > 0 for a in attributes):
                    evidence = True
        if evidence: ties += 1
        else: empty += 1
    return empty, ties


def evaluate_fold(task):
    """一个任务=一个折×一个模型×一个输入设置；OEO 一次 fit 服务三种策略。"""
    from .datasets import load_numeric_dataset
    from .baselines import create_model
    if task['dataset'].get('feature_kind') == 'categorical':
        from .uci_data import load_uci_dataset
        X,y,labels = load_uci_dataset(task['dataset'])
    elif task['dataset'].get('kind') == 'elliptic':
        from .elliptic_experiment import load_elliptic_dataset
        X,y,labels,_,_ = load_elliptic_dataset(task['dataset'])
    elif task['dataset'].get('kind') == 'bitcoin':
        from .bitcoin_data import load_bitcoin_dataset
        X,y,labels,_ = load_bitcoin_dataset(task['dataset'])
    else:
        X,y,labels = load_numeric_dataset(task['dataset'])
    train, test = task['split']['train_indices'], task['split']['test_indices']
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y[train], y[test]
    encoder = None
    start = time.perf_counter()
    # 类别特征直接 one-hot；数值特征保留原分箱后 one-hot 的实现。
    categorical = task['dataset'].get('feature_kind') == 'categorical'
    if categorical or task['model']['input'] == 'context':
        if categorical:
            from oeo_lab.preprocessing.categorical import CategoricalContextEncoder
            encoder = CategoricalContextEncoder()
        else:
            encoder = LeakFreeContextEncoder(list(X.columns), task['max_bins'], task['min_data_in_bin'])
        X_train = encoder.fit_transform(X_train).to_numpy()
        preprocessing_fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        X_test = encoder.transform(X_test).to_numpy()
        preprocessing_test_seconds = time.perf_counter() - start
    else:
        X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
        preprocessing_fit_seconds = preprocessing_test_seconds = 0.0
    model = create_model(task['model'], task['split']['seed'])
    # LightGBM自动生成列名；显式保留相同列名与顺序，避免预测接口警告。
    if task['model']['estimator'] == 'lightgbm':
        columns = [f'feature_{i}' for i in range(X_train.shape[1])]
        X_train = pd.DataFrame(X_train, columns=columns)
        X_test = pd.DataFrame(X_test, columns=columns)
    start = time.perf_counter()
    model.fit(X_train,y_train)
    fit_seconds = time.perf_counter() - start
    is_oeo = task['model']['estimator']=='oeo'
    # 先提取完整规则一次，后续投票/消融仅改变预测，不再次 fit。
    rules = _rule_audit(model,X_test,y_test) if is_oeo else []
    variants = task.get('prediction_variants')
    if variants is None:  # 保留 Bitcoin 等现有入口。
        variants = [{'voting':v} for v in (task['voting'] if is_oeo else ['baseline'])]
    results = []
    for variant in variants:
        voting = variant['voting']
        if is_oeo:
            model.voting = voting
            model.rule_types = tuple(variant.get('rule_types', ('A1','A2','B1','B2')))
        start = time.perf_counter()
        ranking = None
        if is_oeo and 'positive_label' in task:
            # 同一次规则计分产生预测及排序分数，包含在决策耗时中。
            details = [model.predictWithRules(model.RuleSet_, sample, return_details=True) for sample in X_test]
            prediction = np.array([d['prediction'] for d in details], dtype=object)
            candidates = [d['candidates'] for d in details]
            pos = list(model.classes_).index(task['positive_label'])
            ranking = np.array([d['scores'][pos]-d['scores'][1-pos] for d in details])
        elif is_oeo:
            prediction, candidates = model.predict(X_test, return_candidates=True)
        else:
            prediction = normalize_class_predictions(model.predict(X_test), len(y_test))
            candidates = [(label,) for label in prediction]
        if not is_oeo and 'positive_label' in task:
            pos = list(model.classes_).index(task['positive_label'])
            if hasattr(model, 'predict_proba'):
                ranking = model.predict_proba(X_test)[:, pos]
            elif hasattr(model, 'decision_function'):
                ranking = model.decision_function(X_test) * (1 if pos == 1 else -1)
            else:
                raise ValueError('Binary evaluation requires continuous ranking scores')
        predict_seconds = time.perf_counter() - start
        scores = evaluate_predictions(y_test, prediction, list(range(len(labels))))
        scores.update(evaluate_candidates(y_test, candidates, len(labels)))
        if 'positive_label' in task:
            scores.update(evaluate_illicit(y_test, prediction, ranking, task['positive_label']))
        empty,ties = _defer_reasons(model,X_test,prediction,voting) if is_oeo else (0,0)
        assert empty+ties == scores['n_deferred']
        if not np.isfinite(scores['classified_accuracy']): scores['classified_accuracy'] = None
        row = {'voting':voting, **scores, 'predict_seconds':predict_seconds,
               'defer_no_evidence':empty, 'defer_tie':ties}
        if 'prediction_variants' in task:
            row['rule_group'] = variant['rule_group']
            row['enabled_rule_types'] = ','.join(variant['rule_types'])
            for kind in ('A1','A2','B1','B2'):
                row['rules_'+kind] = sum(r['type']==kind for r in rules) if kind in variant['rule_types'] else 0
            row['active_rule_count'] = sum(row['rules_'+k] for k in ('A1','A2','B1','B2')) if is_oeo else None
            # 相同预测可用于多个实验；时间是共享测量值，不代表重复训练。
            results.extend(dict(row, study=study) for study in variant['studies'])
        else:
            results.append(row)
    metadata = None if encoder is None else {
        'columns':[str(c) for c in encoder.columns_],
        'bins':{str(c):[float(b) if np.isfinite(b) else None for b in bounds]
                for c,bounds in encoder.bins_.items()}}
    return {'results':results, 'fit_seconds':fit_seconds,
        'preprocessing_fit_seconds':preprocessing_fit_seconds,
        'preprocessing_test_seconds':preprocessing_test_seconds,
        'train_seconds':preprocessing_fit_seconds+fit_seconds,
        'n_train':len(train), 'attributes':X_train.shape[1],
        'classes':labels, 'encoding':metadata, 'rules':rules,
        'rule_count':len(rules) if is_oeo else None,
        'zero_support_rules':sum(r['train_support']==0 for r in rules) if is_oeo else None}


def evaluate_candidates(y_true, candidates, n_classes):
    """保存可累加计数，先合并各折再计算集合指标；空分母返回 None。"""
    if len(y_true) != len(candidates) or not len(y_true):
        raise ValueError('Candidate sets must match the nonempty test set')
    sizes = np.array([len(c) for c in candidates])
    if np.any(sizes < 1) or np.any(sizes > n_classes):
        raise ValueError('Invalid candidate set size')
    hits = np.array([y in c for y,c in zip(y_true,candidates)])
    partial = (sizes > 1) & (sizes < n_classes)
    count = int(partial.sum())
    return {'candidate_hits':int(hits.sum()), 'candidate_size_sum':int(sizes.sum()),
            'partial_count':count, 'partial_hits':int((hits & partial).sum()),
            'candidate_hit_rate':float(hits.mean()), 'mean_candidate_size':float(sizes.mean()),
            'partial_reduction_rate':count/len(y_true),
            'partial_hit_rate':int((hits & partial).sum())/count if count else None}


def evaluate_illicit(y_true, prediction, ranking, positive_label=1):
    """非法类FN包括延迟决策；AP为阶梯积分，不冒称梯形PR曲线面积。"""
    from sklearn.metrics import average_precision_score
    truth=np.asarray(y_true)==positive_label
    predicted=np.asarray(prediction,dtype=object)==positive_label
    ranking=np.asarray(ranking,dtype=float)
    if ranking.shape!=truth.shape or not np.isfinite(ranking).all():
        raise ValueError('Invalid illicit ranking scores')
    tp=int(np.sum(truth & predicted));fp=int(np.sum(~truth & predicted));fn=int(np.sum(truth & ~predicted))
    return dict(illicit_tp=tp,illicit_fp=fp,illicit_fn=fn,
        illicit_precision=tp/(tp+fp) if tp+fp else 0.0,
        illicit_recall=tp/(tp+fn) if tp+fn else None,
        illicit_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
        illicit_average_precision=float(average_precision_score(truth,ranking)) if truth.any() else None)


def normalize_class_predictions(prediction, n_samples):
    """CatBoost可返回(n,1)类别列；仅压掉这一维，不扁平化概率矩阵。"""
    values = np.asarray(prediction)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or len(values) != n_samples:
        raise ValueError('Expected one class label per test sample')
    return values
