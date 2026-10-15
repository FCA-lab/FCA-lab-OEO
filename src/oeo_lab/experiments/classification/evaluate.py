# -*- coding: utf-8 -*-
"""单任务执行(细粒度): (max_bin, frac, 单模型, 单种子) 跑完一个种子的 5 折。

并行粒度从 (frac × model) 细化到 (frac × model × seed), 任务数 ×len(seeds),
慢模型(SVM/MLP/XGB/LGBM)被切成单种子小任务, 大幅缩短批次末端长尾、吃满核心。

每折流程(无泄漏):
  1) 在原始连续特征上做 StratifiedKFold 划分;
  2) context 模型: 编码器仅在训练折 fit, 测试折只 transform;
     raw 模型: 直接用原始连续特征(不分箱);
  3) 训练集去重: 仅对【需要集合语义的规则方法 rc_3woc】开启, 且仅作用于训练集;
  4) 训练 -> 预测 -> 记录指标。

注意: 单种子任务只产出"原始折行 + 该种子混淆矩阵", 不在此做跨种子聚合;
      mean/std 汇总与混淆矩阵累加由 runner 在收齐一个 (frac, model) 的全部种子后统一完成,
      产出的 xlsx 与旧版逐模型聚合完全一致。
"""
import gc
import time
from copy import deepcopy
from contextlib import nullcontext

import numpy as np
import pandas as pd
from sklearn.base import clone
#from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, matthews_corrcoef)

from .config import COL_NAMES
from .datasets import load_raw, sample_xy, dedup_xy
from .baselines import build_models
from oeo_lab.preprocessing.encoding import LeakFreeContextEncoder

try:
    from threadpoolctl import threadpool_limits
except ImportError:
    threadpool_limits = None

# 仅这些模型需要"相同对象只算一次"的集合语义去重; 其余模型一律不去重。
SET_SEMANTICS_MODELS = {"rc_3woc"}
# classification_report(output_dict) 里的汇总行, 不应混入"每类"报表
_REPORT_AGG_KEYS = {"accuracy", "macro avg", "weighted avg", "micro avg", "samples avg"}

# 参与跨种子聚合(mean/std)的指标列
METRIC_COLS = ["准确率", "宏平均精确率", "宏平均召回率", "宏平均F1值", "加权F1值",
               "MCC", "Runtime", "Num of rules", "条件属性数M", "训练集去重数"]


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


def _fresh(model):
    try:
        return clone(model)
    except TypeError:
        return deepcopy(model)


def _force_single_thread(model):
    try:
        params = model.get_params()
        targets = {k: 1 for k in params if k == "n_jobs" or k.endswith("__n_jobs")}
        if targets:
            model.set_params(**targets)
    except Exception:
        pass
    return model


def _encode_fold(input_type, X_tr_raw, X_te_raw, max_bin):
    """返回 (X_tr, X_te, M_size)。context -> 无泄漏编码; raw -> 原始连续特征。"""
    if input_type == "context":
        enc = LeakFreeContextEncoder(COL_NAMES, max_bin)
        X_tr = enc.fit_transform(X_tr_raw)          # 仅训练折 fit
        X_te = enc.transform(X_te_raw)              # 测试折只 transform
        return X_tr, X_te, enc.get_num_attributes()
    else:  # raw: 不分箱, 直接用连续特征
        return X_tr_raw.reset_index(drop=True), X_te_raw.reset_index(drop=True), X_tr_raw.shape[1]


# ===== 细粒度并行任务: 只跑一个种子的 5 折 =====
def run_one_seed(dataset_csv, max_bin, frac, model_name, cv_seed,
                 fold, random_seed, dedup_train=True, protocol="address"):
    """protocol: 'address' = 地址级随机分层切分(已知实体场景, 原协议);
                'entity'  = 实体级分组切分(未知实体场景, 同一 label_id
                            的全部地址不跨训练/测试集, 无实体泄漏)。"""
    try:
        raw_df, labels = load_raw(dataset_csv)
        estimator, input_type = build_models(random_seed)[model_name]
        do_dedup = dedup_train and (model_name in SET_SEMANTICS_MODELS)

        fold_rows, report_rows = [], []
        cm = np.zeros((len(labels), len(labels)), dtype=int)
        last_shape = None

        limiter = threadpool_limits(limits=1) if threadpool_limits else nullcontext()
        with limiter:
            try:
                X, y, groups = sample_xy(raw_df, frac, cv_seed, return_groups=True)
                last_shape = (X.shape[0], X.shape[1])
                if protocol == "entity":
                    cv = StratifiedGroupKFold(n_splits=fold, shuffle=True, random_state=cv_seed)
                    splits = list(cv.split(X, y, groups=groups))
                else:
                    cv = StratifiedKFold(n_splits=fold, shuffle=True, random_state=cv_seed)
                    splits = list(cv.split(X, y))
            except Exception as exc:
                print(f"[bin={max_bin} frac={frac}] {model_name} seed{cv_seed} 采样/切分失败,跳过: {exc}")
                return _seed_result(max_bin, frac, model_name, input_type, cv_seed,
                                    fold_rows, report_rows, cm, labels, last_shape)

            for fi, (tr, te) in enumerate(splits, start=1):
                X_tr_raw, X_te_raw = X.iloc[tr], X.iloc[te]
                y_tr, y_te = y.iloc[tr].reset_index(drop=True), y.iloc[te].reset_index(drop=True)
                t0 = time.time()
                try:
                    X_tr, X_te, M_size = _encode_fold(input_type, X_tr_raw, X_te_raw, max_bin)
                    if do_dedup:
                        n_before = len(X_tr)
                        X_tr, y_tr = dedup_xy(X_tr, y_tr)
                        n_dropped = n_before - len(X_tr)
                    else:
                        n_dropped = 0
                    m = _force_single_thread(_fresh(estimator))
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_te)
                    rt = time.time() - t0
                    rules_n = m.get_rules_num() if hasattr(m, "get_rules_num") else 0
                    fold_rows.append({
                        "数据集": f"formalContext_{max_bin}", "max_bin": max_bin,
                        "协议": protocol,
                        "输入类型": input_type, "样本比例": frac, "模型": model_name,
                        "随机种子": cv_seed, "折数": f"第{fi}折",
                        "训练集大小": len(X_tr), "测试集大小": len(X_te),
                        "训练集去重数": n_dropped, "条件属性数M": M_size,
                        "准确率": accuracy_score(y_te, y_pred),
                        "宏平均精确率": precision_score(y_te, y_pred, average="macro", zero_division=0),
                        "宏平均召回率": recall_score(y_te, y_pred, average="macro", zero_division=0),
                        "宏平均F1值": f1_score(y_te, y_pred, average="macro", zero_division=0),
                        "加权F1值": f1_score(y_te, y_pred, average="weighted", zero_division=0),
                        "MCC": matthews_corrcoef(y_te, y_pred),
                        "Runtime": rt, "Num of rules": rules_n, "状态": "成功", "错误信息": "",
                    })
                    cm += confusion_matrix(y_te, y_pred, labels=labels)
                    rep = classification_report(y_te, y_pred, labels=labels, output_dict=True, zero_division=0)
                    for cls, met in rep.items():
                        if isinstance(met, dict) and cls not in _REPORT_AGG_KEYS:
                            report_rows.append({
                                "max_bin": max_bin, "协议": protocol, "样本比例": frac,
                                "模型": model_name,
                                "随机种子": cv_seed, "折数": f"第{fi}折", "类别": cls,
                                "precision": met.get("precision"), "recall": met.get("recall"),
                                "f1-score": met.get("f1-score"), "support": met.get("support"),
                            })
                except Exception as exc:
                    fold_rows.append({
                        "数据集": f"formalContext_{max_bin}", "max_bin": max_bin,
                        "协议": protocol,
                        "输入类型": input_type, "样本比例": frac, "模型": model_name,
                        "随机种子": cv_seed, "折数": f"第{fi}折",
                        "状态": "失败", "错误信息": str(exc), "Runtime": time.time() - t0,
                    })
                    print(f"[bin={max_bin} frac={frac}] {model_name} seed{cv_seed} 第{fi}折失败: {exc}")

        print(f"✓ 完成 [bin={max_bin} frac={frac} {protocol}] {model_name}({input_type}) seed{cv_seed} 规模:{last_shape}")
        gc.collect()          # loky 复用 worker 进程, 主动回收编码/训练的中间大对象, 防 RSS 逐任务爬升
        return _seed_result(max_bin, frac, model_name, input_type, cv_seed,
                            fold_rows, report_rows, cm, labels, last_shape)
    except Exception as exc:
        print(f"[bin={max_bin} frac={frac}] {model_name} seed{cv_seed} 整体失败: {exc}")
        gc.collect()          # loky 复用 worker 进程, 主动回收编码/训练的中间大对象, 防 RSS 逐任务爬升
        return {"max_bin": max_bin, "frac": frac, "model_name": model_name,
                "input_type": "?", "cv_seed": cv_seed, "labels": [],
                "fold_rows": [{"max_bin": max_bin, "协议": protocol, "样本比例": frac,
                               "模型": model_name,
                               "随机种子": cv_seed, "状态": "失败", "错误信息": str(exc)}],
                "report_rows": [], "cm": None}


def _seed_result(max_bin, frac, model_name, input_type, cv_seed,
                 fold_rows, report_rows, cm, labels, last_shape):
    return {"max_bin": max_bin, "frac": frac, "model_name": model_name,
            "input_type": input_type, "cv_seed": cv_seed, "labels": list(labels),
            "fold_rows": fold_rows, "report_rows": report_rows, "cm": cm}


# ===== 跨种子聚合: runner 收齐一个 (frac, model) 的全部种子后调用 =====
def aggregate_model(seed_results, n_seeds_expected, fold):
    """把同一 (max_bin, frac, model) 的多个单种子结果合并成
    与旧版一致的 (fold_rows, summary_row, confusion_rows, report_rows)。"""
    seed_results = sorted(seed_results, key=lambda r: r["cv_seed"])
    r0 = seed_results[0]
    max_bin, frac, model_name = r0["max_bin"], r0["frac"], r0["model_name"]
    input_type = next((r["input_type"] for r in seed_results if r["input_type"] != "?"), "?")
    labels = next((r["labels"] for r in seed_results if r["labels"]), [])

    fold_rows, report_rows = [], []
    cm_total = None
    for r in seed_results:
        fold_rows += r["fold_rows"]
        report_rows += r["report_rows"]
        if r["cm"] is not None:
            cm_total = r["cm"].copy() if cm_total is None else cm_total + r["cm"]

    df = pd.DataFrame(fold_rows)
    ok = df[df["状态"] == "成功"] if (not df.empty and "状态" in df) else df
    summary = {"数据集": f"formalContext_{max_bin}", "max_bin": max_bin,
               "输入类型": input_type, "样本比例": frac, "模型": model_name,
               "CV随机种子数量": n_seeds_expected, "每个随机种子折数": fold,
               "总评估次数": len(df), "成功折数": len(ok), "失败折数": len(df) - len(ok)}
    if not ok.empty:
        for c in METRIC_COLS:
            if c in ok:
                summary[f"{c}_mean"], summary[f"{c}_std"] = ok[c].mean(), ok[c].std()

    confusion_rows = []
    if cm_total is not None and labels:
        for i, al in enumerate(labels):
            row = {"max_bin": max_bin, "样本比例": frac, "模型": model_name, "实际类别": al}
            for j, pl in enumerate(labels):
                row[f"预测类别{pl}"] = cm_total[i][j]
            confusion_rows.append(row)

    return {"max_bin": max_bin, "frac": frac, "model_name": model_name,
            "fold_rows": fold_rows, "summary_row": summary,
            "confusion_rows": confusion_rows, "report_rows": report_rows}


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
    if task['dataset'].get('kind') == 'bitcoin':
        from .bitcoin_data import load_bitcoin_dataset
        X,y,labels,_ = load_bitcoin_dataset(task['dataset'])
    else:
        X,y,labels = load_numeric_dataset(task['dataset'])
    train, test = task['split']['train_indices'], task['split']['test_indices']
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y[train], y[test]
    encoder = None
    start = time.perf_counter()
    if task['model']['input'] == 'context':
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
        if is_oeo:
            prediction, candidates = model.predict(X_test, return_candidates=True)
        else:
            prediction = model.predict(X_test)
            candidates = [(label,) for label in prediction]
        predict_seconds = time.perf_counter() - start
        scores = evaluate_predictions(y_test, prediction, list(range(len(labels))))
        scores.update(evaluate_candidates(y_test, candidates, len(labels)))
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
