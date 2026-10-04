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