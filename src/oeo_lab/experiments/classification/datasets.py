# -*- coding: utf-8 -*-
"""数据加载、分层采样、训练集去重。

去重原则(审稿要求): 仅在【划分后】对【训练集】进行, 且 X/y 同步;
测试集绝不去重(否则破坏与 y_test 的 1:1 对齐, 无法正确评测)。

种子解耦(重要): 采样种子 = cv_seed*1000+7, 与 CV 切分种子错开。
    若两者同种子, pandas sample(frac=1.0) 的 permutation 与
    StratifiedKFold(shuffle) 的 Fisher-Yates 洗牌共享同一随机流,
    在 frac=1.0 时两个置换恰好抵消, 折号退化为"按原始行序的连续块",
    而原始 CSV 按实体排序 → 意外变成实体分组切分, 与其它 frac 不可比。
"""
import numpy as np
import pandas as pd

from .config import COL_NAMES, LABEL_FROM, LABEL_OFFSET, BITCOIN_CLASS_IDS

GROUP_COL = "label_id"   # 实体分组键: 同一实体(钱包/服务)控制的所有地址


def load_raw(dataset_csv):
    """读取原始连续特征(分箱前)。label = cate_id - 1, 按配置的原始类别清单过滤。"""
    df = pd.read_csv(dataset_csv).copy()
    # 使用明确类别集合，而不是上界比较，避免纳入意外的类别编号。
    df = df[df[LABEL_FROM].isin(BITCOIN_CLASS_IDS)].copy().reset_index(drop=True)
    df["label"] = df[LABEL_FROM] - LABEL_OFFSET
    labels = sorted(df["label"].unique())
    return df, labels


def sample_xy(raw_df, frac, seed, return_groups=False):
    """按 frac 分层采样原始特征; 采样种子与 CV 种子解耦(见模块 docstring)。
    返回连续特征 X 与标签 y; return_groups=True 时额外返回实体分组键 groups,
    三者行序严格对齐(同一 DataFrame 一次 reset_index 后切列)。"""
    sample_seed = seed * 1000 + 7          # 解耦: 绝不与 StratifiedKFold 同种子
    d = raw_df.groupby("label", group_keys=False)[raw_df.columns].apply(
        lambda x: x.sample(frac=frac, random_state=sample_seed))
    d = d.reset_index(drop=True)           # 统一重置一次, X/y/groups 天然对齐
    X = d[COL_NAMES]
    y = d["label"]
    if return_groups:
        return X, y, d[GROUP_COL]
    return X, y


def dedup_xy(X_df, y):
    """对 (特征行 + 标签) 整体去重, X/y 同步。仅用于训练集。
    适用于形式背景(二值)或原始连续特征两种表示。"""
    tmp = pd.DataFrame(X_df).copy().reset_index(drop=True)
    y_arr = np.asarray(y)
    tmp["__label__"] = y_arr
    keep = ~tmp.duplicated()
    X_out = pd.DataFrame(X_df).reset_index(drop=True).loc[keep.values].reset_index(drop=True)
    y_out = pd.Series(y_arr[keep.values]).reset_index(drop=True)
    return X_out, y_out


def load_numeric_dataset(spec):
    """读取 UCI 数值数据；不采样、不去重、不分箱。类别编码映射随结果返回。"""
    import hashlib
    from pathlib import Path
    path = Path(spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Dataset changed after task creation')
    frame = pd.read_csv(path, header=None, dtype=str)
    if frame.empty or frame.isna().any().any():
        raise ValueError('Empty or missing UCI data')
    labels = sorted(frame[spec['decision_column']].unique())
    mapping = {label: i for i, label in enumerate(labels)}
    y = frame[spec['decision_column']].map(mapping).to_numpy(dtype=int)
    features = frame.drop(columns=[spec['decision_column']] + spec['id_columns'])
    X = features.apply(pd.to_numeric, errors='raise')
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError('Nonfinite numeric features')
    return X, y, labels
