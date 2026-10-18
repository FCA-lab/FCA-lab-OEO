"""UCI 数据读取分支；数值数据继续使用原加载器。"""
import hashlib
from pathlib import Path
import pandas as pd
from .datasets import load_numeric_dataset


def load_uci_dataset(spec):
    """仅读取原始特征和标签；编码统一在划分后的训练折内完成。"""
    kind = spec.get('feature_kind', 'numerical')
    if kind == 'numerical':
        return load_numeric_dataset(spec)
    if kind != 'categorical':
        raise ValueError(f'Unknown feature_kind: {kind}')
    path = Path(spec['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != spec['sha256']:
        raise ValueError('Dataset changed after task creation')
    frame = pd.read_csv(path, header=None, dtype=str,
                        sep=r'\s+' if spec.get('format') == 'whitespace' else ',')
    if spec.get('header', False):
        frame = frame.iloc[1:].reset_index(drop=True)
    if frame.empty or frame.isna().any().any():
        raise ValueError('Empty or missing UCI data')
    labels = sorted(frame[spec['decision_column']].unique())
    mapping = {label: i for i, label in enumerate(labels)}
    y = frame[spec['decision_column']].map(mapping).to_numpy(dtype=int)
    X = frame.drop(columns=[spec['decision_column']] + spec['id_columns'])
    return X, y, labels
