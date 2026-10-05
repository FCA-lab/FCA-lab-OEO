"""生成模型无关的数据划分；仅处理索引，不拟合分箱器或其他预处理。"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from ..common.checkpoints import digest


def _validate_vector(values, name, size=None):
    """拒绝缺失标识，避免将缺失实体当成同一个组或独立地址。"""
    array = np.asarray(values, dtype=object)
    if array.ndim != 1 or not len(array) or (size is not None and len(array) != size):
        raise ValueError(f'{name} must be a nonempty aligned one-dimensional array')
    if pd.isna(array).any() or any(isinstance(v, str) and not v.strip() for v in array):
        raise ValueError(f'{name} contains missing or blank values')
    return array


def make_splits(y, *, protocol='stratified', groups=None, n_splits=5, seeds=(42,)):
    """返回所有模型共享的重复交叉验证索引与审计信息。

    stratified（旧称 address）：逐样本分层；entity：同一实体的全部行只进入一折。
    返回索引相对于传入数据的当前行序；使用方必须保持 X/y/groups 对齐。
    本接口要求训练和测试均覆盖全部任务类别，不满足时显式报错。
    seeds/n_splits 由调用方配置，函数不会搜索更有利的随机划分。
    """
    if protocol not in ('stratified', 'address', 'entity'):
        raise ValueError('protocol must be stratified or entity (address is a legacy alias)')
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError('n_splits must be an integer >= 2')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(
            not isinstance(s, (int, np.integer)) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('seeds must be distinct integers in [0, 2**32)')
    y = _validate_vector(y, 'y')
    # factorize 保留原标签映射，避免字符串/整数混合类型排序改变语义。
    encoded, classes = pd.factorize(y, sort=False)
    group_values = None if groups is None else _validate_vector(groups, 'groups', len(y))
    group_codes = None if group_values is None else pd.factorize(group_values, sort=False)[0]
    if protocol == 'entity' and group_codes is None:
        raise ValueError('Entity protocol requires reliable group identifiers')
    if protocol == 'entity' and len(set(group_codes)) < n_splits:
        raise ValueError('Fewer entities than folds')
    inventory = []
    for c, label in enumerate(classes):
        count = int((encoded == c).sum())
        entities = None if group_codes is None else len(set(group_codes[encoded == c]))
        if count < n_splits:
            raise ValueError(f'Class {label!r}: {count} samples cannot cover {n_splits} folds')
        if protocol == 'entity' and entities < n_splits:
            raise ValueError(f'Class {label!r}: {entities} entities cannot cover {n_splits} folds')
        inventory.append({'label': label, 'samples': count, 'entities': entities})
    rows = []
    for seed in seeds:
        splitter = (StratifiedGroupKFold if protocol == 'entity' else StratifiedKFold)(
            n_splits=n_splits, shuffle=True, random_state=int(seed))
        test_visits = np.zeros(len(y), dtype=int)
        for fold, (train, test) in enumerate(splitter.split(
                np.zeros((len(y), 1)), encoded,
                groups=group_codes if protocol == 'entity' else None), start=1):
            if np.intersect1d(train, test).size or len(train)+len(test) != len(y):
                raise ValueError('Invalid train/test partition')
            overlap = None
            if group_codes is not None:
                overlap = len(set(group_codes[train]) & set(group_codes[test]))
            if protocol == 'entity' and overlap:
                raise ValueError('Train/test entity overlap detected')
            counts = []
            for c, label in enumerate(classes):
                tr, te = int((encoded[train] == c).sum()), int((encoded[test] == c).sum())
                if not tr or not te:
                    raise ValueError(f'seed={seed}, fold={fold}, class={label!r}: '
                                     'missing train/test coverage; revise split design explicitly')
                counts.append({'label': label, 'train': tr, 'test': te})
            test_visits[test] += 1
            rows.append({'seed': int(seed), 'fold': fold,
                         'train_indices': train.tolist(), 'test_indices': test.tolist(),
                         'class_counts': counts, 'entity_overlap': overlap})
        if not np.all(test_visits == 1):
            raise ValueError('Each sample must be tested exactly once per seed')
    # 索引身份绑定标签和实体编码的行序。正式任务还需绑定原始数据文件哈希。
    return {'protocol': protocol, 'n_splits': n_splits, 'seeds': list(map(int, seeds)),
            'inventory': inventory,
            'assignment_sha256': digest({'y': encoded.tolist(),
                'groups': None if group_codes is None else group_codes.tolist(), 'splits': rows}),
            'splits': rows}
