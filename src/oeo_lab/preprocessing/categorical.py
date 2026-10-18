"""类别特征的统一 one-hot 编码：完备性实验和分类实验共用。"""
import numpy as np
import pandas as pd


def encode_categorical(features):
    """在给定数据上确定类别列；保持原完备性实验的列顺序及元数据。"""
    blocks, names = [], []
    for column in features:
        for value in sorted(features[column].unique()):
            blocks.append(features[column].eq(value).to_numpy(dtype=np.int8))
            names.append({'column': int(column), 'value': str(value)})
    return np.column_stack(blocks), {'kind': 'categorical', 'columns': names}


class CategoricalContextEncoder:
    """只在训练集确定类别；测试集沿用训练列，未见类别编码为全零。"""

    def fit_transform(self, features):
        self.feature_columns_ = list(features.columns)
        values, metadata = encode_categorical(features)
        self.categories_ = metadata['columns']
        self.columns_ = [f"{c['column']}={c['value']}" for c in self.categories_]
        self.bins_ = {}  # 类别数据不分箱；兼容公共编码元数据接口。
        return pd.DataFrame(values, index=features.index, columns=self.columns_)

    def transform(self, features):
        if list(features.columns) != self.feature_columns_:
            raise ValueError('Training and test feature columns differ')
        values = np.column_stack([
            features[c['column']].eq(c['value']).to_numpy(dtype=np.int8)
            for c in self.categories_])
        return pd.DataFrame(values, index=features.index, columns=self.columns_)
