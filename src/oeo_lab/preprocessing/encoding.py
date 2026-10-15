# -*- coding: utf-8 -*-
"""LeakFreeContextEncoder —— 无泄漏的"离散化 + 二值化"编码器。

候选分箱逻辑(GreedyFindBin/FindBinWithZeroAsOneBin/GetBins)取自原
Features2Context, 复用历史候选边界逻辑, 仅重组为 sklearn 的 fit/transform:
    fit(X_train)        切分点 + one-hot 列集合 仅由训练集决定
    transform(X)        用训练集参数处理任意数据, 并对齐到训练列
防泄漏: 测试折 unseen 值落最近箱(bins 末位为 Inf), 空候选区间在训练时合并，每个特征恰好激活一个区间属性,
        测试集绝不参与条件属性集 M 的构造。
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin



class LeakFreeContextEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col_names=None, max_bin=350, min_data_in_bin=3):
        self.col_names = col_names
        self.max_bin = max_bin
        self.min_data_in_bin = min_data_in_bin

    # ---------- 以下分箱函数原样取自 Features2Context(逻辑未改) ----------
    def _GreedyFindBin(self, distinct_values, counts, num_distinct_values, max_bin, total_cnt, min_data_in_bin=3):
        bin_upper_bound = list()
        assert (max_bin > 0)
        if max_bin == 1:
            return [float("Inf")]
        if num_distinct_values <= max_bin:
            cur_cnt_inbin = 0
            for i in range(num_distinct_values - 1):
                cur_cnt_inbin += counts[i]
                if cur_cnt_inbin >= min_data_in_bin:
                    bin_upper_bound.append((distinct_values[i] + distinct_values[i + 1]) / 2.0)
                    cur_cnt_inbin = 0
            cur_cnt_inbin += counts[num_distinct_values - 1]
            bin_upper_bound.append(float('Inf'))
        else:
            if min_data_in_bin > 0:
                max_bin = min(max_bin, total_cnt // min_data_in_bin)
                max_bin = max(max_bin, 1)
            mean_bin_size = total_cnt / max_bin
            rest_bin_cnt = max_bin
            rest_sample_cnt = total_cnt
            is_big_count_value = [False] * num_distinct_values
            for i in range(num_distinct_values):
                if counts[i] >= mean_bin_size:
                    is_big_count_value[i] = True
                    rest_bin_cnt -= 1
                    rest_sample_cnt -= counts[i]
            # 所余箱均已预留给高频取值时，无可分配的普通箱。
            # 不再按普通样本均值触发分箱，避免除零；高频取值仍独立触发。
            mean_bin_size = rest_sample_cnt / rest_bin_cnt if rest_bin_cnt > 0 else float('inf')
            upper_bounds = [float('Inf')] * max_bin
            lower_bounds = [float('Inf')] * max_bin
            bin_cnt = 0
            lower_bounds[bin_cnt] = distinct_values[0]
            cur_cnt_inbin = 0
            for i in range(num_distinct_values - 1):
                if not is_big_count_value[i]:
                    rest_sample_cnt -= counts[i]
                cur_cnt_inbin += counts[i]
                if is_big_count_value[i] or cur_cnt_inbin >= mean_bin_size or \
                        is_big_count_value[i + 1] and cur_cnt_inbin >= max(1.0, mean_bin_size * 0.5):
                    upper_bounds[bin_cnt] = distinct_values[i]
                    bin_cnt += 1
                    lower_bounds[bin_cnt] = distinct_values[i + 1]
                    if bin_cnt >= max_bin - 1:
                        break
                    cur_cnt_inbin = 0
                    if not is_big_count_value[i]:
                        rest_bin_cnt -= 1
                        mean_bin_size = rest_sample_cnt / rest_bin_cnt if rest_bin_cnt > 0 else float('inf')
            bin_cnt += 1
            for i in range(bin_cnt - 1):
                bin_upper_bound.append((upper_bounds[i] + lower_bounds[i + 1]) / 2.0)
            bin_upper_bound.append(float('Inf'))
        return bin_upper_bound

    def _FindBinWithZeroAsOneBin(self, distinct_values, counts, num_distinct_values, max_bin, total_cnt, min_data_in_bin=3):
        bin_upper_bound = list()
        assert (max_bin > 0)
        left_cnt_data = cnt_zero = right_cnt_data = 0
        kZeroThreshold = 1e-35
        for i in range(num_distinct_values):
            if distinct_values[i] <= -kZeroThreshold:
                left_cnt_data += counts[i]
            elif distinct_values[i] > kZeroThreshold:
                right_cnt_data += counts[i]
            else:
                cnt_zero += counts[i]
        left_cnt = -1
        for i in range(num_distinct_values):
            if distinct_values[i] > -kZeroThreshold:
                left_cnt = i
                break
        if left_cnt < 0:
            left_cnt = num_distinct_values
        if left_cnt > 0:
            left_max_bin = int(left_cnt_data / (total_cnt - cnt_zero) * (max_bin - 1))
            left_max_bin = max(1, left_max_bin)
            bin_upper_bound = self._GreedyFindBin(distinct_values, counts, left_cnt, left_max_bin,
                                                  left_cnt_data, min_data_in_bin)
            bin_upper_bound[-1] = -kZeroThreshold
        right_start = -1
        for i in range(left_cnt, num_distinct_values):
            if distinct_values[i] > kZeroThreshold:
                right_start = i
                break
        if right_start >= 0:
            right_max_bin = max_bin - 1 - len(bin_upper_bound)
            assert (right_max_bin > 0)
            right_bounds = self._GreedyFindBin(distinct_values[right_start:], counts[right_start:],
                                               num_distinct_values - right_start, right_max_bin,
                                               right_cnt_data, min_data_in_bin)
            bin_upper_bound.append(kZeroThreshold)
            bin_upper_bound += right_bounds
        else:
            bin_upper_bound.append(float('Inf'))
        return bin_upper_bound

    def _GetBins(self, df, col_names, max_bin, min_data_in_bin=3):
        bins = {}

        def _count(arr):
            distinct_values = [arr[0]]
            counts_dict = {arr[0]: 1}
            for i in range(1, len(arr)):
                if arr[i] == arr[i - 1]:
                    counts_dict[arr[i]] += 1
                else:
                    distinct_values.append(arr[i])
                    counts_dict[arr[i]] = 1
            return distinct_values, [counts_dict[x] for x in distinct_values]

        for col in col_names:
            tmp = sorted(df[col].to_list())
            distinct_values, counts = _count(tmp)
            bins[col] = self._FindBinWithZeroAsOneBin(distinct_values, counts, len(distinct_values),
                                                      max_bin, sum(counts), min_data_in_bin=min_data_in_bin)
        return bins

    def _GetCodes(self, df, col, bins):
        # 左闭判定由 side=left 实现：x 等于上界时仍属于当前箱。
        return np.searchsorted(bins[col], df[col].to_numpy(), side='left').tolist()
    # --------------------------------------------------------------------

    def _encode(self, X_df):
        d = {col: self._GetCodes(X_df, col, self.bins_) for col in self.col_names}
        df1 = pd.DataFrame(d, index=X_df.index)
        return pd.get_dummies(df1, columns=self.col_names, dtype='int8')

    def fit(self, X_df, y=None):
        if self.col_names is None:
            self.col_names = list(X_df.columns)
        self.bins_ = self._GetBins(X_df, self.col_names, self.max_bin, self.min_data_in_bin)
        # 2026-10-06：修复未见取值落入空候选箱后被 reindex 丢弃、整组全零的问题。
        # 预留零区间可能无训练对象。移除这些空区间并重新编号，
        # 保留有训练支持的箱边界；首、末箱覆盖训练范围外取值。
        for col in self.col_names:
            used = sorted(set(self._GetCodes(X_df, col, self.bins_)))
            bounds = [self.bins_[col][i] for i in used]
            bounds[-1] = float('inf')
            self.bins_[col] = bounds
        self.columns_ = list(self._encode(X_df).columns)   # 锁定训练折决定的条件属性集 M
        return self

    def transform(self, X_df):
        return self._encode(X_df).reindex(columns=self.columns_, fill_value=0)

    def fit_transform(self, X_df, y=None):
        self.fit(X_df)
        return self._encode(X_df).reindex(columns=self.columns_, fill_value=0)

    def get_num_attributes(self):
        return len(self.columns_)
