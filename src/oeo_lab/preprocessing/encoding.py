# -*- coding: utf-8 -*-
"""LeakFreeContextEncoder —— 无泄漏的"离散化 + 二值化"编码器。

分箱逻辑(GreedyFindBin/FindBinWithZeroAsOneBin/GetBins/GetCodes)原样取自原
Features2Context, 保证与论文方法学完全一致, 仅重组为 sklearn 的 fit/transform:
    fit(X_train)        切分点 + one-hot 列集合 仅由训练集决定
    transform(X)        用训练集参数处理任意数据, 并对齐到训练列
防泄漏: 测试折 unseen 值落最近箱(bins 末位为 Inf), 未见列经 reindex 补 0,
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
            mean_bin_size = rest_sample_cnt / rest_bin_cnt
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
                        mean_bin_size = rest_sample_cnt / rest_bin_cnt
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
        def _find(x, arr):
            if x <= arr[0]:
                return 0
            left, right = 0, len(arr) - 1
            while True:
                mid = (left + right) // 2
                if x <= arr[mid] and x <= arr[mid - 1]:
                    right = mid - 1
                elif x <= arr[mid] and x > arr[mid - 1]:
                    return mid
                elif x > arr[mid] and x <= arr[mid + 1]:
                    return mid + 1
                else:
                    left = mid + 1
        return [_find(x, bins[col]) for x in df[col]]
    # --------------------------------------------------------------------

    def _encode(self, X_df):
        d = {col: self._GetCodes(X_df, col, self.bins_) for col in self.col_names}
        df1 = pd.DataFrame(d, index=X_df.index)
        return pd.get_dummies(df1, columns=self.col_names, dtype='int8')

    def fit(self, X_df, y=None):
        if self.col_names is None:
            self.col_names = list(X_df.columns)
        self.bins_ = self._GetBins(X_df, self.col_names, self.max_bin, self.min_data_in_bin)
        self.columns_ = list(self._encode(X_df).columns)   # 锁定训练折决定的条件属性集 M
        return self

    def transform(self, X_df):
        return self._encode(X_df).reindex(columns=self.columns_, fill_value=0)

    def fit_transform(self, X_df, y=None):
        self.fit(X_df)
        return self._encode(X_df).reindex(columns=self.columns_, fill_value=0)

    def get_num_attributes(self):
        return len(self.columns_)
