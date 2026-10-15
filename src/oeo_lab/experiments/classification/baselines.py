# -*- coding: utf-8 -*-
"""模型库。每个模型标注 input_type:
   'context' -> 消费 per-fold 无泄漏形式背景(二值);
   'raw'     -> 消费原始连续特征(不分箱), 即审稿要求的对照组(树模型的自然输入)。
"""
from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# 本文方法(你的实现); 沙箱缺失自动跳过
try:
    from oeo_lab.algorithms.classification.classifier import RC_3Woc
    HAS_RC = True
except Exception:
    HAS_RC = False
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# ======================================================================
# 向量化集合度量 KNN
#   原实现用 metric=python函数 + algorithm='brute', 每对样本调用一次 Python,
#   O(n²) 纯解释器循环, frac 大时是数小时级的长尾。此处改为矩阵化批量计算:
#     交集 = Q @ X_train.T ; 并集/分母 = |Q| + |X_train| - 交集
#   对二值特征 Jaccard 与 Tanimoto 数学上等价(x*x=x), 故两者公式一致,
#   保留两个名字仅为与既有基线命名/报表对齐(数值会相同, 如需可删其一)。
#   分块查询(默认 1024 行)限制距离矩阵内存, 兼容多进程并行。
# ======================================================================
class BinarySetKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5, metric="jaccard", query_chunk=1024):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.query_chunk = query_chunk

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y)
        self.classes_ = np.unique(self.y_)
        self.x_sum_ = self.X_.sum(axis=1)[None, :]   # 预存训练集行和, 复用
        return self

    def _dist_block(self, Q):
        """返回 Q(块) × X_train 的距离矩阵。二值下 Jaccard==Tanimoto。"""
        Q = np.asarray(Q, dtype=np.float32)
        inter = Q @ self.X_.T                         # 交集计数
        q_sum = Q.sum(axis=1)[:, None]
        den = q_sum + self.x_sum_ - inter             # 并集(Jaccard)/分母(Tanimoto)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(den == 0, 0.0, inter / den)
        return 1.0 - sim

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        n_train = self.X_.shape[0]
        k = min(self.n_neighbors, n_train)
        out = np.empty(X.shape[0], dtype=self.y_.dtype)
        step = max(1, int(self.query_chunk))
        for s in range(0, X.shape[0], step):
            D = self._dist_block(X[s:s + step])
            idx = np.argpartition(D, kth=k - 1, axis=1)[:, :k]   # 每行取 k 个最近
            for r, neigh in enumerate(idx):
                vals, counts = np.unique(self.y_[neigh], return_counts=True)
                out[s + r] = vals[np.argmax(counts)]             # 多数投票
        return out


def build_models(random_seed=42, include_raw=None):
    """返回 OrderedDict: name -> (estimator, input_type)。
    include_raw=None 时读取 config.INCLUDE_RAW_BASELINES。"""
    if include_raw is None:
        from . import config
        include_raw = config.INCLUDE_RAW_BASELINES
    m = OrderedDict()

    # ===== 形式背景模型(input_type='context') =====
    m["Dummy (most_frequent)"] = (DummyClassifier(strategy="most_frequent"), "context")
    m["Dummy (stratified)"] = (DummyClassifier(strategy="stratified", random_state=random_seed), "context")
    if HAS_RC:
        m["rc_3woc"] = (RC_3Woc(rule_type=3), "context")              # 本文方法
    m["DecisionTree"] = (DecisionTreeClassifier(random_state=random_seed), "context")
    m["Random Forest"] = (RandomForestClassifier(n_estimators=100, random_state=random_seed), "context")
    m["Gradient Boosting"] = (GradientBoostingClassifier(random_state=random_seed), "context")
    if HAS_XGB:
        m["XGBoost"] = (XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=1,
                                      eval_metric="mlogloss", random_state=random_seed), "context")
    if HAS_LGBM:
        m["LightGBM"] = (LGBMClassifier(n_estimators=300, n_jobs=1, random_state=random_seed, verbose=-1), "context")
    m["Gaussian NB"] = (GaussianNB(), "context")
    m["KNN Euclidean"] = (make_pipeline(StandardScaler(), KNeighborsClassifier()), "context")
    # 二值集合度量: 向量化实现(不标准化, 直接用 0/1 特征)
    m["KNN Jaccard"] = (BinarySetKNN(n_neighbors=5, metric="jaccard"), "context")
    m["KNN Tanimoto"] = (BinarySetKNN(n_neighbors=5, metric="tanimoto"), "context")
    m["Logistic Regression"] = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_seed)), "context")
    m["SVM"] = (make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=random_seed)), "context")
    m["MLP"] = (make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=random_seed)), "context")

    # ===== 原始连续特征对照组(input_type='raw', 不分箱; 树模型的自然输入) =====
    if include_raw:
        m["Random Forest (raw)"] = (RandomForestClassifier(n_estimators=100, random_state=random_seed), "raw")
        m["Gradient Boosting (raw)"] = (GradientBoostingClassifier(random_state=random_seed), "raw")
        if HAS_XGB:
            m["XGBoost (raw)"] = (XGBClassifier(n_estimators=300, tree_method="hist", n_jobs=1,
                                                eval_metric="mlogloss", random_state=random_seed), "raw")
        if HAS_LGBM:
            m["LightGBM (raw)"] = (LGBMClassifier(n_estimators=300, n_jobs=1, random_state=random_seed, verbose=-1), "raw")
    return m


def create_model(spec, seed):
    """按显式配置创建一个模型；缺失依赖报错，绝不静默跳过。"""
    name, params = spec['estimator'], dict(spec['params'])
    if name == 'dummy':
        return DummyClassifier(**params)
    if name == 'decision_tree':
        return DecisionTreeClassifier(**dict(params, random_state=seed))
    if name == 'random_forest':
        return RandomForestClassifier(**dict(params, random_state=seed, n_jobs=1))
    if name == 'xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(**dict(params, random_state=seed, n_jobs=1))
    if name == 'svm_rbf':
        # Pipeline 确保标准化只拟合训练折；其耗时属于 fit/predict。
        model = SVC(**dict(params, random_state=seed))
        return make_pipeline(StandardScaler(), model) if spec.get('scale', True) else model
    if name == 'logistic':
        return LogisticRegression(**dict(params, random_state=seed))
    if name == 'mlp':
        return MLPClassifier(**dict(params, random_state=seed))
    if name == 'catboost':
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**dict(params, random_seed=seed, thread_count=1))
    if name == 'lightgbm':
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**dict(params, random_state=seed, n_jobs=1))
    if name == 'oeo':
        from oeo_lab.algorithms.classification.classifier import RC_3Woc
        return RC_3Woc(**params)
    raise ValueError(f'Unknown model: {name}')
