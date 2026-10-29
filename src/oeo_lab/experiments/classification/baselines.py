"""Create classifiers from explicit experiment specifications."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def create_model(spec, seed):
    """按显式配置创建一个模型；缺失依赖报错，绝不静默跳过。"""
    name, params = spec['estimator'], dict(spec['params'])
    if name == 'tabpfn_v2':
        from .tabpfn_adapter import TabPFNV2
        return TabPFNV2(random_state=seed, **params)
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
