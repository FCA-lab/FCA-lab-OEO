"""显式 v2 权重适配器；依赖只在选中模型时导入。"""
import hashlib
from pathlib import Path
import numpy as np

class TabPFNV2:
    def __init__(self, model_path, model_sha256, random_state=0, **params):
        self.path = Path(model_path)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != model_sha256:
            raise ValueError('TabPFN checkpoint changed after task creation')
        from tabpfn import TabPFNClassifier
        self.model = TabPFNClassifier(model_path=str(self.path), random_state=random_state, **params)

    def fit(self, X, y):
        # 不绕过原版 v2 的规模限制，不为某个模型单独筛选属性。
        if len(X)>10000 or X.shape[1]>500 or len(np.unique(y))>10:
            raise ValueError('TabPFN v2 limit: 10000 training rows, 500 encoded features, 10 classes')
        self.model.fit(X,y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)
