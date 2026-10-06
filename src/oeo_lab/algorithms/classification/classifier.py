###############################################
# machine learning model based on three-way object-oriented concepts
# 2025.7.24
# designed by zhangzhuo jinyuhao
# zzu.edu.cn
# 使用sklearn 框架，实现模型算法，便于使用sklearn统一接口框架进行实验。
#
#################################################
try:
    import seaborn as sns
    from matplotlib import pyplot as plt
except ImportError:
    sns = None
    plt = None
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
import numpy as np
import time
import pandas as pd
from fca_core.context import formal_context
from fca_core.data_source import data_set
from oeo_lab.algorithms.construction.three_woc import NewThreeWayForObj
#################


class RC_3Woc(ClassifierMixin, BaseEstimator, NewThreeWayForObj):
    def __init__(self, rule_type=1, threshold=0.5, normalize_confidence=True,
                 voting="uniform"):
        self.threshold = threshold
        self.RuleSet_= dict()
        self.rule_type = rule_type  # rule_type: 1,positive; 2,negative; 3, all
        self.normalize_confidence = normalize_confidence
        self.voting = voting  # 新预测方法：uniform、nonzero 或 support。

    @staticmethod
    def get_support_counts(single_att_concepts):
        # 单属性概念的外延大小即支持计数；空外延保留为 0。
        return {attribute: len(extent)
                for extent, attribute in single_att_concepts}

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        # 2026-10-06：补背景使用 1-X，故训练输入必须是二值；布尔值先转整数。
        if not np.all((X == 0) | (X == 1)):
            raise ValueError("Training input must contain only binary values (0 or 1).")
        X = X.astype(np.int8, copy=False)
        self.n_features_in_ = X.shape[1]
        # 每次训练重新初始化，避免保留上一次训练的规则和类别计数。
        self.RuleSet_ = {}
        self.class_count_ = {}
        self.n_samples_fit_ = len(X)  # 支持比例的分母 |G|。
        ## construct the formal decision context (G, M, I, N, J)
        # the formal condition attribute context (G, M ,I)
        K1 = formal_context(data_set(X.tolist()), 1)
        # the complement context of K1=(G, M, I)
        K1_c= formal_context(data_set((1-X).tolist()), 1)
        #the formal decision attribute context (G, N, J)
        decision_df = pd.get_dummies(y, dtype='int8')
        # 决策背景第 n 列、规则键 n 均对应原始类别标签 classes_[n]。
        self.classes_ = decision_df.columns.to_numpy()
        binary_y = decision_df.to_numpy()
        K2 = formal_context(data_set(binary_y.tolist()),1)
        # the complement context of K2=(G, N, J)
        K2_c= formal_context(data_set((1-binary_y).tolist()), 1)
        #
        single_att_concepts = self.get_single_att_concepts(K1)  # 正背景下的单属性概念
        comple_singel_att_concepts = self.get_single_att_concepts(K1_c)  # 负背景下的单属性概念
        # 保存正、负文字项的支持计数；由预测策略决定如何使用。
        self.support_pos_ = self.get_support_counts(single_att_concepts)
        self.support_neg_ = self.get_support_counts(comple_singel_att_concepts)
        base_OEOs = self.get_base_OEOs(single_att_concepts, comple_singel_att_concepts)
        # 合并基本概念，获取类与非类最大概念。
        Max_C = dict()
        for n in range(0,K2.att_num):
            #（W, U, V)
            [W,U,V]=[set(),set(),set()]
            [P,Q,R]=[set(),set(),set()]
            n_class_Objs=K2.att2obj({n})
            self.class_count_[n] = len(n_class_Objs)  # 用于预测时比较训练类别比例。
            non_class_Objs=K2_c.att2obj({n})
            for [extent, [intent, minus]] in base_OEOs:
                intent, minus = set(intent), set(minus)
                if extent.issubset(n_class_Objs):
                    [W,U,V]=[W|extent,U|intent,V|minus]
                if extent.issubset(non_class_Objs):
                    [P,Q,R] = [P | extent, Q | intent, R | minus]
            Max_C[n]= {n: [[W,U,V], [P,Q,R]]}
            # 按 A1、A2、B1、B2 保存前件；后件由类别 n 和规则类型确定。
            self.RuleSet_.update({ n : [U, V, Q, R]})
        return self

    def predict(self, X):
        """Predict original class labels; None denotes deferred classification.

        Uses all four rule types and self.voting. Legacy rule_type,
        threshold and normalize_confidence do not control this procedure.
        """
        check_is_fitted(self, ['classes_', 'RuleSet_', 'support_pos_',
                              'support_neg_', 'class_count_', 'n_samples_fit_'])
        X = check_array(X)
        # object 类型保留原始标签和 None，避免将 defer 转为字符串或数值。
        return np.array([
            self.predictWithRules_2(self.RuleSet_, sample, voting=self.voting)
            for sample in X
        ], dtype=object)

    def get_positive_columns(self, row_data):
        positive_columns = set()
        for idx, value in enumerate(row_data):
            if value > 0:
                positive_columns.add(idx)
        return positive_columns


    def predictWithRules_2(self, class_rules, test_sample, voting="uniform"):
        """Return the original class label, or None for defer.

        voting: "uniform", "nonzero", or "support".
        class_rules uses internal class indices and [U, V, Q, R] antecedents.
        This method always uses all four rule types.
        """
        check_is_fitted(self, ['classes_', 'support_pos_', 'support_neg_',
                              'class_count_', 'n_samples_fit_'])
        if voting not in ("uniform", "nonzero", "support"):
            raise ValueError("voting must be 'uniform', 'nonzero', or 'support'.")
        sample = np.asarray(test_sample)
        if sample.ndim != 1 or len(sample) != len(self.support_pos_):
            raise ValueError("test_sample must be one row with the training feature count.")
        if not np.all((sample == 0) | (sample == 1)):
            raise ValueError("test_sample must contain only binary values (0 or 1).")
        if set(class_rules) != set(range(len(self.classes_))):
            raise ValueError("class_rules must contain every internal class index.")
        present = self.get_positive_columns(sample)

        def weight(support):
            if voting == "uniform":
                return 1
            if voting == "nonzero":
                return int(support > 0)
            # 所有权重共享分母 |G|，先比较整数分子，避免浮点误差影响并列判断。
            return int(support)

        scores = {}
        total_contribution = 0
        for n, (U, V, Q, R) in class_rules.items():
            # A1 属性存在、A2 属性缺失时提供支持证据。
            supporting = (sum(weight(self.support_pos_[a]) for a in U & present)
                          + sum(weight(self.support_neg_[b]) for b in V - present))
            # B1 属性存在、B2 属性缺失时提供排除证据。
            excluding = (sum(weight(self.support_pos_[c]) for c in Q & present)
                         + sum(weight(self.support_neg_[d]) for d in R - present))
            scores[n] = supporting - excluding
            total_contribution += supporting + excluding

        # H=0 表示无有效证据；分数抵消为零或最高分为负并不等于 H=0。
        if total_contribution == 0:
            return None
        best_score = max(scores.values())
        candidates = [n for n, score in scores.items() if score == best_score]
        # 类别比例共享分母 |G|，比较训练类别计数即可。
        largest_count = max(self.class_count_[n] for n in candidates)
        selected = [n for n in candidates if self.class_count_[n] == largest_count]
        return self.classes_[selected[0]] if len(selected) == 1 else None

    def get_rules_num(self):
        count=0
        for key in self.RuleSet_.keys():
            [R1, R2, R3, R4] = self.RuleSet_[key]
            count += (len(R1)+len(R2)+len(R3)+len(R4))
        return count


if __name__ == '__main__':
    frac = 0.4
    #all_sample = pd.read_csv("../bitcoin_classificatiion/sample/binFeature.csv")
    #selected_sample = all_sample.sample(frac=frac)
    #selected_sample.to_csv("../bitcoin_classificatiion/sample/filteredFeature.csv")
    #d = pd.read_csv("../bitcoin_classificatiion/sample/filteredFeature.csv")
    d = pd.read_csv("../../../data/Bitcoin/formalContext_350.csv")
    print("训练集行数为： ", len(d))
    # d = pd.read_csv("../bitcoin_classificatiion/sample/train_pos.csv")
    y = d['label']
    unique_label = y.drop_duplicates().values.tolist()
    X = d.drop([ 'sample_id', 'label'], axis=1)
    # concept_data = d.drop(['sample_id', 'label'], axis=1)
    #ls6 = concept_data.values.tolist()
    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42  # 测试集占30%，固定随机种子保证结果可复现
    )
    start_time = time.time()
    # 5. 初始化分类器
    rc_3woc_ml=RC_3Woc(rule_type=3)
    # 6. 训练模型
    print("训练模型,训练集大小：",len(X_train))
    rc_3woc_ml.fit(X_train, y_train)
    # 7. 模型预测
    print("模型预测")
    y_pred = rc_3woc_ml.predict(X_test)
    end_time = time.time()
    # 8. 模型评估
    print("准确率：", accuracy_score(y_test, y_pred))
    print("\n混淆矩阵：")
    cm=confusion_matrix(y_test, y_pred)
    print()
    print("\n分类报告：")
    print(classification_report(y_test, y_pred))
    print("所用时间为：", end_time - start_time)
    #
    # 4. 方法2：使用seaborn绘制更美观的混淆矩阵
    #class_names=rc_3woc_ml.classes_
    #plt.figure(figsize=(8, 6))
    #sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=class_names,yticklabels=class_names)
    #plt.title('混淆矩阵 (Seaborn)')
    #plt.ylabel('真实标签')
    #plt.xlabel('预测标签')
    #plt.tight_layout()
    #plt.show()

    # 获取预测的准确率
    print('-----Macro-----')
    print('Macro precision', precision_score(y_test, y_pred, average='macro'))
    print('Macro recall', recall_score(y_test, y_pred,  average='macro'))
    print('Macro f1-score', f1_score(y_test, y_pred,  average='macro'))
    print('='*20,frac,'='*20)
    print('-----Weighted-----')
    print('Weighted precision', precision_score(y_test, y_pred, average='weighted'))
    print('Weighted recall', recall_score(y_test, y_pred, average='weighted'))
    print('Weighted f1-score', f1_score(y_test, y_pred, average='weighted'))
    ##
    # 初始化K折交叉验证（5折）
    #kf = KFold(n_splits=5, shuffle=True, random_state=42)  # shuffle=True打乱数据

    # 执行交叉验证
    #scores = cross_val_score(rc_3woc_ml, X, y, cv=kf, scoring='accuracy')
    #print(f"各折准确率: {scores}")
    #print(f"平均准确率: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
