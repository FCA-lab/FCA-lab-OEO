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
#################


class RC_3Woc(BaseEstimator, ClassifierMixin):
    def __init__(self, rule_type=1, threshold=0.5, normalize_confidence=True):
        self.threshold = threshold
        self.RuleSet_= dict()
        self.rule_type = rule_type  # rule_type: 1,positive; 2,negative; 3, all
        self.normalize_confidence = normalize_confidence

    @staticmethod
    def gen_complement_context(ls):
        res = []
        for row in ls:
            tmp = list()
            for element in row:
                if element == 1:
                    tmp.append(0)
                elif element == 0:
                    tmp.append(1)
            res.append(tmp)
        return res

    def get_single_att_concepts(self, cur_context):
        att_num = cur_context.att_num  # 获得所有的属性
        result = list()
        for i in range(att_num):  # 对属性进行遍历
            obj_set = cur_context.att2obj({i})  # 通过属性映射获得对象集合
            # 下面根据对象集合的大小（即数量）进行索引，以便下一步按照集合大小顺序进行生成概念
            # if len(obj_set) > 0:
            result.append([obj_set, i])
        return result

    def get_base_OEOs(self, single_att_concepts, comple_single_att_concepts):
        base_OEOs = list()
        all_sac = list()
        all_sac.extend(single_att_concepts)
        all_sac.extend(comple_single_att_concepts)
        for sac in all_sac:
            if sac[0]:
                X = set(sac[0])
                Y = set()
                Z = set()
                for i in single_att_concepts:
                    if i[0] and set(i[0]).issubset(X):
                        Y.add(i[1])
                for j in comple_single_att_concepts:
                    if j[0] and set(j[0]).issubset(X):
                        Z.add(j[1])
                base_OEOs.append([X, [Y, Z]])
        return base_OEOs

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        ## construct the formal decision context (G, M, I, N, J)
        # the formal condition attribute context (G, M ,I)
        K1 = formal_context(data_set(X.tolist()), 1)
        # the complement context of K1=(G, M, I)
        K1_c= formal_context(data_set((1-X).tolist()), 1)
        #print(np.linalg.matrix_rank(X_test))
        #the formal decision attribute context (G, N, J)
        binary_y=pd.get_dummies(y, dtype='int8').to_numpy()
        K2 = formal_context(data_set(binary_y.tolist()),1)
        # the complement context of K1=(G, N, J)
        K2_c= formal_context(data_set((1-binary_y).tolist()), 1)
        #
        single_att_concepts = self.get_single_att_concepts(K1)  # 正背景下的单属性概念
        comple_singel_att_concepts = self.get_single_att_concepts(K1_c)  # 负背景下的单属性概念
        base_OEOs = self.get_base_OEOs(single_att_concepts, comple_singel_att_concepts)
        # initialize Rule set
        #R_A1=[]; R_A2=[]; R_B1=[];R_B2=[]
        # the form of a rule is [set, target]. the set consisted of necessary attributes.
        Max_C=dict(); RuleSet=dict()
        for n in range(0,K2.att_num):
            #（W, U, V)
            [W,U,V]=[set(),set(),set()]
            [P,Q,R]=[set(),set(),set()]
            n_class_Objs=K2.att2obj({n})
            non_class_Objs=K2_c.att2obj({n})
            for [extent, [intent, minus]] in base_OEOs:
                if extent.issubset(n_class_Objs):
                    [W,U,V]=[W|extent,U|intent,V|minus]
                if extent.issubset(non_class_Objs):
                    [P,Q,R] = [P | extent, Q | intent, R | minus]
            Max_C[n]= {n: [[W,U,V], [P,Q,R]]}
            #  save these ruls
            self.RuleSet_.update({ n : [U, V, Q, R]})
        return self

    def predict(self, X):
        # 预测逻辑
        # X: 待预测的特征数据，形状为 (n_samples, n_features)
        # 返回值: 预测结果，形状为 (n_samples,)

        # 1. 检查模型是否已训练（是否存在拟合后的属性）
        predict_labels_list=list()
        if self.RuleSet_ is None:
            raise NotFittedError(
                "This model has not been fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
        check_is_fitted(self, ['classes_'])
        X = check_array(X)
        for test_obj in X:
            predict_label = self.predictWithRules(self.RuleSet_,test_obj,self.rule_type)
            predict_labels_list.append(predict_label)
        return np.array(predict_labels_list, dtype=self.classes_.dtype)

    def getClassificationRules(concepts, label2objs, att_num):
        # 初始化 分类规则
        class_rules = {}
        veto_rules = {}  # 负规则集合，用于否决分类
        # class_rules_2 = {}
        # veto_rules_2 = {}  # 负规则集合，用于否决分类
        for label in label2objs.keys():
            class_rules[label] = [0, 0]  # 空集的hash就是0,这里的0代表正属性的hash
            veto_rules[label] = [0, 0]  # 空集的hash就是0,这里的0代表负属性的hash
            # class_rules_2[label] = 0
            # veto_rules_2[label] = 0
        modus_ne_cnt = 0
        one_type_concept_cnt = 0
        for concept in concepts:
            if len(concept['modus']) > 0:
                modus_ne_cnt = modus_ne_cnt + 1
            for label, objs in label2objs.items():
                # intersection_length = objs.intersection_length(concept['extent'])
                if len(concept['extent']) > 0 and objs.is_supset(concept['extent']):  # 先判断是否为空集,类极大概念
                    # print("概念的外延大小：", len(concept['extent']))
                    # now_pro_rate = 1 / len(concept['intent'])
                    one_type_concept_cnt = one_type_concept_cnt + 1
                    now_rule = set2hash(concept['intent'])
                    extent_hash, intent_hash = class_rules[label]
                    # if extent_hash != 0 and intent_hash != 0:
                    #     extent = hash2set(extent_hash)
                    #     print(extent[0], '--->', label)
                    class_rules[label] = [extent_hash | set2hash(concept['extent']), now_rule | intent_hash]
                if len(concept['extent']) > 0 and objs.intersection_length(
                        concept['extent']) == 0:  # 如果存在负规则，就从非类极大概念中取
                    veto_rule = set2hash(concept['modus'])
                    extent_hash, modus_hash = veto_rules[label]
                    veto_rules[label] = [extent_hash | set2hash(concept['extent']), veto_rule | modus_hash]
                # if objs.intersection_length(concept['extent']) == 0:
                #     veto_rule = att2hash(concept['modus'])
                #     veto_rules[label].append(veto_rule)
        # print(class_rules)
        # print(len(concepts), modus_ne_cnt, one_type_concept_cnt)
        return class_rules, veto_rules

    def get_positive_columns(self, row_data):
        positive_columns = set()
        for idx, value in enumerate(row_data):
            if value > 0:
                positive_columns.add(idx)
        return positive_columns

    @staticmethod
    def _normalized_support(supports, rule_size):
        if rule_size == 0:
            return 0
        return supports / rule_size

    def predictWithRules(self, class_rules, test_sample, rule_type=3):# rule_type: 1,positive; 2,negative; 3, all
        positive_label = dict()
        negative_label = dict()
        test_feature = self.get_positive_columns(test_sample)
        for class_label in class_rules.keys():
            positive_label[class_label] = 0
            negative_label[class_label] = 0
            [R_A1, R_A2, R_B1, R_B2] = class_rules[class_label]
            if rule_type & 1 >0:
                 # positive rule
                supports =len(R_A1 & test_feature)
                if  supports>0:  # 测试样本中满足正规则中的部分条件属性
                    positive_label[class_label]= supports
            if rule_type & 2 >0:
                supports = len(R_A2 & test_feature)
                if supports> 0:  # 测试样本中满足负规则中的部分条件属性
                    negative_label[class_label]=supports + negative_label[class_label]
                supports = len(R_B1 & test_feature)
                if supports>0:  # 测试样本中满足负规则中的部分条件属性
                    negative_label[class_label]= supports+ negative_label[class_label]
                supports = len(R_B2)-len(R_B2 & test_feature)
                if supports>0:  # 测试样本中满足负规则中的部分条件属性
                    negative_label[class_label]=supports +negative_label[class_label]
        #analysis classified answer
        vote_label = dict()
        for n_class in class_rules.keys():
            vote_label[n_class] = positive_label[n_class]-negative_label[n_class]
        ans_label = max(vote_label, key=lambda k: vote_label[k])
        return ans_label
    def predictWithRules_1(self, class_rules, test_sample, rule_type=3):# rule_type: 1,positive; 2,negative; 3, all
        positive_label = dict()
        negative_label = dict()
        test_feature = self.get_positive_columns(test_sample)
        for class_label in class_rules.keys():
            positive_label[class_label] = 0
            negative_label[class_label] = 0
            [R_A1, R_A2, R_B1, R_B2] = class_rules[class_label]
            if rule_type & 1 >0:
                 # positive rule
                supports =len(R_A1 & test_feature)
                if  supports>0:  # 测试样本中满足正规则中的部分条件属性
                    if self.normalize_confidence:
                        positive_label[class_label] = self._normalized_support(supports, len(R_A1))
                    else:
                        positive_label[class_label]= supports
            if rule_type & 2 >0:
                supports = len(R_A2 & test_feature)
                if supports> 0:  # 测试样本中满足负规则中的部分条件属性
                    if self.normalize_confidence:
                        negative_label[class_label] = (
                            self._normalized_support(supports, len(R_A2)) + negative_label[class_label]
                        )
                    else:
                        negative_label[class_label]=supports + negative_label[class_label]
                supports = len(R_B1 & test_feature)
                if supports>0:  # 测试样本中满足负规则中的部分条件属性
                    if self.normalize_confidence:
                        negative_label[class_label] = (
                            self._normalized_support(supports, len(R_B1)) + negative_label[class_label]
                        )
                    else:
                        negative_label[class_label]= supports+ negative_label[class_label]
                supports = len(R_B2)-len(R_B2 & test_feature)
                if supports>0:  # 测试样本中满足负规则中的部分条件属性
                    if self.normalize_confidence:
                        negative_label[class_label] = (
                            self._normalized_support(supports, len(R_B2)) + negative_label[class_label]
                        )
                    else:
                        negative_label[class_label]=supports +negative_label[class_label]
        #analysis classified answer
        vote_label = dict()
        for n_class in class_rules.keys():
            vote_label[n_class] = positive_label[n_class]-negative_label[n_class]
        ans_label = max(vote_label, key=lambda k: vote_label[k])
        return ans_label
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
