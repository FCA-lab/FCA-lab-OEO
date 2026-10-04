import os
import time

#import pandas as pd
from fca_core.interfaces import abc_algorithm
from fca_core.concept import Three_Way_Concept
from fca_core.lattice import map_lattice
from fca_core.context import formal_context
from fca_core.data_source import data_set

class NewThreeWayForObj(abc_algorithm):
    def __init__(self, context, update_type):
        super().__init__("Three_way_for_obj")
        self._context = context
        self._comple_context = formal_context(
            data_source=data_set(self.gen_complement_context(self._context.get_context())), key_type=1)
        self.obj_set = {i for i in range(context.obj_num)}
        self.att_set = {i for i in range(context.att_num)}
        self.single_att_concepts = list()
        self.comple_singel_att_concepts = list()
        self.base_OEOs = list()
        self.ref = dict()
        #self.num_concepts=0

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
            #if len(obj_set) > 0:
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
                Y = list()
                Z = list()
                for i in single_att_concepts:
                    if i[0] and set(i[0]).issubset(X):
                        Y.append(i[1])
                for j in comple_single_att_concepts:
                    if j[0] and set(j[0]).issubset(X):
                        Z.append(j[1])
                base_OEOs.append([X, [Y, Z]])

        return base_OEOs
    def necessary_obj2att(self, context, objs):
        intent=set()
        for i in range(0,context.att_num):  # 遍历所有属性
            if context.att2obj({i}).issubset(objs):
            # if self.__context.att2obj({i}).isSub(objs):  # 如果当前属性和对象集合符合正算子
                intent.add(i)  # 将当前属性加入到拥有的必然属性集合中
        return intent

    def start(self):
        self.single_att_concepts = self.get_single_att_concepts(self._context)  # 正背景下的单属性概念
        self.comple_singel_att_concepts = self.get_single_att_concepts(self._comple_context)  # 负背景下的单属性概念
        self.base_OEOs = self.get_base_OEOs(self.single_att_concepts, self.comple_singel_att_concepts)
        #initialize ref
        for v in range(1, self._context.obj_num + 1):
            self.ref[v]={}
        objs = self.fn_num2objs(0)
        Y=self.necessary_obj2att(self._context, objs)
        Z =self.necessary_obj2att(self._comple_context, objs)
        self.ref[0] = {0: [0, [h(Y), h(Z)]]} #0 is a hash value

        for [X, [Y, Z]] in self.base_OEOs:
            if h(X) not in self.ref[len(X)].keys():
                self.ref[len(X)][h(X)] = [h(X), [h(Y), h(Z)]]
            else:
                concept = self.ref[len(X)][h(X)]
                self.ref[len(X)][h(X)] = [h(X) | concept[0], [h(Y) | concept[1][0], h(Z) | concept[1][1]]]

        for v in range(2, self._context.obj_num+1):
            for c1 in self.ref[v-1].values():
                for m in range(1, v):
                    for c2 in self.ref[m].values():
                        c3 = M(c1, c2)
                        #print(c3[0])
                        length = length_of_hash(c3[0])
                        if length in self.ref:
                            if c3[0] in self.ref[length].keys():
                                c4 = self.ref[length][c3[0]]
                                self.ref[length][c3[0]] = [c3[0], [c3[1][0] | c4[1][0], c3[1][1] | c4[1][1]]]
                            else:
                                self.ref[length][c3[0]] = c3
                        else:
                            self.ref[length] = {c3[0]: c3}




    def fn_num2objs(self, num):
        res = set()
        for i in self.obj_set:
            t = 1 << i
            if t & num:
                res.add(i)
        return res
    def fn_num2atts(self,num):
        res = set()
        for i in self.att_set:
            t = 1 << i
            if t & num:
                res.add(i)
        return res
    def get_numOfconcepts(self):
        cnt = 0
        for i in self.ref:
            for r in self.ref[i].values():
                cnt += 1
        return cnt

def M(c1, c2):
    [X1, [Y1, Z1]] = c1
    [X2, [Y2, Z2]] = c2
    return [X1 | X2, [Y1 | Y2, Z1 | Z2]]


# 定义一个函数，将对象集合转为数字

def h(objs):
    res = 0
    if len(objs)>0:
        for obj in objs:
            temp = 1 << obj
            res = res | temp
    return res


#@jit(nopython=True)
def length_of_hash(value):
    count = 0
    while value:
        count = count + 1
        value = value & (value - 1)
    return count



if __name__ == '__main__':
    ls = [[1, 0, 1, 1, 1],
          [1, 0, 1, 0, 0],
          [0, 1, 0, 0, 1],
          [1, 0, 0, 0, 0],
          [1, 1, 0, 0, 1]]
    # 刘敏茜例子
    ls1 = [[1, 0, 1, 0, 1, 1, 0],
           [1, 1, 0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 1, 0],
           [1, 1, 0, 0, 0, 0, 1]]

    # 潘爱宝例子
    ls2 = [[0, 0, 1, 0, 1, 0, 1],
           [0, 0, 1, 0, 0, 1, 1],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 1],
           [1, 0, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 1, 0, 1]]

    # 陈希邦例子
    ls3 = [[1, 1, 0, 1, 1, 0],
           [1, 0, 1, 0, 0, 1],
           [0, 1, 0, 1, 0, 0],
           [1, 1, 1, 0, 1, 0],
           [0, 0, 0, 1, 0, 0]]

    # 汪胜楠例子
    ls4 = [[1, 1, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [1, 1, 1, 0, 0]]

    # 钱婷例子
    ls5 = [[1, 0, 1, 1, 1],
           [1, 0, 1, 0, 0],
           [0, 1, 0, 0, 1],
           [1, 0, 0, 0, 0],
           [1, 1, 0, 0, 1]]

    ls6 = [[0, 1, 1, 1, 0, 0, 1, 0],
           [0, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 0, 1, 0, 1],
           [1, 1, 1, 1, 0, 0, 0, 1],
           [1, 1, 1, 0, 1, 0, 1, 0],
           [1, 1, 1, 0, 0, 1, 0, 1],
           [1, 1, 0, 1, 1, 1, 1, 0]]
    #example for rc-3woc
    CAK=[[1,0,0,0,0,0],
         [0,1,0,0,0,0],
         [0,0,1,0,0,1],
         [0,0,0,1,1,1],
         [0,0,0,1,0,1]]

    DAK=[[1,0,0],
         [1,0,0],
         [0,1,0],
         [0,0,1],
         [0,0,1]]
    data = data_set(CAK)
    newFC_start = time.perf_counter()  # 改进后的完备构造

    context = formal_context(data, 1)
    ss = NewThreeWayForObj(context, 1)

    ss.start()
    newFC_end = time.perf_counter()  # 改进后的完备构造
    print("所用时间为：{} s".format(newFC_end - newFC_start))
    print("形式背景的大小为 对象个数：{} 属性个数：{}".format(len(ss.obj_set), len(ss.att_set)))
    cnt = 0
    for i in ss.ref:
        for r in ss.ref[i].values():
            cnt += 1
            # print(r)
            print("(",ss.fn_num2objs(r[0]),",(",ss.fn_num2atts(r[1][0]),",",ss.fn_num2atts(r[1][1]),"))")
    print("概念总个数为：", cnt)
    for c in ss.base_OEOs:
        print(c)
