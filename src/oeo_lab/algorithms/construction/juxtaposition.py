import time

from oeo_lab.algorithms.construction.object_oriented import Object_oriented
from fca_core.interfaces import abc_algorithm
from fca_core.lattice import map_lattice
from fca_core.context import formal_context
from fca_core.data_source import data_set
from fca_core.concept import Three_Way_Concept

import numpy as np


class ThreeWayOC(abc_algorithm):
    def __init__(self, context, update_type):
        super().__init__("Three_way_for_obj")
        self._context = context
        self._comple_context = None  # 补集
        self.att_num = context.att_num
        self.att_set = {i for i in range(context.att_num)}
        self.obj_set = {i for i in range(context.obj_num)}
        self.lattice = map_lattice(context)
        self.all_extent = list()
        self.concepts = list()

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

    def start(self):
        # 先生成补背景
        self._comple_context = formal_context(
            data_source=data_set(self.gen_complement_context(self._context.get_context())), key_type=1)
        # print(self._context.get_context())
        # 再将背景并置
        np_context = np.asarray(self._context.get_context())
        np_com_context = np.asarray(self._comple_context.get_context())
        np_concat = np.concatenate((np_context, np_com_context), axis=1)
        # 然后构造面向对象概念格
        oo_data = data_set(np_concat.tolist())
        oo_context = formal_context(oo_data, 1)
        oo = Object_oriented(oo_context, 1)
        oo.start()
        self.all_extent = oo.all_extent
        for i in oo.all_extent:
            print(i)
        #根据面向对象概念映射三支面向对象概念
        for oo_concept in oo.concepts:
            extent = oo_concept.extent
            intent = set()
            modus = set()
            intent_modus = oo_concept.intent
            for i in intent_modus:
                if i >= self.att_num:
                    modus.add(i - self.att_num)
                else:
                    intent.add(i)
            new_node = Three_Way_Concept(oo_concept.key,extent, intent, modus)  # 根据对象集合和必然属性集合生成概念
            self.concepts.append(new_node)
        # 按照特定格式输出概念列表
        # att_map = ['a', 'b', 'c', 'd', 'e']
        # text_str= ""
        # for i in self.concepts:
        #     extents_plus = "".join([str(t + 1) for t in i.extent])
        #     intents_plus = "".join([att_map[t] for t in i.intent])
        #     modus_plus = "".join([att_map[t] for t in i.modus])
        #     text_str = text_str + "({},({},{})) ,".format(extents_plus, intents_plus, modus_plus)
        # print(text_str)




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

    data = data_set(ls6)
    context = formal_context(data, 1)
    algorithm = ThreeWayOC(context, 1)
    newFC_start = time.perf_counter()  # 改进后的完备构造
    algorithm.start()
    newFC_end = time.perf_counter()  # 改进后的完备构造
    print("改进的完备算法构造生成的概念数量为：{}".format(len(algorithm.concepts)))
    print("所用时间为：{}".format(newFC_end - newFC_start))
    print("形式背景的大小为 对象个数：{} 属性个数：{}".format(len(algorithm.obj_set),algorithm.att_num))
    print('=' * 20)  # 分隔线
    cnt=0
    for c in algorithm.concepts:
        c.print()
        cnt += 1
    print("概念总个数为：", cnt)
