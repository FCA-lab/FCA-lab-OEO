import random
import time

import numpy as np

from oeo_lab.algorithms.construction.three_woc import NewThreeWayForObj
from oeo_lab.algorithms.construction.juxtaposition import ThreeWayOC
from fca_core.context import formal_context
from fca_core.data_source import data_set


# 生成随机的形式背景，其中n是行数，m是列数
def rand_context(n, m):
    result = list()
    for i in range(n):
        row = np.random.randint(0, 2, m)
        # row = np.random.choice([0, 1], size=m, p=[0.9, 0.1])
        result.append(row.tolist())
    return result


# 根据概率生成形式背景
def rand_context_probality(n, m, q):
    result = list()
    for i in range(n):
        # row = np.random.randint(0, 2, m)
        row = np.random.choice([0, 1], size=m, p=[1 - q, q])
        result.append(row.tolist())
    return result


# 生成随机的行数和列数相同的形式背景，
def rand_context2(n):
    return rand_context(n, n)


def print_context(context):
    for r in context:
        print(r)


# 根据对象数，属性数和稠密度生成随机的形式背景
def rand_context_with_desity(obj_num, att_num, desity):
    context = [[0 for i in range(att_num)] for j in range(obj_num)]
    change_num = round(obj_num * att_num * desity)  # 四舍五入进行取整
    t = 0
    while t < change_num:
        x = random.randrange(obj_num)
        y = random.randrange(att_num)
        if (context[x][y] != 1):
            context[x][y] = 1
            t = t + 1
    return context



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

    # 马建明的Table4  文章：Concept acquisition approach of object-oriented concept lattices
    # 也是教材上的 The context of an educational film ‘‘living beings and water’’
    ls7 = [[1, 1, 0, 0, 0, 0, 1, 0, 0],
           [1, 1, 0, 0, 0, 0, 1, 1, 0],
           [1, 1, 1, 0, 0, 0, 1, 1, 0],
           [1, 1, 1, 0, 0, 0, 1, 1, 0],
           [1, 1, 0, 1, 0, 1, 0, 0, 0],
           [1, 1, 1, 1, 0, 1, 0, 0, 0],
           [1, 0, 1, 1, 1, 0, 0, 0, 0],
           [1, 0, 1, 1, 0, 1, 0, 0, 0]]
    #
    ls8 =[[1,0,0,0],
          [1,0,0,0],
          [1,0,0,0],
          [1,0,0,0],
          [1,0,0,0],
          [1,0,0,0]]

    for i in [1,2,3,4,5,6,7,8]:  # 要生成的形式背景的规模列表
        # for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        # context_list = rand_context(i, 8)  # 生成规模为i的形式背景
        #        context_list = rand_context_probality(8, 8, i)
        #  print("形式背景的规模为{}*{}.生成的形式背景如下,关系密度为{}".format(8, 8, i))
        # print_context(context_list)
        # data = data_set(context_list)
        data = data_set(eval("ls" + str(i)))
        print("formal context: "+"ls" + str(i))
        context = formal_context(data, 1)
        levelConstruction = ThreeWayOC(context, 1)
        A_3WOC = NewThreeWayForObj(context, 1)
        level_start_time = time.perf_counter()
        levelConstruction.start()
        level_end_time = time.perf_counter()
        print(
            "Qian's背景并置生成概念的时间消耗为：{},生成的概念数量为：{}".format(level_end_time - level_start_time,
                                                                               len(levelConstruction.concepts)))
        iter_start_time = time.perf_counter()
        A_3WOC.start()
        iter_end_time = time.perf_counter()
        print("3woc生成概念的时间消耗为：{},生成的概念数量为：{}".format(iter_end_time - iter_start_time,
                                                                    A_3WOC.get_numOfconcepts()))
        flag=0
        for c1 in levelConstruction.concepts:
            c2=A_3WOC.ref[len(c1.extent)][c1.key]
            if c2== None:
                flag=1
                break
            if A_3WOC.fn_num2atts(c2[1][0])!=c1.intent:
                flag=2
                break
            if A_3WOC.fn_num2atts(c2[1][1])!=c1.modus:
                flag=3
                break
        if flag>0:
            print("find error code",flag,": (", A_3WOC.fn_num2atts(c2[0]), ",(", A_3WOC.fn_num2atts(c2[1][0]),
                  ",", A_3WOC.fn_num2objs(c2[1][1]), "))")
            c1.print()
        else:
            print("great completeness!")

        cnt=0
        for i in A_3WOC.ref:
            for r in A_3WOC.ref[i].values():

                cnt += 1
                # print(r)
                print("(", A_3WOC.fn_num2objs(r[0]), ",(", A_3WOC.fn_num2atts(r[1][0]), ",", A_3WOC.fn_num2atts(r[1][1]), "))")
        print("概念总个数为：", cnt)
