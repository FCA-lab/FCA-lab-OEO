import os
import random
import time

import pandas as pd
from oeo_lab.algorithms.construction.three_woc import NewThreeWayForObj
from oeo_lab.algorithms.construction.juxtaposition import ThreeWayOC
from fca_core.context import formal_context
from fca_core.data_source import data_set

uci_dir = "./data/uci/"


# data = list()

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


def get_iris_data():
    data = list()
    iris_file = os.path.join(uci_dir, "iris.data")
    with open(iris_file, 'r') as f:
        line = f.readline()
        while line:
            lineStr = line.strip()
            if lineStr != "" and '?' not in lineStr:
                line_list = lineStr.split(',')[0: -1]
                data.append(line_list)
            line = f.readline()
    return data


def get_lung_cancer_data():
    data = list()
    lung_cancer_file = os.path.join(uci_dir, "lung-cancer.data")
    with open(lung_cancer_file, 'r') as f:
        line = f.readline()
        while line:
            lineStr = line.strip()
            if lineStr != "" and '?' not in lineStr:
                # if lineStr != "":
                line_list = lineStr.split(',')[1:]
                # line_list = lineStr.split(',')
                data.append(line_list)
            line = f.readline()
    return data


def get_zoo_data():
    data = list()
    zoo_file = os.path.join(uci_dir, "zoo.data")
    with open(zoo_file, 'r') as f:
        line = f.readline()
        while line:
            lineStr = line.strip()
            if lineStr != "" and '?' not in lineStr:
                line_list = lineStr.split(',')[1: -1]
                data.append(line_list)
            line = f.readline()
    return data


def get_mushroom_data():
    data = list()
    mushroom_file = os.path.join(uci_dir, "mushroom.data")
    with open(mushroom_file, 'r') as f:
        line = f.readline()
        while line:
            lineStr = line.strip()
            if lineStr != "" and '?' not in lineStr:
                line_list = lineStr.split(',')[1: -1]
                data.append(line_list)
            line = f.readline()
    return data


# 根据三支概念格构造算法_汪文威，祁建军，读取数据集，把缺失值也当作一种属性，把类别也当作为一种属性
def get_uci_data(data_file_name):
    data = list()
    parent_path = os.path.dirname(os.getcwd())+"/data/uci"
    file = os.path.join(parent_path, data_file_name)
    with open(file, 'r') as f:
        line = f.readline()
        while line:
            lineStr = line.strip()
            if lineStr != "":
                line_list = lineStr.split(',')
                data.append(line_list)
            line = f.readline()
    return data


iris_obj_num = 150
lung_cancer_num = 27
mushroom_num = 5644
# uci_data_names = ["balance-scale.data", "car.data", "nursery.data",  "mushroom.data", "zoo.data"] # "lung-cancer.data",属性值太多
uci_data_names =["ballon", "post", "lung-cancer", "iris.data","balance-scale.data", "car.data"] # "mushroom.data",
if __name__ == '__main__':
    # data = get_iris_data()
    # data = get_lung_cancer_data()
    # data = get_mushroom_data()
    # data = get_zoo_data()  # time: 1660.419767   concept: 17869
    # data = rand_context_with_desity(150, 125, 0.2)  # 相似大小的随机形式背景，验证程序是否正常。
    # data = get_uci_data(uci_data_names[1])
    # data_pd = pd.DataFrame(data)
    # # data_pd = data_pd.iloc[0:50, :]  # 取前50行数据
    # data_dumies = pd.get_dummies(data_pd)
    # data_dumies = data_dumies.iloc[0:50, :]  # 取前50行数据
    # print(data_dumies)
    # ls = data_dumies.values
    # print("生成的背景的大小为:{} * {}".format(len(ls), len(ls[0])))
    # data = data_set(ls)
    # context = formal_context(data, 1)
    # iterConstruciton = NewThreeWayForObj(context, 1, mushroom_num)
    #
    # iter_start_time = time.perf_counter()
    # iterConstruciton.start()
    # iter_end_time = time.perf_counter()
    # print("迭代生成在鸢尾花背景构造概念的时间为：{}".format(iter_end_time - iter_start_time))
    # print("生成的概念数量为：", len(iterConstruciton.result_ls))

    for uci_data in uci_data_names:
        print("==============UCI:",uci_data,"开始时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        data = get_uci_data(uci_data+".data")
        data_pd = pd.DataFrame(data)
        data_dumies = pd.get_dummies(data_pd)
        #data_dumies = data_dumies.iloc[0:10, :]  # 取前50行数据
        # print(data_dumies)
        ls = data_dumies.values
        print("生成的背景的大小为:{} * {}".format(len(ls), len(ls[0])))
        data = data_set(ls)
        context = formal_context(data, 1)

        A_3WOC = NewThreeWayForObj(context, 1)

        levelConstruction = ThreeWayOC(context,1)


        print("3woc algorithm begin->>")
        iter_start_time = time.perf_counter()
        A_3WOC.start()
        iter_end_time = time.perf_counter()
        print("3woc生成概念的时间消耗为：{},生成的概念数量为：{}".format(iter_end_time - iter_start_time,
                                                                       A_3WOC.get_numOfconcepts()))
        print("Qian's algorithm begin->>")
        level_start_time = time.perf_counter()
        levelConstruction.start()
        level_end_time = time.perf_counter()

        print("Qian's背景并置生成概念的时间消耗为：{},生成的概念数量为：{}".format(level_end_time - level_start_time,
                                                                                 len(levelConstruction.concepts)))
        flag = 0
        for c1 in levelConstruction.concepts:
            c2 = A_3WOC.ref[len(c1.extent)][c1.key]
            if c2 == None:
                flag = 1
                break
            if A_3WOC.fn_num2atts(c2[1][0]) != c1.intent:
                flag = 2
                break
            if A_3WOC.fn_num2atts(c2[1][1]) != c1.modus:
                flag = 3
                break
        if flag > 0:
            print("find error code", flag, ": (", A_3WOC.fn_num2atts(c2[0]), ",(", A_3WOC.fn_num2atts(c2[1][0]),
                  ",", A_3WOC.fn_num2objs(c2[1][1]), "))")
            c1.print()
        else:
            print("great completeness!")
'''
        qianing_start_time = time.perf_counter()
        qianting_constructuin.start()
        qianing_end_time = time.perf_counter()
        print("迭代生成在", uci_data, "背景通过背景并置构造概念的时间为：{}".format(qianing_end_time - qianing_start_time))
        print("生成的概念数量为：", len(qianting_constructuin.concepts))
        print("完成时间：", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
'''