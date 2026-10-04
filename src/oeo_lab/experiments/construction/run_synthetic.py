import time

import pandas as pd
import os
import random

# 根据对象数，属性数和稠密度生成随机的形式背景
from oeo_lab.algorithms.construction.three_woc import NewThreeWayForObj
from oeo_lab.algorithms.construction.juxtaposition import ThreeWayOC
from fca_core.context import formal_context
from fca_core.data_source import data_set


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


data_path = "../data/random/obj-{}-att{}-dens-{}.csv"
result_path = "../data/result/result-for-random-comtext.csv"


def gen_random_data():
    for obj_num in [50, 100, 300, 500, 800, 1000]:
        for att_num in [10, 50, 100, 150, 200, 300, 400, 500]:
            for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
                context = rand_context_with_desity(obj_num, att_num, density)
                data = pd.DataFrame(context)
                data.to_csv(data_path.format(obj_num, att_num, density), index=None)


if __name__ == '__main__':
   # result_data = pd.DataFrame(
   #     columns=['obj_num', 'att_num', 'desity', 'qianting_time', 'qianting_concept_num', 'ba_time', 'ba_concept_num'])
    # result_data = pd.read_csv(result_path)
    for att_num in [ 10,40, 50, 60]:
        for obj_num in [10, 40, 50, 60]:
            for density in [0.1, 0.2, 0.3, 0.4, 0.5]:
                d = pd.read_csv(data_path.format(obj_num, att_num, density))
                print("形式背景: 对象：{},属性：{},密度:{}".format(obj_num,att_num,density))
                ls = d.values
                data = data_set(ls)
                context = formal_context(data, 1)

                iterConstruciton = NewThreeWayForObj(context, 1)# obj_num)
                iter_start_time = time.perf_counter()
                iterConstruciton.start()
                iter_end_time = time.perf_counter()

                ba_time = iter_end_time - iter_start_time
                ba_concept_num = iterConstruciton.get_numOfconcepts()
                print("迭代合并完成,产生三支概念:{}，所用时间为：{} s".format(ba_concept_num,ba_time))

'''
                levelConstruction = ThreeWayOC(context, 1)
                level_start_time = time.perf_counter()
                levelConstruction.start()
                level_end_time = time.perf_counter()
                qianting_concept_num = len(levelConstruction.concepts)
                print("钱婷算法完成")

                qianting_time = level_end_time - level_start_time
               

                data_row = pd.DataFrame(
                    {'obj_num': [obj_num], 'att_num': [att_num], 'desity': [density], 'qianting_time': [qianting_time],
                     'qianting_concept_num': qianting_concept_num,
                     'ba_time': [ba_time], 'ba_concept_num': ba_concept_num})
                # 将数据保存到文件
                result_data = pd.concat([result_data, data_row], axis=0)
                print("obj:{},att:{},desity:{} 生成完毕，背景并置时间：{}，迭代合并时间:{} ---".format(obj_num, att_num, density,
                                                                                    qianting_time, ba_time))
                result_data.to_csv(result_path, index=None)
                print("上面数据已保存")
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

                data_path = os.path.join(os.getcwd(), "../experiment/data/result/result-for-random-comtext.csv")
                result_dir_path = os.path.join(os.getcwd(), "../experiment/data/result/new_pic/")
                view = three_way_oc_data_view(data_path, result_dir_path)
                view.create_line_chart([[10, 20], [20, 10], [30, 10], [40, 10]], ['o-', '^:', 'D--', 'p-.'])
'''
# if __name__ == '__main__':
#     result_data = pd.DataFrame(columns=['obj_num', 'att_num', 'desity', 'qianting_time', 'ba_time'])
#     for obj_num in [50, 100, 300, 500, 800, 1000]:
#         for att_num in [10, 50, 100, 150, 200, 300, 400, 500]:
#             for density in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
#                 d = pd.read_csv(data_path.format(obj_num, att_num, density))
#                 ls = d.values
#                 data = data_set(ls)
#                 context = formal_context(data, 1)
#                 levelConstruction = ThreeWayOC(context, 1)
#                 iterConstruciton = NewThreeWayForObj(context, 1, obj_num)
#
#                 iter_start_time = time.perf_counter()
#                 iterConstruciton.start()
#                 iter_end_time = time.perf_counter()
#
#                 level_start_time = time.perf_counter()
#                 levelConstruction.start()
#                 level_end_time = time.perf_counter()
#
#                 qianting_time = level_end_time - level_start_time
#                 ba_time = iter_end_time - iter_start_time
#                 data_row = pd.DataFrame({'obj_num':[obj_num], 'att_num':[att_num], 'desity':[density], 'qianting_time':[qianting_time], 'ba_time':[ba_time]})
#                 # 将数据保存到文件
#                 result_data = pd.concat([result_data,data_row],axis=0)
#                 print("obj:{},att:{},desity:{} 生成完毕，背景并置时间：{}，迭代合并时间:{} ---".format(obj_num, att_num, density, qianting_time, ba_time))
#                 result_data.to_csv(result_path)
#                 print("上面数据已保存")
#                 print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time())))
