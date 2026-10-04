import itertools

from fca_core.interfaces import abc_algorithm
from fca_core.context import abc_context, prime_type, formal_context
from fca_core.concept import formal_concept as concept, Object_Oriented_cncept
from fca_core.containers import concepts_list, concepts_map, abc_container, container_type
import time
import pandas as pd

from fca_core.data_source import data_set


# 面向对象的概念格构造
# 论文来源：Concept acquisition approach of object-oriented concept lattices
# International Journal of Mache Learning and Cybernetics
class Object_oriented(abc_algorithm):
    def __init__(self, context, sequence_type):
        super().__init__("object oriented")
        self.__context = context
        self.att_num = context.att_num
        self.obj_num = context.obj_num
        self.all_extent = list()
        self.__sequence_type = sequence_type
        if self.__sequence_type == prime_type.att_type:
            self.__sequence_size = context.att_num
        else:
            self.__sequence_size = context.obj_num
        self.concepts = []
        self.att_set = set()
        for i in range(self.__context.att_num):
            self.att_set.add(i)

        self.obj_set = set()
        for i in range(self.__context.obj_num):
            self.obj_set.add(i)

    # 集合不能被hash，因此定义一个函数，将对象集合转为数字
    def objs2hash(self, objs):
        res = 0
        for obj in objs:
            temp = 1 << obj
            res = res | temp
        return res

    def num2objs(self, num):
        res = set()
        for i in self.obj_set:
            t = 1 << i
            if t & num:
                # res.add(i+1)  # 此处是特殊情况，如果是从0开始的背景，则不需要加1
                res.add(i)
        return res

    # 根据可能算子，通过属性映射对应的对象
    def oo_att2num(self, atts):
        result = list()
        for obj in self.obj_set:
            temp = self.__context.obj2att({obj})
            if not temp.isdisjoint(atts):  # 取交集不为空时
                result.append(obj)
        return result

    # 根据外延生成面向对象概念
    def gen_concept(self, objs):
        intent = set()
        for i in range(self.__context.att_num):  # 遍历所有属性
            if self.__context.att2obj({i}).issubset(objs):
                # if self.isSub(self._context.att2obj({i}), objs):  # 如果当前属性和对象集合符合正算子
                intent.add(i)  # 将当前属性加入到拥有的必然属性集合中
        new_concept = Object_Oriented_cncept(objs, intent)
        return new_concept
    def gen_concept_by_hash(self,hash_key):
        objs = self.num2objs(hash_key)
        intent = set()
        for i in range(self.__context.att_num):  # 遍历所有属性
            if self.__context.att2obj({i}).issubset(objs):
            #if self.__context.att2obj({i}).isSub(objs):  # 如果当前属性和对象集合符合正算子
                intent.add(i)  # 将当前属性加入到拥有的必然属性集合中
        new_concept = Object_Oriented_cncept(hash_key,objs, intent)
        return new_concept

    def start(self):
        concept_list = abc_container.gen_container(container_type.map_type,
                                                   prime_type.att_type)  # concepts_list() #concepts_map() #list()
        all_result_extent = list()
        now_result_extent = list()
        all_result_extent.append(0) #L0
        for att in self.att_set:
            objs = self.oo_att2num({att})
            objs_hash = self.objs2hash(objs)
            all_result_extent.append(objs_hash)
            now_result_extent.append(objs_hash)
        for k in range(2, self.att_num + 1):
            att_power_set = itertools.combinations(self.att_set, k)
            temp_list = list()
            for atts in att_power_set:
                temp_objs = self.oo_att2num(atts)
                temp_list.append(self.objs2hash(temp_objs))
            if set(temp_list).issubset(set(now_result_extent)):
                break
            else:
                now_result_extent = temp_list
                all_result_extent.extend(now_result_extent)
        all_result_extent = set(all_result_extent) #
        # print(all_result_extent)
        for extent_hash in all_result_extent:
            #extent=self.num2objs(extent_hash)
            #self.all_extent.append(extent)  # 保存所有的外延
            self.concepts.append(self.gen_concept_by_hash(extent_hash))
            # print(self.num2objs(extent))


if __name__ == '__main__':
    ls = [[1, 0, 1, 1, 1],
          [1, 0, 1, 0, 0],
          [0, 1, 0, 0, 1],
          [1, 0, 0, 0, 0],
          [1, 1, 0, 0, 1]]
    ls2 = [[1, 1, 0, 1, 1],  # 测试数据2
           [1, 1, 1, 0, 0],
           [0, 0, 0, 1, 0],
           [1, 1, 1, 0, 0]]
    ls3 = [[1, 0, 1, 0, 1, 1, 0],  # 测试数据3-刘敏茜论文
           [1, 1, 0, 1, 0, 0, 1],
           [0, 0, 1, 0, 0, 1, 0],
           [0, 0, 1, 0, 0, 1, 0],
           [1, 1, 0, 0, 0, 0, 1]]

    ls4 = [[0, 0, 1, 0, 1, 0, 1],  # 测试数据4-潘爱宝论文
           [0, 0, 1, 0, 0, 1, 1],
           [0, 0, 1, 0, 0, 0, 0],
           [0, 1, 0, 1, 0, 0, 1],
           [1, 0, 1, 0, 1, 1, 1],
           [0, 1, 1, 1, 1, 0, 1]]
    ls5 = [[1, 1, 0, 1, 1, 0],  # 测试数据5-陈希邦论文
           [1, 0, 1, 0, 0, 1],
           [0, 1, 0, 1, 0, 0],
           [1, 1, 1, 0, 1, 0],
           [0, 0, 0, 1, 0, 0]]

    ls7 = [[1, 1, 0, 0],
           [0, 0, 1, 1],
           [1, 0, 1, 0],
           [1, 1, 1, 0],
           [0, 0, 0, 1]]

    ls8 = [[1, 0, 0, 0, 0, 1, 0, 0],
           [1, 0, 0, 0, 0, 1, 1, 0],
           [1, 1, 0, 0, 0, 1, 1, 0],
           [0, 1, 0, 0, 0, 1, 1, 1],
           [1, 0, 1, 0, 1, 0, 0, 0],
           [1, 1, 1, 0, 1, 0, 0, 0],
           [0, 1, 1, 1, 0, 0, 0, 0],
           [0, 1, 1, 0, 1, 0, 0, 0], ]

    d = pd.read_csv("../bitcoin_classificatiion/sample/train_pos.csv")
    label = d['label']
    concept_data = d.drop(['Unnamed: 0', 'sample_id', 'label'], axis=1)
    ls6 = concept_data.values.tolist()

    start_time = time.perf_counter()
    data = data_set(ls8)
    context = formal_context(data, 1)
    oo_builder = Object_oriented(context, 1)
    oo_builder.start()
    end_time = time.perf_counter()
    print("所用时间为：", end_time - start_time,"num of concepts:", len(oo_builder.concepts))
    for concept in oo_builder.concepts:
       concept.print()
