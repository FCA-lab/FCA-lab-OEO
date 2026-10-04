######################################################
#  Purpose: 定义形式概念抽象类。
#  Author:zhang zhuo
#  Date:2020-4-26
#  Modife Date:
#  Purpose:
#
#
#########################################################
from abc import ABCMeta, abstractmethod
from enum import Enum


class galois_op_type(Enum):
    binary = {0, 1}
    fuzzy_2L = {0, 0.5, 1}


class abc_galois(metaclass=ABCMeta):
    __galois_L = None

    def __init__(self, type):
        self.__galois_L = type.value

    @property
    def L_size(self):
        return len(self.__galois_L)

    @property
    def L_type(self):
        return self.__galois_L

    @staticmethod
    def gen_galois_op(type):
        if type == galois_op_type.binary:
            galois_op = classic_galois(type)
        elif type == galois_op_type.fuzzy_2L:
            pass
        return galois_op

    @abstractmethod  # 抽象方法
    def obj2att(self, context, obj):
        pass

    @abstractmethod  # 抽象方法
    def att2obj(self, context, att):
        pass

    def att_closure(self, context, intent):
        return self.obj2att(context, self.att2obj(context, intent))

    def obj_closure(self, context, extent):
        return self.att2obj(context, self.obj2att(context, extent))


class classic_galois(abc_galois):
    def __init__(self, type):
        super().__init__(type)

    def obj2att(self, context, extent):
        obj_num = context.obj_num
        att_num = context.att_num
        # extent and intent 使用压缩表达
        new_intent = set([i for i in range(0, att_num, 1)])
        if len(extent) > 0:
            for obj_index in extent:
                new_intent.intersection_update(context.get_row(obj_index))
        return new_intent

    def att2obj(self, context, intent):
        obj_num = context.obj_num
        att_num = context.att_num
        # extent and intent 使用压缩表达
        new_extent = set([i for i in range(0, obj_num, 1)])
        if len(intent) > 0:
            for att_index in intent:
                new_extent.intersection_update(context.get_col(att_index))
        return new_extent
