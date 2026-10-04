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
from fca_core.concept import abc_concept
from fca_core.operators import galois_op_type, abc_galois
from fca_core.iterators import abc_iterator
from fca_core.data_source import data_set
from enum import Enum
import copy


class prime_type(Enum):
    att_type = 0
    obj_type = 1


# 指定这是一个抽象类
class abc_context(metaclass=ABCMeta):
    #
    _context = None

    def __init__(self, galois_type, data, key_type):
        self.__galois_op = abc_galois.gen_galois_op(galois_type)
        self._context = data.get_dataset()
        self.__key_type = key_type

    #
    @property
    def obj_num(self):
        return len(self._context)

    @property
    def att_num(self):
        return len(self._context[0])

    @property
    def key_type(self):
        return self.__key_type

    @property
    def galois_op(self):
        return self.__galois_op

    # @galois_op.setter
    # def galois_op(self,new_galois):
    # self.__galois_op=new_galois
    ##伽罗瓦联系适配器
    def att_closure(self, intent):
        return self.__galois_op.att_closure(self, intent)

    def obj_closure(self, extent):
        return self.__galois_op.obj_closure(self, extent)

    def obj2att(self, obj):
        return self.__galois_op.obj2att(self, obj)

    def att2obj(self, att):
        return self.__galois_op.att2obj(self, att)

    ##
    def get_row(self, obj_id):
        return self._context[obj_id][:]

    def get_col(self, att_id):
        return self._context[:][att_id]

    def get_data(self, row_id=0, col_id=0):
        # print(self._context[row_id][col_id])
        # print("row",row_id,"col",col_id)
        return self._context[row_id][col_id]

    def att_concept(self, atts):  # 属性概念
        extent = self.att2obj(atts)
        intent = self.obj2att(extent)
        new_concept = self.gen_concept(extent, intent)
        return new_concept

    def obj_concept(self, objs):
        intent = self.obj2att(objs)
        extent = self.att2obj(intent)
        new_concept = self.gen_concept(extent, intent)
        return new_concept

    def gen_concept_by_intent(self, intent):
        extent = self.att2obj(intent)
        new_concept = self.gen_concept(extent, intent)
        return new_concept

    def gen_concept_by_extent(self, extent):
        intent = self.obj2att(extent)
        new_concept = self.gen_concept(extent, intent)
        return new_concept

    def gen_concept(self, extent, intent):
        key = self.gen_key_by_intent(intent)
        if self.key_type == prime_type.att_type:
            key = self.gen_key_by_intent(intent)
        elif self.key_type == prime_type.obj_type:
            key = self.gen_key_by_extent(extent)
        new_concept = abc_concept.gen_concept(extent, intent)
        new_concept.key = key
        return new_concept

    def gen_key_by_intent(self, intent):
        return self.gen_key(intent, self.att_num)

    def gen_key_by_extent(self, extent):  # todo 有问题 既然此方法暴露给外界使用，但是没有实现gen_key这个抽象方法，反而是让子类去实现，
        self.gen_key(extent, self.obj_num)  # # 虽然子类实现了gen_key抽象方法，如果外界通过子类来调用此方法，则又去调用子类的gen_key方法，
        # ,形成了闭环,造成参数传递错误

    def check_pos(self, pos):  # 检测该点是否越界（即该点是否在形式背景范围内）
        result = False
        obj_id = pos[0]
        att_id = pos[1]
        if att_id < self.att_num and obj_id < self.obj_num:
            result = True
        return result

    # 返回复制的形式背景
    def get_context(self):
        return copy.deepcopy(self._context)

    #
    @abstractmethod  # 抽象方法
    def get_row(self, obj_id):
        pass

    @abstractmethod  # 抽象方法
    def get_col(self, att_id):
        pass

    @abstractmethod  # 抽象方法
    def gen_key(self, set, full_set_size):
        pass

    @abstractmethod
    def __iter__(self):
        pass


###############################################################################

class formal_context(abc_context):
    '''
    只是重写父类的抽象方法，其他不变
    '''

    def __init__(self, data_source, key_type):
        super().__init__(galois_op_type.binary, data_source, key_type)

    # 集合压缩表示格式
    def get_row(self, obj_id):
        # row_list=get_row(obj_id)
        row_list_zip = list()
        for att_id in range(0, self.att_num, 1):
            if super().get_data(obj_id, att_id) == 1:
                row_list_zip.append(att_id)
        return row_list_zip

    # 集合压缩表示格式
    def get_col(self, att_id):
        # obj_list = get_row(att_id)
        col_list_zip = list()
        for obj_id in range(0, self.obj_num, 1):
            if super().get_data(obj_id, att_id) == 1:
                col_list_zip.append(obj_id)
        return col_list_zip

    def gen_key(self, zip_set, full_set_size):  # 传进来的参数含义 属性集合，属性全集的个数
        key = 0
        bas = self.galois_op.L_size
        for index in zip_set:
            power_index = full_set_size - index - 1
            key = key + pow(bas, power_index)
        return key

    def __iter__(self):
        return context_iterator(self)


class context_iterator(abc_iterator):
    def __init__(self, context):
        self.__context = context
        self.__counter = 0
        self.__Total = 0
        self.__mode = prime_type.att_type  # default mode

    @property
    def mode(self):
        return self.__mode

    @mode.setter
    def mode(self, type):
        self.__mode = type
        if type == prime_type.att_type:
            self.__Total = self.__context.obj_num
        elif type == prime_type.obj_type:
            self.__Total = self.__context.att_num

    def __next__(self):
        if self.__counter < self.__Total:
            if self.mode == prime_type.att_type:
                value = self.__context.get_row(self.__counter)
            elif self.mode == prime_type.obj_type:
                value = self.__context.get_col(self.__counter)
            self.__counter = self.__counter + 1
            return self.__counter - 1, value
        else:
            raise StopIteration


################################################################
# 模糊形式背景的定义
# 2021.5.24
# zhang zhuo @ zzu
#############
class fuzzy_context(abc_context):
    pass


################################################################
# 三元形式背景的定义
# 2021.5.24
# zhang zhuo @ zzu
#################################################################
class triadic_context(abc_context):
    def __init__(self, data_source_list, key_type):
        context_list = list()
        for i in range(len(data_source_list)):
            single_modus_context = formal_context(data_set(data_source_list[i]), key_type)  # 将子背景包装为data_set对象
            context_list.append(single_modus_context)
        super().__init__(galois_op_type.binary, data_set(context_list), key_type)
        self.objnum = len(data_source_list[1])  # 形式背景中对象个数
        self.attrnum = len(data_source_list[1][0])  # 形式背景中属性个数
        self.modus_num = len(context_list)  # 形式背景中条件个数

    def get_single_context(self, condition):  # 增加一个方法：获取单条件形式背景，传入nextclosure算法
        _context = self.get_context()  # author ：pilanyu
        return _context[condition]

    def get_row(self, obj_id):
        row_list = list()
        for i in range(len(self._context)):
            row_i = self._context[i].get_row(obj_id)
            row_list.append(row_i)
        return row_list

    def get_col(self, att_id):
        col_list = list()
        for i in range(len(self._context)):
            col_i = self._context[i].get_col(att_id)
            col_list.append(col_i)
        return col_list

    def gen_key(self, zip_set, full_set_size):
        key = self._context[0].gen_key(zip_set, full_set_size)
        return key

    def __iter__(self):
        return tricontext_iterator(self)

    def get_sub_context(self, modus):
        return self._context[modus]


class tricontext_iterator(context_iterator):
    def __init__(self, context):
        super().__init__(context)
