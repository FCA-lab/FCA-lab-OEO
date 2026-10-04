###########################################################
#  2020-7-20
#  Author: ZhangZhuo
#  School of Information Engineering, Zhengzhou University
#  Define the interface for all algorithms in this project
## purpose:容器类型及其迭代器定义
##########################################
from enum import Enum
from abc import ABCMeta,abstractmethod
from fca_core.context import prime_type
class container_type(Enum):
    map_type =0
    list_type =1
    buckets_type =2
    buckets_index_type =3
##抽象容器类，定义接口
class abc_container(metaclass= ABCMeta):

    @staticmethod
    def gen_container(type,key_type):  # factory method
        if type == container_type.map_type:
            container = concepts_map()
        elif type == container_type.list_type:
            container = concepts_list()
        elif type == container_type.buckets_index_type:
            container = concepts_buckets(key_type)
        elif type==container_type.buckets_type:
            container = buckets()
        return container


    @abstractmethod  #抽象方法
    def print(self):
        pass
    @abstractmethod  #抽象方法
    def add_element(self,element_index,element):
        pass
    @abstractmethod  #抽象方法
    def find_element(self,element_index,element=None):
        pass
    @abstractmethod  # 抽象方法
    def del_element(self, element_index, element):
        pass
    @abstractmethod  #抽象方法
    def update_element(self,element_index,element):
        pass

    #容器迭代器
    def __iter__(self):
        pass
    #h获得容器大小
    def __len__(self):
        pass


#########################################
##字典类容器
class concepts_map(abc_container):
    def __init__(self):
        self.__space=dict()

    def add_element(self,element_index,element):
        #old_len =len(self)
        old = self.__space.get(element_index)
        if old == None:
            self.__space[element_index]=element
            return True
        #new_len= len(self)
        else:
            return False

    def del_element(self,element_index,element=None):
        return  self.__space.pop(element_index)

    def find_element(self,element_index,element=None):
        return self.__space.get(element_index)

    def update_element(self,element_index,element):
        self.__space[element_index]=element

    def print(self):
        for node in iter(self):
            print(node)

    def __iter__(self):
        return iter(self.__space.values())

    def __len__(self):
        return len(self.__space)

    def get_index_iter(self):  # 修复Bug1，添加此方法即可  2021-6-7 Yue
        return iter(self.__space.keys())

#################################################
#木桶类型容器
class buckets(abc_container):
    def __init__(self):
        self.__buckets=dict()
    def add_element(self,element_index,element):
        #首先判断bucket{i}是否存在
        a_bucket=self.__buckets.get(element_index)
        if a_bucket!=None:
            a_bucket.add(element)
        else:
            a_bucket = {element}
        self.__buckets[element_index]=a_bucket
        return 1

    def del_element(self,element_index,element):#(self,ball_num,key):
        bucket = self.__buckets[element_index]
        return bucket.remove(element)

    def find_element(self,element_index,element):
        element_set = self.__buckets.get(element_index)
        if element_set!=None and len(element_set.intersection(element)):
            return element
        else:
            return None

    def update_element(self,element_index,element):
        return add_element(element_index,element)
    def print(self):
        pass
    def __iter__(self):
       #return iter(self.__buckets.values())
        return iter(self.get_order())
    def get_order(self):
        sorted_list = sorted(self.__buckets.items(), key = lambda item:item[0])
        return sorted_list
    def __len__(self):
        return len(self.__buckets)
######################################################
##该类只适用于经典概念格应用，而且集合表示为压缩表示形式
##
class concepts_buckets(abc_container):
    def __init__(self,key_type):
        self.__space=concepts_map()
        self.__index=buckets()
        self.__key_type=key_type

    def add_element(self,element_index,element):
        self.__space.add_element(element_index,element)
        ball_num=self.__get_ball_num(element)
        self.__index.add_element(ball_num,element_index)
        return ball_num

    def del_element(self,element_index,element=None):
        node = self.__space.del_element(element_index)
        ball_num = self.__get_ball_num(node)
        self.__index.del_element(ball_num,element_index)
        return node

    def find_element(self,element_index,element=None):
        return self.__space.find_element(element_index)

    def update_element(self,element_index,element):
        #key值不变，所以索引不变，故不需要更新索引buckets
        return self.__space.update_element(element_index,element)

    def __get_ball_num(self,node):
        if self.__key_type==prime_type.att_type:
            ball_num=len(node.concept.intent)
        elif self.__key_type==prime_type.obj_type:
            ball_num=len(node.concept.extent)
        return ball_num

    def print(self):
        iter = iter(self)
        for node in iter:
            print(node)

    def __iter__(self):
        #return iter(self.__index)
        return iter(self.__space)

    def get_index_iter(self):
        return iter(self.__index.get_order())

    def __len__(self):
        return len(self.__space)
#################################################################
#列表类型容器
class concepts_list(abc_container):
    def __init__(self):
        self.__space=list()
    def print(self):
        pass

    def add_element(self,element_index,element):
        self.__space.append(element)
        return 1
    def del_element(self,element_index,element):
        self.__space.remove(element)
        return 1
    def find_element(self,element_index,element=None):
        if element_index in range(len(self.__space)):
            return  self.__space[element_index]
        else:
            return None
    def update_element(self,element_index,element):
        old_element =self.__space[element_index]
        self.__space.remove(old_element)
        self.__space.insert(element_index,element)
        return old_element

    #容器迭代器
    def __iter__(self):
        return iter(self.__space)
    def __len__(self):
        return len(self.__space)

