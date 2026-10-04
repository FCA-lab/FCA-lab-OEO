######################################################
#  Purpose: 定义迭代器
#  Author:zhang zhuo
#  Date:2020-7-20
#  Modife Date:
#  Purpose:
#
#
#########################################################
from abc import ABCMeta,abstractmethod

class abc_iterator(metaclass= ABCMeta):
    def __init__(self):
        pass
    def __iter__(self):
        return self
    @abstractmethod  #抽象方法
    def __next__(self):
        pass

class lattice_iterator(abc_iterator):
    def __init__(self):
        pass
    def __next__(self):
        pass

'''


class concepts_list_iter(abc_iterator):
    def __init__(self,container):
        self.__container = container
        self.__counter = 0
    def __next__(self):
        if self.__counter<len(self.__container):
            value = self.__container[self.__counter]
            self.__counter = self.__counter +1
            return value
        else:
            raise StopIteration

'''