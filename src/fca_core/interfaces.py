######################################################
#  Purpose: 定义抽象工厂
#  Author:zhang zhuo
#  Date:2020-4-26
#  Modife Date:
#  Purpose:
#
#
#########################################################
from abc import ABCMeta,abstractmethod
from enum import Enum
##
class alg_name(Enum):
    NextClosure = "next_closure_binary"
    Godin_increment= "godin_increment"
    Bordat = "Bordat"
    Triadic = 'Triadic'

class abc_algorithm(metaclass= ABCMeta):
    #指定这是一个抽象类
    def __init__(self,name):
         self.__alg_name=name
    @property
    def alg_name(self):
        return self.__alg_name

    @alg_name.setter
    def alg_name(self, name):
        self.__alg_name = name

    @abstractmethod  #抽象方法
    def start(self):
        pass

