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
from fca_core.operators import galois_op_type
from enum import Enum


class simliarity_type(Enum):
    Jaccard_index = 1
    Sorenesen_coefficient = 2
    Symmetric_difference = 3


class abc_concept(metaclass=ABCMeta):
    def __init__(self, concept):
        self.concept = concept

    @property
    def id(self):
        return self.concept._id

    @id.setter
    def id(self, new_id):
        self.concept._id = new_id

    @property
    def key(self):
        return self.concept._key

    @key.setter
    def key(self, new_key):
        self.concept._key = new_key

    @staticmethod
    def gen_concept(extent, intent, type=galois_op_type.binary):
        if type == galois_op_type.binary:
            new_concept = formal_concept(extent, intent)  # 调用其子类，在下面
        else:
            pass
        return new_concept

    @staticmethod
    def jaccard_index(set1, set2):
        a = len(set1.intersection(set2))
        b = len(set1.union(set2))
        return a / b

    @staticmethod
    def sorenesen_coefficient(set1, set2):
        a = len(set1.intersect(set2))
        b = len(set1)
        c = len(set2)
        return 2 * a / (b + c)

    @staticmethod
    def symmetric_difference(set1, set2):
        temp_set1 = set1.difference(set2)
        temp_set2 = set2.difference(set1)
        a = len(temp_set1.union(temp_set2))
        b = len(set1.union(set2))
        return 1 - a / b

    @abstractmethod  # 抽象方法
    def print(self):
        pass

    @abstractmethod
    def is_empty(self):
        pass


class formal_concept(abc_concept):
    def __init__(self, extent, intent):
        super().__init__(self)  # 把自己传过去，使自己多了一个属性concept，这个属性的值就是自己
        self.extent = extent
        self.intent = intent
        self._id = 0
        self._key = 0

    def is_empty(self):
        if len(self.extent) == 0 and len(self.intent) == 0:  # 判断是否外延和内涵都为空
            return True
        else:
            return False

    def is_half_empty(self):  # 判断是否外延或内涵为空,用于三元概念产生过程。
        if len(self.extent) > 0 and len(self.intent) > 0:
            return True
        else:
            return False

    def weighted_concept_similarity(self, weight, concept, set_similarity_fun):  # 方法名作为形参还是类名作为形成？？？
        a = set_similarity_fun(self.extent, concept.extent)
        b = set_similarity_fun(self.intent, concept.intent)
        return weight * a + (1 - weight) * b

    def print(self):
        # string="id:" + str(self._id)+ "->extent:"+str(self.extent)+",intent:"+str(self.intent)
        string = "->extent:" + str(self.extent) + ",intent:" + str(self.intent)
        print(string)
        return string

    def printstr(self):
        string = "id:" + str(self._id) + "->extent:" + str(self.extent) + ",intent:" + str(self.intent)
        return string


################################################################
# 结点类型定义
#############
class concept_node(abc_concept):
    def __init__(self, concept):  # 这里的concept应该为 formal_concept 对象
        super().__init__(concept)
        if concept != None:
            self.up_cover = set()
            self.down_cover = set()
        else:
            raise ValueError

    def print(self):
        print("id:{:d},key:{:d}".format(self.id, self.key),
              self.concept.printstr(),
              "upC:", self.up_cover, "downCove:", self.down_cover)

    def is_empty(self):
        return self.concept.is_empty()

    def __str__(self):
        return f"key{self.key} extent{self.concept.extent} intent{self.concept.intent}  up{self.up_cover} down{self.down_cover}"


################################################################
# 三元概念的定义
# 2021.5.24
# zhang zhuo @ zzu
#############
class Triadic_Concept(abc_concept):
    def __init__(self, extent, intent, modus):
        super().__init__(self)  # 把自己传过去，使自己多了一个属性concept，这个属性的值就是自己
        self.extent = extent
        self.intent = intent
        self.modus = modus
        self._id = 0
        self._key = 0

    def is_empty(self):
        if len(self.extent) == 0 and len(self.intent) == 0 and len(self.modus) == 0:
            return True
        else:
            return False

    def print(self):

        string = "id:" + str(self._id) + "->extent:" + str(self.extent) + ",intent:" + str(
            self.intent) + ",modus:" + str(self.modus)
        print(string)

    def printstr(self):
        string = "id:" + str(self._id) + "->extent:" + str(self.extent) + ",intent:" + str(
            self.intent) + ",modus:" + str(self.modus)
        return string
################################################################
# 三支概念的定义
# 2024.5.24
# zhang zhuo @ zzu
#############

class Three_Way_Concept(formal_concept):
    def __init__(self, extent, intent, modus):
        super().__init__(extent, intent)  # 把自己传过去，使自己多了一个属性concept，这个属性的值就是自己
        self.modus = modus
        self.parent = None
    def __init__(self, key,extent, intent, modus):
        super().__init__(extent, intent)  # 把自己传过去，使自己多了一个属性concept，这个属性的值就是自己
        self.modus = modus
        self.parent = None
        self.key=key

    def print(self):
        string = "extent: " + str(self.extent) + " intent: " + str(self.intent) + " modus: " + str(self.modus)
        print(string)


class Object_Oriented_cncept(formal_concept):
    def __init__(self, extent, intent):
        super().__init__(extent, intent)
    def __init__(self,key, extent, intent):
        super().__init__(extent, intent)
        self.key = key
    def print(self):
        string = "extent: " + str(self.extent) + " intent: " + str(self.intent)
        print(string)
