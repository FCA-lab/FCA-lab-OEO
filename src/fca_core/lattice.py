######################################################
#  Purpose: 定义形式概念格结构
#  Author:zhang zhuo
#  Date:2020-4-26
#  Modife Date:2020-7-20
#  Purpose:
#  note:由于算法的专一性，原理不同，而且为追求高效性，算法所使用的的格结构的数据结构，并不通用。
#       例如：godin_increment算法使用木桶容器，方便建立概念间联系时的检索。而map类容器，则不具有这也功能。
#       故针对具有的算法，定义具体适用的格结构类。
#########################################################
from abc import ABCMeta,abstractmethod
from fca_core.context import prime_type,formal_context
from fca_core.containers import container_type,abc_container
from fca_core.concept import concept_node
from fca_core.iterators import abc_iterator
from enum import Enum
'''
class structure_type(Enum):
    concept_lattice_map=0
    concept_lattice_list=1
    concept_lattice_bucket=2
'''
class abc_structure(metaclass= ABCMeta):
    def __init__(self,context,container_type,key_type):
        self._source_context = context
        self._container = abc_container.gen_container(container_type, key_type)
        self.key_type = key_type
    '''
    @staticmethod
    def gen_structure(lattice_type):
        if lattice_type==structure_type.concept_lattice_map:
            new_structure=concept_lattice(container_type.map_type)
        elif lattice_type==structure_type.concept_lattice_list:
            new_structure=concept_lattice(container_type.list_type)
        elif lattice_type==structure_type.concept_lattice_bucket:
            new_structure=concept_lattice(container_type.map_type,prime_type.att_type)
        return new_structure
    '''
    def father_son(self,father_index,son_index):  # 建立父子关系
        father_node = self.find_node(father_index)
        son_node = self.find_node(son_index)
        father_node.down_cover.add(son_index)
        son_node.up_cover.add(father_index)

    def break_father_and_son(self,father_index,son_index):
        father_node = self.find_node(father_index)
        son_node = self.find_node(son_index)
        if son_index in father_node.down_cover:
            father_node.down_cover.remove(son_index)
        if father_index in son_node.up_cover:
            son_node.up_cover.remove(father_index)

    def find_concepts_by_atts(self,atts):
        #获得属性概念
        supre_c = self._source_context.att_concept(atts)
        keys_list =list()
        self.get_sons_and_me(supre_c.key,keys_list)
        return keys_list

    def find_concepts_by_objs(self,objs):
        infi_c = self._source_context.obj_concept(objs)
        keys_list =list()
        self.get_fathers_and_me(infi_c.key,keys_list)
        return keys_list

    def get_sons_and_me(self,concept_key,keys_list):
        cur_node = self.find_node(concept_key)
        keys_list.append(cur_node.key)
        for key in cur_node.down_cover:
            self.get_sons_and_me(key,keys_list)

    def get_fathers_and_me(self,concept_key, keys_list):
        cur_node = self.find_node(concept_key)
        keys_list.append(cur_node.key)
        for key in cur_node.up_cover:
            self.get_fathers_and_me(key,keys_list)


    #容器迭代器
    def __iter__(self):
        pass

    def __len__(self):
        return len(self._container)
    @abstractmethod
    def add_concept(self, concept, fathers, sons):
        pass
    @abstractmethod
    def del_node(self, node_index):
        pass
    @abstractmethod
    def find_node(self, node_index):
        pass
    @abstractmethod
    def replace_concept(self, old_node_index,new_concept):
        pass
    @abstractmethod  #抽象方法
    def print(self):
        pass

class map_lattice(abc_structure):
    def __init__(self,context,container_type=container_type.map_type,
                 key_type=prime_type.att_type):
        super().__init__(context,container_type,key_type)
        self.__key_type = key_type
        self.supre_key=None
        self.infi_key=None
    '''
    @property
    def supre(self):
        node = find(self.supre_key)
        return node
    @supre.setter
    def supre(self,new_key):
        self.supremum_key=new_key

    @property
    def infi(self):
        node = find(self.infimum)
        return node
    @infi.setter
    def infi(self,key):
        self.infimum=key
    '''
    def add_concept(self, concept, fathers, sons):  # todo 差了一个contains方法
        id = len(self._container)
        if self.__key_type==prime_type.att_type:
            key = self._source_context.gen_key_by_intent(concept.intent)
        else:
            key = self._source_context.gen_key_by_extent(concept.extent)
        node=concept_node(concept)
        node.id=id
        node.key=key
        node.up_cover = fathers
        node.down_cover =sons
        self._container.add_element(key,node)
        return key  # 此方法只能根据概念返回key，但是无法确定格中是否已经存在了这个Key


    def del_node(self, node_index):
        deled_node = self._container.del_element(node_index)
        fathers_set=deled_node.up_cover
        if len(fathers_set)>0:
            for key in fathers_set:
                father_node = self.find_node(key)
                father_node.down_cover.remove(node_index)
        sons_set=deled_node.down_cover
        if len(sons_set)>0:
            for key in sons_set:
                son_node = self.find_node(key)
                son_node.up_cover.remove(node_index)
        return deled_node

    def find_node(self, node_index):  # 只能根据index来查找,index指的是key属性
        node=self._container.find_element(node_index)
        return node

    def update_node(self,node_index, node):
        self._container.update_element(node_index,node)
    #2021.1.26
    #zhangzhuo add
    def replace_concept(self, old_node_index,new_concept):
        #prepare the new concept for replace the old node
        #1. delete the old node from lattice forever
        old_node = self._container.del_element(old_node_index)
        #2. copy the old node's id, upcover and downcover
        new_key = self.add_concept(new_concept, old_node.up_cover,old_node.down_cover)
        new_node= self.find_node(new_key)
        new_node.id = old_node.id
        #2. replace the relathionship between in the old node's fathers and sons
        fathers_set = new_node.up_cover
        if len(fathers_set) > 0:
            for key in fathers_set:
                father_node = self.find_node(key)
                father_node.down_cover.remove(old_node_index)
                father_node.down_cover.add(new_key)
        sons_set = new_node.down_cover
        if len(sons_set) > 0:
            for key in sons_set:
                son_node = self.find_node(key)
                son_node.up_cover.remove(old_node_index)
                son_node.down_cover.add(new_key)
        return new_key

    def __iter__(self):
        return iter(self._container)

    def get_index_iter(self):
        return self._container.get_index_iter()  # Bug1 concepts_map 容器并无 get_index_iter方法，yue已修复

    def get_sorted_iter(self):
        return buskets_index_lattice_iter(self)

    def print(self):
        iter1 = iter(self)
        for node in iter1:
            node.print()

##############################################################
#
class buskets_index_lattice(map_lattice):
    def __init__(self,context,key_type):
        super().__init__(context,container_type.buckets_index_type,key_type)

class buskets_index_lattice_iter(abc_iterator):
    def __init__(self,lattice):
        self.__lattice = lattice
        index_iter = lattice.get_index_iter()
        self.__keys_list = list()
        for i,keys in index_iter:
            for key in keys:
                self.__keys_list.append(key)
        self.__counter=0
        self.__key_mode = False # True:return key, otherwise, return node
    @property
    def key_mode(self):
        return self.__key_mode
    @key_mode.setter
    def key_mode(self,new_mode):
        self.__key_mode= new_mode

    def __next__(self):
        if self.__counter<len(self.__keys_list):
            key = self.__keys_list[self.__counter]
            self.__counter = self.__counter +1
            if self.__key_mode:
                return  key
            else:
                return self.__lattice.find_node(key)
        else:
            raise StopIteration
