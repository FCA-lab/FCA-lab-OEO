########################################
# Title: data module
#
# Version:  1.0
#
# Last-Modified: 2020-6-8
#
# Author: iezhangzhuo@zzu.edu.cn (ZhangZhuo)
#
# Status: Active
#
# Type: python file
#
# Created: 2020-6-4
##########################################
from abc import ABCMeta, abstractmethod
import random


class abc_data(metaclass=ABCMeta):

    @abstractmethod
    def get_dataset(self):
        pass


class data_set(abc_data):
    def __init__(self, dataset):
        self.__dataset = dataset

    def get_dataset(self):
        return self.__dataset.copy()


class data_file(abc_data):
    def __init__(self):
        pass

    def get_dataset(self):
        pass

    def load_from_file(self, file):
        pass


class data_database(abc_data):
    #
    def load_from_dataset(self, dataset):
        pass


class MapData(abc_data):
    def __init__(self, w, h, proportion, seed=10):
        self.high = h
        self.width = w
        self.propor = proportion
        self.seed = seed
        self._data = RandomMap(w, h, proportion, randseed=seed)

    def get_dataset(self):
        return self._data

    def get_description(self):
        return f"宽高:{self.width}*{self.high},0占比{self.propor},随机种子:{self.seed}"


def RandomMap(w, h, propor, randseed=10):
    map = [[1 for i in range(w)] for j in range(h)]
    num = int(w * h * propor)
    random.seed(randseed)
    for i in range(num):
        x = random.randrange(h)
        y = random.randrange(w)
        map[x][y] = 0
    return map


if __name__ == '__main__':
    ls = MapData(10, 10, 0.24).data
    print(ls)
