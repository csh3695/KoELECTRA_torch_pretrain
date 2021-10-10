import numpy as np


class BaseAverager(object):
    data = []

    def get(self):
        raise NotImplementedError

    def __getitem__(self, item):
        return self.data[item]

    def __repr__(self):
        return str(self.get())

    def __len__(self):
        return len(self.data)


class Averager(BaseAverager):
    def __init__(self):
        self.data = []

    def reset(self):
        self.data = []

    def add(self, x, **kwargs):
        self.data.append(x)

    def get(self):
        if len(self.data) != 0:
            return sum(self.data) / len(self.data)
        else:
            return 0


class WeightAverager(BaseAverager):
    def __init__(self):
        self.data = []
        self.weight = []

    def reset(self):
        self.data = []
        self.weight = []

    def add(self, x, weight, **kwargs):
        self.data.append(x)
        self.weight.append(weight)

    def get(self):
        if len(self.data) != 0:
            return (np.array(self.data) * np.array(self.weight)).sum() / np.array(
                self.weight
            ).sum()
        else:
            return 0


class MultiAverager(object):
    def __init__(self, averager):
        self._averagers = dict()
        self.spawner = averager

    def __getattr__(self, item):
        if item not in self._averagers:
            self._averagers[item] = self.spawner()
        return self._averagers[item]

    def reset(self):
        for k in set(self._averagers.keys()):
            self._averagers.pop(k)

    def add(self, x: dict, weight=None):
        for k in x:
            self.__getattr__(k).add(x[k], weight=weight)

    def get(self):
        out = dict()
        for k in self._averagers:
            out[k] = self.__getattr__(k).get()
        return out

    def __iter__(self):
        return iter(self._averagers)
