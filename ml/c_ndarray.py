from functools import reduce
from math import exp, tanh
from cndarray import cndarray
import random


class ndarray:
    def __init__(self, data=None, shape=None):
        if data is None: data = []
        if shape is None: shape = []
        if isinstance(data, cndarray):
            self.data = data
            self.size = data.get_size()
            self.dims = data.get_dims()
            self.shape = data.get_shape()
        else:
            if isinstance(data, float):
                self.data = cndarray([data], [])
            elif isinstance(data, int):
                self.data = cndarray([data], [])
            elif isinstance(data, list):
                if shape is None: shape = [len(data)]
                self.data = cndarray(list(data), list(shape))
            elif isinstance(data, tuple):
                if shape is None: shape = [len(data)]
                self.data = cndarray(list(data), list(shape))
            else:
                raise Exception("cannot convert to ndarray")
            self.shape = shape
            self.dims = len(shape)
            self.size = self.data.get_size()

    def __float__(self):
        if self.dims == 0:
            return self.data.get_data()[0]
        else:
            raise Exception("cannot convert to float")

    def __getitem__(self, item):
        return ndarray(self.data.__getitem__(item))

    def __add__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__add__(other.data))

    def __sub__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__sub__(other.data))

    def __mul__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__mul__(other.data))

    def __truediv__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__truediv__(other.data))

    def __pow__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__pow__(other.data))

    def __radd__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(other.data.__add__(self.data))

    def __rsub__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(other.data.__sub__(self.data))

    def __rmul__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(other.data.__mul__(self.data))

    def __rtruediv__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(other.data.__truediv__(self.data))

    def __rpow__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(other.data.__pow__(self.data))

    def __iadd__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__add__(other.data))

    def __isub__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__sub__(other.data))

    def __imul__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__mul__(other.data))

    def __itruediv__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__truediv__(other.data))

    def __ipow__(self, other):
        if not isinstance(other, ndarray):
            other = ndarray(cndarray([other], []))
        return ndarray(self.data.__pow__(other.data))

    def __matmul__(self, other):
        return ndarray(self.data.__matmul__(other.data))

    def __str__(self):
        return str(self.data.get_data())

    def mm(self, other):
        return ndarray(self.data.__matmul__(other.data))

    @property
    def T(self):
        return ndarray(self.data.transpose())

    @staticmethod
    def init(shape, generator=lambda x=None: None):
        size = 1
        for i in range(len(shape)):
            size *= shape[i]
        data = []
        for i in range(size):
            data.append(generator())
        return ndarray(data, list(shape))

    @staticmethod
    def random(shape):
        return ndarray.init(shape, random.random)

    @staticmethod
    def uniform(shape, a=-1, b=1):
        def uniform():
            return random.uniform(a, b)

        return ndarray.init(shape, uniform)

    def sigmoid(self):
        return ndarray(self.data.sigmoid())

    def d_sigmoid(self):
        return ndarray(self.data.d_sigmoid())

    def tanh(self):
        return ndarray(self.data.tanh())

    def d_tanh(self):
        return ndarray(self.data.d_tanh())

    def relu(self):
        return ndarray(self.data.relu())

    def d_relu(self):
        return ndarray(self.data.d_relu())

    def mean(self):
        return self.data.mean()

    def get_data(self):
        return self.data.get_data()

    def max(self):
        return self.data.max()

    def min(self):
        return self.data.min()
