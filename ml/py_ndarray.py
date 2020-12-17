from functools import reduce
from math import exp, tanh
import random


class ndarray:
    def __init__(self, data=None, shape=None):
        if data is None: data = []
        if shape is None: shape = []
        if isinstance(data, float):
            self.data = [data]
        elif isinstance(data, int):
            self.data = [data]
        elif isinstance(data, list):
            self.data = data
        elif isinstance(data, tuple):
            self.data = list(data)
        else:
            raise Exception("cannot convert to ndarray")
        self.dims = len(shape)
        self.shape = tuple(shape)
        self.size, self.sizes = 1, []
        for i in range(len(self.shape)):
            self.size = self.size * self.shape[i]
        for i in range(len(self.shape)):
            self.sizes.append(self.size // self.shape[i])

    def __float__(self):
        if self.is_scalar(self):
            return self.data[0]

    def __getitem__(self, item):
        if isinstance(item, int):
            start = self.sizes[0] * item
            stop = self.sizes[0] * (item + 1)
            return ndarray(self.data[start:stop], shape=self.shape[1:])
        elif isinstance(item, slice):
            start = self.sizes[0] * item.start
            stop = self.sizes[0] * item.stop
            dim = [item.stop - item.start]
            return ndarray(self.data[start:stop], shape=dim + self.shape[1:])
        elif isinstance(item, tuple):
            if isinstance(item[0], int):
                return self[item[0]][item[1:]]
        elif isinstance(item[0], slice):
            pass

    def absolute_index(self, relative_index):
        absolute_index, size = 0, self.size
        for i in range(len(self.shape)):
            size = size // self.shape[i]
            absolute_index += relative_index[i] * size
        return absolute_index

    def relative_index(self, absolute_index):
        relative_index, size = [], self.size
        for i in range(len(self.shape) - 1):
            size = size // self.shape[i]
            relative_index.append(absolute_index // size)
            absolute_index -= absolute_index // size * size
        relative_index.append(absolute_index % self.shape[-1])
        return relative_index

    @property
    def T(self):
        result = ndarray(shape=list(reversed(self.shape)))
        for i in range(len(self.data)):
            relative_index = result.relative_index(i)
            relative_index = list(reversed(relative_index))
            absolute_index = self.absolute_index(relative_index)
            result.data.append(self.data[absolute_index])
        return result

    def reshape(self, shape):
        return ndarray(self.data, shape=shape)

    @staticmethod
    def is_scalar(array):
        if isinstance(array, ndarray):
            return array.dims == 0
        return True

    @staticmethod
    def is_array(array):
        if isinstance(array, ndarray):
            return array.dims != 0
        return False

    @staticmethod
    def binary_operation(ndarray1, ndarray2, operation):
        if ndarray.is_scalar(ndarray1) and isinstance(ndarray1, ndarray):
            ndarray1 = ndarray1.data[0]
        if ndarray.is_scalar(ndarray2) and isinstance(ndarray2, ndarray):
            ndarray2 = ndarray2.data[0]
        if isinstance(ndarray1, ndarray) and isinstance(ndarray2, ndarray):
            if ndarray1.dims == ndarray2.dims:
                assert ndarray1.shape == ndarray2.shape
                result = ndarray(shape=ndarray1.shape)
                for i in range(len(ndarray1.data)):
                    result.data.append(
                        operation(ndarray1.data[i], ndarray2.data[i]))
                return result
            elif ndarray1.dims > ndarray2.dims:
                result = ndarray(shape=ndarray1.shape)
                for i in range(0, ndarray1.size, ndarray2.size):
                    for j in range(0, ndarray2.size):
                        result.data.append(
                            operation(ndarray1.data[i + j], ndarray2.data[j]))
                return result
            elif ndarray1.dims < ndarray2.dims:
                result = ndarray(shape=ndarray1.shape)
                for i in range(0, ndarray2.size, ndarray1.size):
                    for j in range(0, ndarray1.size):
                        result.data.append(
                            operation(ndarray1.data[j], ndarray2.data[i + j]))
                return result

        elif isinstance(ndarray1, ndarray) and not isinstance(ndarray2, ndarray):
            result = ndarray(shape=ndarray1.shape)
            for i in range(len(ndarray1.data)):
                result.data.append(
                    operation(ndarray1.data[i], ndarray2))
            return result
        elif not isinstance(ndarray1, ndarray) and isinstance(ndarray2, ndarray):
            result = ndarray(shape=ndarray2.shape)
            for i in range(len(ndarray2.data)):
                result.data.append(
                    operation(ndarray1, ndarray2.data[i]))
            return result
        elif not isinstance(ndarray1, ndarray) and not isinstance(ndarray2, ndarray):
            return ndarray(operation(ndarray1, ndarray2))

    @staticmethod
    def unary_operation(array, operation):
        result = ndarray(shape=array.shape)
        for i in range(len(array.data)):
            result.data.append(operation(array.data[i]))
        return result

    def __add__(self, other):
        return self.binary_operation(self, other, lambda x, y: x + y)

    def __sub__(self, other):
        return self.binary_operation(self, other, lambda x, y: x - y)

    def __mul__(self, other):
        return self.binary_operation(self, other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self.binary_operation(self, other, lambda x, y: x / y)

    def __pow__(self, other):
        return self.binary_operation(self, other, lambda x, y: x ** y)

    def __radd__(self, other):
        return self.binary_operation(other, self, lambda x, y: x + y)

    def __rsub__(self, other):
        return self.binary_operation(other, self, lambda x, y: x - y)

    def __rmul__(self, other):
        return self.binary_operation(other, self, lambda x, y: x * y)

    def __rtruediv__(self, other):
        return self.binary_operation(other, self, lambda x, y: x / y)

    def __rpow__(self, other):
        return self.binary_operation(other, self, lambda x, y: x ** y)

    def __iadd__(self, other):
        return self.binary_operation(self, other, lambda x, y: x + y)

    def __isub__(self, other):
        return self.binary_operation(self, other, lambda x, y: x - y)

    def __imul__(self, other):
        return self.binary_operation(self, other, lambda x, y: x * y)

    def __itruediv__(self, other):
        return self.binary_operation(self, other, lambda x, y: x / y)

    def __ipow__(self, other):
        return self.binary_operation(self, other, lambda x, y: x ** y)

    def sigmoid(self):
        return ndarray.unary_operation(self, lambda x: 1 / (1 + exp(-x)))

    def d_sigmoid(self):
        return ndarray.unary_operation(self, lambda x: (1 / (1 + exp(-x))) -
                                                       (1 / (1 + exp(-x))) ** 2)

    def tanh(self):
        return ndarray.unary_operation(self, tanh)

    def d_tanh(self):
        return ndarray.unary_operation(self, lambda x: 1 - tanh(x) ** 2)

    def relu(self):
        def relu(x):
            if x > 0: return x
            return 0

        return ndarray.unary_operation(self, lambda x: relu(x))

    def d_relu(self):
        def relu(x):
            if x > 0:
                return 1
            return 0

        return ndarray.unary_operation(self, lambda x: relu(x))

    def mm(self, other):
        assert self.dims == other.dims == 2
        assert self.shape[1] == other.shape[0]
        result = ndarray(shape=[self.shape[0], other.shape[1]])
        for i in range(self.shape[0]):
            for j in range(other.shape[1]):
                element = 0
                idx1, idx2 = i * self.shape[1], j
                for k in range(other.shape[0]):
                    element += self.data[idx1] * other.data[idx2]
                    idx1, idx2 = idx1 + 1, idx2 + other.shape[1]
                result.data.append(element)
        return result

    @staticmethod
    def init(shape, generator=lambda x=None: None):
        result = ndarray(shape=shape)
        for i in range(result.size):
            result.data.append(generator())
        return result

    @staticmethod
    def random(shape):
        return ndarray.init(shape, random.random)

    @staticmethod
    def uniform(shape, a=-1, b=1):
        def uniform():
            return random.uniform(a, b)

        return ndarray.init(shape, uniform)

    @staticmethod
    def zeros(shape):
        def zero(): return 0

        return ndarray.init(shape, zero)

    @staticmethod
    def reduce(array, function):
        if isinstance(array, ndarray):
            return reduce(function, array.data)
        elif isinstance(array, list):
            return reduce(function, array)
        elif isinstance(array, tuple):
            return reduce(function, array)

    def mean(self):
        return ndarray.sum(self.data) / self.size

    @staticmethod
    def sum(array):
        return ndarray.reduce(array, lambda x, y: x + y)

    @staticmethod
    def product(array):
        return ndarray.reduce(array, lambda x, y: x * y)

    def __str__(self):
        return str(self.data)

    def max(self):
        return max(self.data)

    def min(self):
        return min(self.data)

    def get_data(self):
        return self.data
