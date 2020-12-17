from .ndarray import ndarray
from .autograd import GRAPH, BACKWARD, Derivative
import math
import random


class Tensor:
    def __init__(self, data, shape=None, ID=None):
        if isinstance(data, ndarray):
            self.data = data
        elif isinstance(data, int):
            self.data = ndarray(data)
        elif isinstance(data, float):
            self.data = ndarray(data)
        elif isinstance(data, list):
            if shape is None: shape = [len(data)]
            self.data = ndarray(data, shape)
        elif isinstance(data, tuple):
            if shape is None: shape = [len(data)]
            self.data = ndarray(data, shape)
        else:
            self.data = ndarray()

        if ID is None: self.id = id(self)
        else: self.id = ID
        self.parents = []
        self.dependencies = set()
        self.dims = self.data.dims
        self.shape = self.data.shape
        GRAPH.variables[self.id] = self
        GRAPH.derivatives[self.id] = {}
        GRAPH.dependencies[self.id] = []

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def __float__(self):
        if self.data.is_scalar(self.data):
            return self.data.data[0]
        else:
            raise Exception("cannot convert Tensor to float")

    @staticmethod
    def binary_operation(parent1, parent2, operation, derivative1, derivative2,
                         gradient1=lambda x, y: x * y, gradient2=lambda x, y: x * y):
        if not isinstance(parent1, Tensor): parent1 = Tensor(parent1)
        if not isinstance(parent2, Tensor): parent2 = Tensor(parent2)
        result = Tensor(operation(parent1.data, parent2.data))
        result.dependencies.update({parent1.id, parent2.id})
        result.dependencies.update(parent1.dependencies)
        result.dependencies.update(parent2.dependencies)
        GRAPH.parents[result.id] = [parent1.id, parent2.id]
        GRAPH.dependencies[result.id] = result.dependencies
        GRAPH.derivatives[result.id][parent1.id] = Derivative.derivative(
            parent1.id, parent2.id, derivative1, gradient1)
        GRAPH.derivatives[result.id][parent2.id] = Derivative.derivative(
            parent1.id, parent2.id, derivative2, gradient2)
        return result

    @staticmethod
    def unary_operation(parent, operation, derivative,
                        gradient=lambda x, y: x * y):
        if not isinstance(parent, Tensor): parent = Tensor(parent)
        result = Tensor(operation(parent.data))
        result.dependencies.add(parent.id)
        result.dependencies.update(parent.dependencies)
        GRAPH.parents[result.id] = [parent.id]
        GRAPH.dependencies[result.id] = result.dependencies
        GRAPH.derivatives[result.id][parent.id] = Derivative.derivative(
            parent.id, parent.id, derivative, gradient)
        return result

    def backward(self):
        BACKWARD[0] = self

    def grad(self):
        return GRAPH.derivative(BACKWARD[0].id, self.id)()

    def __add__(self, other):
        return self.binary_operation(self, other, lambda x, y: x + y,
                                     lambda x, y: 1, lambda x, y: 1)

    def __sub__(self, other):
        return self.binary_operation(self, other, lambda x, y: x - y,
                                     lambda x, y: 1, lambda x, y: -1)

    def __mul__(self, other):
        return self.binary_operation(self, other, lambda x, y: x * y,
                                     lambda x, y: y, lambda x, y: x)

    def __truediv__(self, other):
        return self.binary_operation(self, other, lambda x, y: x / y,
                                     lambda x, y: 1 / y, lambda x, y: -1 * x / y ** 2)

    def __pow__(self, other):
        return self.binary_operation(self, other, lambda x, y: x ** y,
                                     lambda x, y: y * x ** (y - 1),
                                     lambda x, y: x ** y * math.log(x))

    def __radd__(self, other):
        return self.binary_operation(self, other, lambda x, y: x + y,
                                     lambda x, y: 1, lambda x, y: 1)

    def __rsub__(self, other):
        return self.binary_operation(self, other, lambda x, y: x - y,
                                     lambda x, y: -1, lambda x, y: 1)

    def __rmul__(self, other):
        return self.binary_operation(self, other, lambda x, y: x * y,
                                     lambda x, y: x, lambda x, y: y)

    def __rtruediv__(self, other):
        return self.binary_operation(self, other, lambda x, y: x / y,
                                     lambda x, y: -x / y ** 2, lambda x, y: 1 / y)

    def __rpow__(self, other):
        return self.binary_operation(self, other, lambda x, y: x ** y,
                                     lambda x, y: x ** y * math.log(x),
                                     lambda x, y: y * x ** (y - 1))

    def __iadd__(self, other):
        return self.binary_operation(self, other, lambda x, y: x + y,
                                     lambda x, y: 1, lambda x, y: 1)

    def __isub__(self, other):
        return self.binary_operation(self, other, lambda x, y: x - y,
                                     lambda x, y: 1, lambda x, y: -1)

    def __imul__(self, other):
        return self.binary_operation(self, other, lambda x, y: x * y,
                                     lambda x, y: y, lambda x, y: x)

    def __itruediv__(self, other):
        return self.binary_operation(self, other, lambda x, y: x / y,
                                     lambda x, y: 1 / y, lambda x, y: -x / y ** 2)

    def __ipow__(self, other):
        return self.binary_operation(self, other, lambda x, y: x ** y,
                                     lambda x, y: y * x ** (y - 1),
                                     lambda x, y: x ** y * math.log(x))

    def mm(self, other):
        return self.binary_operation(self, other, lambda x, y: x.mm(y),
                                     lambda x, y: y, lambda x, y: x,
                                     lambda x, y: x.mm(y.T), lambda x, y: y.T.mm(x))

    def sigmoid(self):
        return self.unary_operation(self, lambda x: x.sigmoid(),
                                    lambda x, y: x.d_sigmoid())

    def tanh(self):
        return self.unary_operation(self, lambda x: x.tanh(),
                                    lambda x, y: x.d_tanh())

    def relu(self):
        return self.unary_operation(self, lambda x: x.relu(),
                                    lambda x, y: x.d_relu())

    @staticmethod
    def random(shape):
        return Tensor(ndarray.random(shape))

    @staticmethod
    def uniform(shape):
        return Tensor(ndarray.uniform(shape))

    def mean(self):
        return self.data.mean()

    def __str__(self):
        return str(self.data)

    def max(self):
        return self.data.max()

    def min(self):
        return self.data.min()

    def save(self, fname):
        with open(fname, mode='w', encoding='utf-8') as file:
            data = f"Tensor(ID={self.id}, shape={self.shape}, " \
                   f"data={self.data.data.get_data()})"
            file.write(data)
        file.close()

    @staticmethod
    def load(fname):
        with open(fname, mode='r', encoding='utf-8') as file:
            return eval(file.read())

    def reshape(self, shape):
        return Tensor(ndarray(self.data.get_data(), shape))

    def get_data(self):
        return self.data.get_data()

    @property
    def T(self):
        return Tensor(self.data.T)