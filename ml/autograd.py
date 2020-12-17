from .ndarray import ndarray
from copy import deepcopy


class Derivative:
    @classmethod
    def derivative(cls, id1, id2, function, gradient):
        def derivative():
            return function(GRAPH.variables[id1].data,
                            GRAPH.variables[id2].data)
        return Derivative(derivative, gradient)

    @classmethod
    def derivatives(cls, derivatives):
        def derivative():
            derivative_ = derivatives[0]()
            for i in range(1, len(derivatives)):
                derivative_ = derivatives[i].gradient(
                    derivative_, derivatives[i]())
            return derivative_
        return Derivative(derivative, gradient=None)

    def __init__(self, derivative, gradient):
        self.derivative = derivative
        self.gradient = gradient

    def __call__(self, *args, **kwargs):
        return self.derivative()


class ComputationalGraph:
    def __init__(self):
        self.parents = {}
        self.variables = {}
        self.derivatives = {}
        self.dependencies = {}

    def derivative(self, id1, id2):
        derivatives = []

        def derivative(id1_, id2_):
            if id2_ in self.derivatives[id1_].keys():
                derivatives.append(self.derivatives[id1_][id2_])
            elif id2_ not in self.derivatives[id1_].keys():
                for id3_ in self.parents[id1_]:
                    if id2_ in self.dependencies[id3_]:
                        derivatives.append(self.derivatives[id1_][id3_])
                        derivative(id3_, id2_)

        derivative(id1, id2)
        derivative = Derivative.derivatives(derivatives)
        self.derivatives[id1][id2] = derivative
        return derivative


BACKWARD = [None]
GRAPH = ComputationalGraph()


def getTensorFromId(id):
    return GRAPH.variables[id]
