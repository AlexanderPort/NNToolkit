from .tensor import Tensor


class Activation:
    parameters = []


class Sigmoid(Activation):
    def __call__(self, x: Tensor, *args, **kwargs):
        return x.sigmoid()


class Tanh(Activation):
    def __call__(self, x: Tensor, *args, **kwargs):
        return x.tanh()


class ReLU(Activation):
    def __call__(self, x: Tensor, *args, **kwargs):
        return x.relu()


class Softmax(Activation):
    def __init__(self):
        pass
