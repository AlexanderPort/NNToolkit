from .tensor import Tensor


LAYERS = {}


class Layer:
    def __init__(self, ID=None):
        if ID is None: self.id = id(self)
        else: self.id = ID
        LAYERS[self.id] = self
        self.parameters = []

    def __call__(self, *args, **kwargs):
        return None


class Dense(Layer):
    @classmethod
    def init(cls, weights, bias=None, activation=lambda x: x, model=None, ID=None):
        dense = Dense(1, 1, model=model, ID=ID)
        dense.weights = weights
        if bias is not None:
            dense.bias = bias
            dense.contains_bias = True
        else:
            dense.contains_bias = False
        dense.activation = activation
        return dense

    def __init__(self, input_dim, output_dim, activation=lambda x: x,
                 contains_bias=True, model=None, trainable=False, ID=None):
        super(Dense, self).__init__(ID)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.contains_bias = contains_bias
        if model: self.parameters = model.parameters
        self.weights = Tensor.uniform((input_dim, output_dim))
        self.parameters.append(self.weights)
        if self.contains_bias:
            self.bias = Tensor.uniform((output_dim,))
            self.parameters.append(self.bias)
        else: self.bias = None

    def __call__(self, x: Tensor):
        if self.contains_bias:
            return self.activation(x.mm(self.weights) + self.bias)
        else:
            return self.activation(x.mm(self.weights))


class RNNCell(Layer):
    def __init__(self):
        super(RNNCell, self).__init__()


def getLayerFromId(id: int) -> Layer:
    return LAYERS[id]