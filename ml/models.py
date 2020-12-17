from .autograd import getTensorFromId
from .layers import *
from .activations import *


class Model:
    def __init__(self, layers=None):
        if layers is None: layers = []
        self.layers = layers
        self.parameters = []
        for layer in self.layers:
            self.parameters.extend(layer.parameters)

    def addLayer(self, layer):
        self.layers.append(layer)
        self.parameters.extend(layer.parameters)

    def __call__(self, *args, **kwargs):
        pass

    def save(self, fname):
        with open(fname, mode='w', encoding='utf-8') as file:
            for parameter in self.parameters:
                tensor = f'Tensor(ID={parameter.id}, data={parameter.get_data()}, shape={parameter.shape})\n'
                file.write(tensor)
            layers = []
            for layer in self.layers:
                if isinstance(layer, Dense):
                    if layer.contains_bias:
                        bias = f'getTensorFromId({layer.bias.id})'
                    else: bias = 'None'
                    layer = f'Dense.init(weights=getTensorFromId({layer.weights.id}), ' \
                            f'bias={bias}, activation={layer.activation.__class__.__name__}()) '
                if isinstance(layer, Sigmoid):
                    layer = 'Sigmoid()'
                layers.append(layer)
            model = f'{self.__class__.__name__}(layers={layers})'
            model = model.replace("'", '').replace('"', '')
            file.write(model)
        file.close()

    @staticmethod
    def load(fname):
        with open(fname, mode='r', encoding='utf-8') as file:
            lines = file.readlines()
            for i in range(len(lines) - 1): eval(lines[i])
            return eval(lines[-1])

    def copy(self):
        return self.__class__(layers=self.layers[:])


class Sequential(Model):
    def __init__(self, layers):
        super().__init__(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

