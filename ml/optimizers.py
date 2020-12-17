class Optimizer:
    def __init__(self, parameters, learning_rate=1):
        self.gradient = None
        self.parameters = parameters
        self.learning_rate = learning_rate

    def step(self):
        for parameter in self.parameters:
            parameter.data -= parameter.grad() * \
                              self.learning_rate

    def zero_grad(self):
        self.gradient = 0


class Momentum:
    def __init__(self, parameters, learning_rate=1, alpha=0.99):
        self.parameters = parameters
        self.gradients = [0] * len(parameters)
        self.learning_rate = learning_rate
        self.alpha = alpha

    def get_grad(self, gradient, i):
        self.gradients[i] = self.alpha * gradient + \
                            (1 - self.alpha) * self.gradients[i]
        return self.gradients[i] * self.learning_rate

    def step(self):
        for i in range(len(self.parameters)):
            self.parameters[i].data -= \
                self.get_grad(self.parameters[i].grad(), i)

    def zero_grad(self):
        self.gradients = [0] * len(self.parameters)


class RMSprop(Optimizer):
    def __init__(self, parameters):
        super().__init__(parameters)


class Adam(Optimizer):
    def __init__(self, parameters):
        super().__init__(parameters)
