from ml import *
from time import time


tensor1 = Tensor.uniform((1000, 1000))
tensor2 = Tensor.random((1000, 1000))
'''
model = Sequential([
    Dense(100, 100, activation=Sigmoid(), contains_bias=False),
    Dense(100, 100, activation=Sigmoid(), contains_bias=False),
    Dense(100, 100, activation=Sigmoid(), contains_bias=False),
    Dense(100, 100, activation=Sigmoid(), contains_bias=False),
    Dense(100, 100, activation=Sigmoid(), contains_bias=False),
    Dense(100, 100, activation=Sigmoid(), contains_bias=False),
])
'''

loss = MSELoss()
# optim = Optimizer(model.parameters, learning_rate=3)
t1 = time()
t = tensor1[0][0]
'''
for i in range(100):
    l = loss(tensor2, model(tensor1))
    l.backward()
    print(l.mean())
    optim.step()
    optim.learning_rate = i / 10
'''
t2 = time()

print(t2 - t1)