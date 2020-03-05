# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np


class Neuron(object):

    def __init__(self, num_inputs, activation_fn):
        super().__init__()
        # weight bector and bias value
        self.W = np.random.rand(num_inputs)
        self.b = np.random.rand(1)
        self.activation_fn = activation_fn

    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        return self.activation_fn(z)

class FullyConnectedLayer(object):
    def __init__(self, num_inputs, layer_size, activation_fn):
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn

    def forward(self, x):
        z = np.dot(x, self.W) + self.b
        return self.activation_fn(z)


np.random.seed(42)
x = np.random.rand(3).reshape(1, 3)
step_fn = lambda y: 0 if y <= 0 else 1
perceptron = Neuron(num_inputs=x.size, activation_fn=step_fn)
out = perceptron.forward(x)
print(out)

# testing of dense network

np.random.seed(42)
x1 = np.random.uniform(-1, 1, 2).reshape(1, 2)

x2 = np.random.uniform(-1, 1, 2).reshape(1, 2)
relu_fn = lambda y: np.maximum(y, 0)

# constructs a single layer
layer = FullyConnectedLayer(2, 3, relu_fn)
out1 = layer.forward(x1)
print(out1)
out2 = layer.forward(x2)
print(out2)

