# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np
import mnist


class Neuron(object):

    def __init__(self, num_inputs, activation_fn):
        super().__init__()
        # weight vector and bias value
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


def sigmoid(value):
    return 1 / (1 + np.exp(-value))


class SimpleNetwork(object):

    def __init__(self, num_inputs, num_outputs, hidden_layers):
        super().__init__()
        sizes = [num_inputs, *hidden_layers, num_outputs]
        self.layers = [
            FullyConnectedLayer(sizes[i], sizes[i + 1], sigmoid)
            for i in range(len(sizes) - 1)]

    def forward(self, x_in):
        for current_layer in self.layers:
            x_in = current_layer.forward(x_in)
        return x_in

    def predict(self, x_in):
        estimations = self.forward(x_in)
        best_class = np.argmax(estimations)
        return best_class

    def evaluate_accuracy(self, X_val, y_val):
        num_corrects = 0
        for i in range(len(X_val)):
            if self.predict(X_val[i]) == y_val[i]:
                num_corrects += 1
        return num_corrects / len(X_val)


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

np.random.seed(42)
# np.random.seed(54)
num_classes = 10
X_train, y_train = mnist.train_images(), mnist.train_labels()
X_test, y_test = mnist.test_images(), mnist.test_labels()

X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
y_train = np.eye(num_classes)[y_train]

mnist_classifier = SimpleNetwork(X_train.shape[1], num_classes, [64, 32])
accuracy = mnist_classifier.evaluate_accuracy(X_test, y_test)
print("accuracy = {:.2f}%".format(accuracy * 100))
