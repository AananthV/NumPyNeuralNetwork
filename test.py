import numpy as np
import matplotlib.pyplot as plt
from neuralnetwork import *

relu_activation = {
    'function': lambda x: np.maximum(0, x),
    'gradient': lambda x: x > 0
}

linear_activation = {
    'function': lambda x: x,
    'gradient': lambda x: 1
}

[X_train, Y_train] = np.hsplit(np.loadtxt(open('train.csv', 'rb'), delimiter=','), 2)
[X_test, Y_test] = np.hsplit(np.loadtxt(open('test.csv', 'rb'), delimiter=','), 2)

layers = []

layers.append({
    'inputs': 1,
    'outputs': 1,
    'activation': linear_activation,
    'learning_rate': 0.001
})

nn = NeuralNetwork(layers)

nn.train(X_train, Y_train)

predictions = nn.test(X_test, Y_test)

figure, axes = plt.subplots(1, 2)

axes[0].set_xlabel('X Values')
axes[0].set_ylabel('Y Values')
axes[0].scatter(X_test, Y_test, label = 'Original')
axes[0].scatter(X_test, predictions, label = 'Predicted')
axes[0].grid()

axes[1].set_xlabel('X Index')
axes[1].scatter(np.array([i + 1 for i in range(len(X_test))]), Y_test, label = 'Original')
axes[1].scatter(np.array([i + 1 for i in range(len(X_test))]), predictions, label = 'Predicted')
axes[1].grid()

plt.show()
