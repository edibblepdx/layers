#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, layers=[], learning_rate: float=0.01):
        self.layers = layers
        self.learning_rate = learning_rate

    def fit(self, X, Y, max_iterations=100, tolerance=1e-4, callbacks=[]):
        """Fit the model to provided data.

        Keyword arguments:
        X -- The input data
        Y -- The target labels
        max_iterations -- The max number of training iterations (default 50)
        tolerance -- The tolerance at which to stop training based on the magnitude of the gradient (default 1e-4)
        """
        # Gradient Descent and Back Propagation
        for i in range(max_iterations):
            for sample, label in zip(X, Y):
                # Forward Phase
                activations_list = [sample]
                for layer in self.layers:
                    activations_list.append(layer._forward(activations_list[-1]))

                # Backward Phase
                error = activations_list[-1] - label
                for j, layer in enumerate(reversed(self.layers)):
                    error = layer._backward(activations_list[-(j+2)], error, self.learning_rate)

            # Callbacks
            for callback in callbacks:
                callback(self, i)

    def predict(self, x):
        """Return value in range [0, n_classes-1]"""
        activations = x
        for layer in self.layers:
            activations = layer._forward(activations)
        return np.argmax(activations)

class Layer:
    def __init__(self, input_size:int , output_size: int, activation="sigmoid"):
        self.input_size = input_size
        self.output_size = output_size
        # Xavier Weight Initialization (sigmoid)
        u = 1 / math.sqrt(input_size)
        self.weights = np.random.uniform(-u, u, (output_size, input_size))
        self.bias = np.zeros(output_size)

    def _activation(self, z):
        # Sigmoid function
        return 1 / (1 + np.exp(-z))

    def _forward(self, x):
        # Pre-activation 'z' and activation 'a'
        z = np.dot(self.weights, x) + self.bias
        self._a = self._activation(z)
        # Forward propagate the activation
        return self._a

    def _backward(self, x, error, learning_rate):
        # Get the gradient 'delta'
        delta = error * self._a * (1 - self._a) # Hadamard product
        # Update weights and bias
        self.weights -= learning_rate * np.outer(delta, x) # Multiply by x transpose
        self.bias -= learning_rate * delta
        # Back propagate the error term
        return self.weights.T @ delta

def one_hot(a, n_classes):
    """One hot encode target labels."""
    b = np.zeros((a.size, n_classes))
    b[np.arange(a.size), a] = 1
    return b

if __name__ == "__main__":
    # log accuracy after each iteration
    log = []
    def logger(model, iter):
        num_correct = 0
        count = 0
        for (sample, label) in zip(X_test, y_test):
            prediction = model.predict(sample)
            count += 1
            if prediction == label:
                num_correct += 1

        if count != 0:
            log.append(num_correct / count)

    # load iris dataset (75-25 split)
    X_train, X_test, y_train, y_test = train_test_split(*load_iris(return_X_y=True), random_state=4)
    Y_train, Y_test = one_hot(y_train, 3), one_hot(y_test, 3)

    # Put model together and fit it to the data
    iris_model = Model([
        #Layer(4, 10),  # hidden layer (4 features, 10 neurons)
        #Layer(10, 3),  # output layer (10 inputs, 3 classifications)
        Layer(4, 3)
    ])
    logger(iris_model, 0) # initial accuracy
    iris_model.fit(X_train, Y_train, max_iterations=50, callbacks=[logger])

    # plot accuracy
    fig, ax = plt.subplots()
    ax.plot(log)
    ax.set_xlabel('iteration')
    ax.set_ylabel('accuracy')
    plt.show()

