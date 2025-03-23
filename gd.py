#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Model:
    def __init__(self, layers=[], learning_rate: float=0.01):
        """A multi-layered model for classification and regression. Batch size of 1.

        Keyword arguments:
        layers -- A list of hidden and one output layer
        learning_rate -- Determines how fast the model learns
        """
        self.layers = layers
        self.learning_rate = learning_rate

    def fit(self, X, Y, max_iterations=100, tolerance=1e-4, callbacks=[]):
        """Fit the model to provided data.

        Keyword arguments:
        X -- The input data
        Y -- The target labels
        max_iterations -- The max number of training iterations (default 100)
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
        """Make a prediction to the provided input.
        Return prediction in range [0, n_classes-1]

        Keyword arguments:
        x -- The input
        """
        activations = x
        for layer in self.layers:
            activations = layer._forward(activations)
        return activations

    def push(layer):
        layers.append(Layer)

    def pop():
        if not layers:
            return None
        return layers.pop()

class Layer:
    def __init__(self, input_size:int , output_size: int, activation="sigmoid"):
        """A model layer. The loss function is mean squared error.

        Keyword arguments:
        input_size -- Must match the size of the output from the previous layer
        output_size -- Must match the size of the input to the next layer
        activation -- The activation function (default "sigmoid")
        """
        # Xavier Weight Initialization (sigmoid)
        u = 1 / math.sqrt(input_size)
        self._weights = np.random.uniform(-u, u, (output_size, input_size))
        self._bias = np.zeros(output_size)
        # Activation function
        self._activation, self._activation_derivative = self._get_activation(activation)

    def _forward(self, x):
        # Pre-activation 'z' and activation 'a'
        self._z = np.dot(self._weights, x) + self._bias
        self._a = self._activation()
        # Forward propagate the activation
        return self._a

    def _backward(self, x, error, learning_rate):
        # Get the gradient 'delta'
        delta = error * self._activation_derivative() # Hadamard product
        # Update weights and bias
        self._weights -= learning_rate * np.outer(delta, x) # Multiply by x transpose
        self._bias -= learning_rate * delta
        # Back propagate the error term
        return self._weights.T @ delta

    def _get_activation(self, activation):
        match activation:
            case "linear":
                return self._linear, self._linear_derivative
            case "sigmoid":
                return self._sigmoid, self._sigmoid_derivative
            case "relu":
                return self._relu, self._relu_derivative
            case "tanh":
                return self._tanh, self._tanh_derivative

    def _linear(self):
        return self._z

    def _linear_derivative(self):
        return np.ones_like(self._z)

    def _sigmoid(self):
        return 1 / (1 + np.exp(-self._z))

    def _sigmoid_derivative(self):
        return self._a * (1 - self._a)

    def _relu(self):
        return np.maximum(0, self._z)

    def _relu_derivative(self):
        return (self._a > 0) * 1

    def _tanh(self):
        return np.tanh(self._z)

    def _tanh_derivative(self):
        return 1 - self._a**2

    def _softmax(self):
        e_z = np.exp(self._z - np.max(self._z))
        return e_z / e_z.sum(axis=0)

def one_hot(a, n_classes):
    """One hot encode target labels."""
    b = np.zeros((a.size, n_classes))
    b[np.arange(a.size), a] = 1
    return b

def classification():
    # log accuracy after each iteration
    log = []
    def logger(model, iter):
        num_correct = 0
        count = 0
        for (sample, label) in zip(X_test, y_test):
            prediction = np.argmax(model.predict(sample))
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
        Layer(4, 3, activation="sigmoid")
    ])
    logger(iris_model, 0) # initial accuracy
    iris_model.fit(X_train, Y_train, max_iterations=50, callbacks=[logger])

    # plot accuracy
    fig, ax = plt.subplots()
    ax.plot(log, linewidth=5.0)
    ax.set_xlabel('iteration', size=20)
    ax.set_ylabel('accuracy', size=20)

def linear_regression():
    # Load iris dataset
    iris = load_iris()
    y = iris.target
    sepal_length = iris.data[:, 0].reshape(-1, 1)
    petal_length = iris.data[:, 2].reshape(-1, 1)

    iris_model = Model([
        Layer(1, 1, activation="linear")
    ])
    iris_model.fit(sepal_length, petal_length)

    # Generate a range of sepal lengths to plot
    x_range = np.linspace(sepal_length.min(), sepal_length.max(), 100).reshape(-1, 1)
    # Get petal length predictions from the model
    y_pred = np.array([iris_model.predict(x) for x in x_range])

    # plot sepal length by petal length
    fig, ax = plt.subplots()
    ax.scatter(sepal_length, petal_length, c=y, linewidths=5.0)
    ax.plot(x_range, y_pred, c="red", linewidth=5.0)
    ax.set_xlabel('sepal length', size=20)
    ax.set_ylabel('petal length', size=20)

def decision_boundaries():
    # !!! Method to plot the boundary is taken from here:
    # https://dnmtechs.com/plotting-decision-boundary-with-matplotlibs-pyplot-in-python-3/

    # Create mock data
    X, y = make_blobs(n_samples=200, centers=2, random_state=4)

    # Fit the model
    model = Model([
        Layer(2, 1, activation="sigmoid")  # 2 inputs, 1 output (binary classification)
    ])
    model.fit(X, y.reshape(-1, 1), max_iterations=100)

    # Define the range of the plot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Flatten grid and predict for each point
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = np.array([model.predict(point) for point in grid_points])
    predictions = predictions.reshape(xx.shape)

    # Plot decision boundary
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, predictions, alpha=0.5, cmap=plt.cm.coolwarm)

    ax.scatter(X[:, 0], X[:, 1], c=y, linewidths=5.0)
    ax.set_xlabel('feature 1', size=20)
    ax.set_ylabel('feature 2', size=20)

if __name__ == "__main__":
    classification()
    linear_regression()
    decision_boundaries()
    plt.show()

