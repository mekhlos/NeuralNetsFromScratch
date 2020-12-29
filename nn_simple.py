"""
MLP implementation using for loops
"""

from typing import Callable
from typing import List
from typing import Optional

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

import data_utils


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


# (n_data * n_features) -> ((n_data * n_features) * (n_data * n_features))
def sigmoid_prime(X: np.ndarray) -> np.ndarray:
    s = sigmoid(X)
    return s * (1 - s)


# n_data * n_features -> n_data * n_features
def softmax(X: np.ndarray) -> np.ndarray:
    return np.exp(X) / np.exp(X).sum(1, keepdims=True)


# (n_data * n_features) -> ((n_data * n_features) * (n_data * n_features))
def softmax_derivative(X: np.ndarray) -> np.ndarray:
    J = []
    for x in X:
        S = softmax(x.reshape((1, -1))).T
        J.append(np.diagflat(S) - np.dot(S, S.T))

    return np.array(J)


# n_data * n_out -> 1
def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.where(y_true == 1, -np.log(y_pred), 0).mean()


# n_data * n_out -> n_data * n_out
def cross_entropy_prime(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.where(y_true == 1, -1 / y_pred, 0)


class Layer:
    def __init__(self, d_in, d_out, initialiser: Callable, activation: Callable):
        self.d_in = d_in
        self.d_out = d_out
        self.b = initialiser(d_out)
        self.W = initialiser(d_in, d_out)

        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.pre_activation: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

        self.compute_activation = activation

    def forward(self, X) -> np.ndarray:
        self.input = X
        self.pre_activation = self.input.dot(self.W) + self.b
        self.output = self.compute_activation(self.pre_activation)
        return self.output

    def __repr__(self):
        return f"Layer {self.d_in}x{self.d_out}: {self.W} {self.b}"


class MLP:
    def __init__(self, dimensions: List[int], initialiser: Callable, learning_rate: float):
        self.learning_rate = learning_rate
        self.layers: List[Layer] = []

        for d_in, d_out in zip(dimensions[:-1], dimensions[1:-1]):
            self.layers.append(Layer(d_in, d_out, initialiser, sigmoid))

        self.layers.append(Layer(dimensions[-2], dimensions[-1], initialiser, softmax))

    def loss(self, X, y):
        y_pred = self.forward(X)

        return cross_entropy(y, y_pred)

    def update(self):
        pass

    def forward(self, X):
        h = X
        for layer in self.layers:
            h = layer.forward(h)

        return h

    def backward(self, y_pred: np.ndarray, y: np.ndarray):
        dloss_do = cross_entropy_prime(y, y_pred)

        for layer_ix, layer in enumerate(self.layers[::-1]):
            dz_dW = layer.input
            dz_db = np.ones(len(layer.input))
            dz_dx = layer.W

            if layer_ix == 0:
                do_dz = softmax_derivative(layer.pre_activation)
                dloss_dz = np.array([a.dot(b) for a, b in zip(dloss_do, do_dz)])
            else:
                do_dz = sigmoid_prime(layer.pre_activation)
                dloss_dz = dloss_do * do_dz

            dloss_dW = dloss_dz.T.dot(dz_dW) / len(y_pred)
            dloss_db = dloss_dz.T.dot(dz_db) / len(y_pred)
            dloss_dX = dloss_dz.dot(dz_dx.T)

            layer.dW = dloss_dW.T
            layer.db = dloss_db
            dloss_do = dloss_dX

        for layer in self.layers:
            layer.W -= self.learning_rate * layer.dW
            layer.b -= self.learning_rate * layer.db

    def fit(self, X, y):
        pass

    def __repr__(self):
        return f"Network: {' '.join(map(str, self.layers))}"


def train(
    network: MLP,
    train_data_manager: data_utils.DataManager,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    n_epochs: int
) -> None:
    for i in range(n_epochs):
        for x_batch, y_batch in train_data_manager.get_batches():
            y_pred = network.forward(x_batch)
            network.backward(y_pred, y_batch)

        y_pred = network.forward(x_valid)
        loss_valid = cross_entropy(y_valid, y_pred)
        acc = (y_valid.argmax(1) == y_pred.argmax(1)).sum() / len(y_pred)
        print(f"loss: {loss_valid}, acc: {acc}")


def test(network: MLP, x_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred = network.forward(x_test)
    acc = (y_test.argmax(1) == y_pred.argmax(1)).sum() / len(y_pred)
    print(f"acc: {acc}")


if __name__ == '__main__':
    print("Testing network on scikit-learn wine dataset...")

    np.random.seed(1)
    scaler = StandardScaler()

    print("\nLoading data...")
    data = load_wine()

    print("\nPreprocessing data...")
    x = data['data']
    y = data['target']
    y = data_utils.to_one_hot(y)
    x_train, y_train, x_test, y_test, x_valid, y_valid = data_utils.train_test_valid_split(
        x,
        y,
        0.7,
        0.2,
        should_shuffle=True
    )

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_valid = scaler.transform(x_valid)

    dataset = data_utils.Dataset(x_train, y_train)
    data_manager = data_utils.DataManager(dataset, batch_size=8)

    print("\nBuilding network...")
    n_inputs = len(x[0])
    n_hidden = 2
    n_outputs = 3

    network = MLP([n_inputs, n_hidden, n_outputs], initialiser=np.random.rand, learning_rate=0.01)

    print("\nTraining network...")
    train(network, data_manager, x_valid, y_valid, n_epochs=250)

    print("\nTesting trained network...")
    test(network, x_test, y_test)

    print("\nFinal network:")
    print(network)

    print("\nDone.")
