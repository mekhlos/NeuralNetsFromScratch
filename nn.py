"""
MLP implementation using PyTorch like architecture
"""

from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

import data_utils


class Sigmoid:
    def __init__(self):
        super().__init__()
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        self.output = 1 / (1 + np.exp(-X))
        return self.output

    def backward(self, d_prev: np.ndarray) -> np.ndarray:
        S = self.output
        J = S * (1 - S)
        return d_prev * J


class Softmax:
    def __init__(self):
        super().__init__()
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.input = X
        exp = np.exp(X)
        self.output = exp / exp.sum(1, keepdims=True)
        return self.output

    def backward(self, d_prev: np.ndarray) -> np.ndarray:
        S = self.output
        J = []
        for d, s in zip(d_prev, S):
            x = np.diagflat(s) - np.einsum('i,j->ij', s, s)
            J.append(d @ x)

        return np.array(J)


class CrossEntropy:
    def __init__(self):
        pass

    def forward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.where(y_true == 1, -np.log(y_pred), 0).mean()

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return np.where(y_true == 1, -1 / y_pred, 0)


class LinearLayer:
    def __init__(self, d_in, d_out, initialiser: Callable):
        super().__init__()
        self.input: Optional[np.ndarray] = None
        self.output: Optional[np.ndarray] = None
        self.W = initialiser(d_out, d_in)
        self.b = initialiser(d_out)
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None

    def forward(self, X) -> np.ndarray:
        self.input = X
        self.output = X.dot(self.W.T) + self.b
        return self.output

    def backward(self, d_prev: np.ndarray) -> np.ndarray:
        self.dW = d_prev.T @ self.input
        self.db = d_prev.sum(0)

        return d_prev @ self.W

    def __repr__(self):
        return f"Layer {self.W.shape}"


class Layer(LinearLayer):
    def __init__(self, d_in, d_out, initialiser: Callable, activation: Union[Sigmoid, Softmax]):
        super().__init__(d_in, d_out, initialiser)
        self.activation = activation

    def forward(self, X) -> np.ndarray:
        self.input = X
        self.output = self.activation.forward(super().forward(X))
        return self.output

    def backward(self, d_prev: np.ndarray) -> np.ndarray:
        d_prev = self.activation.backward(d_prev)
        return super().backward(d_prev)


class SoftmaxLayer(Layer):
    def __init__(self, d_in, d_out, initialiser: Callable):
        super().__init__(d_in, d_out, initialiser, Softmax())


class SigmoidLayer(Layer):
    def __init__(self, d_in, d_out, initialiser: Callable):
        super().__init__(d_in, d_out, initialiser, Sigmoid())


class MLP:
    def __init__(self, dimensions: List[int], initialiser: Callable, learning_rate: float):
        self.learning_rate = learning_rate
        self.layers: List[Layer] = []

        for d_in, d_out in zip(dimensions[:-1], dimensions[1:-1]):
            self.layers.append(SigmoidLayer(d_in, d_out, initialiser))

        self.layers.append(SoftmaxLayer(dimensions[-2], dimensions[-1], initialiser))

    def forward(self, X):
        h = X
        for layer in self.layers:
            h = layer.forward(h)

        return h

    def backward(self, d_values: np.ndarray) -> np.ndarray:
        for layer in self.layers[::-1]:
            d_values = layer.backward(d_values)

        return d_values

    def __repr__(self):
        return f"Network: {' '.join(map(str, self.layers))}"


def backward(loss: CrossEntropy, mlp: MLP, y_true, y_pred):
    d_values = loss.backward(y_true, y_pred)
    d_values = mlp.backward(d_values)

    return d_values


def update_weights(mlp: MLP):
    for layer in mlp.layers:
        layer.b -= mlp.learning_rate * layer.db
        layer.W -= mlp.learning_rate * layer.dW


def train(
    network: MLP,
    loss_function: CrossEntropy,
    train_data_manager: data_utils.DataManager,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    n_epochs: int
) -> None:
    for i in range(n_epochs):
        for x_batch, y_batch in train_data_manager.get_batches():
            y_pred = network.forward(x_batch)
            loss = loss_function.forward(y_batch, y_pred)
            backward(loss_function, network, y_batch, y_pred)
            update_weights(network)

        y_pred = network.forward(x_valid)
        loss_valid = loss_function.forward(y_valid, y_pred)
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

    x = data['data']
    y = data['target']
    y = data_utils.to_one_hot(y)
    x_train, y_train, x_test, y_test, x_valid, y_valid = data_utils.train_test_valid_split(
        data_x=x,
        data_y=y,
        train_ratio=0.7,
        test_ratio=0.2,
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

    cross_entropy = CrossEntropy()
    network = MLP([n_inputs, n_hidden, n_outputs], initialiser=np.random.rand, learning_rate=0.01)

    print("\nTraining network...")
    train(network, cross_entropy, data_manager, x_valid, y_valid, n_epochs=25)

    print("\nTesting trained network...")
    test(network, x_test, y_test)

    print("\nFinal network:")
    print(network)

    print("\nDone.")
