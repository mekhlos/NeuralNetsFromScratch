from typing import Tuple

import numpy as np


def train_test_valid_split(data_x, data_y, train_ratio, test_ratio, should_shuffle: bool) -> Tuple:
    n = len(data_x)

    if should_shuffle:
        ix = np.random.choice(n, n, replace=False)
        data_x = data_x[ix]
        data_y = data_y[ix]

    test_start_ix = round(n * train_ratio)
    x_train = data_x[:test_start_ix]
    y_train = data_y[:test_start_ix]

    valid_start_ix = round(n * (train_ratio + test_ratio))

    x_test = data_x[test_start_ix:valid_start_ix]
    y_test = data_y[test_start_ix:valid_start_ix]

    x_valid = data_x[valid_start_ix:]
    y_valid = data_y[valid_start_ix:]

    return x_train, y_train, x_test, y_test, x_valid, y_valid


def to_one_hot(x):
    n_values = np.max(x) + 1
    return np.eye(n_values)[x]


def from_one_hot(x):
    return np.argmax(x, 1)


class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X
        self.y = y

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)


class DataManager:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int
    ):
        self.dataset = dataset
        self.batch_size = batch_size

    def get_batches(self):
        batch_size = self.batch_size
        n_batches = len(self.dataset) // batch_size
        for batch_ix in range(n_batches):
            start_ix = batch_ix * batch_size
            end_ix = start_ix + batch_size
            yield self.dataset[start_ix:end_ix]


if __name__ == '__main__':
    x = np.random.rand(100, 10)
    y = np.random.rand(100, 3)

    data = train_test_valid_split(x, y, 0.7, 0.2, should_shuffle=False)

    print([d.shape for d in data])
    print(x[:4, :4])
    print(data[0][:4, :4])

    print(to_one_hot([4, 4, 1, 0, 0]))
    print(from_one_hot(to_one_hot([4, 4, 1, 0, 0])))
