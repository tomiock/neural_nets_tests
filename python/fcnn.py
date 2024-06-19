import numpy as np
from typing import List


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def BCE(output_activations, y):
    return -np.sum(y * np.log(output_activations) + (1 - y) * np.log(1 - output_activations))


def BCE_prime(output_activations, y):
    return (output_activations - y) / (output_activations * (1 - output_activations))


# TODO: Do not make a class?
class FCNN_BinaryDigits:
    def __init__(self, sizes: List[int]) -> None:
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = []
        self.weights = []

    def init_parameters(self):
        np.random.seed(42)  # For reproducibility
        self.biases = [np.random.randn(y, ) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0/x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        return self.biases, self.weights

    def predict(self, x):
        activation = x
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            activation = sigmoid(z)
        return activation

    def forwardprop(self, x: np.ndarray):
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        return activation, activations, zs

    def backprop_bce(self, image, label):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation, activations, zs = self.forwardprop(image)
        delta = BCE_prime(activations[-1], label) * sigmoid_prime(zs[-1])

        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].reshape(1,
                             activations[-2].shape[0]))

        for l in range(2, self.num_layers):
            z = zs[-l]
            delta = np.dot(
                self.weights[-l + 1].transpose(), delta) * sigmoid_prime(z)
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta.reshape(delta.shape[0], 1),
                                 activations[-l - 1].reshape(1, activations[-l - 1].shape[0]))
        return nabla_b, nabla_w, activations[-1]

    def train_mini_batch(self, mini_batch, learning_rate):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for image, label in zip(mini_batch[0], mini_batch[1]):
            delta_nabla_b, delta_nabla_w, _ = self.backprop_bce(image, label)

            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


if __name__ == '__main__':
    sizes = [256, 30, 1]
    fcnn = FCNN_BinaryDigits(sizes)
    fcnn.init_parameters()
