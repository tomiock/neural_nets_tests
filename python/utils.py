import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))


def BCE(output_activations, y):
    return -np.sum(
        y * np.log(output_activations) + (1 - y) * np.log(1 - output_activations)
    )


def BCE_prime(output_activations, y):
    return (output_activations - y) / (output_activations * (1 - output_activations))
