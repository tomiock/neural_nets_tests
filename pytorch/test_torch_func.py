import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.func import functional_call
import torchopt


from collections.abc import Callable
from enum import Enum
from torch import Tensor

class Optimizer(Enum):
    ADAM = (torchopt.adam, )
    SGD = (torchopt.sgd, )
    ADAGRAD = (torchopt.adagrad, )
    ADAMAX = (torchopt.adamax, )
    ADADELTA = (torchopt.adadelta, )
    ADAMW = (torchopt.adamw, )
    RADAM = (torchopt.radam, )
    RMSPROP = (torchopt.rmsprop, )

    def __new__(cls, optmizer_class):
        obj = object.__new__(cls)
        obj._value_ = optmizer_class
        return obj

    def get_optimizer(self, lr, **kwargs):
        return torchopt.FuncOptimizer(self.value(lr=lr, **kwargs))


class SimpleNN(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        num_neurons: int = 5,
    ) -> None:
        """Basic neural network architecture with linear layers
        Args:
            num_layers (int, optional): number of hidden layers
            num_neurons (int, optional): neurons for each hidden layer
        """
        super().__init__()

        layers = []

        # input layer
        layers.append(nn.Linear(1, num_neurons))

        # hidden layers
        for _ in range(num_layers):
            layers.extend([nn.Linear(num_neurons, num_neurons), nn.Tanh()])

        # output layer
        layers.append(nn.Linear(num_neurons, 1))

        # build the network
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x.reshape(-1, 1)).squeeze()


def make_functional_fwd(
    model: torch.nn.Module,
) -> Callable[[Tensor, tuple[Tensor, ...]], Tensor]:
    """Make a functional forward pass for a generic module
    This function is compatible with the torchopt library which
    returns the updated parameters as a tuple while the `functional_call`
    routine requires parameters dictionary. This conversion is automatically
    implemented by this function
    """

    keys = list(dict(model.named_parameters()).keys())

    def fn(data: Tensor, parameters: tuple[Tensor, ...]) -> Tensor:
        params_dict = {k: v for k, v in zip(keys, parameters)}
        return functional_call(model, params_dict, (data,))

    return fn


def get_data(n_points=20, noise_std=0.1) -> tuple[Tensor, Tensor]:
    x = torch.rand(n_points, 1) * 2.0 * torch.pi  # Shape: [n_points, 1]
    y = 2.0 * torch.sin(x + 2.0 * torch.pi)  # Original y-values

    noise = torch.randn(n_points, 1) * noise_std
    y_noisy = y + noise

    return x, y_noisy


if __name__ == "__main__":
    torch.manual_seed(42)

    model = SimpleNN(num_layers=2)
    model_func = make_functional_fwd(model)

    x_train, y_train = get_data(n_points=40)
    x_test, y_test = get_data(n_points=10)

    # choose optimizer with functional API using functorch
    num_epochs = 1000
    lr = .01

    loss_curves = {}

    for opt in Optimizer:
        optimizer = opt.get_optimizer(lr=lr)
        loss_fn = torch.nn.MSELoss()

        # train the model
        loss_evolution = []
        params = tuple(model.parameters())

        for i in range(num_epochs):
            # update the parameters
            y = model_func(x_train, params)
            loss = loss_fn(y, y_train)
            params = optimizer.step(loss, params)

            loss_evolution.append(float(loss))

        loss_curves[opt.name] = loss_evolution

        # performance on the model on the test set
        y_pred = model_func(x_test, params)
        print(f"Loss on the test set: {loss_fn(y_pred, y_test)} with opt {opt.name}")

    plt.figure(figsize=(10, 6))
    for opt_name, losses in loss_curves.items():
        plt.plot(np.arange(len(losses)), losses, label=f'{opt_name}')

    plt.title("Loss function evolution for different optimizers")
    plt.xlabel("Epochs")
    #plt.xscale('log')
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
