import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor
from torch.func import functional_call
import torchopt


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


def make_functional_fwd(model: torch.nn.Module):
    """Make a functional forward pass for a generic module
    This function is compatible with the torchopt library which
    returns the updated parameters as a tuple while the `functional_call`
    routine requires parameters dictionary. This conversion is automatically
    implemented by this function
    """

    keys = list(dict(model.named_parameters()).keys())

    def fn(data: Tensor, parameters: tuple[Tensor, ...]):
        params_dict = {k: v for k, v in zip(keys, parameters)}
        return functional_call(model, params_dict, (data,))

    return fn


def get_data(n_points: int = 20) -> tuple[Tensor, Tensor]:
    """Prepare the input data for training/test sets"""
    x = torch.rand(n_points) * 2.0 * torch.pi
    y = 2.0 * torch.sin(x + 2.0 * torch.pi)
    return x, y


if __name__ == "__main__":
    torch.manual_seed(42)

    model = SimpleNN(num_layers=2)
    model_func = make_functional_fwd(model)

    x_train, y_train = get_data(n_points=40)
    x_test, y_test = get_data(n_points=10)

    # choose optimizer with functional API using functorch
    num_epochs = 500
    lr = 0.01
    optimizer = torchopt.FuncOptimizer(torchopt.adam(lr=lr))
    loss_fn = torch.nn.MSELoss()

    # train the model
    loss_evolution = []
    params = tuple(model.parameters())

    for i in range(num_epochs):
        # update the parameters
        y = model_func(x_train, params)
        loss = loss_fn(y, y_train)
        params = optimizer.step(loss, params)

        if i % 100 == 0:
            print(f"Iteration {i} with loss {float(loss)}")
        loss_evolution.append(float(loss))

    plt.plot(np.arange(len(loss_evolution)), loss_evolution)
    plt.title('Loss function')
    plt.show()

    # performance on the model on the test set
    y_pred = model_func(x_test, params)
    print(f"Loss on the test set: {loss_fn(y_pred, y_test)}")
