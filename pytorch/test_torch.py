import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import Tensor

# Define the model class


class SimpleNN(nn.Module):
    def __init__(self, num_layers: int = 1, num_neurons: int = 5) -> None:
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
        return self.network(x).squeeze()


def get_data(n_points=20, noise_std=0.1) -> tuple[Tensor, Tensor]:
    x = torch.rand(n_points, 1) * 2.0 * torch.pi  # Shape: [n_points, 1]
    y = 2.0 * torch.sin(x + 2.0 * torch.pi)  # Original y-values

    noise = torch.randn(n_points, 1) * noise_std
    y_noisy = y + noise

    return x, y_noisy


if __name__ == "__main__":
    x_train, y_train = get_data(n_points=500)
    x_test, y_test = get_data(n_points=100)

    plt.scatter(x_train, y_train)
    plt.title("Data")
    plt.show()

    model = SimpleNN(num_layers=2)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 10000
    loss_evolution = []

    for i in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train.squeeze())
        loss.backward()
        optimizer.step()

        loss_evolution.append(float(loss.item()))

        if i % 100 == 0:
            print(f"Iteration {i} with loss {float(loss.item())}")

    plt.plot(np.arange(len(loss_evolution)), loss_evolution)
    plt.title("Loss function")
    plt.show()

    model.eval()
    with torch.no_grad():
        y_test_pred = model(x_test)
        test_loss = loss_fn(y_test_pred, y_test.squeeze())

    print(f"Loss on the test set: {test_loss.item()}")
