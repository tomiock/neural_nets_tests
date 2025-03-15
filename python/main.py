import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import create_greyscale_digit_normal, generate_dataset_normal, plot_greyscale_image, load_dataset
from fcnn_naive import FCNN_BinaryDigits_naive
from utils import BCE


def prepare_data(num_samples, img_size, noise_level):
    dataset = generate_dataset_normal(num_samples, img_size, noise_level)
    np.random.shuffle(dataset)
    X = np.array([img for img, _ in dataset])
    y = np.array([label for _, label in dataset])
    return X, y


def load_data(filename: str):
    dataset = load_dataset(filename)
    np.random.shuffle(dataset)
    X = np.array([img for img, _ in dataset])
    y = np.array([label for _, label in dataset])
    return X, y


def evaluate_network(network, X_test, y_test):
    test_results = [(np.round(network.predict(x)), y)
                    for (x, y) in zip(X_test, y_test)]
    test_accuracy = sum(int(x.item() == y.item())
                        for (x, y) in test_results) / len(y_test)
    return test_accuracy


def evaluate_loss(network, X_test, y_test):
    loss = sum(BCE(network.predict(x), y)
               for x, y in zip(X_test, y_test)) / len(y_test)
    return loss


def predict(network, X):
    return np.array([np.round(network.predict(x)) for x in X])


def main():
    img_size = (28, 28)
    noise_level = 0.1
    num_samples = 1000
    X, y = load_data("python/dataset.txt")

    train_size = int(num_samples * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    start_t = time.time()

    network = FCNN_BinaryDigits_naive([img_size[0] * img_size[1], 500, 100, 10, 1])
    network.init_parameters()

    accuracy = []
    loss = []
    epochs = 10
    mini_batch_size = 10
    learning_rate = .1

    n = len(X_train)
    for epoch in tqdm(range(epochs)):
        mini_batches = [
            (X_train[k:k+mini_batch_size], y_train[k:k+mini_batch_size])
            for k in range(0, n, mini_batch_size)
        ]
        for mini_batch in mini_batches:
            network.train_mini_batch(mini_batch, learning_rate)
        test_accuracy = evaluate_network(network, X_test, y_test)
        accuracy.append(test_accuracy)
        test_loss = evaluate_loss(network, X_test, y_test)
        loss.append(test_loss)

    end_t = time.time()

    final_test_accuracy = evaluate_network(network, X_test, y_test)
    print(f"Test Accuracy: {final_test_accuracy:.2%}")

    """
    # Plot example images
    digit = 1
    img = create_greyscale_digit_normal(digit, img_size, noise_level)
    plot_greyscale_image(img)

    digit = 0
    img = create_greyscale_digit_normal(digit, img_size, noise_level)
    plot_greyscale_image(img)

    # Plot test accuracy
    plt.plot(accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.show()

    # Predict
    predictions = predict(network, X_test[:10])
    print(predictions)
    print(y_test[:10])

    """

    # Plot test loss
    plt.title(f'Time taken: {end_t - start_t:.2f}s')
    plt.plot(loss)
    plt.xlabel("Epoch")
    plt.ylabel("Test Loss")
    plt.show()

if __name__ == "__main__":
    main()
