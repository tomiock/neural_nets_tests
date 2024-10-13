import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_greyscale_digit_normal(digit, img_size, noise_level=0.1):
    img = np.zeros(img_size, dtype=float)
    offset_x = np.random.randint(-img_size[0]//8, img_size[0]//8)
    offset_y = np.random.randint(-img_size[1]//8, img_size[1]//8)

    if digit == 0:
        x, y = np.indices(img_size)
        center = (img_size[0]//2, img_size[1]//2) + \
            np.array([offset_x, offset_y])
        radius = min(img_size) // 4
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        img[mask] = 1.0
    elif digit == 1:
        x, y = np.indices(img_size)
        center_x = img_size[1] // 2 + offset_y
        width = img_size[1] // 8
        mask = (y > center_x - width) & (y < center_x + width)
        img[mask] = 1.0

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, img_size)
    img = np.clip(img + noise, 0, 1)

    # Deform the borders
    for _ in range(int(noise_level * 10000)):
        x, y = np.random.randint(
            0, img.shape[0]), np.random.randint(0, img.shape[1])
        if img[x, y] == 1.0:
            deform_x = x + np.random.choice([-1, 1])
            deform_y = y + np.random.choice([-1, 1])
            if 0 <= deform_x < img.shape[0] and 0 <= deform_y < img.shape[1]:
                img[deform_x, deform_y] = np.random.uniform(.7, .95)

    return img


def generate_dataset_normal(num_samples, img_size, noise_level=0.1):
    dataset = []
    for _ in tqdm(range(num_samples)):
        label = np.random.choice([0, 1])
        img = create_greyscale_digit_normal(label, img_size, noise_level)
        dataset.append((img.flatten(), label))
    return dataset


"""
def print_greyscale_image(img):
    for row in img:
        print(" ".join([f"{pixel:.2f}" for pixel in row]))
"""


def plot_greyscale_image(img):
    img = np.array(img)
    if img.ndim == 1:
        img = img.reshape((int(np.sqrt(img.size)), -1))

    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.show()


def save_dataset(dataset, filename):
    with open(filename, 'w') as f:
        for img, label in dataset:
            f.write(f"{label} {' '.join(map(str, img))}\n")


def load_dataset(filename):
    dataset = []
    with open(filename, 'r') as f:
        for line in f:
            label, *img = map(float, line.strip().split())
            img = np.array(img)
            dataset.append((img, int(label)))
    return dataset


if __name__ == '__main__':
    img_size = (512, 512)
    num_samples = 1000
    noise_level = 0.5

    dataset = generate_dataset_normal(num_samples, img_size, noise_level)
    plot_greyscale_image(dataset[0][0])
    print(dataset[0][1])

    save_dataset(dataset, 'dataset.txt')
    print("Dataset saved.")
    print("Sample image:")

    dataset = load_dataset('dataset.txt')
    plot_greyscale_image(dataset[0][0])
