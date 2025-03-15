import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = torchvision.datasets.MNIST('.data/', train=True, download=True)
test_set = torchvision.datasets.MNIST('.data/', train=False, download=True)

class MNIST_dataset(Dataset):
    def __init__(self, data, partition="train"):
        self.data = data
        self.partition = partition
        print("total len", len(self.data))

    def __len__(self):
        return len(self.data)

    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __getitem__(self, idx):
        image = self.data[idx][0]
        image_tensor = self.from_pil_to_tensor(image)
        image_tensor = image_tensor.view(-1)

        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()

        return {"img": image_tensor, "label": label}

train_dataset = MNIST_dataset(train_set, partition="train")
test_dataset = MNIST_dataset(test_set, partition="test")

batch_size = 10
num_workers = torch.multiprocessing.cpu_count()-1

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_workers)

class NN(nn.Module):
    def __init__(self, num_classes):
        super(NN, self).__init__()
        self.l1 = nn.Linear(784, 1024)
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.l3 = nn.Linear(1024, 1024)
        self.relu3 = nn.ReLU()
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.relu1(self.l1(x))
        x = self.relu2(self.l2(x))
        x = self.relu3(self.l3(x))
        x = self.classifier(x)
        return x

num_classes = 10
net = NN(num_classes)
print(net)

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=.01, weight_decay=1e-6, momentum=.9)
epochs = 1

net.to(device)

print("\n---- Start Training ----")
best_accuracy = -1
best_epoch = 0
for epoch in range(epochs):
    # TRAIN NETWORK
    train_loss, train_correct = 0, 0
    net.train()
    with tqdm(iter(train_dataloader), desc="Epoch " + str(epoch), unit="batch") as tepoch:
        for batch in tepoch:

            # Returned values of Dataset Class
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = net(images)
            loss = criterion(outputs, labels)

            # Calculate gradients
            loss.backward()

            # Update gradients
            optimizer.step()

            # one hot -> labels
            labels = torch.argmax(labels, dim=1)
            pred = torch.argmax(outputs, dim=1)
            train_correct += pred.eq(labels).sum().item()

            # print statistics
            train_loss += loss.item()

    train_loss /= len(train_dataloader.dataset)

    # TEST NETWORK
    test_loss, test_correct = 0, 0
    net.eval()
    with torch.no_grad():
      with tqdm(iter(test_dataloader), desc="Test " + str(epoch), unit="batch") as tepoch:
          for batch in tepoch:

            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # Forward
            outputs = net(images)
            test_loss += criterion(outputs, labels)

            # one hot -> labels
            labels = torch.argmax(labels, dim=1)
            pred = torch.argmax(outputs, dim=1)

            test_correct += pred.eq(labels).sum().item()

    test_loss /= len(test_dataloader.dataset)
    test_accuracy = 100. * test_correct / len(test_dataloader.dataset)

    print("[Epoch {}] Train Loss: {:.6f} - Test Loss: {:.6f} - Train Accuracy: {:.2f}% - Test Accuracy: {:.2f}%".format(
        epoch + 1, train_loss, test_loss, 100. * train_correct / len(train_dataloader.dataset), test_accuracy
    ))

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch

        # Save best weights
        torch.save(net.state_dict(), "best_model.pt")

print("\nBEST TEST ACCURACY: ", best_accuracy, " in epoch ", best_epoch)
