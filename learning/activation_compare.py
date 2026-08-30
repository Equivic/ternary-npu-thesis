import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MNISTNet(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation_type = activation
        if activation == "prelu":
            self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.fc1(x)
        if self.activation_type == "relu":
            x = torch.relu(x)
        elif self.activation_type == "leaky_relu":
            x = F.leaky_relu(x, negative_slope=0.01)
        elif self.activation_type == "prelu":
            x = self.prelu(x)
        elif self.activation_type == "elu":
            x = F.elu(x)
        x = self.fc2(x)
        return x


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    model.train()
    return correct / total


def train_and_eval(activation, epochs=50):
    model = MNISTNet(activation=activation)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    start = time.time()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start

    acc = evaluate(model, test_loader)
    print(f"{activation:12s} acc: {acc * 100:.2f}%   time: {elapsed:.2f}s")
    return model, acc, elapsed


if __name__ == "__main__":
    accs = {}
    times = {}
    for act in ["relu", "leaky_relu", "prelu", "elu"]:
        model, acc, elapsed = train_and_eval(act)
        accs[act] = acc
        times[act] = elapsed

    print("\nsummary:")
    for act in accs:
        print(f"  {act:12s} {accs[act] * 100:.2f}%   {times[act]:.2f}s")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    names = list(accs.keys())
    ax1.bar(names, [accs[a] * 100 for a in names], color="tab:blue")
    ax1.set_ylabel("test accuracy (%)")
    ax1.set_title("accuracy by activation")
    ax1.set_ylim(min(accs.values()) * 100 - 1, max(accs.values()) * 100 + 1)

    ax2.bar(names, [times[a] for a in names], color="tab:orange")
    ax2.set_ylabel("training time (s)")
    ax2.set_title("training time by activation")

    plt.tight_layout()
    plt.savefig("activation_compare.png")
    plt.show()
