import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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


def ternary_ste(w, threshold_multiplier=1.0):
    threshold = threshold_multiplier * w.abs().mean()
    mask = (w.abs() >= threshold).float()
    q = torch.sign(w) * mask
    q_scaled = q * threshold
    return w + (q_scaled - w).detach()


class QATNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        w1 = ternary_ste(self.fc1.weight)
        x = F.linear(x, w1, self.fc1.bias)
        x = torch.relu(x)
        w2 = ternary_ste(self.fc2.weight)
        x = F.linear(x, w2, self.fc2.bias)
        x = torch.relu(x)
        w3 = ternary_ste(self.fc3.weight)
        x = F.linear(x, w3, self.fc3.bias)
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


def run_seed(seed, epochs=25):
    torch.manual_seed(seed)
    model = QATNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        for images, labels in train_loader:
            images = images.view(images.shape[0], -1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    live_acc = evaluate(model, test_loader)

    # Freeze weights permanently to their ternary values
    with torch.no_grad():
        model.fc1.weight.data = ternary_ste(model.fc1.weight)
        model.fc2.weight.data = ternary_ste(model.fc2.weight)
        model.fc3.weight.data = ternary_ste(model.fc3.weight)
    frozen_acc = evaluate(model, test_loader)

    print(
        f"seed {seed}: live-quantized acc {live_acc * 100:.2f}%   frozen acc {frozen_acc * 100:.2f}%"
    )
    return live_acc, frozen_acc


if __name__ == "__main__":
    seeds = [0, 1, 2]
    results = [run_seed(s) for s in seeds]

    frozen_accs = [r[1] for r in results]
    mean_frozen = sum(frozen_accs) / len(frozen_accs)
    spread = max(frozen_accs) - min(frozen_accs)

    print(f"\nfrozen accuracy across seeds: {[f'{a * 100:.2f}%' for a in frozen_accs]}")
    print(
        f"mean: {mean_frozen * 100:.2f}%   spread (max-min): {spread * 100:.2f} points"
    )
