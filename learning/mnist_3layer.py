import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import time

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

small_tran_dataset = Subset(train_dataset, range(1000))
# train_loader = DataLoader(small_tran_dataset, batch_size=batch_size, shuffle=True)


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
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


model = MNISTNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

epoch_losses = []
epoch_accuracies = []

start_time = time.time()

for epoch in range(25):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(images.shape[0], -1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # if epoch == 0 and i == 0:
        #   print("fc1 weight grad shape:", model.fc1.weight.grad.shape)
        #   print("fc1 weight grad sample (first row):", model.fc1.weight.grad[0][:5])
        # optimizer.step()

    avg_loss = running_loss / len(train_loader)
    epoch_losses.append(avg_loss)

    accuracy = evaluate(model, test_loader)
    epoch_accuracies.append(accuracy)
    print(
        f"epoch {epoch}, avg loss {avg_loss:.4f}, test accuracy: {accuracy * 100:.2f}%"
    )


end_time = time.time()

print(f"training took {end_time - start_time:.2f} seconds")

fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.set_xlabel("epoch")
ax1.set_ylabel("training loss", color="tab:red")
ax1.plot(epoch_losses, color="tab:red", marker="o", label="train loss")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.set_ylabel("test accuracy (%)", color="tab:blue")
ax2.plot(
    [a * 100 for a in epoch_accuracies],
    color="tab:blue",
    marker="s",
    label="test accuracy",
)
ax2.tick_params(axis="y", labelcolor="tab:blue")

plt.title("MNIST training: loss vs accuracy")
fig.tight_layout()
plt.savefig("training_curve.png")
plt.show()

import json

w1 = model.fc1.weight.detach().numpy()
with open("fc1_weights.json", "w") as f:
    json.dump(w1.tolist(), f)
print("exported fc1_weights.json, shape:", w1.shape)

torch.save(model.state_dict(), "trained_model_3layer.pt")
print("saved trained_model.pt")
