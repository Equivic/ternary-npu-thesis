import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
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

epoch_accuracies = []


def ternary_ste(w, threshold_multiplier=1.0):
    threshold = threshold_multiplier * w.abs().mean()
    mask = (w.abs() >= threshold).float()
    q = torch.sign(w) * mask
    q_scaled = q * threshold
    # straight-through: forward uses q_scaled, backward acts like identity on w
    return w + (q_scaled - w).detach()


class QATNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # TODO: this is the key difference from your normal MNISTNet.
        # Instead of x = self.fc1(x), you need to:
        #   1. Get fc1's real weight and bias: self.fc1.weight, self.fc1.bias
        #   2. Quantize the weight using ternary_ste(...)
        #   3. Manually do the linear operation with the QUANTIZED weight
        #      (hint: F.linear(input, weight, bias) does exactly what nn.Linear
        #       does internally -- it's the raw function version)
        #   4. Apply relu, then pass through fc2 normally (leave fc2 in full
        #      precision for now -- we're only QAT-ing fc1 to start simple)
        w = self.fc1.weight
        b = self.fc1.bias
        w = ternary_ste(w)
        x = F.linear(x, w, b)
        x = torch.relu(x)
        w = self.fc2.weight
        b = self.fc2.bias
        w = ternary_ste(w)
        x = F.linear(x, w, b)
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


model = QATNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(25):
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    acc = evaluate(model, test_loader)
    epoch_accuracies.append(acc)
    print(f"epoch {epoch}, test accuracy: {acc * 100:.4f}%")

# After training, check: is fc1's weight actually ternary now?
with torch.no_grad():
    w_final = ternary_ste(model.fc1.weight)
    print(f"\nfinal QAT accuracy (weights genuinely ternary at eval time): ", end="")

model.fc1.weight.data = w_final
final_acc = evaluate(model, test_loader)
print(f"{final_acc * 100:.4f}%")

torch.save(model.state_dict(), "qat_full_model.pt")
print("saved qat_full_model.pt")

# --- Plot the training curve ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([a * 100 for a in epoch_accuracies], marker="o", color="tab:blue")
ax.axhline(
    y=85.64,
    color="tab:red",
    linestyle="--",
    label="naive post-training ternary (85.64%)",
)
ax.axhline(
    y=final_acc * 100,
    color="tab:green",
    linestyle="--",
    label=f"QAT frozen final ({final_acc * 100:.2f}%)",
)
ax.set_xlabel("epoch")
ax.set_ylabel("test accuracy (%)")
ax.set_title("QAT training curve (fc1 + fc2 ternary)")
ax.legend()
plt.tight_layout()
plt.savefig("qat_full_curve.png")
plt.show()

# --- Neuron receptive fields: fp32-trained baseline vs QAT-ternary fc1 weights ---
w_qat_frozen = model.fc1.weight.detach().numpy()  # already frozen ternary at this point

neuron_picks = [0, 5, 12, 20, 33, 47, 55, 68, 79, 90, 101, 115]
fig2, axes = plt.subplots(1, len(neuron_picks), figsize=(18, 2.2))
for ax, idx in zip(axes, neuron_picks):
    ax.imshow(w_qat_frozen[idx].reshape(28, 28), cmap="RdBu")
    ax.set_title(f"n{idx}", fontsize=9)
    ax.axis("off")
plt.suptitle("QAT-trained ternary fc1 receptive fields (frozen, post-training)")
plt.tight_layout()
plt.savefig("qat_neuron_grid.png")
plt.show()
