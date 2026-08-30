import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
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


def quantize_ternary(w):
    threshold = w.abs().mean()
    mask = w.abs() >= threshold
    w_masked = w * mask
    q = torch.sign(w_masked)
    w_reconstructed = q * threshold
    return w_reconstructed, threshold


# --- Baseline: ternary, no fine-tuning (should match precision_compare.py's 85.64%) ---
model_ternary = MNISTNet()
model_ternary.load_state_dict(torch.load('trained_model.pt'))
model_ternary.eval()

with torch.no_grad():
    w_quant, threshold = quantize_ternary(model_ternary.fc1.weight)
    model_ternary.fc1.weight.copy_(w_quant)

acc_ternary_no_ft = evaluate(model_ternary, test_loader)
print(f"ternary, no fine-tune:     {acc_ternary_no_ft*100:.4f}%")

# --- Fine-tune: start from the same ternary-quantized weights, train a few more epochs ---
model_ternary_ft = MNISTNet()
model_ternary_ft.load_state_dict(torch.load('trained_model.pt'))

with torch.no_grad():
    w_quant, threshold = quantize_ternary(model_ternary_ft.fc1.weight)
    model_ternary_ft.fc1.weight.copy_(w_quant)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_ternary_ft.parameters(), lr=0.01)

for epoch in range(5):
    for images, labels in train_loader:
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        outputs = model_ternary_ft(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    acc = evaluate(model_ternary_ft, test_loader)
    print(f"  epoch {epoch}, test accuracy: {acc*100:.4f}%")

acc_ternary_ft = evaluate(model_ternary_ft, test_loader)
print(f"\nternary, fine-tuned (5ep): {acc_ternary_ft*100:.4f}%")
print(f"recovered: {(acc_ternary_ft - acc_ternary_no_ft)*100:.2f} points")

# Check: are the fine-tuned weights still actually ternary, or did they drift?
w_ft = model_ternary_ft.fc1.weight.detach()
unique_vals = torch.unique(w_ft)
print(f"\nnumber of distinct weight values after fine-tuning: {unique_vals.numel()}")
print("first 10 weights, neuron 0, after fine-tuning:")
print(w_ft[0][:10])
