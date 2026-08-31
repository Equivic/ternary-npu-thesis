import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


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


def evaluate(model, test_loader, dtype=torch.float32):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.shape[0], -1).to(dtype)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


def quantize_int8(w):
    scale = w.abs().max() / 127
    q = torch.round(w / scale)
    q = torch.clamp(q, -128, 127)
    w_reconstructed = q * scale
    return w_reconstructed, scale


def quantize_int4(w):
    scale = w.abs().max() / 7
    q = torch.round(w / scale)
    q = torch.clamp(q, -8, 7)
    w_reconstructed = q * scale
    return w_reconstructed, scale


def quantize_ternary(w):
    threshold = w.abs().mean()
    mask = w.abs() >= threshold
    w_masked = w * mask
    q = torch.sign(w_masked)
    w_reconstructed = q * threshold
    return w_reconstructed, threshold


def quantize_ternary_07(w):
    threshold = 0.7 * w.abs().mean()
    mask = w.abs() >= threshold
    w_masked = w * mask
    q = torch.sign(w_masked)
    w_reconstructed = q * threshold
    return w_reconstructed, threshold


def quantize_binary(w):
    q = torch.sign(w)
    scale = w.abs().mean()
    w_reconstructed = q * scale
    return w_reconstructed, scale


# Load your trained fp32 model
model_fp32 = MNISTNet()
model_fp32.load_state_dict(torch.load("trained_model_3layer.pt"))
model_fp32.eval()

# Create fp16 and bf16 copies
model_fp16 = MNISTNet()
model_fp16.load_state_dict(torch.load("trained_model_3layer.pt"))
model_fp16.half()

model_bf16 = MNISTNet()
model_bf16.load_state_dict(torch.load("trained_model_3layer.pt"))
model_bf16.bfloat16()

# Create INT8 model copy
model_int8 = MNISTNet()
model_int8.load_state_dict(torch.load("trained_model_3layer.pt"))
model_int8.eval()

# Create INT4 model copy
model_int4 = MNISTNet()
model_int4.load_state_dict(torch.load("trained_model_3layer.pt"))
model_int4.eval()

# Create Ternary model copy (threshold = 1.0 x mean)
model_ternary = MNISTNet()
model_ternary.load_state_dict(torch.load("trained_model_3layer.pt"))
model_ternary.eval()

# Create Ternary model copy (threshold = 0.7 x mean)
model_ternary_07 = MNISTNet()
model_ternary_07.load_state_dict(torch.load("trained_model_3layer.pt"))
model_ternary_07.eval()

# Create Binary model copy
model_binary = MNISTNet()
model_binary.load_state_dict(torch.load("trained_model_3layer.pt"))
model_binary.eval()

with torch.no_grad():
    w_quant, scale = quantize_int8(model_int8.fc1.weight)
    model_int8.fc1.weight.copy_(w_quant)
    w_quant, scale = quantize_int8(model_int8.fc2.weight)
    model_int8.fc2.weight.copy_(w_quant)
    w_quant, scale = quantize_int8(model_int8.fc3.weight)
    model_int8.fc3.weight.copy_(w_quant)

with torch.no_grad():
    w_quant, scale = quantize_int4(model_int4.fc1.weight)
    model_int4.fc1.weight.copy_(w_quant)
    w_quant, scale = quantize_int4(model_int4.fc2.weight)
    model_int4.fc2.weight.copy_(w_quant)
    w_quant, scale = quantize_int4(model_int4.fc3.weight)
    model_int4.fc3.weight.copy_(w_quant)

with torch.no_grad():
    w_quant, scale = quantize_ternary(model_ternary.fc1.weight)
    model_ternary.fc1.weight.copy_(w_quant)
    w_quant, scale = quantize_ternary(model_ternary.fc2.weight)
    model_ternary.fc2.weight.copy_(w_quant)
    w_quant, scale = quantize_ternary(model_ternary.fc3.weight)
    model_ternary.fc3.weight.copy_(w_quant)

with torch.no_grad():
    w_quant, scale = quantize_ternary_07(model_ternary_07.fc1.weight)
    model_ternary_07.fc1.weight.copy_(w_quant)
    w_quant, scale = quantize_ternary_07(model_ternary_07.fc2.weight)
    model_ternary_07.fc2.weight.copy_(w_quant)
    w_quant, scale = quantize_ternary_07(model_ternary_07.fc3.weight)
    model_ternary_07.fc3.weight.copy_(w_quant)

with torch.no_grad():
    w_quant, scale = quantize_binary(model_binary.fc1.weight)
    model_binary.fc1.weight.copy_(w_quant)
    w_quant, scale = quantize_binary(model_binary.fc2.weight)
    model_binary.fc2.weight.copy_(w_quant)
    w_quant, scale = quantize_binary(model_binary.fc3.weight)
    model_binary.fc3.weight.copy_(w_quant)

acc_fp32 = evaluate(model_fp32, test_loader, dtype=torch.float32)
acc_fp16 = evaluate(model_fp16, test_loader, dtype=torch.float16)
acc_bf16 = evaluate(model_bf16, test_loader, dtype=torch.bfloat16)
acc_int8 = evaluate(model_int8, test_loader, dtype=torch.float32)
acc_int4 = evaluate(model_int4, test_loader, dtype=torch.float32)
acc_ternary = evaluate(model_ternary, test_loader, dtype=torch.float32)
acc_ternary_07 = evaluate(model_ternary_07, test_loader, dtype=torch.float32)
acc_binary = evaluate(model_binary, test_loader, dtype=torch.float32)

print(f"fp32 accuracy:        {acc_fp32 * 100:.4f}%")
print(f"fp16 accuracy:        {acc_fp16 * 100:.4f}%")
print(f"bf16 accuracy:        {acc_bf16 * 100:.4f}%")
print(f"int8 accuracy:        {acc_int8 * 100:.4f}%")
print(f"int4 accuracy:        {acc_int4 * 100:.4f}%")
print(f"ternary accuracy:     {acc_ternary * 100:.4f}%")
print(f"ternary(0.7) accuracy:{acc_ternary_07 * 100:.4f}%")
print(f"binary accuracy:      {acc_binary * 100:.4f}%")

# Compare a few actual weight values across precisions
w_fp32 = model_fp32.fc1.weight[0][:5]
w_fp16 = model_fp16.fc1.weight[0][:5]
w_bf16 = model_bf16.fc1.weight[0][:5]
w_int8 = model_int8.fc1.weight[0][:5]
w_int4 = model_int4.fc1.weight[0][:5]
w_ternary = model_ternary.fc1.weight[0][:5]
w_ternary_07 = model_ternary_07.fc1.weight[0][:5]
w_binary = model_binary.fc1.weight[0][:5]
print("\nfirst 5 weights, neuron 0:")
print("fp32:        ", w_fp32)
print("fp16:        ", w_fp16)
print("bf16:        ", w_bf16)
print("int8:        ", w_int8)
print("int4:        ", w_int4)
print("ternary:     ", w_ternary)
print("ternary(0.7):", w_ternary_07)
print("binary:      ", w_binary)

# Visualize: several neurons' receptive fields across every precision level
neuron_picks = [0, 5, 12, 20, 33, 47, 55, 68, 79, 90, 101, 115, 20, 44, 60, 88]
models = [
    model_fp32,
    model_fp16,
    model_bf16,
    model_int8,
    model_int4,
    model_ternary,
    model_ternary_07,
    model_binary,
]
names = ["fp32", "fp16", "bf16", "int8", "int4", "ternary", "ternary_0.7", "binary"]

fig, axes = plt.subplots(
    len(models), len(neuron_picks), figsize=(len(neuron_picks) * 1.3, len(models) * 1.5)
)
for row, (model, name) in enumerate(zip(models, names)):
    for col, neuron_idx in enumerate(neuron_picks):
        ax = axes[row][col]
        w = model.fc1.weight[neuron_idx].float().detach().numpy().reshape(28, 28)
        ax.imshow(w, cmap="RdBu")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(name, fontsize=10, rotation=0, ha="right", va="center")

plt.tight_layout()
plt.savefig("precision_compare_grid.png", dpi=130)
plt.show()
