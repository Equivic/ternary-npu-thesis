import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---- Load and prepare data (shared across all experiments below) ----
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

MAJORITY_BASELINE = max((y_test == 0).mean(), (y_test == 1).mean()) * 100
print(f"'always guess majority class' floor: {MAJORITY_BASELINE:.2f}%\n")


# ---- Quantization functions (same as MNIST work) ----
def quantize_ternary(w, threshold_multiplier=1.0):
    threshold = threshold_multiplier * w.abs().mean()
    mask = w.abs() >= threshold
    w_masked = w * mask
    q = torch.sign(w_masked)
    return q * threshold


def quantize_binary(w):
    q = torch.sign(w)
    scale = w.abs().mean()
    return q * scale


def ternary_ste(w, threshold_multiplier=1.0):
    threshold = threshold_multiplier * w.abs().mean()
    mask = (w.abs() >= threshold).float()
    q = torch.sign(w) * mask
    q_scaled = q * threshold
    return w + (q_scaled - w).detach()


def binary_ste(w):
    scale = w.abs().mean()
    q_scaled = torch.sign(w) * scale
    return w + (q_scaled - w).detach()


# ---- Model definitions ----
class BreastCancerNet(nn.Module):
    """Plain fp32 network -- used for training the baseline and for naive post-training quantization."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)
        return x


class BreastCancerQATNet(nn.Module):
    """QAT network -- quantizes every layer's weight on every forward pass via STE."""
    def __init__(self, quant_fn):
        super().__init__()
        self.fc1 = nn.Linear(30, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.quant_fn = quant_fn

    def forward(self, x):
        x = F.linear(x, self.quant_fn(self.fc1.weight), self.fc1.bias)
        x = torch.relu(x)
        x = F.linear(x, self.quant_fn(self.fc2.weight), self.fc2.bias)
        x = torch.relu(x)
        x = F.linear(x, self.quant_fn(self.fc3.weight), self.fc3.bias)
        x = torch.sigmoid(x)
        return x


# ---- Evaluation ----
def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predicted = (outputs >= 0.5).float()
        correct = (predicted == y).sum().item()
        total = y.shape[0]
    model.train()
    return correct / total


# ---- Part 1: train fp32 baseline ----
print("=== Training fp32 baseline ===")
torch.manual_seed(42)
model_fp32 = BreastCancerNet()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model_fp32.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model_fp32(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

acc_fp32_train = evaluate(model_fp32, X_train_t, y_train_t)
acc_fp32_test = evaluate(model_fp32, X_test_t, y_test_t)
print(f"fp32 baseline: train {acc_fp32_train*100:.2f}%  test {acc_fp32_test*100:.2f}%\n")

torch.save(model_fp32.state_dict(), 'breast_cancer_model.pt')

# ---- Part 2: naive post-training quantization ----
print("=== Naive post-training quantization ===")

model_naive_ternary = BreastCancerNet()
model_naive_ternary.load_state_dict(torch.load('breast_cancer_model.pt'))
with torch.no_grad():
    for layer in [model_naive_ternary.fc1, model_naive_ternary.fc2, model_naive_ternary.fc3]:
        layer.weight.copy_(quantize_ternary(layer.weight))
acc_naive_ternary = evaluate(model_naive_ternary, X_test_t, y_test_t)
print(f"naive ternary: test {acc_naive_ternary*100:.2f}%")

model_naive_binary = BreastCancerNet()
model_naive_binary.load_state_dict(torch.load('breast_cancer_model.pt'))
with torch.no_grad():
    for layer in [model_naive_binary.fc1, model_naive_binary.fc2, model_naive_binary.fc3]:
        layer.weight.copy_(quantize_binary(layer.weight))
acc_naive_binary = evaluate(model_naive_binary, X_test_t, y_test_t)
print(f"naive binary:  test {acc_naive_binary*100:.2f}%\n")


# ---- Part 3: QAT, multi-seed sweep ----
def run_qat_seed(quant_fn, seed, epochs=100):
    torch.manual_seed(seed)
    model = BreastCancerQATNet(quant_fn)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    live_acc = evaluate(model, X_test_t, y_test_t)

    with torch.no_grad():
        model.fc1.weight.data = quant_fn(model.fc1.weight)
        model.fc2.weight.data = quant_fn(model.fc2.weight)
        model.fc3.weight.data = quant_fn(model.fc3.weight)
    frozen_acc = evaluate(model, X_test_t, y_test_t)

    return live_acc, frozen_acc


def sweep(quant_fn, name, seeds=(0, 1, 2, 3, 4)):
    print(f"=== QAT {name}, {len(seeds)}-seed sweep ===")
    results = [run_qat_seed(quant_fn, s) for s in seeds]
    live_accs = [r[0] for r in results]
    frozen_accs = [r[1] for r in results]
    for s, (live, frozen) in zip(seeds, results):
        print(f"  seed {s}: live {live*100:.2f}%  frozen {frozen*100:.2f}%")
    mean_frozen = sum(frozen_accs) / len(frozen_accs)
    spread = max(frozen_accs) - min(frozen_accs)
    print(f"  frozen mean: {mean_frozen*100:.2f}%   spread: {spread*100:.2f} points\n")
    return mean_frozen, spread


mean_ternary, spread_ternary = sweep(ternary_ste, "ternary")
mean_binary, spread_binary = sweep(binary_ste, "binary")

# ---- Diagnostic: check for zero-collapse in QAT ternary ----
print("=== Diagnostic: QAT ternary weight collapse check (seed 0) ===")
torch.manual_seed(0)
diag_model = BreastCancerQATNet(ternary_ste)
diag_optimizer = torch.optim.SGD(diag_model.parameters(), lr=0.1)
diag_criterion = nn.BCELoss()

for epoch in range(100):
    diag_optimizer.zero_grad()
    outputs = diag_model(X_train_t)
    loss = diag_criterion(outputs, y_train_t)
    loss.backward()
    diag_optimizer.step()

with torch.no_grad():
    for name, layer in [('fc1', diag_model.fc1), ('fc2', diag_model.fc2), ('fc3', diag_model.fc3)]:
        w_frozen = ternary_ste(layer.weight)
        total = w_frozen.numel()
        zeros = (w_frozen == 0).sum().item()
        pos = (w_frozen > 0).sum().item()
        neg = (w_frozen < 0).sum().item()
        print(f"  {name}: {total} weights -> {zeros} zero, {pos} positive, {neg} negative "
              f"({zeros/total*100:.1f}% zero)")
        print(f"    bias: {layer.bias.data}")

    # Freeze the diagnostic model fully and look at actual raw output values
    diag_model.fc1.weight.data = ternary_ste(diag_model.fc1.weight)
    diag_model.fc2.weight.data = ternary_ste(diag_model.fc2.weight)
    diag_model.fc3.weight.data = ternary_ste(diag_model.fc3.weight)
    diag_model.eval()
    raw_outputs = diag_model(X_test_t)
    print(f"\n  raw sigmoid outputs, first 15 test samples: {raw_outputs[:15].squeeze()}")
    print(f"  output min: {raw_outputs.min().item():.6f}  max: {raw_outputs.max().item():.6f}")
    print(f"  unique output values (rounded to 3dp): {torch.unique(torch.round(raw_outputs*1000)/1000).numel()}")

# ---- Summary ----
print("=== Summary ===")
print(f"'always guess majority' floor: {MAJORITY_BASELINE:.2f}%")
print(f"fp32 baseline:                 {acc_fp32_test*100:.2f}%")
print(f"naive ternary:                 {acc_naive_ternary*100:.2f}%")
print(f"naive binary:                  {acc_naive_binary*100:.2f}%")
print(f"QAT ternary (frozen mean):     {mean_ternary*100:.2f}%  (spread {spread_ternary*100:.2f}pt)")
print(f"QAT binary (frozen mean):      {mean_binary*100:.2f}%  (spread {spread_binary*100:.2f}pt)")
