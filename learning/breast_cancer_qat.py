from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn


class BreastCancerNet(nn.Module):
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


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        predicted = (outputs >= 0.5).float()
        correct = (predicted == y).sum().item()
        total = y.shape[0]
    model.train()
    return correct / total


data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("train samples:", X_train.shape[0])
print("train samples:", X_test.shape[0])

print("X shape:", X.shape)
print("y shape:", y.shape)
print("feature name:", data.feature_names[:5], "...")
print("target names:", data.target_names)
print("class counts:", (y == 0).sum(), (y == 1).sum())

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled.mean(axis=0)[:5])
print(X_train_scaled.std(axis=0)[:5])

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

model = BreastCancerNet()
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss {loss.item():.4f}")

train_acc = evaluate(model, X_train_t, y_train_t)
test_acc = evaluate(model, X_test_t, y_test_t)
print(f"\ntrain accuracy: {train_acc * 100:.2f}")
print(f"test accuracy: {test_acc * 100:.2f}")
