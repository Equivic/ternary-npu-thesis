import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


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


model = MNISTNet()
model.load_state_dict(torch.load("trained_model.pt"))
model.eval()

w1 = model.fc1.weight.detach().numpy()

# --- Full weight matrix heatmap ---
fig1, ax1 = plt.subplots(figsize=(10, 6))
im = ax1.imshow(w1, cmap="RdBu", aspect="auto", vmin=-abs(w1).max(), vmax=abs(w1).max())
ax1.set_xlabel("input pixel (784)")
ax1.set_ylabel("hidden neuron (128)")
ax1.set_title("fc1 weight matrix")
plt.colorbar(im, ax=ax1)
plt.tight_layout()
plt.savefig("weight_heatmap.png")

# --- All 128 neurons as reshaped 28x28 receptive fields ---
fig2, axes = plt.subplots(8, 16, figsize=(20, 10))
for ax, idx in zip(axes.flat, range(128)):
    ax.imshow(w1[idx].reshape(28, 28), cmap="RdBu")
    ax.axis("off")
plt.tight_layout()
plt.savefig("neuron_grid_all.png", dpi=150)

plt.show()
