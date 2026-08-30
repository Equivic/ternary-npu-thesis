import numpy as np

# ---- Dataset: XOR, all 4 possible input combinations ----
# X shape: (4 examples, 2 features)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# y shape: (4 examples, 1 output) -- reshape so it's a column, matches network output shape
y = np.array([[0], [0], [0], [1]])

# ---- Network shape ----
n_input = 2
n_hidden = 2
n_output = 1

# ---- Weight initialization ----
# Why random, not zero? Think about it: if ALL weights start at exactly 0,
# every hidden neuron computes the exact same z, the exact same gradient,
# and updates identically forever -- they'd never differentiate from each other.
# Small random values break that symmetry.
np.random.seed(7)  # fixed seed so results are reproducible while you debug
W1 = np.random.randn(n_input, n_hidden) * 0.5  # weights: input -> hidden
b1 = np.zeros((1, n_hidden))  # biases for hidden layer
W2 = np.random.randn(n_hidden, n_output) * 0.5  # weights: hidden -> output
b2 = np.zeros((1, n_output))  # bias for output layer


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(a):
    return a * (1 - a)


def forward(X):
    """
    Run one forward pass for ALL 4 examples at once (that's what the matrix
    shapes buy you -- no python for-loop over examples needed).
    """
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2


def compute_loss(y_true, y_pred):
    """
    Binary cross-entropy, averaged over all 4 examples.
    """
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def backward(X, y, z1, a1, z2, a2):
    """
    Backprop through both layers, averaged over the batch of 4 examples.
    """
    m = X.shape[0]

    dz2 = a2 - y
    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = dz2 @ W2.T
    dz1 = da1 * sigmoid_derivative(a1)
    dW1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


# ---- Training loop skeleton ----
learning_rate = 0.25
epochs = 5000

for epoch in range(epochs):
    z1, a1, z2, a2 = forward(X)
    loss = compute_loss(y, a2)
    dW1, db1, dW2, db2 = backward(X, y, z1, a1, z2, a2)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    if epoch % 500 == 0:
        print(f"epoch {epoch}, loss {loss:.4f}")

# ---- Final check ----
_, _, _, final_pred = forward(X)
print("\nFinal predictions vs targets:")
for i in range(4):
    print(f"{X[i]} -> predicted {final_pred[i][0]:.4f}, target {y[i][0]}")
