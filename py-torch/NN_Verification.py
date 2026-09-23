import numpy as np

# --- 1. ACTIVATION & LOSS FUNCTIONS ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    return a * (1 - a)

# --- 2. INPUT DATA & TARGETS ---
X = np.array([[1.0, 2.0]])     # Shape: (1, 2)
y = np.array([[1.0]])          # Shape: (1, 1)

# --- 3. WEIGHTS & BIASES ---
W1 = np.array([[0.5, 0.9], 
               [0.7, 0.4]])    # Shape: (2, 2)
b1 = np.array([[0.0, 0.0]])

W2 = np.array([[0.5], 
               [0.6]])         # Shape: (2, 1)
b2 = np.array([[0.0]])

# --- 4. FORWARD PASS ---
# Layer 1 (Hidden)
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)

# Layer 2 (Output)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)

output = a2
loss = 0.5 * np.sum((output - y) ** 2)

# --- 5. BACKWARD PASS ---
# Output Layer Gradient
dL_da2 = output - y
da2_dz2 = sigmoid_derivative(a2)
dz2 = dL_da2 * da2_dz2

dW2 = np.dot(a1.T, dz2)
db2 = np.sum(dz2, axis=0, keepdims=True)

# Hidden Layer 1 Gradient
da1 = np.dot(dz2, W2.T)
dz1 = da1 * sigmoid_derivative(a1)

dW1 = np.dot(X.T, dz1)
db1 = np.sum(dz1, axis=0, keepdims=True)

# --- 6. PRINT RESULTS ---
print("--- FORWARD PASS ---")
print("z1:\n", z1)
print("a1:\n", a1)
print("z2:\n", z2)
print("a2 / Output:\n", output)
print("Loss:\n", loss)

print("\n--- BACKWARD PASS (GRADIENTS) ---")
print("dW2:\n", dW2)
print("db2:\n", db2)
print("dW1:\n", dW1)
print("db1:\n", db1)