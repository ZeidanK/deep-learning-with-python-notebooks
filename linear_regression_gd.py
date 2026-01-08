#!/usr/bin/env python3
"""
Linear Regression with Gradient Descent (from scratch)

Produces 4 graphs required by class:
1) Scatter of the 6 input points + fitted regression line.
2) w (weight) over iterations.
3) b (bias) over iterations.
4) Loss (MSE) over iterations.

How to run:
    python3 linear_regression_gd.py

Notes:
- No sklearn/pytorch/etc. — everything is implemented manually.
- Adjust the LEARNING_RATE and MAX_EPOCHS below if needed.
- If convergence is slow/unstable, try reducing the learning rate.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Hyperparameters & Settings
# -----------------------------
LEARNING_RATE = 0.05     # size of the update step for w,b
MAX_EPOCHS    = 1000     # max number of iterations
TOL           = 1e-8     # early stopping threshold on |loss change|

# -----------------------------
# 2) Example Input: 6 Points
#    Replace these with your own input loader as needed.
# -----------------------------
# (x, y) pairs — for demo we choose a line with noise.
rng = np.random.default_rng(42)
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])          # 6 x-values
true_w, true_b = 1.7, -0.8
y = true_w * x + true_b + rng.normal(0, 0.5, size=x.shape)  # noisy y-values

# Ensure shapes are 1D numpy arrays
x = x.astype(float).ravel()
y = y.astype(float).ravel()

# -----------------------------
# 3) Model: y_hat = w*x + b
#    Loss: MSE = (1/n) * sum (y_hat - y)^2
#    Gradients:
#       dL/dw = (2/n) * sum( (y_hat - y) * x )
#       dL/db = (2/n) * sum( y_hat - y )
# -----------------------------
def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def predict(w, b, x):
    return w * x + b

# -----------------------------
# 4) Training Loop (Gradient Descent)
# -----------------------------
w = 0.0  # initialize weight
b = 0.0  # initialize bias

loss_history = []
w_history = []
b_history = []

prev_loss = np.inf

for epoch in range(1, MAX_EPOCHS + 1):
    # Forward pass: compute predictions
    y_hat = predict(w, b, x)
    # Compute loss
    loss = mse_loss(y, y_hat)

    # Book-keeping for plots
    loss_history.append(loss)
    w_history.append(w)
    b_history.append(b)

    # Early stopping if loss barely changes
    if abs(prev_loss - loss) < TOL:
        print(f"Early stopping at epoch {epoch}, loss={loss:.6f}")
        break
    prev_loss = loss

    # Gradients
    n = x.shape[0]
    grad_w = (2.0 / n) * np.sum((y_hat - y) * x)
    grad_b = (2.0 / n) * np.sum(y_hat - y)

    # Parameter update
    w -= LEARNING_RATE * grad_w
    b -= LEARNING_RATE * grad_b

print(f"Final parameters: w={w:.6f}, b={b:.6f}")
print(f"Final MSE: {loss_history[-1]:.6f}  (epochs: {len(loss_history)})")

# -----------------------------
# 5) PLOTS (4 figures, as requested in class)
# -----------------------------

# (1) Scatter of the 6 input points + fitted line
plt.figure()
plt.title("Linear Regression: Data & Fitted Line")
plt.xlabel("x")
plt.ylabel("y")
plt.scatter(x, y, label="data (6 points)")
# Draw line across a nice range
x_line = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
y_line = predict(w, b, x_line)
plt.plot(x_line, y_line, label=f"fitted line: y={w:.2f}x+{b:.2f}")
plt.legend()
plt.tight_layout()
plt.savefig("linear_plot_data_and_line.png", dpi=150)

# (2) w over iterations
plt.figure()
plt.title("Parameter w over Iterations")
plt.xlabel("Iteration")
plt.ylabel("w")
plt.plot(range(1, len(w_history) + 1), w_history)
plt.tight_layout()
plt.savefig("linear_plot_w_over_time.png", dpi=150)

# (3) b over iterations
plt.figure()
plt.title("Parameter b over Iterations")
plt.xlabel("Iteration")
plt.ylabel("b")
plt.plot(range(1, len(b_history) + 1), b_history)
plt.tight_layout()
plt.savefig("linear_plot_b_over_time.png", dpi=150)

# (4) Loss over iterations
plt.figure()
plt.title("Loss (MSE) over Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.plot(range(1, len(loss_history) + 1), loss_history)
plt.tight_layout()
plt.savefig("linear_plot_loss_over_time.png", dpi=150)

print("Saved 4 figures:")
print("  - linear_plot_data_and_line.png")
print("  - linear_plot_w_over_time.png")
print("  - linear_plot_b_over_time.png")
print("  - linear_plot_loss_over_time.png")
