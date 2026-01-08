#!/usr/bin/env python3
"""
Cubic Polynomial Regression with Gradient Descent (from scratch)

Model:
    y = w3*x^3 + w2*x^2 + w1*x + b

Data:
    - x in [-2, 2] with step 0.1
    - y generated from a chosen "true" cubic polynomial
    - Add random noise in the range [-10%, +10%] *relative to y*

Outputs:
    - Scatter of noisy data + the fitted polynomial curve
    - Parameter traces over iterations (w3, w2, w1, b)  [4 separate figs]
    - Loss (MSE) over iterations

How to run:
    python3 cubic_polynomial_gd.py

Tips:
- If it does not converge, try NORMALIZE=True or reduce LEARNING_RATE.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Hyperparameters & Options
# -----------------------------
LEARNING_RATE = 1e-3      # works well when features aren't normalized
MAX_EPOCHS    = 3000
TOL           = 1e-10     # early stopping threshold
NORMALIZE     = False      # if True, min-max normalize x and y to [0,1]

# -----------------------------
# 2) Generate Synthetic Data
# -----------------------------
rng = np.random.default_rng(123)

# True coefficients for data generation (choose any values you like)
TRUE_W3, TRUE_W2, TRUE_W1, TRUE_B = 0.8, -0.4, 1.2, -0.3

# x values from -2 to 2 inclusive at step 0.1
x = np.arange(-2.0, 2.0 + 1e-9, 0.1, dtype=float)

def poly_eval(w3, w2, w1, b, x):
    """Compute y = w3*x^3 + w2*x^2 + w1*x + b"""
    return w3*(x**3) + w2*(x**2) + w1*x + b

# noiseless y
y_clean = poly_eval(TRUE_W3, TRUE_W2, TRUE_W1, TRUE_B, x)

# Relative noise in [-10%, +10%] of y value.
# We'll add multiplicative noise: y_noisy = y * (1 + eps), eps ~ U(-0.1, 0.1).
eps = rng.uniform(-0.1, 0.1, size=x.shape)
y_noisy = y_clean * (1.0 + eps)

# Optional normalization (can help if gradients explode or converge slowly).
if NORMALIZE:
    def minmax(a):
        amin, amax = a.min(), a.max()
        if np.isclose(amin, amax):
            return a, amin, amax
        return (a - amin) / (amax - amin), amin, amax

    x_norm, x_min, x_max = minmax(x)
    y_norm, y_min, y_max = minmax(y_noisy)
    x_used = x_norm
    y_used = y_norm
else:
    x_used = x
    y_used = y_noisy

# -----------------------------
# 3) Design Matrix for Cubic Model
#    We can write y_hat = X @ theta, where
#    theta = [w3, w2, w1, b]^T and each row of X is [x^3, x^2, x, 1].
# -----------------------------
def build_design_matrix(xvec):
    return np.column_stack([xvec**3, xvec**2, xvec, np.ones_like(xvec)])

X = build_design_matrix(x_used)

# -----------------------------
# 4) Loss & Gradient
#    Loss: MSE = (1/n) * ||X @ theta - y||^2
#    Gradient: (2/n) * X^T (X @ theta - y)
# -----------------------------
def mse_loss(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def predict(theta, X):
    return X @ theta

# -----------------------------
# 5) Gradient Descent Loop
# -----------------------------
n, d = X.shape
theta = np.zeros(d)  # [w3, w2, w1, b]

loss_history = []
theta_history = []  # will store copies of theta each iteration

prev_loss = np.inf

for epoch in range(1, MAX_EPOCHS + 1):
    # Forward: predictions and loss
    y_hat = predict(theta, X)
    loss = mse_loss(y_used, y_hat)

    loss_history.append(loss)
    theta_history.append(theta.copy())

    if abs(prev_loss - loss) < TOL:
        print(f"Early stopping at epoch {epoch}, loss={loss:.10f}")
        break
    prev_loss = loss

    # Gradient for MSE wrt theta
    grad = (2.0 / n) * (X.T @ (y_hat - y_used))

    # Parameter update
    theta -= LEARNING_RATE * grad

w3, w2, w1, b = theta
print(f"Final parameters: w3={w3:.6f}, w2={w2:.6f}, w1={w1:.6f}, b={b:.6f}")
print(f"Final MSE: {loss_history[-1]:.6f}  (epochs: {len(loss_history)})")

# -----------------------------
# 6) PLOTS
#    (a) Data + fitted curve
#    (b-e) Parameter traces (w3, w2, w1, b)
#    (f) Loss
# -----------------------------

# (a) Data + fitted curve
plt.figure()
plt.title("Cubic Polynomial: Data (noisy) & Fitted Curve")
plt.xlabel("x (used)")
plt.ylabel("y (used)")
plt.scatter(x_used, y_used, label="noisy data", s=12)
# Smooth curve for the fitted polynomial
x_curve = np.linspace(x_used.min(), x_used.max(), 400)
X_curve = build_design_matrix(x_curve)
y_curve = predict(theta, X_curve)
plt.plot(x_curve, y_curve, label="fitted cubic curve")
plt.legend()
plt.tight_layout()
plt.savefig("cubic_plot_data_and_curve.png", dpi=150)

# Unpack parameter histories
theta_hist_arr = np.array(theta_history)  # shape: (T, 4)
iters = range(1, theta_hist_arr.shape[0] + 1)

# (b) w3 over iterations
plt.figure()
plt.title("w3 over Iterations")
plt.xlabel("Iteration")
plt.ylabel("w3")
plt.plot(iters, theta_hist_arr[:, 0])
plt.tight_layout()
plt.savefig("cubic_plot_w3_over_time.png", dpi=150)

# (c) w2 over iterations
plt.figure()
plt.title("w2 over Iterations")
plt.xlabel("Iteration")
plt.ylabel("w2")
plt.plot(iters, theta_hist_arr[:, 1])
plt.tight_layout()
plt.savefig("cubic_plot_w2_over_time.png", dpi=150)

# (d) w1 over iterations
plt.figure()
plt.title("w1 over Iterations")
plt.xlabel("Iteration")
plt.ylabel("w1")
plt.plot(iters, theta_hist_arr[:, 2])
plt.tight_layout()
plt.savefig("cubic_plot_w1_over_time.png", dpi=150)

# (e) b over iterations
plt.figure()
plt.title("b over Iterations")
plt.xlabel("Iteration")
plt.ylabel("b")
plt.plot(iters, theta_hist_arr[:, 3])
plt.tight_layout()
plt.savefig("cubic_plot_b_over_time.png", dpi=150)

# (f) Loss over iterations
plt.figure()
plt.title("Loss (MSE) over Iterations")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.plot(range(1, len(loss_history) + 1), loss_history)
plt.tight_layout()
plt.savefig("cubic_plot_loss_over_time.png", dpi=150)

print("Saved figures:")
print("  - cubic_plot_data_and_curve.png")
print("  - cubic_plot_w3_over_time.png")
print("  - cubic_plot_w2_over_time.png")
print("  - cubic_plot_w1_over_time.png")
print("  - cubic_plot_b_over_time.png")
print("  - cubic_plot_loss_over_time.png")
