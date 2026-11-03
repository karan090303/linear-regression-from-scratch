import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# 1. Load data with Pandas
df = pd.read_csv("data/Housing.csv")

target_column = "medv"

X_df = df.drop(columns=[target_column])
y = df[target_column].to_numpy(dtype=float)

# Convert feature columns to numeric values.
X_df = X_df.apply(pd.to_numeric, errors="coerce")

# Split before learning preprocessing values.
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=0.2, random_state=42
)

# 2. Handle missing values using training medians only.
medians = X_train_df.median()
X_train_df = X_train_df.fillna(medians)
X_test_df = X_test_df.fillna(medians)

# Remove any feature that has no variation in training data.
std_check = X_train_df.std(ddof=0)
valid_columns = std_check[std_check > 0].index

X_train_df = X_train_df[valid_columns]
X_test_df = X_test_df[valid_columns]

# 3. Standardize using training-set statistics.
means = X_train_df.mean()
scales = X_train_df.std(ddof=0)

X_train = (
    (X_train_df - means) / scales
).to_numpy(dtype=float)

X_test = (
    (X_test_df - means) / scales
).to_numpy(dtype=float)

# Add the intercept column (bias).
X_train = np.column_stack(
    [np.ones(X_train.shape[0]), X_train]
)
X_test = np.column_stack(
    [np.ones(X_test.shape[0]), X_test]
)


# 4. Hypothesis / prediction function
def predict(X, theta):
    return X @ theta


# 5. Cost function: half mean squared error
def compute_cost(X, y, theta):
    errors = predict(X, theta) - y
    return np.sum(errors ** 2) / (2 * len(y))


# 6. Gradient descent
def gradient_descent(X, y, learning_rate=0.03, iterations=5000):
    m, n = X.shape
    theta = np.zeros(n)
    cost_history = []

    for iteration in range(iterations):
        errors = predict(X, theta) - y

        # Gradient of the cost with respect to every parameter.
        gradient = (X.T @ errors) / m

        # Update all parameters together.
        theta = theta - learning_rate * gradient

        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history


# 7. Train our own model
theta, cost_history = gradient_descent(
    X_train, y_train,
    learning_rate=0.03,
    iterations=5000
)

# 8. Evaluate manually on unseen test data
predictions = predict(X_test, theta)

mse = np.mean((y_test - predictions) ** 2)

ss_res = np.sum((y_test - predictions) ** 2)
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_total)

print("FROM-SCRATCH MODEL")
print("Initial cost:", round(cost_history[0], 4))
print("Final cost:", round(cost_history[-1], 4))
print("Test MSE:", round(mse, 4))
print("Test R²:", round(r2, 4))


# 9. Plot convergence
plt.plot(cost_history)
plt.xlabel("Iteration")
plt.ylabel("Cost (half MSE)")
plt.title("Gradient Descent Cost Convergence")
plt.grid(True)
plt.tight_layout()
plt.savefig("cost_convergence.png")
plt.show()


# 10. Final sanity-check using scikit-learn
# Same training/test split and same standardized features.
baseline = LinearRegression()
baseline.fit(X_train[:, 1:], y_train)

sk_predictions = baseline.predict(X_test[:, 1:])
sk_mse = np.mean((y_test - sk_predictions) ** 2)

sk_ss_res = np.sum((y_test - sk_predictions) ** 2)
sk_r2 = 1 - (sk_ss_res / ss_total)

print("\nSCIKIT-LEARN COMPARISON")
print("From-scratch MSE:", round(mse, 4))
print("Scikit-learn MSE:", round(sk_mse, 4))
print("From-scratch R²:", round(r2, 4))
print("Scikit-learn R²:", round(sk_r2, 4))

print("\nFrom-scratch coefficients:")
print(theta)

print("\nScikit-learn coefficients:")
print("Bias:", baseline.intercept_)
print(baseline.coef_)