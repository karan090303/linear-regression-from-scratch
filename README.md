# Linear Regression from Scratch

## Objective
Implement Linear Regression using NumPy and Gradient
Descent without using a machine-learning library for
the model implementation.

## Tools
- Python
- NumPy: predictions, cost, gradients, and metrics
- Pandas: loading and cleaning data
- Matplotlib: cost-convergence visualization
- scikit-learn: train/test split and final baseline comparison

## Mathematical approach

### Hypothesis
The model predicts a continuous value using:

y_hat = X @ theta

### Cost function
J(theta) = (1 / 2m) * sum((X @ theta - y)^2)

The objective is to find parameters that minimize the cost.

### Gradient derivation
Let e = X @ theta - y.

Differentiating the cost with respect to theta gives:

gradient = (1 / m) * X.T @ e

### Gradient descent update
theta = theta - learning_rate * gradient

This update repeats for a fixed number of iterations.
The cost is stored at every iteration to monitor convergence.

## Data preprocessing
1. Split the dataset into training and testing sets.
2. Fill missing feature values using training-set medians.
3. Standardize features using training-set means and standard deviations.
4. Add an intercept column.

The test set does not determine preprocessing statistics.

## Evaluation
Calculate test Mean Squared Error and R-squared manually.
Compare results with scikit-learn LinearRegression
using the same data split.

## How to run
pip install -r requirements.txt
python linear_regression.py

## Output
- Cost-convergence plot
- Test MSE and R-squared
- Coefficients from both models
- Final performance comparison
-