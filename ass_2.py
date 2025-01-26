import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
X = pd.read_csv('logisticX.csv', header=None).values
y = pd.read_csv('logisticY.csv', header=None).values.flatten()

# Normalize features
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Sigmoid function
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

# Compute the cost
def compute_cost(features, labels, weights):
    num_samples = len(labels)
    predictions = sigmoid_function(features @ weights)
    cost = -1 / num_samples * np.sum(
        labels * np.log(predictions + 1e-8) + (1 - labels) * np.log(1 - predictions + 1e-8)
    )
    return cost

# Perform gradient descent
def optimize_weights(features, labels, weights, learning_rate, num_iterations):
    num_samples = len(labels)
    cost_history = []
    weight_history = []

    for _ in range(num_iterations):
        predictions = sigmoid_function(features @ weights)
        gradient = features.T @ (predictions - labels) / num_samples
        weights -= learning_rate * gradient

        cost_history.append(compute_cost(features, labels, weights))
        weight_history.append(weights.copy())

    return weights, cost_history, weight_history

# Add bias term to the features
features_with_bias = np.column_stack([np.ones(X.shape[0]), X])

# Initialize weights
weights = np.zeros(features_with_bias.shape[1])

# Hyperparameters
learning_rate = 0.1
num_iterations = 1000

# Train the model
weights, cost_history, weight_history = optimize_weights(
    features_with_bias, y, weights, learning_rate, num_iterations
)

# Output Results
print("Trained Coefficients:")
print(weights)
print(f"Final Cost Value: {cost_history[-1]:.6f}\n")

# Manual Confusion Matrix
def compute_confusion_matrix(actual, predicted):
    true_negative = np.sum((actual == 0) & (predicted == 0))
    false_positive = np.sum((actual == 0) & (predicted == 1))
    false_negative = np.sum((actual == 1) & (predicted == 0))
    true_positive = np.sum((actual == 1) & (predicted == 1))
    return np.array([[true_negative, false_positive], [false_negative, true_positive]])

# Predictions
predicted_labels = (sigmoid_function(features_with_bias @ weights) >= 0.5).astype(int)

# Confusion matrix
conf_matrix = compute_confusion_matrix(y, predicted_labels)
tn, fp, fn, tp = conf_matrix.ravel()

# Metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("Confusion Matrix:")
print(conf_matrix)
print("\nPerformance Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1_score:.4f}")

# Plot Cost vs Iterations
plt.figure(figsize=(8, 5))
plt.plot(range(len(cost_history[:50])), cost_history[:50], color='blue', marker='o', linestyle='-', linewidth=1)
plt.title('Cost Reduction Over Iterations', fontsize=14)
plt.xlabel('Number of Iterations', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('cost_vs_iterations_v2.png')
plt.close()

# Visualize Dataset with Decision Boundary
plt.figure(figsize=(8, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.7, edgecolor='black')
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.7, edgecolor='black')

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
boundary = sigmoid_function(np.column_stack([np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel()]) @ weights)
boundary = boundary.reshape(xx.shape)
plt.contour(xx, yy, boundary, levels=[0.5], colors='black', linestyles='--')

plt.title('Dataset with Decision Boundary', fontsize=14)
plt.xlabel('Feature 1 (Normalized)', fontsize=12)
plt.ylabel('Feature 2 (Normalized)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('decision_boundary_v2.png')
plt.close()

# Extend Dataset with Squared Features
X_squared = np.column_stack([X, X ** 2])
features_squared_with_bias = np.column_stack([np.ones(X_squared.shape[0]), X_squared])

# Train with Squared Features
initial_weights_squared = np.zeros(features_squared_with_bias.shape[1])
weights_squared, cost_squared_history, _ = optimize_weights(
    features_squared_with_bias, y, initial_weights_squared, learning_rate, num_iterations
)

# Plot Decision Boundary for Squared Features
plt.figure(figsize=(8, 5))
plt.scatter(X_squared[y == 0, 0], X_squared[y == 0, 1], color='cyan', label='Class 0', alpha=0.7, edgecolor='black')
plt.scatter(X_squared[y == 1, 0], X_squared[y == 1, 1], color='magenta', label='Class 1', alpha=0.7, edgecolor='black')

boundary_squared = sigmoid_function(
    np.column_stack(
        [np.ones(xx.ravel().shape[0]), xx.ravel(), yy.ravel(), xx.ravel() ** 2, yy.ravel() ** 2]
    ) @ weights_squared
)
boundary_squared = boundary_squared.reshape(xx.shape)
plt.contour(xx, yy, boundary_squared, levels=[0.5], colors='orange', linestyles='--')

plt.title('Squared Features Decision Boundary', fontsize=14)
plt.xlabel('Feature 1 (Normalized)', fontsize=12)
plt.ylabel('Feature 2 (Normalized)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig('squared_features_boundary_v2.png')
plt.close()
