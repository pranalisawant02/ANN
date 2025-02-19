#Application 1

import numpy as np
import matplotlib.pyplot as plt

# Step 2: Define activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

# Step 3: Generate input values
x = np.linspace(-5, 5, 100)

# Step 4: Compute activation outputs
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

# Step 5: Plot the activation functions
plt.figure(figsize=(10, 6))

plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
plt.plot(x, y_relu, label='ReLU', color='red')
plt.plot(x, y_tanh, label='Tanh', color='green')

plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.legend()
plt.title("Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid()
plt.show()

# Step 6: Analyze properties and impact
print("Sigmoid: Smooth, bounded between (0,1), suffers from vanishing gradients.")
print("ReLU: Zero for negative values, unbounded for positive values, avoids vanishing gradients but has dead neurons issue.")
print("Tanh: Smooth, bounded between (-1,1), stronger gradients than sigmoid.")

# Student Performance Prediction using a Single-Layer Neural Network
# Step 1: Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Generate synthetic dataset
np.random.seed(0)
study_hours = np.random.uniform(1, 10, 100)
exam_scores = 5 * study_hours + np.random.normal(0, 5, 100)  # Linear relation with noise

# Normalize data
study_hours = (study_hours - np.mean(study_hours)) / np.std(study_hours)
exam_scores = (exam_scores - np.mean(exam_scores)) / np.std(exam_scores)

# Step 3: Initialize weights and bias
W = np.random.randn()
b = np.random.randn()
learning_rate = 0.01
epochs = 1000

# Activation function choice (Change between sigmoid, relu, and tanh)
def activation(x, func='sigmoid'):
    if func == 'sigmoid':
        return sigmoid(x)
    elif func == 'relu':
        return relu(x)
    elif func == 'tanh':
        return tanh(x)
    else:
        raise ValueError("Unsupported activation function")

# Step 4: Train the model
losses = []
for epoch in range(epochs):
    y_pred = activation(W * study_hours + b, func='sigmoid')
    loss = np.mean((y_pred - exam_scores) ** 2)
    losses.append(loss)

    # Compute gradients
    dW = np.mean(2 * (y_pred - exam_scores) * study_hours)
    db = np.mean(2 * (y_pred - exam_scores))

    # Update weights
    W -= learning_rate * dW
    b -= learning_rate * db

# Step 5: Plot training loss
plt.figure(figsize=(8, 5))
plt.plot(range(epochs), losses, label='Loss over epochs', color='purple')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.grid()
plt.show()

# Step 6: Analyze results
print(f"Final Weights: W={W:.4f}, b={b:.4f}")
