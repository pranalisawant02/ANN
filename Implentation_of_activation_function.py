#Exp_3

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
