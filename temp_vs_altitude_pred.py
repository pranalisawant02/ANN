#Que 2

import numpy as np
import matplotlib.pyplot as plt

x = np.array([100, 200, 300, 400, 500])#altitudes
y = np.array([50, 40, 30, 20, 10])#temperature

x = (x - np.mean(x)) / np.std(x)

m = 0  # initial guess for weight
b = 0  # initial guess for bias
learning_rate = 0.1
num_iterations = 100

# Gradient Descent
for i in range(num_iterations):
    # Predicted output
    y_pred = w * x + b

    error = y_pred - y

    dw = (2/len(x)) * np.sum(error * x)
    db = (2/len(x)) * np.sum(error)


    w = w - learning_rate * dw
    b = b - learning_rate * db


    if i % 100 == 0:
        mse = np.mean(error**2)
        print(f"Iteration {i}: w = {w:.4f}, b = {b:.4f}, MSE = {mse:.4f}")

        # Predict exam scores based on study hours
def predict(altitudes):
    return m * altitudes + b

# Plot the result
plt.scatter(x * np.std(x) + np.mean(x), y, color='blue', label='True values')
plt.plot(x * np.std(x) + np.mean(x), w * x + b, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Temperature prediction with altitudes')
plt.legend()
plt.show()


print(f"Final weight (w) = {w:.4f}, Final bias (b) = {b:.4f}")

# Example prediction
altitudes = 500
predicted_temp = predict(altitudes)
print(f"Predicted temperature for altitudes {altitudes}  : {predicted_temp:.2f}")
