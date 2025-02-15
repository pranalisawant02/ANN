#Q1: Student Performance Prediction:-
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset (hours studied vs. exam scores)
# Example dataset: Study hours (x) and corresponding exam scores (y)
x = np.array([1, 2, 3, 4, 5,6,7,8,9])  # Study hours
y = np.array([20, 30, 40, 50, 60,70,80,90,100])  # Exam scores

# Initialize parameters
m = 0  # Slope
b = 0.1  # Intercept
learning_rate = 0.01
iterations = 500  # Number of iterations

# Number of data points
n = len(x)

# Gradient Descent implementation
for iterations in range(iterations):
    # Predictions
    y_pred = m * x + b

    # Calculate gradients
    cost = (1/n) * np.sum((y - y_pred)**2)
    dm = -(2 / n) * np.sum(x * (y - y_pred))  # Partial derivative with respect to m
    db = -(2 / n) * np.sum(y - y_pred)       # Partial derivative with respect to b

    # Update parameters
    m -= learning_rate * dm
    b -= learning_rate * db
    print(f"Iterations {iterations} m :{m} b :{b} cost :{cost} ")

# Final parameters
print(f"Final slope (m): {m:.2f}")
print(f"Final intercept (b): {b:.2f}")
print(f"Final cost (cost) : {cost:.2f}")

# Predict exam scores based on study hours
def predict(hours):
    return m * hours + b

# Visualize the dataset and the best-fit line
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, predict(x), color='red', label='Best-fit Line')
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Student Performance Prediction')
plt.legend()
plt.show()

# Example prediction
study_hours = 7
predicted_score = predict(study_hours)
print(f"Predicted exam score for studying {study_hours} hours daily: {predicted_score:.2f}")
