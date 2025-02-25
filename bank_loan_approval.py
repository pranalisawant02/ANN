import numpy as np

# Sample data for loan approval prediction
# Features: [credit_score, annual_income, past_defaults]
X = np.array([[650, 50000, 0],   # Sample 1
              [700, 60000, 0],   # Sample 2
              [750, 75000, 0],   # Sample 3
              [800, 80000, 1],   # Sample 4
              [620, 45000, 1]])  # Sample 5

# Target output (1 for approved, 0 for denied)
y = np.array([0, 1, 1, 1, 0])

# Initialize weights and bias
w = np.random.rand(3)  # 3 features, so 3 weights
b = np.random.rand(1)  # 1 bias term

# Set learning rate and iterations
learning_rate = 0.0001
iterations = 1000

# Sigmoid function with overflow handling
def sigmoid(x):
    # Clip values to avoid overflow
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Train the model
for i in range(iterations):
    # Calculate the predicted values (z = Xw + b)
    z = np.dot(X, w) + b
    
    # Apply the sigmoid activation function to get probabilities
    y_predicted = sigmoid(z)
    
    # Error is the difference between predicted and actual values
    error = y_predicted - y
    
    # Calculate gradients for weights and bias
    dw = (2 / len(X)) * np.dot(X.T, error)  # Gradient for weights
    db = (2 / len(X)) * np.sum(error)       # Gradient for bias
    
    # Update weights and bias using gradient descent
    w -= learning_rate * dw
    b -= learning_rate * db

# Print final output
print(f"Final weights: {w}")
print(f"Final bias: {b}")
print(f"Final error: {error}")
print(f"Final predicted values: {y_predicted}")
print(f"Actual output: {y}")
