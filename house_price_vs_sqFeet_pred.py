#Q3: House Price Prediction:-
import numpy as np
import matplotlib.pyplot as plt

x = np.array([10,20,30,40,50])#sq_feet
y = np.array([20,40,60,80,100])#house price

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
def predict(sq_feet):
    return m * sq_feet + b

# Plot the result
plt.scatter(x * np.std(x) + np.mean(x), y, color='blue', label='True values')
plt.plot(x * np.std(x) + np.mean(x), w * x + b, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('House Price Prediction')
plt.legend()
plt.show()


print(f"Final weight (w) = {w:.4f}, Final bias (b) = {b:.4f}")

# Example prediction
sq_feet = 50
predicted_house_price = predict(sq_feet)
print(f"Predicted house_price for sq_feet {sq_feet}  : {predicted_house_price:.2f}")
