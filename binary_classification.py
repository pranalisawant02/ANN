def artificial_neuron(inputs, weights, bias):
    # Calculate the weighted sum (similar to your current code)
    weighted_sum = 0
    for i in range(len(inputs)):
        weighted_sum += inputs[i] * weights[i]
    weighted_sum += bias
    
    # Apply sigmoid activation (used for binary classification)
    output = 1 / (1 + np.exp(-weighted_sum))  # Sigmoid activation
    return output

# For binary image classification, example inputs might be pixel values of a processed image
inputs = [0.5, 0.2, 0.8, 0.3]  # These would be the processed pixel values or feature vectors
weights = [0.6, -0.8, 0.4, 0.1]  # These would be learned during training
bias = -0.1  # Bias also learned during training

# Run through the artificial neuron for one image
output = artificial_neuron(inputs, weights, bias)
if output >= 0.5:
    print("The image contains a cat!")
else:
    print("The image does not contain a cat!")
