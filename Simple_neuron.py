#implementing simple neuron


input1 = 2
input2 = 3
w1= 0.5
w2= 0.8
b = 0.1

sum= (w1*input1)+(w2*input2)+b
print("weightedsum:",sum)

def step_function(x, threshold=0):
    return 1 if x > threshold else 0

output= step_function(sum)

print("output",output)

#simple neuron
#using AND gate
def artificial_neuron(inputs,weights,bias):
    #calculating weighted sum
    weighted_sum=0
    for i in range(len(inputs)):
        weighted_sum+=inputs[i]*weights[i]
    weighted_sum+=bias
    output=1 if  weighted_sum>0 else 0
    return output
inputs=[(0,0),(0,1),(1,0),(1,1)]
weights=[1,1]
bias=-1

for input_combination in inputs:
    output = artificial_neuron(input_combination, weights, bias)
    print(f"Input: {input_combination}, Output: {output}")

#output= artificial_neuron(inputs,weights,bias)
#print("output is :",output)

#simple neuron
#using OR gate
def artificial_neuron(inputs,weights,bias):
    #calculating weighted sum
    weighted_sum=0
    for i in range(len(inputs)):
        weighted_sum+=inputs[i]*weights[i]
    weighted_sum+=bias
    output=1 if  weighted_sum>0 else 0
    return output
inputs=[(0,0),(0,1),(1,0),(1,1)]
weights=[2,2]
bias=-1

for input_combination in inputs:
    output = artificial_neuron(input_combination, weights, bias)
    print(f"Input: {input_combination}, Output: {output}")

#output= artificial_neuron(inputs,weights,bias)
#print("output is :",output)

#simple neuron
#using NAND gate
def artificial_neuron(inputs,weights,bias):
    #calculating weighted sum
    weighted_sum=0
    for i in range(len(inputs)):
        weighted_sum+=inputs[i]*weights[i]
    weighted_sum+=bias
    output=1 if  weighted_sum>0 else 0
    return output
inputs=[(0,0),(0,1),(1,0),(1,1)]
weights=[-1,-1]
bias=2

for input_combination in inputs:
    output = artificial_neuron(input_combination, weights, bias)
    print(f"Input: {input_combination}, Output: {output}")

#output= artificial_neuron(inputs,weights,bias)
#print("output is :",output)

#simple neuron
#using NOR gate
def artificial_neuron(inputs,weights,bias):
    #calculating weighted sum
    weighted_sum=0
    for i in range(len(inputs)):
        weighted_sum+=inputs[i]*weights[i]
    weighted_sum+=bias
    output=1 if  weighted_sum>0 else 0
    return output
inputs=[(0,0),(0,1),(1,0),(1,1)]
weights=[-2,-2]
bias=1

for input_combination in inputs:
    output = artificial_neuron(input_combination, weights, bias)
    print(f"Input: {input_combination}, Output: {output}")

#output= artificial_neuron(inputs,weights,bias)
#print("output is :",output)

# Simple neuron using NOT gate
def artificial_neuron(inputs, weights, bias):
    # Calculating weighted sum
    weighted_sum = 0
    for i in range(len(inputs)):
        weighted_sum += inputs[i] * weights[i]
    weighted_sum += bias
    output = 1 if weighted_sum > 0 else 0
    return output

inputs = [0, 1]
weights = [-3]
bias = 2

for input_value in inputs:
    output = artificial_neuron([input_value], weights, bias)
    print(f"Input: {input_value}, Output: {output}")







