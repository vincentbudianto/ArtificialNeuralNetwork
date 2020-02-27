import math
# import numpy as np

# Activation function
def sigmoid(x):
	return (1 / (1 + math.exp(-x)))

# Check final result
def finalResult(x):
	if (x > 0.5):
		return 1
	else:
		return 0

# Mean Squared Error function
def mse(expected, reality):
	return ((0.5 * sum(((expected - reality) ** 2))), (reality - expected))

# Cross-entropy calculation per output
def crossentropyCount(expected, reality):
	return (reality * math.log10(expected) + (1 - reality) * math.log10(1 - expected))

# Count the error (primitive way)
def errorCount(expected, reality):
	return math.pow((reality - expected), 2)
