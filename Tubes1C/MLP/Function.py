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
