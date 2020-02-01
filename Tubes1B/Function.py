import numpy as np

# Function entropyFunction(data):
# calculate the entropy value of a dataset
def entropyFunction(data):
	value, count = np.unique(data, return_counts=True)
	entropy = 0

	for i in range(len(value)):
		entropy += np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count))])

	return entropy

# Function informationGainFunction(data):
# calculate the information gain value of a dataset
def informationGainFunction(data, attr, target):
	entropy = entropyFunction(data[target])
	values, counts = np.unique(data[attr], return_counts=True)
	gain = 0

	for i in range(len(values)):
		gain = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[attr] == values[i]).dropna()[target])])

	informationGain = entropy - gain

	return informationGain
