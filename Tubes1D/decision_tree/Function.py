import numpy as np

# Function entropyFunction(data):
# calculate the entropy value of a dataset
def entropyFunction(data):
	value, count = np.unique(data, return_counts=True)
	entropy = 0

	for i in range(len(value)):
		entropy += np.sum([(-count[i] / np.sum(count)) * np.log2(count[i] / np.sum(count))])

	return entropy

# Funcion splitInInformation(data, index):
# Calculate the value of splitInInformation of certain index in dataset
# Total length = length of the total data uncategorized
def splitInInformation(splittedClassContainer, totalLength):
	splitInformation = 0

	for i in range(len(splittedClassContainer)):
		if (len(splittedClassContainer[i]) != 0):
			splitInformation += (-len(splittedClassContainer[i]) / totalLength) * np.log2(len(splittedClassContainer[i]) / totalLength)
	return splitInformation

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
