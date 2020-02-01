# Main program to fit the data read

import pandas as pd
import numpy as np
import Function as f
from DecisionTree import DecisionTree

# Get data from csv
def getCSVData(fileName):
    data = pd.read_csv(fileName)
    dataHead = list(data.columns)
    data = np.array(data)
    return (np.array(dataHead), data)


# Splitters and translaters

# Data splitter (splits X and Y)
def splitXY(dataHead, data):
    newX = []
    newY = []
    size = len(dataHead)
    for i in range(len(data)):
        tempArray = []
        for j in range(size - 1):
            tempArray.append(data[i, j])
        newX.append(tempArray)
        newY.append(data[i, size - 1])
    return (np.array(newX), np.array(newY))

# Coding the Y axis
# Returns a dictionary of Y and translated value of Y
# Dictionary index represents the number while the value of the index represents the number mapping
def translateY(dataTarget):
    classDictionary = []
    for i in range(len(dataTarget)):
        if dataTarget[i] not in classDictionary:
            classDictionary.append(dataTarget[i])
            dataTarget[i] = len(classDictionary) - 1
        else:
            dataTarget[i] = classDictionary.index(dataTarget[i])
    return (dataTarget, np.array(classDictionary))

# Data splitter (splits data based on X attribute)
def dataSplitter(dataX, dataY):
    result = DecisionTree()

    # Checking the number of data in Y and the variety
    tempY = []
    tempYCounter = []
    for i in range(len(dataY)):
        if dataY[i] not in tempDictionary:
            tempY.append(dataY[i])
            tempYCounter.append(1)
        else:
            tempCounter.index(dataY[i]) += 1
    
    # If all examples are negative, Return the single-node tree Root, with label = +
    # If all examples are negative, Return the single-node tree Root, with label = -
    if (len(tempY) == 1):
        result.setRootValue(tempY[0])
        return result
    
    # If number of predicting attributes is empty, then Return the single node tree Root,
    # with label = most common value of the target attribute in the examples.
    if len(dataX) == 0:
        # Count most common value
        maxIdx = tempYCounter.index(max(tempYCounter))
        result.setRootValue(dataY[maxIdx])
        return result

    
    # Otherwise...
    # Splits into a couple of attributes
    for i in range(len(dataX)):
        # 1. List every attribute in temporary dictionary
        tempDictionary = []
        for j in range(len(dataX[i])):
            if dataX[i, j] not in tempDictionary:
                tempDictionary.append(dataX[i, j])

        # 2. Transform the X equipment into numbers
    




    # Sp
    return result



# Create a basic fitying algorithn
def fit(dataX, dataY):
    decisionTree = DecisionTree()
    
    # Checking current entropy
    currentEntropy = f.entropyFunction(dataY)
    print(currentEntropy)
    
    # Finding nodes that is most effective


# Node splitter function
def attributeSplitter(dataX):
    print("test")
    for i in range(len(dataX)):
        print("test_too")




    
	

# dataHead, data = getCSVData("tennis.csv")
# newX, newY = splitXY(dataHead, data)
# newY, classDictionary = translateY(newY)
# fit(newX, newY)
