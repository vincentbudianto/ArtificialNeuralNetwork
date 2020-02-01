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

# Data splitter (splits data)
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

# Splitting header... Get attributes and 

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


# Create a basic fitying algorithn
def fit(dataX, dataY):
    decisionTree = DecisionTree()
    
    currentEntropy = f.entropyFunction(dataY)
    print(currentEntropy)
    
    # Simple looping to find root algorithm


    
	

dataHead, data = getCSVData("tennis.csv")
newX, newY = splitXY(dataHead, data)
newY, classDictionary = translateY(newY)
fit(newX, newY)
