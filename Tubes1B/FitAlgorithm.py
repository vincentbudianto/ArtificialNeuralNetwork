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
    return (dataHead, data)

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
    return (newX, newY)

# Coding the Y axis
# Returns a dictionary of Y and translated value of Y
# Dictionary index represents the number while the value of the index represents the number mapping
def translateY(dataTarget):
    dic = []
    for i in range(len(dataTarget)):
        if dataTarget[i] not in dic:
            dic.append(dataTarget[i])
            dataTarget[i] = len(dic) - 1
        else:
            dataTarget[i] = dic.index(dataTarget[i])
    return (dataTarget, dic)
        


# Parsing ke bentuk number
def numberParser(data):
    print("test")
    

# Create a basic fitying algorithn
def fit(data):
    decisionTree = DecisionTree()
	

dataHead, data = getCSVData("tennis.csv")
newX, newY = splitXY(dataHead, data)
# print(newY)
newY, dic = translateY(newY)
print(newY)
print(dic)
