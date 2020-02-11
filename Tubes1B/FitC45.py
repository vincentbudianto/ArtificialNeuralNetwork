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

# Splitters and translaters
# Data splitter (splits X and Y (attributes and target attributes))
def splitXY(data):
    newX = []
    newY = []
    for datum in data:
        newX.append(datum[:-1])
        newY.append(datum[-1])
    return (newX, newY)

# Translates target values
# Save the value in classDictionary and codes the value
def translateY(dataTarget):
    classDictionary = []
    for i in range(len(dataTarget)):
        if dataTarget[i] not in classDictionary:
            classDictionary.append(dataTarget[i])
            dataTarget[i] = len(classDictionary) - 1
        else:
            dataTarget[i] = classDictionary.index(dataTarget[i])
    return (dataTarget, classDictionary)

# Translates the attronite values
# Saves the value in attributeDictionary and codes the value
# If is continuous, then 
def translateX(dataX, dataY):
    attributeIsDiscrete = []
    attributeDictionary = []
    for i in range(len(dataX[0])):
        # Check if the attribute is discrete or note
        isDiscrete = True
        for j in range(len(dataX)):
            if isinstance((dataX[j][i]), float):
                isDiscrete = False
                break

        # List every possible values in temporary dictionary
        tempDictionary = []

        # If the data is discrete, encode data
        if (isDiscrete):
            for j in range(len(dataX)):
                if dataX[j][i] not in tempDictionary:
                    tempDictionary.append(dataX[j][i])
                    dataX[j][i] = len(tempDictionary) - 1
                else:
                    dataX[j][i] = tempDictionary.index(dataX[j][i])
        
        # If the data is continuous, change data into splitters
        else:
            dataX = sorted(dataX, key=lambda x:x[i])

            firstIdx = 0
            for j in range(1, len(dataX)):
                # If the Y value is different, then assign splitter
                if dataY[j] != dataY[j - 1]:
                    split = (dataX[j][i] + dataX[j - 1][i]) / 2    
                    for k in range(firstIdx, j):
                        dataX[k][i] = len(tempDictionary)
                    tempDictionary.append(split)
                    firstIdx = j
                
            # Encode the last value
            tempDictionary.append(dataX[len(dataX) - 1][i] + 0.5)
            for j in range(firstIdx, len(dataX)):
                dataX[j][i] = len(tempDictionary) - 1

        attributeDictionary.append(tempDictionary)
        attributeIsDiscrete.append(isDiscrete)
    
    return (dataX, attributeDictionary, attributeIsDiscrete)

         

# Gata data of attributes, target, and their names
dataHead, data = getCSVData("iris.csv")
dataX, dataY = splitXY(data)
dataY, classDictionary = translateY(dataY)
dataX, attributeDictionary, attributeIsDiscrete = translateX(dataX, dataY)

# print(dataX)
# print(dataY)
# print(classDictionary)
print(attributeDictionary)
print(attributeIsDiscrete)