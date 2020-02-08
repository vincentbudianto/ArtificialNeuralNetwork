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
# Data splitter (splits X and Y (attributes and target attributes))
def splitXY(dataHead, data):
    newX = []
    newY = []
    size = len(dataHead)
    for datum in data:
        newX.append(datum[:-1])
        newY.append(datum[-1])
    return (newX, newY)

# Merging array X and Y (merging attributes array and target array)
def attributeMerger(dataX, dataY):
    for i in range(len(dataX)):
        dataX[i].append(dataY[i])
    return dataX

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

# Coding the X axis
# Returns a dictionary of X and translated value of X
def translateX(dataX):
    attributeDictionary = []
    for i in range(len(dataX[0])):
        # List every attribute in temporary dictionary
        tempDictionary = []
        for j in range(len(dataX)):
            if dataX[j][i] not in tempDictionary:
                tempDictionary.append(dataX[j][i])
                dataX[j][i] = len(tempDictionary) - 1
            else:
                dataX[j][i] = tempDictionary.index(dataX[j][i])
        attributeDictionary.append(tempDictionary)
    return (dataX, np.array(attributeDictionary))


# Data assessment (returns the result decision tree)
def dataAssessment(dataX, dataY, oldEntropy, dataHead, attributeDictionary, usedAttribute = []):
    # Empty result variable
    result = DecisionTree()

    # Checking the number of data in Y and the variety
    tempY = []
    tempYCounter = []
    for i in range(len(dataY)):
        if dataY[i] not in tempY:
            tempY.append(dataY[i])
            tempYCounter.append(1)
        else:
            tempYCounter[tempY.index(dataY[i])] += 1

    # If all examples are negative, Return the single-node tree Root, with label = +
    # If all examples are negative, Return the single-node tree Root, with label = -
    if (len(tempY) == 1):
        result.setRootValue(tempY[0])
        return result

    # If number of predicting attributes is empty, then Return the single node tree Root,
    # with label = most common value of the target attribute in the examples.
    if len(dataX[0]) == 0:
        # Count most common value
        maxIdx = tempYCounter.index(max(tempYCounter))
        result.setRootValue(dataY[maxIdx])
        return result


    # Otherwise...
    # Splits into a couple of attributes, find entropy, and recursively update this
    bestInformationGain = 0 # Change this with minimum entropy gain
    bestAttribute = None
    bestSplitter = None

    # For each attributes
    for i in range(len(dataX[0])):
        # 1. Assign splitters
        # Breakpoints are placed for every integer slip
        for j in range(1, len(attributeDictionary[i])):
            # Breakpoints are the (element value - 0.5)
            bp = j - 0.5

            # Splits data based on classes (more and less than break points)
            smallerClassY = []
            biggerClassY = []

            # Loop through the data
            # Put into the next data according to breakpoint position
            for k in range(len(dataX)):
                if (dataX[k][i] <= bp):
                    smallerClassY.append(dataY[k])
                else:
                    biggerClassY.append(dataY[k])
            
            # Find the entropy of each class
            smallerEntropy = f.entropyFunction(smallerClassY)
            biggerEntropy = f.entropyFunction(biggerClassY)

            # Find the information gain
            informationGain = oldEntropy - (smallerEntropy / len(dataX) * len(smallerClassY) + biggerEntropy / len(dataX) * len(biggerClassY))

            if informationGain > bestInformationGain:
                bestInformationGain = informationGain
                bestAttribute = i
                bestSplitter = j
    
    # Add attribute dictionary used
    print(dataHead[bestAttribute])
    usedAttribute.append((dataHead[bestAttribute] + " = " + str(bestSplitter - 0.5)))
    
    # Know best splitting area for best information gain
    # If no information is gained then
    # Change the zero if the pruning is based on minimum information gain
    if (bestInformationGain == 0):
        # Count most common value
        maxIdx = tempYCounter.index(max(tempYCounter))
        result.setRootValue(dataY[maxIdx])
        return result
    
    # Else, loop for next entropy gain
    smallerClassX = []
    biggerClassX = []
    bp = bestSplitter - 0.5

    for i in range(len(dataX)):
        if (dataX[i][bestAttribute] <= bp):
            smallerClassX.append(dataX[i])
        else:
            biggerClassX.append(dataX[i])

    # Recursively add the next tree node based on this...
    # result.setLeft(dataAssessment(smallerClassX, smallerClassY, smallerEntropy, dataHead, attributeDictionary, usedAttribute))
    # result.setRight(dataAssessment(biggerClassX, biggerClassY, biggerEntropy, dataHead, attributeDictionary, usedAttribute))
    return usedAttribute



# Create a basic fitying algorithn
# After translating X and Y
def fit(dataX, dataY, dataHead, attributeDictionary):
    decisionTree = DecisionTree()

    # Checking current entropy
    currentEntropy = f.entropyFunction(dataY)
    print(currentEntropy)

    # Testing
    print(dataAssessment(dataX, dataY, currentEntropy, dataHead, attributeDictionary))

    # Finding nodes that is most effective



dataHead, data = getCSVData("tennis.csv")
newX, newY = splitXY(dataHead, data)
newY, classDictionary = translateY(newY)
newX, attributeDictionary = translateX(newX)
print(attributeDictionary)
fit(newX, newY, dataHead, attributeDictionary)
