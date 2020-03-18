# Main program to fit the data read
import pandas as pd
import numpy as np
import Function as f
from DecisionTree import DecisionTree
from collections import defaultdict

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
def dataAssessment(dataX, dataY, oldEntropy, dataHead, attributeDictionary, classDictionary, usableAttribute, oldAttribute = None):
    # Empty result variable
    result = DecisionTree()

    # Insert attribute value
    if (oldAttribute != None):
        result.setAttributeValue(oldAttribute)

    # Checking the number of data in Y and the variety
    tempY = []
    tempYCounter = []
    for i in range(len(dataY)):
        if dataY[i] not in tempY:
            tempY.append(dataY[i])
            tempYCounter.append(1)
        else:
            tempYCounter[tempY.index(dataY[i])] += 1

    # If all examples are positive, Return the single-node tree Root, with label = +
    # If all examples are negative, Return the single-node tree Root, with label = -
    if (len(tempY) == 1):
        result.setRootValue(classDictionary[tempY[0]])
        return result

    # If number of predicting attributes is empty, then Return the single node tree Root,
    # with label = most common value of the target attribute in the examples.
    if len(usableAttribute) == 0:
        # Count most common value
        maxIdx = tempYCounter.index(max(tempYCounter))
        result.setRootValue(classDictionary[dataY[maxIdx]])
        return result

    # Otherwise...
    # Splits into a couple of attributes, find entropy, and recursively update this
    bestInformationGain = 0 # Change this with minimum entropy gain
    bestAttribute = None
    bestClassContainer = []
    bestTargetContainer = []
    bestEntropy = []

    # For each attributes
    for i in range(len(usableAttribute)):
        # Breakpoints are placed for every integer slip
        splittedClassContainer = []
        splittedTargetContainer = []
        entropyContainer = []

        # Translate usableAttribute to dataHead index
        idx = dataHead.tolist().index(usableAttribute[i])

        # For each value of the class in the attributeDictionary => create an array of X
        for j in range(len(attributeDictionary[idx])):
            tempXContainer = []
            tempYContainer = []

            for k in range(len(dataX)):
                if (dataX[k][idx]) == j:
                    tempXContainer.append(dataX[k])
                    tempYContainer.append(dataY[k])

            splittedClassContainer.append(tempXContainer)
            splittedTargetContainer.append(tempYContainer)
            entropyContainer.append(f.entropyFunction(tempYContainer))


        # Find the total entropy
        totalEntropy = 0
        for j in range(len(splittedClassContainer)):
            dataCount = len(splittedClassContainer[j])
            totalEntropy += dataCount / len(dataX) * entropyContainer[j]

        # Get information gain
        informationGain = oldEntropy - totalEntropy

        # Check if information gain is better than the last one
        # If yes, set best entropy, information gain, attribute, class container, and target container
        if informationGain > bestInformationGain:
            bestInformationGain = informationGain
            bestAttribute = usableAttribute[i]
            bestClassContainer = splittedClassContainer
            bestTargetContainer = splittedTargetContainer
            bestEntropy = entropyContainer

    # Know best splitting area for best information gain
    # If no information is gained then
    # Change the zero if the pruning is based on minimum information gain
    if (bestInformationGain == 0):
        # Count most common value
        maxIdx = tempYCounter.index(max(tempYCounter))
        result.setRootValue(classDictionary[tempY[maxIdx]])
        return result
    else:
        # Remove the best attribute
        idx = usableAttribute.tolist().index(bestAttribute)
        usableAttribute = np.delete(usableAttribute, idx)
        result.setRootValue(bestAttribute)

        # Check index where
        bestIdx = dataHead.tolist().index(bestAttribute)
        # bestIdx = np.where(dataHead == bestAttribute)[0][0]

        # Recursively add the next tree node based on this...
        for i in range(len(bestClassContainer)):
            # Set next old attribute
            nextAttribute = bestAttribute + " = " + str(attributeDictionary[bestIdx][i])
            result.setNodes(dataAssessment(bestClassContainer[i], bestTargetContainer[i], bestEntropy[i], dataHead, attributeDictionary, classDictionary, usableAttribute, nextAttribute))

        return result



# Create a basic fitying algorithn
# After translating X and Y
def fit(dataX, dataY, dataHead, attributeDictionary, classDictionary):
    # Checking current entropy
    currentEntropy = f.entropyFunction(dataY)
    print("Initial entropy:", currentEntropy)

    # Testing
    usableAttribute = dataHead[:-1]
    result = dataAssessment(dataX, dataY, currentEntropy, dataHead, attributeDictionary, classDictionary, usableAttribute)
    print()
    print('Tree Result:')
    result.printTree()

    # Finding nodes that is most effective

# Procedure that feels missing values for each attributes in dataX
def feelEmptyData(dataX):
    dataXSample = dataX[0]
    # do for each attribute
    for attrIdx in range(len(dataXSample)):
        attrCountDict = defaultdict(int)
        missingValueRows = []

        # do for each row of the data
        for rowIdx, eachRow in enumerate(dataX):
            cellValue = eachRow[attrIdx]

            # if not missing, put into the counter
            if not pd.isna(cellValue):
                attrCountDict[cellValue] += 1
            else: # if missing put into list that contain index of rows having the missing value
                missingValueRows.append(rowIdx)

        # get most common value for current attr
        mostCommonValueAttr = max(attrCountDict, key=attrCountDict.get)

        # for those missing the value, replace the nan with the most common value
        for missingRowIdx in missingValueRows:
            dataX[missingRowIdx][attrIdx] = mostCommonValueAttr

# Get data and head
# dataHead, data = getCSVData("iris.csv")
dataHead, data = getCSVData("../dataset/iris.csv")
# Get x and y
newX, newY = splitXY(dataHead, data)
feelEmptyData(newX)
newY, classDictionary = translateY(newY)
newX, attributeDictionary = translateX(newX)
fit(newX, newY, dataHead, attributeDictionary, classDictionary)