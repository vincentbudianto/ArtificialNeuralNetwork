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

# Create a basic fitying algorithn
# After translating X and Y
def fit(dataX, dataY, dataHead, attributeDictionary, attributeIsDiscrete, classDictionary):
    decisionTree = DecisionTree()

    # Checking current entropy
    currentEntropy = f.entropyFunction(dataY)
    print()
    print("Initial entropy:", currentEntropy)

    # Getting tree through recursive function
    usableAttribute = dataHead[:-1]
    print("Attributes:", usableAttribute)
    result = dataAssessment(dataX, dataY, currentEntropy, dataHead, attributeDictionary, attributeIsDiscrete, classDictionary, usableAttribute)
    print()
    print('Tree Result:')
    result.printTree()

# Data assessment (returns the result decision tree)
def dataAssessment(dataX, dataY, oldEntropy, dataHead, attributeDictionary, attributeIsDiscrete, classDictionary, usableAttribute, oldAttribute = None):
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
        # Start information gain
        informationGain = 0

        # Breakpoints are placed for every integer slip
        splittedClassContainer = []
        splittedTargetContainer = []
        entropyContainer = []

        # Translate usableAttribute to dataHead index
        idx = dataHead.index(usableAttribute[i])

        if attributeIsDiscrete[i]:    
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
        
        else:
            # Breakpoints are assigned in the attributeDictionary
            # Need to find value with the best information gain
            splittedClassContainer = []
            splittedTargetContainer = []
            entropyContainer = []

            # Translate usableAttribute to dataHead index
            idx = dataHead.index(usableAttribute[i])

            # The attributes are already encoded so.....
            for j in range(len(attributeDictionary[idx])):
                smallerClassContainer = []
                smallerTargetContainer = []
                biggerClassContainer = []
                biggerTargetContainer = []

                # Class lower or equal.... <= j and class bigger
                for k in range(len(dataX)):
                    if (data[k][idx]) <= j:
                        smallerClassContainer.append(dataX[k])
                        smallerTargetContainer.append(dataY[k])
                    else:
                        biggerClassContainer.append(dataX[k])
                        biggerTargetContainer.append(dataY[k])
                
                # Check entropy for each class
                smallerEntropy = f.entropyFunction(smallerTargetContainer)
                biggerEntropy = f.entropyFunction(biggerTargetContainer)

                # Get the information gain
                totalEntropy = smallerEntropy * len(smallerClassContainer) / len(dataY) + biggerEntropy * len(biggerClassContainer) / len(dataY)
                tempGain = oldEntropy - totalEntropy

                # If it is the better information gain
                if tempGain > informationGain:
                    informationGain = tempGain
                    splittedClassContainer = [smallerClassContainer, biggerClassContainer]
                    splittedTargetContainer = [smallerTargetContainer, biggerTargetContainer]
                    entropyContainer = [smallerEntropy, biggerEntropy]

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
        usableAttribute.remove(bestAttribute)
        result.setRootValue(bestAttribute)

        # Check index where
        bestIdx = dataHead.index(bestAttribute)
        # bestIdx = np.where(dataHead == bestAttribute)[0][0]

        # Recursively add the next tree node based on this...
        for i in range(len(bestClassContainer)):
            # Set next old attribute
            if (attributeIsDiscrete[bestIdx]):
                nextAttribute = bestAttribute + " = " + str(attributeDictionary[bestIdx][i])
            else:
                if (i == 0):
                    nextAttribute = bestAttribute + " <= " + str(attributeDictionary[bestIdx][i])
                elif (i == len(bestClassContainer) - 1):
                    nextAttribute = bestAttribute + " > " + str(attributeDictionary[bestIdx][i-1])
                else:
                    nextAttribute = str(attributeDictionary[bestIdx][i - 1]) + " < " + bestAttribute + " <= " + str(attributeDictionary[bestIdx][i])
            result.setNodes(dataAssessment(bestClassContainer[i], bestTargetContainer[i], bestEntropy[i], dataHead, attributeDictionary, attributeIsDiscrete, classDictionary, usableAttribute, nextAttribute))

        return result



# Gata data of attributes, target, and their names
dataHead, data = getCSVData("iris.csv")
dataX, dataY = splitXY(data)
dataY, classDictionary = translateY(dataY)
dataX, attributeDictionary, attributeIsDiscrete = translateX(dataX, dataY)
fit(dataX, dataY, dataHead, attributeDictionary, attributeIsDiscrete, classDictionary)

# print(dataX)
# print(dataY)
# print(classDictionary)
# print(attributeDictionary)
# print(attributeIsDiscrete)
# print(f.entropyFunction(dataY))