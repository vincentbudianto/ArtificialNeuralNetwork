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
def translateX(data):
    attributeIsDiscrete = []
    attributeDictionary = []
    for i in range(len(data[0]) - 1):
        # Check if the attribute is discrete or note
        isDiscrete = True
        for j in range(len(data)):
            if isinstance((data[j][i]), float):
                isDiscrete = False
                break

        # List every possible values in temporary dictionary
        tempDictionary = []

        # If the data is discrete, encode data
        if (isDiscrete):
            for j in range(len(data)):
                if data[j][i] not in tempDictionary:
                    tempDictionary.append(data[j][i])
                    data[j][i] = len(tempDictionary) - 1
                else:
                    data[j][i] = tempDictionary.index(data[j][i])

        # If the data is continuous, change data into splitters
        else:
            data = sorted(data, key=lambda x:x[i])

            firstIdx = 0
            for j in range(1, len(data)):
                # If the Y value is different, then assign splitter
                if data[j][-1] != data[j - 1][-1] and data[j][i] != data[j - 1][i]:
                    split = (data[j][i] + data[j - 1][i]) / 2
                    for k in range(firstIdx, j):
                        data[k][i] = len(tempDictionary)
                    tempDictionary.append(split)
                    firstIdx = j

            # Encode the last value
            tempDictionary.append(data[len(data) - 1][i] + 0.5)
            for j in range(firstIdx, len(data)):
                data[j][i] = len(tempDictionary) - 1


        attributeDictionary.append(tempDictionary)
        attributeIsDiscrete.append(isDiscrete)

    return (data, attributeDictionary, attributeIsDiscrete)

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
    bestInformationGainRatio = 0 # Change this with minimum entropy gain (if pre-prune wants to be applied)
    bestAttribute = None
    bestClassContainer = []
    bestTargetContainer = []
    bestEntropy = []
    bestSplitted = 0

    # For each attributes
    for i in range(len(usableAttribute)):
        # Start information gain
        informationGainRatio = 0

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
            informationGainRatio = (oldEntropy - totalEntropy) / f.splitInInformation(splittedClassContainer, len(dataX))

        else:
            # Breakpoints are assigned in the attributeDictionary
            # Need to find value with the best information gain
            splittedClassContainer = []
            splittedTargetContainer = []
            entropyContainer = []
            tempSplitted = 0

            # Translate usableAttribute to dataHead index
            idx = dataHead.index(usableAttribute[i])

            # The attributes are already encoded so.....
            for j in range(len(attributeDictionary[idx]) - 1):
                smallerClassContainer = []
                smallerTargetContainer = []
                biggerClassContainer = []
                biggerTargetContainer = []

                # Class lower or equal.... <= j and class bigger
                for k in range(len(dataX) - 1):
                    if dataX[k][idx] <= j:
                        smallerClassContainer.append(dataX[k])
                        smallerTargetContainer.append(dataY[k])
                    else:
                        biggerClassContainer.append(dataX[k])
                        biggerTargetContainer.append(dataY[k])

                # Check entropy for each class
                smallerEntropy = f.entropyFunction(smallerTargetContainer)
                biggerEntropy = f.entropyFunction(biggerTargetContainer)

                if (len(smallerClassContainer) != 0 and len(biggerClassContainer) != 0):
                    # Get the information gain
                    totalEntropy = smallerEntropy * len(smallerClassContainer) / len(dataY) + biggerEntropy * len(biggerClassContainer) / len(dataY)
                    tempGainRatio = (oldEntropy - totalEntropy) / f.splitInInformation([smallerClassContainer, biggerClassContainer], len(dataX))

                    # If it is the better information gain
                    if tempGainRatio > informationGainRatio:
                        informationGainRatio = tempGainRatio
                        splittedClassContainer = [smallerClassContainer, biggerClassContainer]
                        splittedTargetContainer = [smallerTargetContainer, biggerTargetContainer]
                        entropyContainer = [smallerEntropy, biggerEntropy]
                        tempSplitted = j
        

        # Check if information gain is better than the last one
        # If yes, set best entropy, information gain, attribute, class container, and target container
        if informationGainRatio > bestInformationGainRatio:
            bestInformationGainRatio = informationGainRatio
            bestAttribute = usableAttribute[i]
            bestClassContainer = splittedClassContainer
            bestTargetContainer = splittedTargetContainer
            bestEntropy = entropyContainer
            if not attributeIsDiscrete[i]:
                bestSplitted = tempSplitted

    # Know best splitting area for best information gain
    # If no information is gained then
    # Change the zero if the pruning is based on minimum information gain
    # print(bestAttribute)
    # if (bestAttribute == "petal.width"):
    #     print(bestClassContainer)
    #     print(bestTargetContainer)
    #     print(bestSplitted)
    #     print(bestInformationGainRatio)
    if (bestInformationGainRatio == 0):
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
                    nextAttribute = bestAttribute + " <= " + str(attributeDictionary[bestIdx][bestSplitted])
                else:
                    nextAttribute = bestAttribute + " > " + str(attributeDictionary[bestIdx][bestSplitted])
            
            # Set new array of attributes
            tempUsableAttribute = usableAttribute[:]
            
            result.setNodes(dataAssessment(bestClassContainer[i], bestTargetContainer[i], bestEntropy[i], dataHead, attributeDictionary, attributeIsDiscrete, classDictionary, tempUsableAttribute, nextAttribute))

        return result



# Gata data of attributes, target, and their names
dataHead, data = getCSVData("tennis.csv")
data, attributeDictionary, attributeIsDiscrete = translateX(data)
dataX, dataY = splitXY(data)
dataY, classDictionary = translateY(dataY)
fit(dataX, dataY, dataHead, attributeDictionary, attributeIsDiscrete, classDictionary)

# print(dataX)
# print(dataY)
# print(classDictionary)
# print(attributeDictionary)
# print(attributeIsDiscrete)
# print(f.entropyFunction(dataY))