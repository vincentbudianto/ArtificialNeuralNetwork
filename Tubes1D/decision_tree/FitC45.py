import pandas as pd
import numpy as np
import Function as f
import copy as cp
import random
from DecisionTree import DecisionTree
from collections import defaultdict

# Get data from csv
def getCSVData(fileName):
    data = pd.read_csv(fileName)
    dataCopy = cp.copy(data)
    dataHead = list(dataCopy.columns)
    dataProcessed = np.array(dataCopy)
    return (dataHead, dataProcessed, data)

# Splitters and translaters
# Data splitter (splits X and Y (attributes and target attributes))
def splitXY(data):
    newX = []
    newY = []
    for datum in data:
        newX.append(datum[:-1])
        newY.append(datum[-1])
    return (newX, newY)

# Merging array X and Y (merging attributes array and target array)
def attributeMerger(dataX, dataY):
    data = []
    for i in range(len(dataX)):
        dataX[i].append(dataY[i])
    return dataX

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
def createAttributeandIsDiscrete(data):
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

        # If the data is continuous, change data into splitters
        else:
            data = sorted(data, key=lambda x:x[i])

            firstIdx = 0
            for j in range(1, len(data)):
                # If the Y value is different, then assign splitter
                if data[j][-1] != data[j - 1][-1] and data[j][i] != data[j - 1][i]:
                    split = (data[j][i] + data[j - 1][i]) / 2
                    tempDictionary.append(split)

            tempDictionary.append(data[len(data) - 1][i] + 0.5)

        attributeDictionary.append(tempDictionary)
        attributeIsDiscrete.append(isDiscrete)

    return (attributeDictionary, attributeIsDiscrete)


# Translates the attronite values
# Saves the value in attributeDictionary and codes the value
# If is continuous, then
def translateX(data, attributeDictionary, attributeIsDiscrete):
    # Testing
    for i in range(len(data[0]) - 1):
        if (attributeIsDiscrete[i]):
            for j in range(len(data)):
                data[j][i] = attributeDictionary[i].index(data[j][i])

        # If the data is continuous, change data into splitters
        else:
            for j in range(len(data)):
                # Loop through the nodes
                for k in range(len(attributeDictionary[i])):
                    if (data[j][i] <= attributeDictionary[i][k]):
                        data[j][i] = k
                        break
    return data

# Create a basic fitying algorithn
# After translating X and Y
def fit(dataX, dataY, dataHead, attributeDictionary, attributeIsDiscrete, classDictionary):
    # Checking current entropy
    currentEntropy = f.entropyFunction(dataY)
    # print()
    # print("Initial entropy:", currentEntropy)

    # Getting tree through recursive function
    usableAttribute = dataHead[:-1]
    # print("Attributes:", usableAttribute)
    result = dataAssessment(dataX, dataY, currentEntropy, dataHead, attributeDictionary, attributeIsDiscrete, classDictionary, usableAttribute)
    print('Tree Result:')
    return result

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
        # If the attributes in dataX is the same, then... pass the attribute
        tempContainer = []
        for j in range(len(dataX)):
            if (dataX[j][i] not in tempContainer):
                tempContainer.append(dataX[j][i])
        if (len(tempContainer) <= 1):
            continue


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
            if (len(dataX) > 0):
                totalEntropy = 0
                for j in range(len(splittedClassContainer)):
                    dataCount = len(splittedClassContainer[j])
                    totalEntropy += dataCount / len(dataX) * entropyContainer[j]

                # Get information gain
                informationGainRatio = (oldEntropy - totalEntropy) / f.splitInInformation(splittedClassContainer, len(dataX))
            else:
                informationGainRatio = -1

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
                for k in range(len(dataX)):
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

    if (bestInformationGainRatio == 0):
        # Count most common value
        result.setRootValue(classDictionary[0])

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
            if len(bestClassContainer) == 0:
                # Count most common value
                maxIdx = tempYCounter.index(max(tempYCounter))
                result.setRootValue(classDictionary[dataY[maxIdx]])
                return result
            # Set next old attribute
            if (attributeIsDiscrete[bestIdx]):
                nextAttribute = bestAttribute + " == " + str(attributeDictionary[bestIdx][i])
            else:
                if (i == 0):
                    impurity = [[x,bestTargetContainer[0].count(x)] for x in set(bestTargetContainer[0])]

                    if (len(impurity) > 1):
                        nextAttribute = bestAttribute + " <= " + str(attributeDictionary[bestIdx][bestSplitted]) + " : " + str(impurity[0][1])

                        for j in range (1, len(impurity)):
                            nextAttribute += "/" + str(impurity[j][1])
                    else:
                        nextAttribute = bestAttribute + " <= " + str(attributeDictionary[bestIdx][bestSplitted]) + " : " + str(impurity[0][1])

                else:
                    impurity = [[x,bestTargetContainer[1].count(x)] for x in set(bestTargetContainer[1])]

                    if (len(impurity) > 1):
                        nextAttribute = bestAttribute + " > " + str(attributeDictionary[bestIdx][bestSplitted]) + " : " + str(impurity[0][1])

                        for j in range (1, len(impurity)):
                            nextAttribute += "/" + str(impurity[j][1])
                    else:
                        nextAttribute = bestAttribute + " > " + str(attributeDictionary[bestIdx][bestSplitted]) + " : " + str(impurity[0][1])

            # Set new array of attributes
            tempUsableAttribute = usableAttribute[:]

            result.setNodes(dataAssessment(bestClassContainer[i], bestTargetContainer[i], bestEntropy[i], dataHead, attributeDictionary, attributeIsDiscrete, classDictionary, tempUsableAttribute, nextAttribute))

        return result

# def createRuleSet()

#################
# MAIN FUNCTION #
#################
def prune(dataHead, data, dataRaw):
    # Generate tree from training data
    attributeDictionary, attributeIsDiscrete = createAttributeandIsDiscrete(data)

    # Splits the element based on
    trainingData = dataRaw.sample(frac=0.8)


    testingData = dataRaw.drop(trainingData.index)
    trainingData = np.array(trainingData)
    testingData = np.array(testingData)

    # Parsing training data
    trainingData = translateX(trainingData, attributeDictionary, attributeIsDiscrete)

    # Generate tree from training data
    dataX, dataY = splitXY(trainingData)
    feelEmptyData(dataX)
    dataY, classDictionary = translateY(dataY)
    treeResult = fit(dataX, dataY, dataHead, attributeDictionary, attributeIsDiscrete, classDictionary)

    # Create set of rules
    treeResult.printTree()
    ruleList = treeToRules(treeResult, dataHead)
    # print(ruleList)
    functionRules = []
    for i in range(len(ruleList)):
        functionRules.append(createRule(ruleList[i]))

    # Checking the amount of errors
    errorCount = 0
    for i in range(len(testingData)):
        res = testResult(functionRules, testingData[i], dataHead)
        if (res != testingData[i][-1]):
            errorCount += 1
    
    # Testing pruning....
    # For each rule
    for ruleIdx, rule in enumerate(ruleList):
        tempFunctionRules = cp.copy(functionRules)
        tempRule = cp.copy(rule)

        # For each attr in the rule
        attrIdx = 0
        while attrIdx < (len(tempRule)-1):
            experimentRule = cp.copy(tempRule)
            del experimentRule[attrIdx]

            experimentFunctionRules = cp.copy(tempFunctionRules)
            experimentFunctionRules[ruleIdx] = createRule(experimentRule)

            tempErrorCount = getErrorCount(testingData, experimentFunctionRules, dataHead)

            # Do pruning
            if (tempErrorCount <= errorCount):
                tempRule = experimentRule
                tempFunctionRules = experimentFunctionRules
                errorCount = tempErrorCount
            else: # Don't do pruning
                attrIdx += 1

        ruleList[ruleIdx] = tempRule
        functionRules = tempFunctionRules

    print(ruleList)
    print(errorCount)

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

# Get error count from a set of rules
def getErrorCount(testData, functionRules, dataHead):
    errorCount = 0
    for testDatum in testData:
        res = testResult(functionRules, testDatum, dataHead)
        if (res != testDatum[-1]):
            errorCount +=1
    return errorCount


def treeToRules(treeResult, attributeList, lastRule = []):
    rule = lastRule
    if (treeResult.attribute is not None):
        splittedRule = treeResult.attribute.split()
        splittedRule = splittedRule[0] + " " + splittedRule [1] + " " + splittedRule[2]
        rule.append(splittedRule)
    if treeResult.root in attributeList:
        tempRules = []
        for i in range(len(treeResult.nodes)):
            duplicateRule = cp.copy(rule)
            tempResult = treeToRules(treeResult.nodes[i], attributeList, duplicateRule)
            if isinstance(tempResult[0], list):
                tempRules.extend(tempResult)
            else:
                tempRules.append(tempResult)
        return tempRules
    else:
        rule.append(treeResult.root)
        return rule

def createRule(rules):
    def resultRule(testedData, dataHead):
        isTrue = True
        for i in range(len(rules) - 1):
            tempRule = rules[i].split()
            idx = dataHead.index(tempRule[0])
            toBeEvaluated = "\"" + str(testedData[idx]) + "\" " + tempRule[1] + " \"" + str(tempRule[2]) + "\""
            if (not eval(toBeEvaluated)):
                isTrue = False
                break
        if (isTrue):
            return rules[-1]
        else:
            return -9999
    return resultRule

# find rules that predict the data correctly, if not found return -9999
def testResult(functionRules, testedData, dataHead):
    for i in range(len(functionRules)):
        tempResult = functionRules[i](testedData, dataHead)
        if (tempResult != -9999):
            return tempResult
    return -9999


# Gata data of attributes, target, and their names
dataHead, data, dataRaw = getCSVData("../dataset/iris.csv")
prune(dataHead, data, dataRaw)


# print(dataX)
# print(dataY)
# print(classDictionary)
# print(attributeDictionary)
# print(attributeIsDiscrete)
# print(f.entropyFunction(dataY))