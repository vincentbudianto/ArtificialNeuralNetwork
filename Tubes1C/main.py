'''
Main class
Program runs from here....
'''
from MLP import MLP
from Layer import Layer, ActivationFunction
import pandas as pd
import numpy as np


'''
Model generator
Generates layer to be put in MLP class and return MLP Model

Current layer:
    x
        x   x
    x
GK      x   x
    x
        x   x
    x

Abaikan GK
Layer paling kiri adalah keempat input dari 4 atribut iris.csv
Layer tengah adalah hidden layer
Layer output berfungsi sebagai yang dijelaskan di fungsi outputCheck
'''
def generateModel(learningRate):
    layer0 = Layer(4, 0, 4, ActivationFunction.linear)
    layer1 = Layer(3, 1, 4, ActivationFunction.sigmoid)
    layer2 = Layer(3, 2, 3, ActivationFunction.sigmoid)
    layers = []
    layers.append(layer0)
    layers.append(layer1)
    layers.append(layer2)
    return MLP(layers, learningRate)


'''
Main function
'''
def main():
    # Read data from csv
    data = pd.read_csv("iris.csv")
    predictData = data
    dataHead = list(data.columns)

    # Shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    # Creates data by 10 (indexes of data)
    dataSplitCount = 10
    dataDict = {n: data.iloc[n:n+dataSplitCount, :]
           for n in range(0, len(data), dataSplitCount)}

    # Do data training
    '''
    Create a function that returns the tuple of output node
    '''
    def nodeOutputCheck(str):
        if (str == "Versicolor"):
            return [0, 0, 1]
        elif (str == "Virginica"):
            return [0, 1, 0]
        elif (str == "Setosa"):
            return [1, 0, 0]
        else:
            return [1, 1, 1]

    model = generateModel(0.05)
    model.learn(dataDict, dataSplitCount, nodeOutputCheck, maxIteration=100, minError=1, divergingMaxCount=10)
    model.predict(predictData, nodeOutputCheck)

    # Test result
    for i in range(len(model.layers)):
        print("Layer: {}".format(i))
        print(np.matrix(model.layers[i].weight))

main()



'''
Create the output dictionary
Rule:
1. Output 1 = 1 -> The value is "Veriscolor"
2. Ouptut 2 = 1 -> The value is "Virginica"
3. Both output = 0 -> The value is "Setosa"
'''
def outputCheck(a, b):
    if (a >= 0.5 and b < 0.5):
        return "Veriscolor"
    elif (a < 0.5 and b >= 0.5):
        return "Virginica"
    elif (a < 0.5 and b < 0.5):
        return "Setosa"
    else:
        return None

def nodeOutputCheckExternal(str):
    if (str == "Versicolor"):
        return [0, 0, 1]
    elif (str == "Virginica"):
        return [0, 1, 0]
    elif (str == "Setosa"):
        return [1, 0, 0]
    else:
        return [1, 1, 1]

# Just for test
    # model = MLP(layers, 0.1)
    # print(result.inputSize)

    # for i in range(len(model.layers)):
    #     print(model.layers[i].weight)
    #     print(model.layers[i].deltaWeight)

    # model.layers[1].deltaWeight[0][0] += 1
    # model.layers[2].deltaWeight[0][0] += 1

    # for i in range(len(model.layers)):
    #     print(model.layers[i].weight)
    #     print(model.layers[i].deltaWeight)

    # model.flushDelta()

    # for i in range(len(model.layers)):
    #     print(model.layers[i].weight)
    #     print(model.layers[i].deltaWeight)
    # print(model.learningRate)


