'''
Main class
Program runs from here....
'''
from MLP import MLP
from Layer import Layer
import pandas as pd


'''
Layer generator
Generates layer to be put in MLP class

Current layer:
    x   x
    x   x   x
GK  x   x   x
    x   x

Abaikan GK
Layer paling kiri adalah keempat input dari 4 atribut iris.csv
Layer tengah adalah hidden layer
Layer output berfungsi sebagai yang dijelaskan di fungsi outputCheck
'''
def generateLayers(learningRate):
    layer0 = Layer(4, 0, 'sigmoid')
    layer1 = Layer(4, 1, 'sigmoid')
    layer2 = Layer(2, 2, 'sigmoid')
    print(layer0)
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
    data = pd.read_csv("../iris.csv")
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
        if (str == "Veriscolor"):
            return [1, 0]
        elif (str == "Virginica"):
            return [0, 1]
        elif (str == "Setosa"):
            return [0, 0]
        else:
            return [1, 1]

    model = generateLayers(0.1)
    model.oneEpoch(dataDict, dataSplitCount, nodeOutputCheck)

    # Test result
    for i in range(len(model.layers)):
        print(i)
        print(model.layers[i].weight)

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


    