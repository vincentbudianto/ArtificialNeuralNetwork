'''
Main class
Program runs from here....
'''
from .MLP import MLP
from .Layer import Layer, ActivationFunction
import pandas as pd
import numpy as np
import pickle


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
Learning function
'''
def learn(data, dataHead, predictData):
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

    model = None
    try:
        file = open('save_file_ann', 'rb')
        decision = input("ANN model save_file found, do you want to load model (y/n)?")
        if decision[0].lower() == 'y':
            model = pickle.load(file)
            print('ANN model is loaded from file successfully!')
        else:
            print('ANN model is not loaded from file!')
        file.close()
    except IOError:
        print('ANN model save_file not found, initiating new model!')

    if model is None:
        model = generateModel(0.05)
        model.learn(dataDict, dataSplitCount, nodeOutputCheck, maxIteration=50, minError=1, divergingMaxCount=10)

    model.predict(predictData, nodeOutputCheck)

    file = open('save_file_ann', 'wb')
    pickle.dump(model, file)
    file.close()

    print('Saving newest ANN model!')

    return model

'''
Main function
'''
def main():
    # Read data from csv
    data = pd.read_csv("../dataset/iris.csv")
    predictData = data
    dataHead = list(data.columns)

    model = learn(data, dataHead, predictData)

    # Test result
    for i in range(len(model.layers)):
        print("Layer: {}".format(i))
        print(np.matrix(model.layers[i].weight))

# main()



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
