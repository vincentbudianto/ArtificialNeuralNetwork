# from Layer import Layer
from .Layer import Layer
from typing import List
from .Function import sigmoid, mse, crossentropyCount, errorCount, softmax
import numpy as np
import copy as cp

'''
MLP class
MLP class as a whole
- Layers        : Stores list of layers (starting from the starting layer)
- LearningRate  : Learning rate of the class
- Error         : The error function
- InputNodeSize : Size of the input for training

Terdapat beberapa perubahan yang dilakukan dari desain MLP sebelumnya...
Maaf...

Jadi weight disimpan pada node target, bukan pada node source
Misalnya:
Node j -> Node k
Maka value dari Node k = LayerK.Weight[k][j] * LayerJ.Value[j]

Yang belum diimplementasikan:
- Error function (menghitung error untuk setiap mini-batch jika < error, stop proses learning)
- Full learning (bukan 1 x epoch)
- Prediction function (feed forward + cek nilai dari 2 node di layer terakhir + gunakan output check dari main.py)
- Testing akurasi
- Mungkin ada error lain dalam kode ini yang belum terdeteksi :)

Setiap node non-output memiliki bias senilai sigmoid(1)
'''

class MLP:
    '''
    Constructor
    '''
    def __init__(self, layers: List[Layer], learningRate):
        INPUT_LAYER = 0
        self.inputNodeSize = layers[INPUT_LAYER].node_count
        self.layers : List[Layer] = layers
        # self.layers = self.generateWeightsAndBias(layers)
        self.learningRate = learningRate
        self.error = 0
        self.countError = 0
        self.caseNumber = 0

    '''
    Flush after every iteration of data
    Empty the input and output
    '''
    def flush(self, oldLayer):
        for i in range(len(self.layers)):
            self.layers[i].input = []
            self.layers[i].output = []
            self.layers[i].weight = oldLayer[i].weight

    '''
    Flush after every batch
    Add the value of to the delta to the layers
    '''
    def flushDelta(self):
        # Add delta weight ot weight as well as emptying the delta weight
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].weight)):
                for k in range(len(self.layers[i].weight[j])):
                    self.layers[i].weight[j][k] += self.layers[i].deltaWeight[j][k]
                    self.layers[i].deltaWeight[j][k] = 0
        # self.error = 0

    '''
    Feed forward algorithm
    Adds the value from the beginning of the node to reach the output value
    '''
    def feedForward(self, data):
        # First layer
        self.layers[0].getOutputFromInput(data)

        # Hidden layers and output layer
        for layerIdx in range(1, len(self.layers)):
            prevLayerOutput = self.layers[layerIdx-1].getOutput()
            self.layers[layerIdx].insertInput(prevLayerOutput)
            self.layers[layerIdx].calculateOutput()

        # Make into softmax
        # self.layers[-1].output = softmax(self.layers[-1].output)

    '''
    Back propagation algorithm
    Adds the value of delta of a node as well as delta weight of the said node
    1. Find out the value of delta attribute
    2. Find delta hidden unit
    3. Find the delta weight
    '''
    def backPropagation(self, targetValue):
        # Output layer
        lastLayer = self.layers[-1]

        tempError = 0
        for i in range(len(lastLayer.output)):
            tempError += errorCount(lastLayer.output[i], targetValue[i])
        tempError = tempError / len(lastLayer.output)
        self.error += tempError

        maxProbIdx = np.argmax(lastLayer.output)
        if targetValue[maxProbIdx] != 1:
            self.countError += 1

        # Update the value of the delta
        # For every value in lastLayer, get the delta by comparing it with the target value
        # delta = output (1 - output) (target - output)
        for i in range(len(lastLayer.output)):
            lastLayer.delta[i] = lastLayer.output[i] * (1 - lastLayer.output[i]) * (targetValue[i] - lastLayer.output[i])

        # Update the output of deltaweight
        # i = list of output nodes (weight lists)
        # j = list of input nodes (weight for a node)
        # for i in range(len(lastLayer.delta)):
            # for j in range(len(lastLayer.input)):
            #     lastLayer.deltaWeight[i][j] += self.learningRate * lastLayer.delta[i] * lastLayer.input[j]
        for nodeIdx in range(len(lastLayer.weight)):
            for prevNodeIdx in range(len(lastLayer.weight[nodeIdx])):
                lastLayer.deltaWeight[nodeIdx][prevNodeIdx] += self.learningRate * lastLayer.delta[nodeIdx] * lastLayer.input[prevNodeIdx]


        # Hidden layers
        # i = loop from second last node to second first node (all hidden layers)
        for i in range(len(self.layers) - 2, 0, -1):
            # Update the output of the delta
            # j = loop for every output in layers -> get the delta
            for j in range(len(self.layers[i].output)):
                totalSigma = 0

                # Loop for every target node
                # k = loop for every delta in the next node (get the sigma of node)
                # We need to plus one j because
                # delta = output (1 - output) (sigma(delta * weight))
                for frontLayerNodeIdx in range(self.layers[i + 1].node_count):
                    totalSigma += self.layers[i + 1].weight[frontLayerNodeIdx][j + 1] * self.layers[i + 1].delta[frontLayerNodeIdx]
                self.layers[i].delta[j] = self.layers[i].output[j] * (1 - self.layers[i].output[j]) * totalSigma


            # Update the output of deltaweight nodes
            # j = number of weight lists in a layer
            for nodeIdx in range(len(self.layers[i].delta)):
                # k = for every node that the weight list points to
                for k in range(len(self.layers[i].input)):
                    self.layers[i].deltaWeight[nodeIdx][k] += self.learningRate * self.layers[i].delta[nodeIdx] * self.layers[i].input[k]


    '''
    Give prediction output
    '''
    def predictionValue(self):
        return self.layers[len(self.layers) - 1].output


    '''
    One epoch loop. Splits data X and dataY by splitter
    Then do a couple of mini-batches
    Data = dataX dan dataY yang dikumpulkan per batch

    '''
    def oneEpoch(self, data, loopJump, outputCheck):
        # Splits the data into mini-batches
        for i in range(0, len(data)):
            # Execute the learning for every data
            for j in range(len(data[i * loopJump])):
                # oldLayer = cp.deepcopy(self.layers)
                # Loop for every data...
                tempData = []
                for k in range(len(data[i * loopJump].iloc[j]) - 1):
                    tempData.append(data[i * loopJump].iloc[j][k])
                self.feedForward(tempData)
                self.backPropagation(outputCheck(data[i * loopJump].iloc[j][len(data[i * loopJump].iloc[j]) - 1]))
                self.caseNumber += 1
            # print("Error per minibatch =", self.error)
            self.flushDelta()


    '''
    Learning function
    Executes the learning algorithm
    Stop function:
    - Reaches max iteration
    - Reaches minimum error
    - Starts to diverge
    '''
    def learn(self, data, loopJump, outputCheck, maxIteration, minError, divergingMaxCount):
        divergeCounter = 0
        lastLayerError = 0

        for i in range(maxIteration):
            # Executes one epoch
            self.oneEpoch(data, loopJump, outputCheck)
            accuracy = (self.caseNumber-self.countError)/self.caseNumber*100;
            # print('Iteration: {}, Wrong Prediction: {}, Total Case: {}, Error: {}, Accuracy: {}%'.format(i+1, self.countError, self.caseNumber, round(self.error, 5), round(accuracy, 2)))

            # Checks if smaller than the minimum error
            if self.error < minError:
                break

            # Checks if it starts to diverge
            if i > 0 and self.error > lastLayerError:
                divergeCounter += 1
            else:
                divergeCounter = 0
            # If the counter > 1
            if divergeCounter >= divergingMaxCount:
                break

            # If doesn't diverge 10 times, and doesn't reach minimum error, resets error
            lastLayerError = self.error
            self.error = 0
            self.countError = 0
            self.caseNumber = 0

    def predict(self, data, nodeOutputCheck):
        rightCount = 0
        for dataIdx in range(len(data)):
            numericList = []
            result = nodeOutputCheck(data.iloc[dataIdx][-1])
            for attrIdx in range(len(data.iloc[dataIdx])-1):
                numericList.append(data.iloc[dataIdx][attrIdx])
            self.feedForward(numericList)
            predict = self.layers[-1].getOutput()
            maxIdx = predict.index(max(predict))
            processedPredict = [0 if i != maxIdx else 1 for i in range(len(predict))]
            # print("Test Prediction:")
            # print(result, processedPredict, result == processedPredict)
            if (result == processedPredict):
                rightCount += 1
        
        return rightCount / len(data)





