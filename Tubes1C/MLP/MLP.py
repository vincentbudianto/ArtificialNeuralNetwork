# from Layer import Layer
from Layer import Layer
from typing import List
from Function import sigmoid, mse, crossentropyCount

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

    '''
    Register weight baru terhadap layer-layer yang ada
    - Semua weight dari layer dengan nilai lebih dari 1 diberi nilai 0
    - Setiap layer diberi sebuah node dengan value 1 (bias)
    '''
    def generateWeightsAndBias(self, layers):
        ## Generate bias, value, and delta
        for layerIdx in range(len(layers) - 1):
            layers[layerIdx].node_count += 1 #for bias layer
            layers[layerIdx].value = [0] * layers[layerIdx].node_count
            layers[layerIdx].value[0] = sigmoid(1)
            layers[layerIdx].delta = [0] * layers[layerIdx].node_count

        # Last value
        layers[-1].value = [0] * layers[len(layers) - 1].node_count
        layers[-1].delta = [0] * layers[len(layers) - 1].node_count

        ## Generate weight
        for i in range(1, len(layers)):
            weight = []
            deltaWeight = []

            weight_iteration = layers[i].node_count - 1
            if (i == len(layers) - 1):
                weight_iteration += 1
            for j in range(weight_iteration):
                weight.append([0] * layers[i - 1].node_count)
                deltaWeight.append([0] * layers[i - 1].node_count)
            layers[i].weight = weight
            layers[i].deltaWeight = deltaWeight

        return layers

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
            self.layers[layerIdx].calculateOutput(prevLayerOutput)
        

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

        # Cross-entropy
        # crossentropy = -(1/n) * Sigma(yi x log(Oouti) + (1 - yi) x log(1 - Oouti))
        crossentropy = 0
        for i in range(len(lastLayer.output)):
            crossentropy += crossentropyCount(lastLayer.output[i], targetValue[i])
        crossentropy = crossentropy / len(lastLayer.output) * -1
        self.error += crossentropy

        # Update the value of the delta
        # For every value in lastLayer, get the delta by comparing it with the target value
        # delta = output (1 - output) (target - output)
        for i in range(len(lastLayer.output)):
            lastLayer.delta[i] = lastLayer.output[i] * (1 - lastLayer.output[i]) * (targetValue[i - 1] - lastLayer.output[i])

        # Update the output of deltaweight
        # i = list of output nodes (weight lists)
        # j = list of input nodes (weight for a node)
        for i in range(len(lastLayer.delta)):
            for j in range(len(lastLayer.input)):
                lastLayer.deltaWeight[i][j] += self.learningRate * lastLayer.delta[i] * lastLayer.input[j]

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
                for k in range(len(self.layers[i + 1].delta)):
                    totalSigma += self.layers[i + 1].weight[k][j + 1] * self.layers[i + 1].delta[k]
                self.layers[i].delta[j] = self.layers[i].output[j] * (1 - self.layers[i].output[j]) * totalSigma


            # Update the output of deltaweight nodes
            # j = number of weight lists in a layer
            for j in range(len(self.layers[i].delta)):
                # k = for every node that the weight list points to
                for k in range(len(self.layers[i].input)):
                    self.layers[i].deltaWeight[j][k] += self.learningRate * self.layers[i].delta[j] * self.layers[i].input[k]


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
            # Execute the learning for every data batch
            for j in range(len(data[i * loopJump])):
                oldLayer = cp.deepcopy(self.layers)
                # Loop for every data...
                tempData = []
                for k in range(len(data[i * loopJump].iloc[j]) - 1):
                    tempData.append(data[i * loopJump].iloc[j][k])
                self.feedForward(tempData)
                self.backPropagation(outputCheck(data[i * loopJump].iloc[j][len(data[i * loopJump].iloc[j]) - 1]))
                self.flush(oldLayer)
            self.flushDelta()
    

    '''
    Learning function
    Executes the learning algorithm
    Stop function:
    - Reaches max iteration
    - Reaches minimum error
    - Starts to diverge
    '''
    def learn(self, data, loopJump, outputCheck, maxIteration, minError):
        divergeCounter = 0
        lastLayerError = 0

        for i in range(maxIteration):
            # Executes one epoch
            self.oneEpoch(data, loopJump, outputCheck)
            print("Error =", self.error)

            # Checks if smaller than the minimum error
            if self.error < minError:
                break
            
            # Checks if it starts to diverge
            if i > 0 and self.error > lastLayerError:
                divergeCounter += 1
            else:
                divergeCounter = 0
            # If the counter > 1
            if divergeCounter == 10:
                break
            
            # If doesn't diverge 3 times, and doesn't reach minimum error, resets error
            lastLayerError = self.error
            self.error = 0


            
            



