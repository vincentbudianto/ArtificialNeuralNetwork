from Layer import Layer
from Function import sigmoid, mse

import copy as cp

'''
MLP class
MLP class as a whole
- Layers        : Stores list of layers (starting from the starting layer)
- LearningRate  : Learning rate of the class
- Error         : The error function
- InputSize     : Size of the input for training

Terdapat beberapa perubahan yang dilakukan dari desain MLP sebelumnya...
Maaf...

Jadi weight disimpan pada node target, bukan pada node source
Misalnya:
Node j -> Node k
Maka value dari Node k = LayerK.Weight[k][j] * LayerJ.Value[j]

Yang belum diimplementasikan:
- Error function (menghitung error untuk setiap mini-batch jika < error, stop proses learning)
- Full learning (bukan 1 x epoch)
- Testing akurasi
- Mungkin ada error lain dalam kode ini yang belum terdeteksi :)

Setiap node non-output memiliki bias senilai sigmoid(1)
'''

class MLP:
    '''
    Constructor
    '''
    def __init__(self, layers, learningRate):
        self.inputSize = layers[0].node_count
        self.layers = self.generateWeightsandBias(layers)
        self.learningRate = learningRate
        self.error = None

    '''
    Register weight baru terhadap layer-layer yang ada
    - Semua weight dari layer dengan nilai lebih dari 1 diberi nilai 0
    - Setiap layer diberi sebuah node dengan value 1 (bias)
    '''
    def generateWeightsandBias(self, layers):
        ## Generate bias, value, and delta
        for i in range(len(layers) - 1):
            layers[i].node_count += 1
            layers[i].value = [0] * layers[i].node_count
            layers[i].value[0] = sigmoid(1)
            layers[i].delta = [0] * layers[i].node_count

        # Last value
        layers[len(layers) - 1].value = [0] * layers[len(layers) - 1].node_count
        layers[len(layers) - 1].delta = [0] * layers[len(layers) - 1].node_count

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
    Return the value of every node in a layer to the value before the last count
    '''
    def flush(self, oldLayer):
        for i in range(1, len(self.layers)):
            for j in range(len(self.layers[i].weight)):
                self.layers[i].weight = oldLayer[i].weight
    
    '''
    Flush after every batch
    Add the value of to the delta to the layers
    '''
    def flushDelta(self):
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
        for i in range(1, len(self.layers[0].value)):
            self.layers[0].value[i] = data[i - 1]

        # Hidden layers
        for i in range(1, len(self.layers) - 1):
            for j in range(1, len(self.layers[i].weight)):
                self.layers[i].value[j] = 0
                for k in range(len(self.layers[i].weight[j])):
                    self.layers[i].value[j] += self.layers[i].weight[j][k] * self.layers[i - 1].value[k]
                self.layers[i].value[j] = sigmoid(self.layers[i].value[j])
        
        # Output layer
        for j in range(len(self.layers[len(self.layers) - 1].weight)):
            self.layers[len(self.layers) - 1].value[j] = 0
            for k in range(len(self.layers[len(self.layers) - 1].weight[j])):
                self.layers[len(self.layers) - 1].value[j] += self.layers[len(self.layers) - 1].weight[j][k] * self.layers[len(self.layers) - 1 - 1].value[k]
            self.layers[len(self.layers) - 1].value[j] = sigmoid(self.layers[len(self.layers) - 1].value[j])
    

    '''
    Back propagation algorithm
    Adds the value of delta of a node as well as delta weight of the said node
    1. Find out the value of delta attribute
    2. Find delta hidden unit
    3. Find the delta weight
    '''
    def backPropagation(self, targetValue):
        # Output layer
        lastLayer = self.layers[len(self.layers) - 1]

        # Update the value of the delta
        for i in range(len(lastLayer.value)):
            lastLayer.delta[i] = lastLayer.value[i] * (1 - lastLayer.value[i]) * (targetValue[i] - lastLayer.value[i])

        # Update the value of deltaweight
        for i in range(len(lastLayer.weight)):
            for j in range(len(lastLayer.weight[i])):
                lastLayer.deltaWeight[i][j] += self.learningRate * lastLayer.delta[i] * self.layers[len(self.layers) - 2].value[j]
        
        # Hidden layers
        for i in range(len(self.layers) - 2, 0, -1):
            # Update the value of the delta
            for j in range(len(self.layers[i].value)):
                totalSigma = 0
                minValue = 1
                if (i == len(self.layers) - 1):
                    minValue = 0
                
                # Loop for every target node
                for k in range(len(self.layers[i + 1].value)):
                    totalSigma += self.layers[i + 1].weight[k][j] * self.layers[i + 1].delta[k]
                self.layers[i].delta[j] = self.layers[i].value[j] * (1 - self.layers[i].value[j]) * totalSigma
                
            
            # Update the value of deltaweight node
            for j in range(len(self.layers[i].weight)):
                for k in range(len(self.layers[i].weight[j])):
                    self.layers[i].deltaWeight[j][k] += self.learningRate * self.layers[i].delta[j] * self.layers[i - 1].value[k]


    '''
    Give prediction value
    '''
    def predictionValue(self):
        return self.layers[len(self.layers) - 1].value
    
    
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


