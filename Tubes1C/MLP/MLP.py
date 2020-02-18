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
            layers[i].value[0] = 1
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
            lastLayer.delta[i] = lastLayer[i].value * (1 - lastLayer[i].value) * (targetValue[i] - lastLayer[i])

        # Update the value of deltaweight
        for i in range(len(lastLayer.weight)):
            for j in range(len(lastLayer.weigth[i])):
                lastLayer.deltaWeight[i][j] += self.learningRate * lastLayer.delta[i] * self.layers[len(self.layers) - 2].value[k]
        
        # Hidden layers
        for i in range(len(self.layers) - 1, 1, -1):
            # Update the value of the delta
            for j in range(len(self.layers[i].value)):
                totalSigma = 0
                minValue = 1
                if (i == len(self.layers) - 1):
                    minValue = 0
                for k in range(minValue, len(self.layers[i + 1].value)):
                    totalSigma += self.layers[i + 1].weight[k][j] * self.layers[i + 1].delta[k]
                self.layers[i].delta[j] = self.layers[i].value * (1 - self.layers[i].value) * totalSigma
            
            # Update the value of deltaweight
            for j in range(len(self.layers[i].weight)):
                for k in range(len(self.layers[i].weight[j])):
                    self.layers[i].deltaWeight[j][k] += self.learningRate * self.layers[i].delta[j] * self.layers[i - 1].value[k]

    '''
    Give prediction value
    '''
    def predictionValue(self):
        return self.layers[len(self.layers) - 1].value
    
    '''
    One epoch loop
    '''
    def oneEpoch(self, dataX, dataY):
        oldEntropy = cp.deepcopy(self.layers)
        for i in range(dataX):
            self.feedForward(dataX)
            self.backPropagation(dataY)
            self.flush(oldEntropy)
        self.flushDelta()
        
            
            





        

layer0 = Layer(2, 0, 'sigmoid')
layer1 = Layer(3, 1, 'sigmoid')
layer2 = Layer(1, 2, 'sigmoid')
print(layer0)
layers = []
layers.append(layer0)
layers.append(layer1)
layers.append(layer2)

result = MLP(layers, 0.1)
print(result.inputSize)
for i in range(len(result.layers)):
    print(result.layers[i].weight)
    print(result.layers[i].deltaWeight)

result.layers[1].deltaWeight[0][0] += 1
result.layers[2].deltaWeight[0][0] += 1

for i in range(len(result.layers)):
    print(layers[i].weight)
    print(layers[i].deltaWeight)

result.flushDelta()

for i in range(len(result.layers)):
    print(layers[i].weight)
    print(layers[i].deltaWeight)
print(result.learningRate)
