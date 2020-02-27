import enum
from typing import List
import math

'''
Layer class
Handles information
- Node_count        : the number of nodes in the said layer
- LayerNo           : the level of the layer (0 being the starting layer)
- IsStart           : boolean (if the layer is the starting layer)
- Weight            : list of list ([current_layer.node_count][previouslayer.node_count + 1])
- ActivationType    : the type of activation function of the layer (like sigmoid)
- Output            : the output of all node in the layer (list) (after sigmoid)
- Delta             : the value of the delta of a node (list)
- DeltaWeight       : temporary value that will be flushed after every epoch
'''
class ActivationFunction(enum.Enum):
    # def linear(self, value):
    #     return value
    linear = 1
    sigmoid = 2

class Layer:
    def __init__(self, node_count, layerNo, previousLayerNodeCount, activationType: ActivationFunction):
        self.node_count = node_count
        self.previousLayerNodeCount = previousLayerNodeCount
        self.layerNo = layerNo
        self.isStart = (layerNo == 0)
        self.activationType = activationType
        self.weight = [[0 for i in range(previousLayerNodeCount + 1)] for j in range(node_count)] if self.isStart == False else [[1] for j in range(node_count)]
        self.input = []
        self.output = []
        self.delta = [0 for i in range(node_count)]
        self.deltaWeight = [[0 for i in range(previousLayerNodeCount + 1)] for j in range(node_count)] if self.isStart == False else [[1] for j in range(node_count)]

    def insertInput(self, prevLayerOutput : list):
        if len(prevLayerOutput) != self.previousLayerNodeCount:
            raise Exception('Input length and previous layer node count is not the same')
        BIAS = [1]
        self.input += BIAS
        self.input += prevLayerOutput #add bias

    def calculateOutput(self, prevLayerOutput: List[int]):
        activationFunction = self.getActivationFunction()

        for nodeIdx, eachNodeWeights  in enumerate(self.weight): #for each node in this layer
            nodeNet = 0
            for weightIdx, eachWeight  in enumerate(eachNodeWeights):
                nodeNet += eachWeight * self.input[weightIdx]
            nodeOutput = activationFunction(nodeNet)
            self.output.append(nodeOutput)


    def getOutputFromInput(self, prevLayerOutput: List[int]) -> List[int]:
        self.insertInput(prevLayerOutput)
        self.calculateOutput(prevLayerOutput)
        return self.getOutput()

    def getOutput(self) -> List[int]:
        return self.output;

    def getActivationFunction(self):
        if self.activationType == ActivationFunction.linear:
            def linear(value):
                return value
            return linear
        elif self.activationType == ActivationFunction.sigmoid:
            def sigmoid(value):
                return (1 / (1 + math.exp(-value)))
            return sigmoid




# Class LayersGenerator:
#     def __init__(self, request : [int, ActivationFunction]):
#         self.layers = None
